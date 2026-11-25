"""Quality checks for preprocessed ``[N, 7]`` face scans (xyz | normals | curvature).

``audit_scans`` measures every file, ``flag_scans`` turns the measurements into
a list of rejection reasons per scan (empty list = keep). Two kinds of rule:

* fixed rules for scans that are broken regardless of the rest of the data
  (unreadable, wrong shape, NaNs, too few or duplicated points, not centred /
  not unit-sphere scaled, invalid normals, constant curvature), and a check that
  the scan faces the same way as the rest of the data (mean normal direction);
* robust outlier rules: values more than ``robust_z`` scaled MADs from the
  dataset median (overall shape of the cloud, point spacing). These catch
  partial scans, non-face clutter and resampling problems.
"""
from __future__ import absolute_import, division, print_function

from multiprocessing import Pool

import numpy as np

DEFAULT_RULES = dict(
    min_points=2048,              # fewer points get resampled with duplicates
    min_finite_frac=0.99,         # fraction of points without NaN / inf
    min_unique_frac=0.8,          # fraction of distinct xyz positions
    max_centroid_offset=0.2,      # data should be zero-mean ...
    radius_range=(0.5, 1.5),      # ... and scaled into the unit sphere
    normal_norm_range=(0.9, 1.1),  # normals should be unit vectors
    max_zero_normal_frac=0.05,
    min_curv_std=1e-6,            # constant curvature = missing curvature
    max_normal_angle_deg=60.0,    # mean normal vs the dataset's typical facing direction
    robust_z=5.0,
    robust_keys=("pca_mid_ratio", "pca_flat_ratio", "nn_dist_median"),
)


def scan_stats(path, nn_sample=1024, seed=0):
    """Measurements of one scan file; ``error`` is non-empty if it cannot be used at all."""
    out = {"path": path, "error": ""}
    try:
        a = np.load(path)
    except Exception as e:  # truncated / empty / not a .npy
        out["error"] = "unreadable ({})".format(type(e).__name__)
        return out
    if a.ndim != 2 or a.shape[1] != 7 or len(a) < 16:
        out["error"] = "bad shape {}".format(a.shape)
        return out
    a = a.astype(np.float64)
    finite = np.isfinite(a).all(axis=1)
    out["num_points"] = int(len(a))
    out["finite_frac"] = float(finite.mean())
    a = a[finite]
    if len(a) < 16:
        out["error"] = "no finite points"
        return out
    xyz, nrm, curv = a[:, :3], a[:, 3:6], a[:, 6]

    out["unique_frac"] = float(len(np.unique(np.round(xyz, 6), axis=0)) / len(xyz))
    out["centroid_offset"] = float(np.linalg.norm(xyz.mean(axis=0)))
    out["max_radius"] = float(np.linalg.norm(xyz, axis=1).max())
    ev = np.sort(np.linalg.eigvalsh(np.cov((xyz - xyz.mean(axis=0)).T)))[::-1]
    out["pca_mid_ratio"] = float(ev[1] / max(ev[0], 1e-12))    # width vs height
    out["pca_flat_ratio"] = float(ev[2] / max(ev[0], 1e-12))   # depth vs height

    n_norm = np.linalg.norm(nrm, axis=1)
    out["normal_norm_mean"] = float(n_norm.mean())
    out["zero_normal_frac"] = float((n_norm < 1e-3).mean())
    for i, k in enumerate("xyz"):
        out["mean_normal_" + k] = float(nrm[:, i].mean())

    out["curv_min"] = float(curv.min())
    out["curv_max"] = float(curv.max())
    out["curv_std"] = float(curv.std())

    from scipy.spatial import cKDTree
    idx = np.random.RandomState(seed).choice(len(xyz), min(nn_sample, len(xyz)), replace=False)
    d, _ = cKDTree(xyz).query(xyz[idx], k=2)
    out["nn_dist_median"] = float(np.median(d[:, 1]))
    return out


def audit_scans(files, workers=4, progress=True):
    """``scan_stats`` for every file (in order), in parallel."""
    it = None
    if workers > 1:
        pool = Pool(workers)
        it = pool.imap(scan_stats, files, chunksize=64)
    else:
        pool, it = None, map(scan_stats, files)
    if progress:
        try:
            from tqdm.auto import tqdm
            it = tqdm(it, total=len(files), desc="auditing scans")
        except ImportError:  # pragma: no cover
            pass
    stats = list(it)
    if pool is not None:
        pool.close()
        pool.join()
    return stats


def flag_scans(stats, rules=None):
    """Rejection reasons for every scan (same order as ``stats``); [] means keep.

    Also adds ``normal_angle_deg`` (angle between the scan's mean normal and the
    dataset's median facing direction) to each usable stats dict.
    """
    r = dict(DEFAULT_RULES, **(rules or {}))
    reasons = [[s["error"]] if s["error"] else [] for s in stats]
    ok = [i for i, s in enumerate(stats) if not s["error"]]
    if not ok:
        return reasons

    def check(i, cond, msg):
        if cond:
            reasons[i].append(msg)

    # typical facing direction of the dataset
    mean_normals = np.array([[stats[i]["mean_normal_" + k] for k in "xyz"] for i in ok])
    facing = np.median(mean_normals, axis=0)
    facing /= max(np.linalg.norm(facing), 1e-12)

    for j, i in enumerate(ok):
        s = stats[i]
        m = mean_normals[j]
        cos = float(m @ facing / max(np.linalg.norm(m), 1e-12))
        s["normal_angle_deg"] = float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
        check(i, s["num_points"] < r["min_points"], "too few points")
        check(i, s["finite_frac"] < r["min_finite_frac"], "NaN / inf values")
        check(i, s["unique_frac"] < r["min_unique_frac"], "duplicated points")
        check(i, s["centroid_offset"] > r["max_centroid_offset"], "not centred")
        check(i, not r["radius_range"][0] <= s["max_radius"] <= r["radius_range"][1], "not unit-sphere scaled")
        check(i, not r["normal_norm_range"][0] <= s["normal_norm_mean"] <= r["normal_norm_range"][1],
              "normals not unit length")
        check(i, s["zero_normal_frac"] > r["max_zero_normal_frac"], "missing normals")
        check(i, s["curv_std"] < r["min_curv_std"], "constant curvature")
        check(i, s["normal_angle_deg"] > r["max_normal_angle_deg"], "faces another direction")

    # robust outliers relative to the rest of the dataset
    for key in r["robust_keys"]:
        v = np.array([stats[i][key] for i in ok])
        med = np.median(v)
        mad = np.median(np.abs(v - med)) * 1.4826
        if mad <= 0:
            continue
        for i, z in zip(ok, (v - med) / mad):
            check(i, abs(z) > r["robust_z"], "{} outlier".format(key))
    return reasons
