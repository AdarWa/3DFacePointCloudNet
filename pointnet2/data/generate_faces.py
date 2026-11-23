"""Generate synthetic training faces from the Basel Face Model 2017 (pure Python).

This replaces the MATLAB (``Data/GenerateTrainData.m``) + PCL steps of the
original pipeline. For every synthetic subject a random shape coefficient vector
is drawn, and for every scan of that subject a random expression; the resulting
mesh gets per-vertex normals and PCL-style curvature, is subsampled to a fixed
number of points and written as::

    <out>/<subject>/<subject>_<scan>.npy      float32 [num_points, 7]
        0-2 XYZ   (normalised, see --normalize)
        3-5 unit normals
        6   curvature = surface variation l0 / (l0 + l1 + l2)  (raw, like PCL)

This is exactly the format read by ``TripletNpyFaceDataset`` and therefore by
both ``train_cls`` and ``train_triplet``.

Example
-------
python -m pointnet2.data.generate_faces \
    --bfm Data/model2017-1_face12_nomouth.h5 --out Data/synthetic \
    --num_subjects 1000 --scans_per_subject 20 --workers 8

Generation is deterministic per subject (seed = --seed + subject index) and
resumable: subjects whose folder is already complete are skipped.
"""
from __future__ import absolute_import, division, print_function

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

# BFM2017 vertex index of the nose tip (8157 in MATLAB's 1-based indexing).
BFM_NOSE_TIP = 8156


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Generate synthetic BFM2017 face point clouds ([N,7] .npy)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--bfm", required=True,
                   help="BFM2017 h5 file, e.g. model2017-1_face12_nomouth.h5 "
                        "(https://faces.dmi.unibas.ch/bfm/bfm2017.html)")
    p.add_argument("--out", required=True, help="Output folder")
    p.add_argument("--num_subjects", type=int, default=1000,
                   help="Number of synthetic identities (classes)")
    p.add_argument("--scans_per_subject", type=int, default=20,
                   help="Scans (expressions) per identity")
    p.add_argument("--start_subject", type=int, default=0,
                   help="Index of the first subject (to add more subjects later)")
    p.add_argument("--num_points", type=int, default=4096, help="Points per scan")
    p.add_argument("--shape_components", type=int, default=199,
                   help="Number of shape PCA components used")
    p.add_argument("--expression_components", type=int, default=100,
                   help="Number of expression PCA components used")
    p.add_argument("--shape_std", type=float, default=1.0,
                   help="Std of shape coefficients (in units of PCA std)")
    p.add_argument("--expression_std", type=float, default=1.0,
                   help="Std of expression coefficients (0 disables expressions)")
    p.add_argument("--neutral_first", action="store_true",
                   help="Make scan 000 of each subject a neutral expression")
    p.add_argument("--noise_std", type=float, default=0.0,
                   help="Gaussian vertex noise in mm added before normals/curvature "
                        "(simulates scanner noise)")
    p.add_argument("--curvature_k", type=int, default=20,
                   help="Neighbourhood size for normals-free curvature estimation")
    p.add_argument("--normalize", choices=["unit_sphere", "nose_mm100"],
                   default="unit_sphere",
                   help="unit_sphere: zero-mean, max radius 1 (matches the real-data "
                        ".npy format used by train_triplet). nose_mm100: nose tip at "
                        "the origin and mm / 100 (the original paper's .bcnc format)")
    p.add_argument("--nose_index", type=int, default=BFM_NOSE_TIP,
                   help="Nose-tip vertex index (0-based) for --normalize nose_mm100")
    p.add_argument("--prefix", default="syn",
                   help="Subject id prefix (must not contain '_')")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--preview", type=int, default=1,
                   help="Write this many example scans as .ply to <out>/_preview")
    args = p.parse_args(argv)
    if "_" in args.prefix:
        p.error("--prefix must not contain '_' (it separates subject and scan id)")
    if args.scans_per_subject < 2:
        p.error("--scans_per_subject must be >= 2")
    return args


# ---------------------------------------------------------------------------
# Morphable model
# ---------------------------------------------------------------------------
class BFM(object):
    """Shape + expression PCA model and triangle list loaded from a BFM2017 h5."""

    def __init__(self, path, shape_components=199, expression_components=100):
        import h5py
        with h5py.File(path, "r") as f:
            def model(group, n):
                g = f[group]["model"]
                mean = np.asarray(g["mean"], dtype=np.float64)
                basis = np.asarray(g["pcaBasis"], dtype=np.float32)
                var = np.asarray(g["pcaVariance"], dtype=np.float64)
                if basis.shape[0] != mean.shape[0]:  # stored transposed
                    basis = basis.T
                n = min(n, basis.shape[1])
                # Pre-scale the basis by the PCA std so coefficients are N(0, 1).
                basis = basis[:, :n] * np.sqrt(var[:n]).astype(np.float32)
                return mean, np.ascontiguousarray(basis)

            self.shape_mean, self.shape_basis = model("shape", shape_components)
            if expression_components > 0 and "expression" in f:
                self.exp_mean, self.exp_basis = model("expression", expression_components)
            else:
                self.exp_mean = np.zeros_like(self.shape_mean)
                self.exp_basis = np.zeros((self.shape_mean.shape[0], 0), np.float32)
            cells = np.asarray(f["shape"]["representer"]["cells"])
        if cells.shape[0] == 3 and cells.shape[1] != 3:
            cells = cells.T
        self.triangles = np.ascontiguousarray(cells, dtype=np.int64)
        self.num_vertices = self.shape_mean.shape[0] // 3
        if self.triangles.max() >= self.num_vertices:  # 1-based indices
            self.triangles = self.triangles - 1

    @property
    def n_shape(self):
        return self.shape_basis.shape[1]

    @property
    def n_exp(self):
        return self.exp_basis.shape[1]

    def vertices(self, alpha, beta):
        v = self.shape_mean + self.exp_mean + self.shape_basis @ alpha.astype(np.float32)
        if self.n_exp:
            v = v + self.exp_basis @ beta.astype(np.float32)
        return v.reshape(-1, 3)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def vertex_normals(verts, tris):
    """Area-weighted vertex normals, oriented towards +z (the 'camera')."""
    v0, v1, v2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    n = np.zeros_like(verts)
    for k in range(3):
        np.add.at(n, tris[:, k], fn)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    if n[:, 2].mean() < 0:
        n = -n
    return n


def surface_variation(verts, neighbors):
    """PCL-style curvature l0 / (l0 + l1 + l2) of each vertex's neighbourhood."""
    pts = verts[neighbors]                              # [N, k, 3]
    pts = pts - pts.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", pts, pts) / neighbors.shape[1]
    ev = np.linalg.eigvalsh(cov)                        # ascending
    return ev[:, 0] / np.maximum(ev.sum(axis=1), 1e-12)


def normalize_cloud(xyz, mode, nose=None):
    if mode == "nose_mm100":
        return (xyz - nose) / 100.0
    xyz = xyz - xyz.mean(axis=0)
    return xyz / np.linalg.norm(xyz, axis=1).max()


def write_ply(path, cloud):
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\nelement vertex {}\n".format(len(cloud)))
        f.write("property float x\nproperty float y\nproperty float z\n"
                "property float nx\nproperty float ny\nproperty float nz\n"
                "property float curvature\nend_header\n")
        np.savetxt(f, cloud, fmt="%.6f")


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------
_G = {}


def _init_worker(args):
    from scipy.spatial import cKDTree
    np.seterr(all="ignore")
    bfm = BFM(args.bfm, args.shape_components, args.expression_components)
    # The mesh topology is fixed, so curvature neighbourhoods are computed once
    # on the mean face and reused for every generated face.
    mean = (bfm.shape_mean + bfm.exp_mean).reshape(-1, 3)
    _, nbrs = cKDTree(mean).query(mean, k=args.curvature_k)
    _G.update(args=args, bfm=bfm, nbrs=nbrs)


def subject_name(args, i):
    return "{}{:06d}".format(args.prefix, i)


def _generate_subject(i):
    args, bfm, nbrs = _G["args"], _G["bfm"], _G["nbrs"]
    name = subject_name(args, i)
    folder = os.path.join(args.out, name)
    paths = [os.path.join(folder, "{}_{:03d}.npy".format(name, j))
             for j in range(args.scans_per_subject)]
    if all(os.path.isfile(p) for p in paths):
        return i, 0
    os.makedirs(folder, exist_ok=True)

    rng = np.random.RandomState(args.seed + i)
    alpha = rng.randn(bfm.n_shape) * args.shape_std
    for j, path in enumerate(paths):
        beta = rng.randn(bfm.n_exp) * args.expression_std
        if args.neutral_first and j == 0:
            beta[:] = 0
        choice = rng.choice(bfm.num_vertices, args.num_points,
                            replace=bfm.num_vertices < args.num_points)
        noise = rng.randn(bfm.num_vertices, 3) if args.noise_std > 0 else None
        if os.path.isfile(path):  # keep the RNG stream identical when resuming
            continue
        verts = bfm.vertices(alpha, beta).astype(np.float64)
        if noise is not None:
            verts = verts + noise * args.noise_std
        normals = vertex_normals(verts, bfm.triangles)
        curv = surface_variation(verts, nbrs)
        xyz = normalize_cloud(verts, args.normalize, verts[args.nose_index]
                              if args.normalize == "nose_mm100" else None)
        cloud = np.concatenate([xyz, normals, curv[:, None]], axis=1)[choice]
        cloud = np.nan_to_num(cloud).astype(np.float32)
        tmp = path + ".part"
        with open(tmp, "wb") as fh:
            np.save(fh, cloud)
        os.replace(tmp, path)  # atomic: an interrupted run never leaves half files
    return i, 1


def main(argv=None):
    args = parse_args(argv)
    if not os.path.isfile(args.bfm):
        sys.exit("BFM file '{}' not found. Download model2017-1_face12_nomouth.h5 from "
                 "https://faces.dmi.unibas.ch/bfm/bfm2017.html".format(args.bfm))
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "generation_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    _init_worker(args)
    bfm = _G["bfm"]
    if args.normalize == "nose_mm100" and args.nose_index >= bfm.num_vertices:
        sys.exit("--nose_index {} out of range ({} vertices)".format(
            args.nose_index, bfm.num_vertices))
    print("BFM: {} vertices, {} triangles, {} shape / {} expression components".format(
        bfm.num_vertices, len(bfm.triangles), bfm.n_shape, bfm.n_exp))

    subjects = list(range(args.start_subject, args.start_subject + args.num_subjects))
    total = len(subjects) * args.scans_per_subject
    print("Generating {} subjects x {} scans = {} clouds of {} points into {}".format(
        len(subjects), args.scans_per_subject, total, args.num_points, args.out))

    t0 = time.time()
    done = skipped = 0
    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover
        tqdm = None
    if args.workers > 1:
        pool = Pool(args.workers, initializer=_init_worker, initargs=(args,))
        it = pool.imap_unordered(_generate_subject, subjects, chunksize=4)
    else:
        pool, it = None, map(_generate_subject, subjects)
    if tqdm is not None:
        it = tqdm(it, total=len(subjects), unit="subject")
    for _, generated in it:
        done += generated
        skipped += 1 - generated
    if pool is not None:
        pool.close()
        pool.join()
    print("Done in {:.1f}s ({} subjects generated, {} already complete)".format(
        time.time() - t0, done, skipped))

    if args.preview > 0:
        prev = os.path.join(args.out, "_preview")
        os.makedirs(prev, exist_ok=True)
        name = subject_name(args, subjects[0])
        for j in range(min(args.preview, args.scans_per_subject)):
            src = os.path.join(args.out, name, "{}_{:03d}.npy".format(name, j))
            write_ply(os.path.join(prev, "{}_{:03d}.ply".format(name, j)), np.load(src))
        print("Preview .ply files (open in MeshLab / CloudCompare): {}".format(prev))


def generate(bfm, out, **options):
    """Python API for the command line, e.g. from a notebook::

        generate("Data/model2017-1_face12_nomouth.h5", "Data/synthetic",
                 num_subjects=1000, scans_per_subject=20, neutral_first=True)

    Keyword names are the command-line options without ``--``. A boolean option
    is passed when True and left out when False.
    """
    argv = ["--bfm", str(bfm), "--out", str(out)]
    for key, value in options.items():
        if isinstance(value, bool):
            if value:
                argv.append("--" + key)
        elif value is not None:
            argv += ["--" + key, str(value)]
    main(argv)


if __name__ == "__main__":
    main()
