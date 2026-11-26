"""Turn raw depth-camera point clouds into the ``[N, 7]`` format the network expects.

Raw files are named ``Raw_{image_id}_{subject_id}_{timestamp}_points.npy`` with a
matching ``..._color.png``. A points file may be

* organized: ``[H, W, 3+]``, one 3D point per pixel;
* flat and organized: ``[H * W, 3+]`` where H, W is the colour image size;
* unorganized: ``[N, 3+]``.

Only the first three columns (x, y, z in camera coordinates) are used. Invalid
points (NaN / inf / all zero) are dropped. The pipeline (``raw_to_cloud``):

1. crop the face: if the points are organized and a 2D face box is given (from the
   colour image), keep the points inside the box; then keep the points within
   ``face_radius_mm`` of the nose tip (the valid point closest to the camera);
2. remove isolated outlier points;
3. estimate normals (oriented towards the camera) and PCL-style curvature
   (surface variation) from the ``curvature_k`` nearest neighbours;
4. rotate from camera coordinates into the face frame used for training
   (nose towards +z, y up);
5. centre, scale into the unit sphere and sample ``num_points`` points.
"""
from __future__ import absolute_import, division, print_function

import os
import re
import zlib
from collections import defaultdict

import numpy as np

RAW_NAME = re.compile(r"^Raw_(?P<image_id>[^_]+)_(?P<subject>[^_]+)_(?P<timestamp>.+)_"
                      r"(?P<kind>color\.png|points\.npy)$")

# Camera conventions -> rotation into the face frame (x right, y up, nose towards +z).
CAMERA_TO_FACE = {
    "opencv": np.diag([1.0, -1.0, -1.0]),   # x right, y down, z forward (RealSense, Kinect, OpenCV)
    "opengl": np.eye(3),                    # x right, y up, z towards the viewer
}


# ---------------------------------------------------------------------------
# Dataset index
# ---------------------------------------------------------------------------
def parse_raw_name(path):
    """``{"image_id", "subject", "timestamp", "kind"}`` for a raw file name, else None."""
    m = RAW_NAME.match(os.path.basename(path))
    return m.groupdict() if m else None


def find_raw_pairs(root):
    """Index colour/points pairs under ``root`` (searched recursively).

    Returns ``(pairs, unpaired)``: ``pairs`` is a list of dicts with keys ``key``,
    ``image_id``, ``subject``, ``timestamp``, ``color`` and ``points``, sorted by
    subject then timestamp; ``unpaired`` lists files without their partner.
    """
    found = defaultdict(dict)
    for dirpath, _, fnames in os.walk(os.path.expanduser(root)):
        for fname in fnames:
            info = parse_raw_name(fname)
            if info is None:
                continue
            key = "Raw_{image_id}_{subject}_{timestamp}".format(**info)
            entry = found[key]
            entry.update(key=key, image_id=info["image_id"], subject=info["subject"],
                         timestamp=info["timestamp"])
            entry["color" if info["kind"] == "color.png" else "points"] = os.path.join(dirpath, fname)
    pairs = [e for e in found.values() if "color" in e and "points" in e]
    unpaired = [e.get("color", e.get("points")) for e in found.values()
                if not ("color" in e and "points" in e)]
    pairs.sort(key=lambda e: (e["subject"], e["timestamp"], e["image_id"]))
    return pairs, sorted(unpaired)


# ---------------------------------------------------------------------------
# Loading and cropping
# ---------------------------------------------------------------------------
def load_raw_points(path, image_size=None):
    """Valid xyz points of a raw scan, plus their pixel coordinates when organized.

    ``image_size`` is the colour image's ``(width, height)``; a flat array with
    exactly width * height rows is treated as organized at that size.
    Returns ``(xyz [M, 3] float64, pixels [M, 2] (col, row) or None, grid (W, H) or None)``.
    """
    a = np.load(path, allow_pickle=False)
    grid = None
    if a.ndim == 3 and a.shape[2] >= 3:
        h, w = a.shape[:2]
        a = a.reshape(-1, a.shape[2])
        grid = (w, h)
    elif a.ndim == 2 and a.shape[1] >= 3:
        if image_size is not None and len(a) == image_size[0] * image_size[1]:
            grid = tuple(image_size)
    else:
        raise ValueError("unsupported points array shape {}".format(a.shape))
    xyz = a[:, :3].astype(np.float64)
    valid = np.isfinite(xyz).all(axis=1) & (np.abs(xyz).sum(axis=1) > 0)
    pixels = None
    if grid is not None:
        idx = np.arange(len(xyz))
        pixels = np.stack([idx % grid[0], idx // grid[0]], axis=1)[valid]
    return xyz[valid], pixels, grid


def to_metres(xyz):
    """Scale factor that converts the cloud to metres (raw clouds come in m or mm)."""
    return 0.001 if np.median(np.linalg.norm(xyz, axis=1)) > 20 else 1.0


def remove_outliers(xyz, k=8, std_ratio=2.0):
    """Statistical outlier removal: drop points whose mean neighbour distance is unusually large."""
    from scipy.spatial import cKDTree
    if len(xyz) <= k:
        return xyz
    d, _ = cKDTree(xyz).query(xyz, k=k + 1)
    mean_d = d[:, 1:].mean(axis=1)
    return xyz[mean_d <= mean_d.mean() + std_ratio * mean_d.std()]


def crop_face(xyz, pixels=None, box=None, image_size=None, grid=None, face_radius_mm=95.0,
              box_margin=0.15):
    """Face points (metres, camera coordinates) and the nose tip.

    With organized points and a face ``box`` (x1, y1, x2, y2 in colour-image pixels)
    only points inside the (enlarged) box are considered. The nose tip is the point
    closest to the camera after outlier removal; the face is everything within
    ``face_radius_mm`` of it.
    """
    xyz = xyz * to_metres(xyz)
    if box is not None and pixels is not None:
        sx = grid[0] / float(image_size[0]) if image_size else 1.0
        sy = grid[1] / float(image_size[1]) if image_size else 1.0
        x1, y1, x2, y2 = box
        mx, my = (x2 - x1) * box_margin, (y2 - y1) * box_margin
        inside = ((pixels[:, 0] >= (x1 - mx) * sx) & (pixels[:, 0] <= (x2 + mx) * sx) &
                  (pixels[:, 1] >= (y1 - my) * sy) & (pixels[:, 1] <= (y2 + my) * sy))
        xyz = xyz[inside]
    if len(xyz) < 100:
        raise ValueError("too few valid points near the face ({})".format(len(xyz)))
    clean = remove_outliers(xyz)
    nose = clean[np.argmin(np.linalg.norm(clean, axis=1))]   # closest point to the camera
    face = clean[np.linalg.norm(clean - nose, axis=1) <= face_radius_mm / 1000.0]
    return face, nose


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def normals_curvature(xyz, k=20, viewpoint=(0.0, 0.0, 0.0)):
    """Normals (oriented towards ``viewpoint``) and curvature l0/(l0+l1+l2) from k neighbours."""
    from scipy.spatial import cKDTree
    k = min(k, len(xyz))
    _, nbrs = cKDTree(xyz).query(xyz, k=k)
    pts = xyz[nbrs] - xyz[nbrs].mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", pts, pts) / k
    evals, evecs = np.linalg.eigh(cov)                   # ascending eigenvalues
    normals = evecs[:, :, 0]
    flip = ((np.asarray(viewpoint) - xyz) * normals).sum(axis=1) < 0
    normals[flip] *= -1
    curvature = evals[:, 0] / np.maximum(evals.sum(axis=1), 1e-12)
    return normals, curvature


def raw_to_cloud(points_path, image_size=None, box=None, num_points=2048, face_radius_mm=95.0,
                 curvature_k=20, camera="opencv", seed=None):
    """Full preprocessing of one raw scan -> float32 ``[num_points, 7]``. Raises ValueError on failure."""
    xyz, pixels, grid = load_raw_points(points_path, image_size)
    face, _ = crop_face(xyz, pixels, box, image_size, grid, face_radius_mm)
    if len(face) < num_points // 2:
        raise ValueError("face crop has only {} points (< {})".format(len(face), num_points // 2))
    normals, curvature = normals_curvature(face, curvature_k)   # camera at the origin
    rot = CAMERA_TO_FACE[camera]
    face, normals = face @ rot.T, normals @ rot.T
    face = face - face.mean(axis=0)
    face = face / np.linalg.norm(face, axis=1).max()
    if seed is None:
        seed = zlib.crc32(os.path.basename(points_path).encode())
    rng = np.random.RandomState(seed)
    choice = rng.choice(len(face), num_points, replace=len(face) < num_points)
    cloud = np.concatenate([face, normals, curvature[:, None]], axis=1)[choice]
    return cloud.astype(np.float32)
