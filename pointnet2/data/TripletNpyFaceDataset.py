"""Triplet dataset for preprocessed ``.npy`` face point clouds.

Expected input
--------------
* File naming: ``{image_id}_{subject}.npy``. The subject is taken as the text
  after the *last* underscore, so image ids may themselves contain underscores.
  Files may sit flat in one folder or in nested sub-folders.
* Array shape: ``[N, 7]`` (normally ``N = 4096``) with channels
    0-2 XYZ (zero-mean, scaled into the unit sphere)
    3-5 surface normals
    6   curvature

Because the clouds are already cleaned, centred, unit-sphere scaled and
sampled to a fixed size, no coordinate normalisation is applied here (unlike the
legacy ``.bcnc`` loader, which divided raw millimetre coordinates by 100).
"""
from __future__ import print_function

import os
from collections import defaultdict

import numpy as np
import torch
import torch.utils.data as data
from tqdm import tqdm

NUM_CHANNELS = 7


def parse_subject(path):
    """Return the subject id of a ``{image_id}_{subject}.npy`` file."""
    stem = os.path.splitext(os.path.basename(path))[0]
    if "_" not in stem:
        raise ValueError(
            "File name '{}' does not follow '{{image_id}}_{{subject}}.npy'".format(path)
        )
    return stem.rsplit("_", 1)[1]


def find_npy_files(root):
    files = []
    for dirpath, _, fnames in os.walk(os.path.expanduser(root)):
        for fname in fnames:
            if fname.lower().endswith(".npy"):
                files.append(os.path.join(dirpath, fname))
    return sorted(files)


def group_by_subject(files, min_samples=0):
    """Map subject -> sorted list of files, dropping subjects with < min_samples."""
    groups = defaultdict(list)
    for f in files:
        groups[parse_subject(f)].append(f)
    return {s: sorted(fs) for s, fs in sorted(groups.items()) if len(fs) >= min_samples}


def split_by_subject(groups, val_ratio=0.2, seed=0):
    """Per-subject stratified split into (train_groups, val_groups).

    Every subject appears in both splits so the validation set measures how well
    unseen scans of known identities are matched. At least two samples per
    subject are kept for training (needed for anchor/positive pairs) and, when
    val_ratio > 0, at least two for validation (one gallery + one probe).
    """
    rng = np.random.RandomState(seed)
    train, val = {}, {}
    for subject, files in groups.items():
        files = list(files)
        rng.shuffle(files)
        n_val = int(round(len(files) * val_ratio)) if val_ratio > 0 else 0
        if val_ratio > 0:
            n_val = max(n_val, 2)
        n_val = min(n_val, len(files) - 2)
        if n_val > 0:
            val[subject] = sorted(files[:n_val])
        train[subject] = sorted(files[n_val:])
    return train, val


def load_npy_cloud(path, n_points=None, curvature_power=1.0):
    """Load one ``[N, 7]`` cloud as a float32 tensor.

    * Non-finite values (should already be gone) are zeroed defensively.
    * If ``n_points`` is given and differs from N, points are resampled
      (with replacement only when N < n_points).
    * ``curvature_power`` != 1 applies ``curv ** p`` on channel 6 (the original
      pipeline used p = 0.1 on raw PCL curvature to spread small values, which
      matters because curvature is used as a 4th dimension by the
      curvature-aware farthest point sampling).
    """
    arr = np.load(path).astype(np.float32, copy=False)
    if arr.ndim != 2 or arr.shape[1] != NUM_CHANNELS:
        raise ValueError(
            "{}: expected shape [N, {}], got {}".format(path, NUM_CHANNELS, arr.shape)
        )
    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if n_points is not None and len(arr) != n_points:
        choice = np.random.choice(len(arr), n_points, replace=len(arr) < n_points)
        arr = arr[choice]
    arr = np.array(arr, dtype=np.float32)  # own, writable copy for in-place transforms
    if curvature_power != 1.0:
        arr[:, 6] = np.power(np.clip(arr[:, 6], 0.0, None), curvature_power)
    return torch.from_numpy(arr)


class TripletNpyFaceDataset(data.Dataset):
    """Randomly generated (anchor, positive, negative) triplets of ``.npy`` clouds.

    Parameters
    ----------
    groups : dict subject -> list of file paths (e.g. from ``split_by_subject``)
    n_triplets : number of triplets per epoch (regenerate with ``regenerate()``)
    n_points : points per cloud (4096 for the preprocessed data)
    transforms : optional callable applied to each cloud independently
    curvature_power : see ``load_npy_cloud``
    """

    def __init__(self, groups, n_triplets, n_points=4096, transforms=None,
                 curvature_power=1.0):
        self.subjects = sorted(groups.keys())
        self.class_to_idx = {s: i for i, s in enumerate(self.subjects)}
        self.indices = {self.class_to_idx[s]: list(groups[s]) for s in self.subjects}
        self.n_triplets = n_triplets
        self.npoints = n_points
        self.transforms = transforms
        self.curvature_power = curvature_power

        if len(self.subjects) < 2:
            raise RuntimeError("Need at least 2 subjects for triplet training, found {}"
                               .format(len(self.subjects)))
        if not any(len(v) >= 2 for v in self.indices.values()):
            raise RuntimeError("No subject has >= 2 samples; cannot form positive pairs")

        self.regenerate()

    @property
    def num_classes(self):
        return len(self.subjects)

    def regenerate(self):
        print('Generating {} triplets over {} subjects'.format(self.n_triplets, self.num_classes))
        self.training_triplets = self.generate_triplets(self.indices, self.n_triplets)

    @staticmethod
    def generate_triplets(indices, num_triplets):
        classes = sorted(indices.keys())
        pos_classes = [c for c in classes if len(indices[c]) >= 2]
        triplets = []
        for _ in tqdm(range(num_triplets)):
            c1 = pos_classes[np.random.randint(len(pos_classes))]
            c2 = c1
            while c2 == c1:
                c2 = classes[np.random.randint(len(classes))]
            n1, n2 = np.random.choice(len(indices[c1]), 2, replace=False)
            n3 = np.random.randint(len(indices[c2]))
            triplets.append([indices[c1][n1], indices[c1][n2], indices[c2][n3], c1, c2])
        return triplets

    def _load(self, path):
        cloud = load_npy_cloud(path, self.npoints, self.curvature_power)
        if self.transforms is not None:
            cloud = self.transforms(cloud)
        return cloud

    def __getitem__(self, index):
        a, p, n, c1, c2 = self.training_triplets[index]
        return self._load(a), self._load(p), self._load(n), c1, c2

    def __len__(self):
        return len(self.training_triplets)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Sanity-check an .npy face dataset")
    parser.add_argument("root")
    parser.add_argument("--min_samples", type=int, default=20)
    a = parser.parse_args()
    g = group_by_subject(find_npy_files(a.root), a.min_samples)
    print({s: len(f) for s, f in g.items()})
    tr, va = split_by_subject(g, 0.2)
    ds = TripletNpyFaceDataset(tr, n_triplets=10)
    x = ds[0]
    print(x[0].shape, x[0].dtype, x[3], x[4])
