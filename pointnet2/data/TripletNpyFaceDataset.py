"""Triplet dataset for preprocessed ``.npy`` face point clouds.

Expected input
--------------
* File naming: ``{subject}_{image_id}.npy``. The subject is taken as the text
  before the *first* underscore, so image ids may themselves contain underscores.
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
import random
from collections import defaultdict

import numpy as np
import torch
import torch.utils.data as data
from tqdm import tqdm

NUM_CHANNELS = 7


def parse_subject(path):
    """Return the subject id of a ``{subject}_{image_id}.npy`` file."""
    stem = os.path.splitext(os.path.basename(path))[0]
    if "_" not in stem:
        raise ValueError(
            "File name '{}' does not follow '{{subject}}_{{image_id}}.npy'".format(path)
        )
    return stem.split("_", 1)[0]


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


def groups_to_samples(groups):
    """dict subject -> files  =>  ([(path, label_idx), ...], [subject per label_idx])."""
    subjects = sorted(groups.keys())
    samples = [(path, idx) for idx, subject in enumerate(subjects)
               for path in groups[subject]]
    return samples, subjects


class SingleScanNpyDataset(data.Dataset):
    """Single ``.npy`` scans for online (in-batch) triplet mining.

    ``samples`` is a list of ``(file_path, label_idx)``; items are
    ``(points [num_points, 7] float tensor, label_idx)``.

    With ``augment=True`` each scan gets a small random yaw rotation (applied to
    both XYZ and normals) and Gaussian XYZ jitter. ``transforms`` is an optional
    extra callable applied to the ``[num_points, 7]`` tensor afterwards.
    """

    def __init__(self, samples, num_points=4096, curvature_power=0.1, augment=False,
                 max_yaw=0.15, jitter_std=0.005, transforms=None):
        self.samples = list(samples)
        self.num_points = num_points
        self.curvature_power = curvature_power
        self.augment = augment
        self.max_yaw = max_yaw
        self.jitter_std = jitter_std
        self.transforms = transforms

    @property
    def labels(self):
        return [label for _, label in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        pts = load_npy_cloud(path, self.num_points, self.curvature_power)
        if self.augment:
            theta = np.random.uniform(-self.max_yaw, self.max_yaw)
            c, s = np.cos(theta), np.sin(theta)
            rot = torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
                               dtype=pts.dtype)
            pts[:, 0:3] = pts[:, 0:3] @ rot.t()
            pts[:, 3:6] = pts[:, 3:6] @ rot.t()
            pts[:, 0:3] += torch.randn(pts.size(0), 3, dtype=pts.dtype) * self.jitter_std
        if self.transforms is not None:
            pts = self.transforms(pts)
        return pts, label


class PKSampler(data.Sampler):
    """Batch sampler yielding ``P`` random subjects x ``K`` scans each.

    Subjects with fewer than ``K`` scans are sampled with replacement. Use it as
    ``DataLoader(dataset, batch_sampler=PKSampler(...))``.
    """

    def __init__(self, labels, p=8, k=4, batches_per_epoch=500, seed=None):
        self.label_to_indices = defaultdict(list)
        for idx, lbl in enumerate(labels):
            self.label_to_indices[int(lbl)].append(idx)
        self.unique_labels = sorted(self.label_to_indices.keys())
        if len(self.unique_labels) < 2:
            raise ValueError("PKSampler needs at least 2 subjects, got {}"
                             .format(len(self.unique_labels)))
        if p > len(self.unique_labels):
            print("PKSampler: P={} > {} subjects, using P={}".format(
                p, len(self.unique_labels), len(self.unique_labels)))
            p = len(self.unique_labels)
        if k < 2:
            raise ValueError("PKSampler needs K >= 2 to form positive pairs")
        self.p = p
        self.k = k
        self.batches_per_epoch = batches_per_epoch
        self.rng = random.Random(seed)

    @property
    def batch_size(self):
        return self.p * self.k

    def __iter__(self):
        for _ in range(self.batches_per_epoch):
            batch = []
            for lbl in self.rng.sample(self.unique_labels, self.p):
                idxs = self.label_to_indices[lbl]
                if len(idxs) >= self.k:
                    batch.extend(self.rng.sample(idxs, self.k))
                else:
                    batch.extend(self.rng.choices(idxs, k=self.k))
            yield batch

    def __len__(self):
        return self.batches_per_epoch


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Sanity-check an .npy face dataset")
    parser.add_argument("root")
    parser.add_argument("--min_samples", type=int, default=20)
    a = parser.parse_args()
    g = group_by_subject(find_npy_files(a.root), a.min_samples)
    print({s: len(f) for s, f in g.items()})
    tr, va = split_by_subject(g, 0.2)
    samples, subjects = groups_to_samples(tr)
    ds = SingleScanNpyDataset(samples)
    batch = next(iter(PKSampler(ds.labels, p=4, k=4, batches_per_epoch=1)))
    print('PK batch labels:', [ds.samples[i][1] for i in batch])
    x, y = ds[batch[0]]
    print(x.shape, x.dtype, subjects[y])
