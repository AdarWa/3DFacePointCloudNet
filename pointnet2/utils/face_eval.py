"""Losses, embedding extraction and face-recognition metrics.

All distances / similarities are computed on L2-normalised embeddings.
"""
from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics as skm
from torch.utils.data import DataLoader

from pointnet2.data.TripletNpyFaceDataset import SingleScanNpyDataset


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
def triplet_loss(anchor, positive, negative, margin=0.3, active_only=False):
    """Triplet loss on given (anchor, positive, negative) embeddings.

    With ``active_only`` the loss is averaged over the triplets that violate the
    margin only, so already-satisfied (easy) triplets don't dilute the gradient.
    Returns ``(loss, stats)`` where ``stats`` has the mean anchor-positive and
    anchor-negative distances and the fraction of triplets violating the margin.
    """
    a, p, n = (F.normalize(x, p=2, dim=1) for x in (anchor, positive, negative))
    d_ap = (a - p).norm(dim=1)
    d_an = (a - n).norm(dim=1)
    losses = F.relu(d_ap - d_an + margin)
    stats = {"d_ap": d_ap.mean().item(), "d_an": d_an.mean().item(),
             "active": (losses > 0).float().mean().item()}
    if active_only:
        active = losses > 0
        return (losses[active].mean() if active.any() else losses.sum() * 0.0), stats
    return losses.mean(), stats


def batch_hard_triplet_loss(embeddings, labels, margin=0.3):
    """Online batch-hard triplet loss.

    For every anchor in the batch: hardest positive = furthest same-label scan
    (excluding itself), hardest negative = closest other-label scan. Distances
    are Euclidean on L2-normalised embeddings, so they lie in [0, 2].

    Returns ``(loss, num_hard)``; the loss is averaged over the anchors that
    violate the margin only, so easy anchors don't dilute the gradient.
    """
    embeddings = F.normalize(embeddings, p=2, dim=1)
    # ||a - b||^2 = 2 - 2 cos(a, b) for unit vectors. The clamp keeps sqrt's
    # gradient finite for identical scans (e.g. duplicates drawn with replacement).
    sim = embeddings @ embeddings.t()
    dist = torch.sqrt(torch.clamp(2.0 - 2.0 * sim, min=1e-12))

    labels = labels.view(-1, 1)
    same_id = labels.eq(labels.t())
    eye = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)

    pos_mask = same_id & ~eye
    hardest_pos = dist.masked_fill(~pos_mask, -1.0).max(dim=1).values
    hardest_neg = dist.masked_fill(same_id, 1e6).min(dim=1).values

    valid = pos_mask.any(dim=1) & (~same_id).any(dim=1)
    losses = F.relu(hardest_pos - hardest_neg + margin) * valid
    active = losses > 0
    num_hard = int(active.sum().item())

    loss = losses[active].mean() if num_hard > 0 else losses.sum() * 0.0
    return loss, num_hard


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_embeddings(model, files, num_points=4096, curvature_power=0.1,
                       batch_size=64, num_workers=4, device="cuda"):
    """Embed ``.npy`` clouds; returns an L2-normalised float32 array [len(files), D]."""
    dataset = SingleScanNpyDataset([(f, 0) for f in files], num_points, curvature_power)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    was_training = model.training
    model.eval()
    feats = []
    for points, _ in loader:
        feats.append(F.normalize(model(points.to(device, non_blocking=True)), dim=1).cpu())
    model.train(was_training)
    return torch.cat(feats).numpy().astype(np.float32)


def gallery_probe_split(groups):
    """First (sorted) scan of each subject -> gallery, the other scans -> probes.

    Returns ``(gallery_files, gallery_labels, probe_files, probe_labels)``.
    """
    g_files, g_lab, p_files, p_lab = [], [], [], []
    for subject, files in sorted(groups.items()):
        if len(files) < 2:
            continue
        g_files.append(files[0])
        g_lab.append(subject)
        p_files.extend(files[1:])
        p_lab.extend([subject] * (len(files) - 1))
    return g_files, g_lab, p_files, p_lab


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def verification_metrics(g_feat, g_lab, p_feat, p_lab, fars=(1e-3, 1e-2), max_rank=10):
    """Identification + verification metrics for a gallery / probe protocol.

    ``g_feat`` / ``p_feat`` must be L2-normalised. Returns ``(metrics, curves)``:

    * metrics: ``rank1``, ``auc``, ``eer`` and ``tar@far=<far>`` for each far
    * curves:  ``fpr``, ``tpr`` (ROC), ``cmc`` (rank-1..max_rank identification
      rate), ``genuine`` / ``impostor`` similarity scores
    """
    return similarity_metrics(g_feat @ p_feat.T, g_lab, p_lab, fars, max_rank)


def similarity_metrics(sim, g_lab, p_lab, fars=(1e-3, 1e-2), max_rank=10):
    """Same as ``verification_metrics`` but from a ``[n_gallery, n_probe]`` similarity
    matrix, e.g. a weighted sum of the similarities of two modalities."""
    g_lab, p_lab = np.asarray(g_lab), np.asarray(p_lab)
    same = g_lab[:, None] == p_lab[None, :]

    # identification: rank of the correct gallery entry for every probe
    order = np.argsort(-sim, axis=0)                    # gallery indices, best first
    hit = g_lab[order] == p_lab[None, :]
    has_match = hit.any(axis=0)
    rank = np.where(has_match, hit.argmax(axis=0), np.iinfo(np.int64).max)
    max_rank = min(max_rank, len(g_lab))
    cmc = np.array([(rank < k).mean() for k in range(1, max_rank + 1)])

    # verification: every gallery/probe pair is a genuine or impostor comparison
    scores, labels = sim.ravel(), same.ravel().astype(np.int64)
    fpr, tpr, _ = skm.roc_curve(labels, scores)
    fnr = 1.0 - tpr
    i_eer = int(np.nanargmin(np.abs(fnr - fpr)))

    metrics = {"rank1": float(cmc[0]), "auc": float(skm.auc(fpr, tpr)),
               "eer": float((fpr[i_eer] + fnr[i_eer]) / 2)}
    for far in fars:
        i = max(int(np.searchsorted(fpr, far, side="right")) - 1, 0)
        metrics["tar@far={:g}".format(far)] = float(tpr[i])
    curves = {"fpr": fpr, "tpr": tpr, "cmc": cmc,
              "genuine": sim[same], "impostor": sim[~same]}
    return metrics, curves
