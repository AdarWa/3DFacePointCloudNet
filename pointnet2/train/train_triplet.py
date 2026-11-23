"""Triplet fine-tuning on preprocessed ``.npy`` face point clouds.

Data: a folder (searched recursively) of ``{subject}_{image_id}.npy`` files,
each a ``[4096, 7]`` array = XYZ (centred, unit sphere) | normals | curvature.

Example
-------
python -m pointnet2.train.train_triplet \
    -data_root /path/to/npy_faces \
    -model_checkpoint checkpoints/20191028_1000cls_model_best.pth.tar \
    -P 8 -K 4 -batches_per_epoch 500 -lr 5e-5 --margin 0.3

Training uses P x K batches (P subjects, K scans each) and online batch-hard
triplet mining: one forward pass per batch, and each scan is trained against its
hardest positive (furthest same-subject scan) and hardest negative (closest
other-subject scan) in the batch, on L2-normalised embeddings.

A held-out, per-subject split of the data (``-val_ratio``) is used for
validation unless ``-val_root`` points at a separate folder. Validation uses
one gallery scan per subject and all other scans as probes, and reports rank-1
identification, ROC AUC and TAR @ FAR=1e-3.
"""
from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import argparse
import os
import os.path as osp
import shutil
import time

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn import metrics

from pointnet2.models import Pointnet2FaceClsSSG as Pointnet
from pointnet2.data.TripletNpyFaceDataset import (
    SingleScanNpyDataset,
    PKSampler,
    groups_to_samples,
    find_npy_files,
    group_by_subject,
    split_by_subject,
    load_npy_cloud,
)
from pointnet2.utils.pytorch_utils import load_checkpoint
from pointnet2.utils.face_eval import batch_hard_triplet_loss

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

FEAT_DIM = 512


def parse_args():
    parser = argparse.ArgumentParser(
        description="Arguments for triplet training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # data
    parser.add_argument("-data_root", type=str, required=True,
                        help="Folder of {subject}_{image_id}.npy files ([N,7] clouds)")
    parser.add_argument("-val_root", type=str, default=None,
                        help="Optional separate folder of .npy files for validation. "
                             "If omitted, a per-subject split of -data_root is used.")
    parser.add_argument("-val_ratio", type=float, default=0.2,
                        help="Fraction of each subject's scans held out for validation")
    parser.add_argument("-min_samples", type=int, default=20,
                        help="Drop subjects with fewer samples than this")
    parser.add_argument("-split_seed", type=int, default=0, help="Seed for train/val split")
    parser.add_argument("-num_points", type=int, default=4096,
                        help="Points per cloud (resampled if a file differs)")
    parser.add_argument("-curvature_power", type=float, default=0.1,
                        help="Apply curv**p to channel 6 (0.1 matches the original "
                             "pipeline / pretrained checkpoint; use 1.0 to leave as is)")
    parser.add_argument("-augment", action="store_true",
                        help="Small random yaw rotation + XYZ jitter on training clouds")
    parser.add_argument("-num_workers", type=int, default=4)
    # optimisation
    parser.add_argument("-P", type=int, default=8, help="Subjects per batch")
    parser.add_argument("-K", type=int, default=4,
                        help="Scans per subject per batch (with replacement if a "
                             "subject has fewer)")
    parser.add_argument("-batches_per_epoch", type=int, default=500)
    parser.add_argument("-weight_decay", type=float, default=1e-5, help="L2 regularization coeff")
    parser.add_argument("-lr", type=float, default=5e-5, help="Initial learning rate")
    parser.add_argument("-epochs", type=int, default=20, help="Number of epochs to train for")
    parser.add_argument("-print_freq", type=int, default=20, help="Iterations between log lines")
    parser.add_argument("-model_checkpoint", type=str, default=None,
                        help="Checkpoint (.pth.tar) to initialise the network from")
    parser.add_argument("-resume_optimizer", action="store_true",
                        help="Also restore optimizer state from -model_checkpoint")
    parser.add_argument("-checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("-log_file", type=str, default="./log/triplet_log.txt")
    # triplet loss
    parser.add_argument('--margin', type=float, default=0.3, metavar='MARGIN',
                        help='margin of the batch-hard triplet loss (Euclidean '
                             'distance on L2-normalised embeddings)')
    return parser.parse_args()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def log(msg):
    print(msg)
    log_f.write(msg + '\n')
    log_f.flush()


def adjust_learning_rate(optimizer, epoch):
    if epoch in [int(args.epochs * 0.3), int(args.epochs * 0.6), int(args.epochs * 0.9)]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1


def train(train_loader, model, optimizer, epoch, margin):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    hard_frac = AverageMeter()

    model.train()

    end = time.time()
    for i, (data, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)
        data = data.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        embeddings = model(data)  # one forward pass for all P*K scans
        loss, num_hard = batch_hard_triplet_loss(embeddings, labels, margin=margin)

        if num_hard > 0:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losses.update(loss.item(), data.size(0))
        hard_frac.update(num_hard / data.size(0), data.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            log('Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Hard {hard}/{bs} (avg {hard_avg:.1%})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, hard=num_hard,
                    bs=data.size(0), hard_avg=hard_frac.avg))


def seed_worker(worker_id):
    # DataLoader seeds torch/random per worker but not NumPy; without this,
    # forked workers would draw identical augmentations / resampling.
    np.random.seed(torch.initial_seed() % 2 ** 32)


def extract_features(model, file_list):
    feats = torch.zeros((len(file_list), FEAT_DIM))
    for i, fname in enumerate(file_list):
        cloud = load_npy_cloud(fname, args.num_points, args.curvature_power)
        cloud = cloud.unsqueeze(0).contiguous().to("cuda", non_blocking=True)
        feats[i] = model(cloud).squeeze(0).cpu()
    return feats


def validate(model, gfile_list, gclass_list, pfile_list, pclass_list):
    model.eval()
    with torch.no_grad():
        g_feature = extract_features(model, gfile_list)
        p_feature = extract_features(model, pfile_list)

    g_feature = torch.nn.functional.normalize(g_feature, dim=1)
    p_feature = torch.nn.functional.normalize(p_feature, dim=1)
    sim = torch.mm(g_feature, p_feature.t()).numpy()  # [n_gallery, n_probe]

    gclass = np.asarray(gclass_list)
    pclass = np.asarray(pclass_list)
    label = (gclass[:, None] == pclass[None, :]).astype(np.int64) * 2 - 1

    top1 = float(np.mean(gclass[np.argmax(sim, axis=0)] == pclass))
    fpr, tpr, _ = metrics.roc_curve(label.reshape(-1), sim.reshape(-1), pos_label=1)
    auc = metrics.auc(fpr, tpr)
    i = int(np.searchsorted(fpr, 1e-3, side='left'))
    i = min(i, len(fpr) - 1)

    log('\nValidation: top1 {:.4f} | auc {:.4f} | tpr {:.4f} @ fpr {:.4f} '
        '(gallery {}, probe {})\n'.format(top1, auc, tpr[i], fpr[i],
                                          len(gfile_list), len(pfile_list)))
    return top1


def make_gallery_probe(groups):
    """First (sorted) scan of each subject -> gallery, the rest -> probe."""
    g_files, g_cls, p_files, p_cls = [], [], [], []
    for subject, files in sorted(groups.items()):
        if len(files) < 2:
            continue
        g_files.append(files[0])
        g_cls.append(subject)
        p_files.extend(files[1:])
        p_cls.extend([subject] * (len(files) - 1))
    return g_files, g_cls, p_files, p_cls


def checkpoint_state(model=None, optimizer=None, best_prec=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.DataParallel):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    return {
        'epoch': epoch,
        'it': it,
        'best_prec': best_prec,
        'model_state': model_state,
        'optimizer_state': optim_state
    }


def save_checkpoint(state, is_best, filename='checkpoint', bestname='model_best'):
    filename = '{}.pth.tar'.format(filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '{}.pth.tar'.format(bestname))


def describe(name, groups):
    counts = [len(v) for v in groups.values()]
    log('{}: {} subjects, {} scans (min {}, max {} per subject)'.format(
        name, len(groups), sum(counts), min(counts) if counts else 0,
        max(counts) if counts else 0))


if __name__ == "__main__":
    args = parse_args()
    log_dir = osp.dirname(args.log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    log_f = open(args.log_file, 'w')

    # ---- data -------------------------------------------------------------
    all_groups = group_by_subject(find_npy_files(args.data_root), args.min_samples)
    if not all_groups:
        raise RuntimeError("No subjects with >= {} .npy files found under {}".format(
            args.min_samples, args.data_root))
    if args.val_root:
        train_groups = all_groups
        val_groups = group_by_subject(find_npy_files(args.val_root), 2)
    else:
        train_groups, val_groups = split_by_subject(all_groups, args.val_ratio, args.split_seed)
    describe('Train', train_groups)
    describe('Val', val_groups)

    train_samples, _ = groups_to_samples(train_groups)
    train_dataset = SingleScanNpyDataset(
        train_samples,
        num_points=args.num_points,
        curvature_power=args.curvature_power,
        augment=args.augment,
    )
    pk_sampler = PKSampler(
        labels=train_dataset.labels,
        p=args.P,
        k=args.K,
        batches_per_epoch=args.batches_per_epoch,
    )
    log('PK batches: P={} subjects x K={} scans = {} per batch, {} batches/epoch'.format(
        pk_sampler.p, pk_sampler.k, pk_sampler.batch_size, len(pk_sampler)))
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=pk_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
    )

    eg_files, eg_cls, ep_files, ep_cls = make_gallery_probe(val_groups)
    do_val = len(eg_files) > 0 and len(ep_files) > 0
    if not do_val:
        log('No validation data: checkpoints will be saved every epoch without "best" tracking.')

    # ---- model ------------------------------------------------------------
    # 7 channels in: xyz -> coordinates, normals (3) -> point features,
    # curvature -> curvature-aware farthest point sampling.
    model = Pointnet(input_channels=3, use_xyz=True)
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.model_checkpoint is not None:
        if load_checkpoint(model, optimizer if args.resume_optimizer else None,
                           args.model_checkpoint) is None:
            raise FileNotFoundError(args.model_checkpoint)
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_name = osp.join(args.checkpoint_dir, "pointnet2_triplet")
    best_name = osp.join(args.checkpoint_dir, "pointnet2_triplet_best")

    best_top1 = -1.0
    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch)
        train(train_loader, model, optimizer, epoch, args.margin)
        top1 = validate(model, eg_files, eg_cls, ep_files, ep_cls) if do_val else 0.0
        is_best = do_val and top1 > best_top1
        best_top1 = max(best_top1, top1)
        save_checkpoint(
            checkpoint_state(model, optimizer, top1, epoch, epoch),
            is_best,
            filename=checkpoint_name,
            bestname=best_name)
    log_f.close()
