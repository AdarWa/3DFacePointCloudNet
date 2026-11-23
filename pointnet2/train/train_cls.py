"""Classification pre-training on (synthetic) ``.npy`` face point clouds.

Data: a folder (searched recursively) of ``{subject}_{image_id}.npy`` files,
each ``[N, 7]`` = XYZ | normals | curvature, e.g. produced by
``python -m pointnet2.data.generate_faces``. Every subject is one class.

Example
-------
python -m pointnet2.train.train_cls -data_root Data/synthetic \
    -epochs 10 -batch_size 32 --classifier_type AL

A per-subject split (``-val_ratio``) of each subject's scans is held out to
report classification accuracy. The backbone is saved as
``<checkpoint_dir>/pointnet2_model[_best].pth.tar`` -- pass it to
``train_triplet -model_checkpoint`` to fine-tune on real scans -- and the
classifier head as ``pointnet2_cls[_best].pth.tar``. ``-resume`` continues an
interrupted run from ``pointnet2_model.pth.tar`` / ``pointnet2_cls.pth.tar``.
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
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms as T

from pointnet2.models import Pointnet2FaceClsSSG as Pointnet
from pointnet2.data.TripletNpyFaceDataset import (
    SingleScanNpyDataset,
    find_npy_files,
    group_by_subject,
    groups_to_samples,
    split_by_subject,
)
import pointnet2.data.data_utils as d_utils
from pointnet2.train import layer
from pointnet2.utils.pytorch_utils import load_checkpoint

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

FEAT_DIM = 512


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Classification pre-training on .npy face point clouds",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # data
    parser.add_argument("-data_root", type=str, required=True,
                        help="Folder of {subject}_{image_id}.npy files ([N,7] clouds)")
    parser.add_argument("-num_classes", type=int, default=0,
                        help="Use only the first N subjects (0 = all)")
    parser.add_argument("-min_samples", type=int, default=4,
                        help="Drop subjects with fewer scans than this")
    parser.add_argument("-val_ratio", type=float, default=0.1,
                        help="Fraction of each subject's scans held out for validation")
    parser.add_argument("-split_seed", type=int, default=0)
    parser.add_argument("-num_points", type=int, default=4096,
                        help="Points per cloud (resampled if a file differs)")
    parser.add_argument("-curvature_power", type=float, default=0.1,
                        help="Apply curv**p to channel 6 (0.1 = original pipeline)")
    parser.add_argument("-no_augment", action="store_true",
                        help="Disable the small x/y/z rotations + jitter on training clouds")
    parser.add_argument("-num_workers", type=int, default=4)
    # optimisation
    parser.add_argument("-batch_size", type=int, default=32)
    parser.add_argument("-lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("-weight_decay", type=float, default=1e-5)
    parser.add_argument("-epochs", type=int, default=10)
    parser.add_argument("-lr_milestones", type=str, default="",
                        help="Comma-separated epochs at which the LR is divided by 10 "
                             "(default: 50%%, 80%%, 90%% of -epochs)")
    parser.add_argument("-print_freq", type=int, default=50)
    parser.add_argument("-model_checkpoint", type=str, default=None,
                        help="Initialise the backbone from this checkpoint")
    parser.add_argument("-cls_checkpoint", type=str, default=None,
                        help="Initialise the classifier head from this checkpoint")
    parser.add_argument("-resume", action="store_true",
                        help="Resume backbone, head, optimizer and epoch from "
                             "-checkpoint_dir")
    parser.add_argument("-checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("-log_file", type=str, default="./log/train_log.txt")
    # classifier
    parser.add_argument("--classifier_type", type=str, default="AL",
                        choices=["MCP", "AL", "L"],
                        help="MCP: CosFace margin, AL: SphereFace A-Softmax, L: linear")
    return parser.parse_args(argv)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):
    def __init__(self, path):
        d = osp.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        self.f = open(path, "a")

    def __call__(self, msg):
        print(msg)
        self.f.write(msg + "\n")
        self.f.flush()


def seed_worker(worker_id):
    np.random.seed(torch.initial_seed() % 2 ** 32)


def build_classifier(kind, num_classes):
    return {
        "MCP": lambda: layer.MarginCosineProduct(FEAT_DIM, num_classes),
        "AL": lambda: layer.AngleLinear(FEAT_DIM, num_classes),
        "L": lambda: nn.Linear(FEAT_DIM, num_classes, bias=False),
    }[kind]()


def train_logits(classifier, feat, target):
    if isinstance(classifier, nn.Linear):
        return classifier(feat)
    return classifier(feat, target)


def predict_logits(classifier, feat):
    """Margin-free logits for accuracy (cosine for MCP/AL, plain linear for L)."""
    if isinstance(classifier, nn.Linear):
        return classifier(feat)
    return F.linear(F.normalize(feat), F.normalize(classifier.weight))


def train_epoch(loader, model, classifier, criterion, optimizer, epoch, device, log, print_freq):
    batch_time, data_time = AverageMeter(), AverageMeter()
    losses, top1 = AverageMeter(), AverageMeter()
    model.train()
    classifier.train()
    end = time.time()
    for i, (points, target) in enumerate(loader):
        data_time.update(time.time() - end)
        points = points.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True).view(-1)

        feat = model(points)
        loss = criterion(train_logits(classifier, feat, target), target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            acc = (predict_logits(classifier, feat).argmax(-1) == target).float().mean().item()
        losses.update(loss.item(), points.size(0))
        top1.update(acc, points.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if i % print_freq == 0:
            log("Epoch: [{0}][{1}/{2}]\t"
                "Time {bt.val:.3f} ({bt.avg:.3f})\t"
                "Data {dt.val:.3f} ({dt.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                "LR {lr:.2e}".format(epoch, i, len(loader), bt=batch_time, dt=data_time,
                                     loss=losses, top1=top1,
                                     lr=optimizer.param_groups[0]["lr"]))
    return losses.avg, top1.avg


def validate(loader, model, classifier, criterion, device, log):
    losses, top1 = AverageMeter(), AverageMeter()
    model.eval()
    classifier.eval()
    with torch.no_grad():
        for points, target in loader:
            points = points.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True).view(-1)
            feat = model(points)
            logits = predict_logits(classifier, feat)
            if isinstance(classifier, nn.Linear):
                losses.update(criterion(logits, target).item(), points.size(0))
            acc = (logits.argmax(-1) == target).float().mean().item()
            top1.update(acc, points.size(0))
    log("\nValidation: accuracy {:.4f}{} ({} scans)\n".format(
        top1.avg, " | loss {:.4f}".format(losses.avg) if losses.count else "", top1.count))
    return top1.avg


def save(state, path, best_path=None):
    torch.save(state, path)
    if best_path is not None:
        shutil.copyfile(path, best_path)


def describe(log, name, groups):
    counts = [len(v) for v in groups.values()]
    log("{}: {} subjects, {} scans (min {}, max {} per subject)".format(
        name, len(groups), sum(counts), min(counts) if counts else 0,
        max(counts) if counts else 0))


def main(argv=None):
    args = parse_args(argv)
    log = Logger(args.log_file)
    log("Args: {}".format(vars(args)))
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required (the pointnet2 ops are CUDA-only)")
    device = torch.device("cuda")

    # ---- data -------------------------------------------------------------
    groups = group_by_subject(find_npy_files(args.data_root), args.min_samples)
    if not groups:
        raise RuntimeError("No subjects with >= {} .npy files under {}".format(
            args.min_samples, args.data_root))
    if args.num_classes > 0:
        groups = dict(list(groups.items())[:args.num_classes])
    train_groups, val_groups = split_by_subject(groups, args.val_ratio, args.split_seed)
    describe(log, "Train", train_groups)
    describe(log, "Val", val_groups)

    train_samples, subjects = groups_to_samples(train_groups)
    label_of = {s: i for i, s in enumerate(subjects)}
    val_samples = [(p, label_of[s]) for s, files in sorted(val_groups.items()) for p in files]
    num_classes = len(subjects)

    augment = None if args.no_augment else T.Compose([
        d_utils.PointcloudRotate(axis=np.array([1, 0, 0])),
        d_utils.PointcloudRotate(axis=np.array([0, 1, 0])),
        d_utils.PointcloudRotate(axis=np.array([0, 0, 1])),
        d_utils.PointcloudJitter(std=0.002),
    ])
    train_set = SingleScanNpyDataset(train_samples, args.num_points, args.curvature_power,
                                     transforms=augment)
    val_set = SingleScanNpyDataset(val_samples, args.num_points, args.curvature_power)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              drop_last=True, worker_init_fn=seed_worker)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    if len(train_loader) == 0:
        raise RuntimeError("Fewer training scans ({}) than -batch_size ({})".format(
            len(train_set), args.batch_size))

    # ---- model ------------------------------------------------------------
    model = Pointnet(input_channels=3, use_xyz=True).to(device)
    classifier = build_classifier(args.classifier_type, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        [{"params": model.parameters()}, {"params": classifier.parameters()}],
        lr=args.lr, weight_decay=args.weight_decay)

    if args.lr_milestones:
        milestones = [int(m) for m in args.lr_milestones.split(",") if m.strip()]
    else:
        milestones = sorted({max(1, int(args.epochs * f)) for f in (0.5, 0.8, 0.9)})
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    model_path = osp.join(args.checkpoint_dir, "pointnet2_model.pth.tar")
    model_best = osp.join(args.checkpoint_dir, "pointnet2_model_best.pth.tar")
    cls_path = osp.join(args.checkpoint_dir, "pointnet2_cls.pth.tar")
    cls_best = osp.join(args.checkpoint_dir, "pointnet2_cls_best.pth.tar")

    start_epoch, best_acc = 0, -1.0
    if args.resume:
        if not osp.isfile(model_path):
            raise FileNotFoundError("-resume: {} not found".format(model_path))
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if ckpt.get("scheduler_state"):
            scheduler.load_state_dict(ckpt["scheduler_state"])
        load_checkpoint(classifier, None, cls_path)
        start_epoch = ckpt["epoch"] + 1
        best_acc = ckpt.get("best_acc", -1.0)
        log("Resumed at epoch {} (best val acc {:.4f})".format(start_epoch, best_acc))
    else:
        if args.model_checkpoint and load_checkpoint(model, None, args.model_checkpoint) is None:
            raise FileNotFoundError(args.model_checkpoint)
        if args.cls_checkpoint and load_checkpoint(classifier, None, args.cls_checkpoint) is None:
            raise FileNotFoundError(args.cls_checkpoint)

    log("{} classes, {} train / {} val scans, {} batches/epoch, LR milestones {}".format(
        num_classes, len(train_set), len(val_set), len(train_loader), milestones))

    for epoch in range(start_epoch, args.epochs):
        train_epoch(train_loader, model, classifier, criterion, optimizer, epoch,
                    device, log, args.print_freq)
        scheduler.step()
        acc = validate(val_loader, model, classifier, criterion, device, log) \
            if len(val_set) else 0.0
        is_best = acc > best_acc
        best_acc = max(best_acc, acc)
        common = {"epoch": epoch, "it": epoch, "best_prec": acc, "best_acc": best_acc,
                  "num_classes": num_classes, "classifier_type": args.classifier_type,
                  "subjects": subjects}
        save(dict(common, model_state=model.state_dict(),
                  optimizer_state=optimizer.state_dict(),
                  scheduler_state=scheduler.state_dict()),
             model_path, model_best if is_best else None)
        save(dict(common, model_state=classifier.state_dict(), optimizer_state=None),
             cls_path, cls_best if is_best else None)
        log("Saved {}{}".format(model_path, " (new best)" if is_best else ""))
    log("Finished. Best validation accuracy {:.4f}; backbone: {}".format(best_acc, model_best))


if __name__ == "__main__":
    main()
