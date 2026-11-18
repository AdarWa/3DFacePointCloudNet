"""Minimal, dependency-free replacements for the parts of ``etw_pytorch_utils``
used by this project (``SharedMLP``, ``Seq``/``FC`` and
``feature_dropout_no_scaling``).

The module layout (and therefore the ``state_dict`` key names) matches
etw_pytorch_utils 1.1.1, so checkpoints trained with it -- e.g.
``checkpoints/20191028_1000cls_model_best.pth.tar`` -- load unchanged:

    SharedMLP : layer{i}.conv.weight, layer{i}.normlayer.bn.{weight,bias,running_*}
    Seq.fc    : {i}.fc.weight, {i}.fc.bias
"""
from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn


class _BatchNorm(nn.Sequential):
    """Wraps a BatchNorm layer under the name ``bn`` (key: ``...normlayer.bn.*``)."""

    def __init__(self, in_size, batch_norm=nn.BatchNorm2d):
        super(_BatchNorm, self).__init__()
        self.add_module("bn", batch_norm(in_size))
        nn.init.constant_(self.bn.weight, 1.0)
        nn.init.constant_(self.bn.bias, 0.0)


class Conv2d(nn.Sequential):
    """conv -> (batch norm) -> (activation), 1x1 by default."""

    def __init__(self, in_size, out_size, kernel_size=(1, 1), stride=(1, 1),
                 padding=(0, 0), activation=None, bn=False,
                 init=nn.init.kaiming_normal_, bias=True):
        super(Conv2d, self).__init__()
        bias = bias and not bn
        conv = nn.Conv2d(in_size, out_size, kernel_size=kernel_size,
                         stride=stride, padding=padding, bias=bias)
        if init is not None:
            init(conv.weight)
        if bias:
            nn.init.constant_(conv.bias, 0.0)
        self.add_module("conv", conv)
        if bn:
            self.add_module("normlayer", _BatchNorm(out_size, nn.BatchNorm2d))
        if activation is not None:
            self.add_module("activation", activation)


class SharedMLP(nn.Sequential):
    """Stack of 1x1 Conv2d layers applied point-wise: args = [c_in, c1, c2, ...]."""

    def __init__(self, args, bn=False, activation=None):
        super(SharedMLP, self).__init__()
        if activation is None:
            activation = nn.ReLU(inplace=True)
        for i in range(len(args) - 1):
            self.add_module(
                "layer{}".format(i),
                Conv2d(args[i], args[i + 1], bn=bn, activation=activation),
            )


class FC(nn.Sequential):
    """linear -> (batch norm) -> (activation)."""

    def __init__(self, in_size, out_size, activation=None, bn=False, init=None):
        super(FC, self).__init__()
        fc = nn.Linear(in_size, out_size, bias=not bn)
        if init is not None:
            init(fc.weight)
        if not bn:
            nn.init.constant_(fc.bias, 0.0)
        self.add_module("fc", fc)
        if bn:
            self.add_module("normlayer", _BatchNorm(out_size, nn.BatchNorm1d))
        if activation is not None:
            self.add_module("activation", activation)


_DEFAULT = object()


class Seq(nn.Sequential):
    """Chainable builder: ``Seq(512).fc(256, bn=True).dropout(0.5).fc(10, activation=None)``."""

    def __init__(self, input_channels):
        super(Seq, self).__init__()
        self.count = 0
        self.current_channels = input_channels

    def _add(self, module):
        self.add_module(str(self.count), module)
        self.count += 1
        return self

    def fc(self, size, activation=_DEFAULT, bn=False, init=None):
        if activation is _DEFAULT:
            activation = nn.ReLU(inplace=True)
        self._add(FC(self.current_channels, size, activation=activation, bn=bn, init=init))
        self.current_channels = size
        return self

    def dropout(self, p=0.5):
        return self._add(nn.Dropout(p=p))


def feature_dropout_no_scaling(x, p=0.5, train=False, inplace=False):
    """Drop whole feature channels with probability ``p`` *without* rescaling the
    survivors. ``x`` is (B, C, ...); one mask entry per (sample, channel)."""
    if not train or p <= 0:
        return x
    mask_shape = (x.size(0), x.size(1)) + (1,) * (x.dim() - 2)
    mask = torch.empty(mask_shape, device=x.device, dtype=x.dtype).bernoulli_(1.0 - float(p))
    return x.mul_(mask) if inplace else x * mask


def load_checkpoint(model=None, optimizer=None, filename="checkpoint"):
    """Load a checkpoint saved as ``{'model_state', 'optimizer_state', 'it',
    'epoch', 'best_prec'}``. ``filename`` may include or omit ``.pth.tar``.

    Returns ``(it, epoch, best_prec)`` or ``None`` if the file does not exist.
    """
    import os

    if not filename.endswith(".pth.tar") and not os.path.isfile(filename):
        filename = "{}.pth.tar".format(filename)
    if not os.path.isfile(filename):
        print("==> Checkpoint '{}' not found".format(filename))
        return None

    print("==> Loading from checkpoint '{}'".format(filename))
    try:
        ckpt = torch.load(filename, map_location="cpu", weights_only=False)
    except TypeError:  # torch < 1.13 has no weights_only argument
        ckpt = torch.load(filename, map_location="cpu")
    if not isinstance(ckpt, dict) or "model_state" not in ckpt:
        ckpt = {"model_state": ckpt}  # a bare state_dict
    if model is not None and ckpt.get("model_state") is not None:
        model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and ckpt.get("optimizer_state") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    print("==> Done")
    return ckpt.get("it", 0), ckpt.get("epoch"), ckpt.get("best_prec")
