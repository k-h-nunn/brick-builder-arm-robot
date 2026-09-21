"""The network: a compact encoder-decoder with four per-pixel heads.

Nothing here is unusual. It is a U-Net (Ronneberger et al., 2015) with a five-stage strided
encoder, a dilated bottleneck, skip connections into a four-stage decoder, batch normalization and
ReLU throughout, and four 1x1 output heads. Every layer is a plain 3x3 or 1x1 convolution; there
is no attention, no deformable sampling, no recurrence, nothing that a 2016 framework could not
run. That is the point: this is the *conventional* model, the one whose behaviour is easy to
explain to a teammate and easy to port.

The four heads
--------------
``class`` (4 channels)
    background/table, baseplate, arm (links and both jaws), brick. Argmax gives the class map.

``centre`` (1 channel, a logit)
    a heat map that peaks at the centre of each brick. Its peaks are how two bricks are told
    apart; the class map alone cannot, and neither can colour when the two bricks are the same
    colour and touching.

``offset`` (2 channels)
    for each brick pixel, the vector to its own brick's centre, in units of
    ``OFFSET_SCALE`` pixels of the 256x256 window. A brick pixel is assigned to whichever
    predicted centre its own offset points nearest to -- the standard bottom-up grouping of
    Panoptic-DeepLab (Cheng et al., 2020), which needs no proposals, no region-of-interest
    pooling and no non-maximum suppression over boxes.

``depth`` (1 channel)
    relative depth, normalized per frame (see `train.py`'s target). Only the *order* of the
    per-slot medians is ever used, which is all the slot contract's ordinal `depth` field needs.

Resolution
----------
The decoder stops at half resolution (128x128) and the four heads' outputs are bilinearly
upsampled to 256x256. Running the last decoder stage at full resolution would roughly double the
whole network's cost for a boundary that moves by about one pixel; a 1x1 head commutes with
bilinear upsampling, so upsampling the head outputs is exactly equivalent to upsampling the
features and then applying the head, for a quarter of the arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .labels import CLASS_NAMES, N_CLASSES

__all__ = ["SceneModelNet", "CLASS_NAMES", "N_CLASSES", "OFFSET_SCALE", "HeadOutputs",
           "count_macs", "pack_weights", "unpack_weights"]

OFFSET_SCALE = 64.0
"""The offset head's unit, in pixels of the 256x256 window. A 2x4 DUPLO brick is roughly 40-90
pixels across in these windows, so a half-diagonal of about 0.7 units keeps the regression target
comfortably inside a range a `tanh`-free linear head predicts well."""


@dataclass
class HeadOutputs:
    """What one forward pass returns, all at the input's own resolution."""

    class_logits: torch.Tensor   # [B, 4, H, W]
    centre_logits: torch.Tensor  # [B, 1, H, W]
    offset: torch.Tensor         # [B, 2, H, W], units of OFFSET_SCALE pixels
    depth: torch.Tensor          # [B, 1, H, W], normalized relative depth


def _conv(cin: int, cout: int, stride: int = 1, dilation: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride=stride, padding=dilation, dilation=dilation, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class _Stage(nn.Module):
    """One encoder stage: a strided convolution followed by a plain one."""

    def __init__(self, cin: int, cout: int, stride: int = 2):
        super().__init__()
        self.a = _conv(cin, cout, stride=stride)
        self.b = _conv(cout, cout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.b(self.a(x))


class _Up(nn.Module):
    """One decoder stage: upsample, concatenate the matching encoder features, two convolutions."""

    def __init__(self, cin: int, cskip: int, cout: int):
        super().__init__()
        self.a = _conv(cin + cskip, cout)
        self.b = _conv(cout, cout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.b(self.a(torch.cat([x, skip], dim=1)))


class SceneModelNet(nn.Module):
    """The whole network. `width` scales every channel count, for a smaller or larger variant.

    A forward pass takes `[B, 3, H, W]` float in `[0, 1]` and returns `HeadOutputs` at `[H, W]`.
    `H`/`W` need only be multiples of 32; the network is fully convolutional, which is what lets
    training run on 160x160 crops of the 256x256 windows and inference run on the full window.
    """

    def __init__(self, width: float = 1.0, n_classes: int = N_CLASSES):
        super().__init__()

        def c(n: int) -> int:
            return max(8, int(round(n * width / 8.0)) * 8)

        w1, w2, w3, w4, w5 = c(16), c(32), c(64), c(128), c(192)
        d4, d3, d2, d1 = c(128), c(96), c(48), c(24)

        self.stem = _conv(3, w1, stride=2)                       # 1/2
        self.enc2 = _Stage(w1, w2)                               # 1/4
        self.enc3 = _Stage(w2, w3)                               # 1/8
        self.enc4 = _Stage(w3, w4)                               # 1/16
        self.enc5 = _Stage(w4, w5)                               # 1/32
        self.bottleneck = nn.Sequential(_conv(w5, w5, dilation=2), _conv(w5, w5, dilation=4))
        self.dec4 = _Up(w5, w4, d4)                              # 1/16
        self.dec3 = _Up(d4, w3, d3)                              # 1/8
        self.dec2 = _Up(d3, w2, d2)                              # 1/4
        self.dec1 = _Up(d2, w1, d1)                              # 1/2

        self.head_class = nn.Conv2d(d1, n_classes, 1)
        self.head_centre = nn.Conv2d(d1, 1, 1)
        self.head_offset = nn.Conv2d(d1, 2, 1)
        self.head_depth = nn.Conv2d(d1, 1, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # A centre heat map is almost all background; starting its bias low stops the first few
        # hundred steps being spent unlearning "everything is a centre".
        nn.init.constant_(self.head_centre.bias, -4.0)

    def forward(self, x: torch.Tensor) -> HeadOutputs:
        size = x.shape[-2:]
        s1 = self.stem(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        s5 = self.bottleneck(self.enc5(s4))
        y = self.dec4(s5, s4)
        y = self.dec3(y, s3)
        y = self.dec2(y, s2)
        y = self.dec1(y, s1)

        def up(t: torch.Tensor) -> torch.Tensor:
            return F.interpolate(t, size=size, mode="bilinear", align_corners=False)

        return HeadOutputs(
            class_logits=up(self.head_class(y)),
            centre_logits=up(self.head_centre(y)),
            offset=up(self.head_offset(y)),
            depth=up(self.head_depth(y)),
        )

    # -- bookkeeping ---------------------------------------------------------------------------

    def n_parameters(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))


def count_macs(model: SceneModelNet, size: int = 256) -> int:
    """Multiply-accumulates for one forward pass at `size` x `size`, counted by hooking every
    convolution -- a measurement of this model as built, not a figure copied from a paper."""
    total = 0

    def hook(module, inputs, output):
        nonlocal total
        out_elems = output.shape[-1] * output.shape[-2] * output.shape[1]
        total += out_elems * module.in_channels // module.groups * module.kernel_size[0] * module.kernel_size[1]

    handles = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, nn.Conv2d)]
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, size, size))
    model.train(was_training)
    for h in handles:
        h.remove()
    return int(total)


def pack_weights(state: dict) -> dict:
    """Store convolution weights at half precision, everything else untouched.

    A 2.4-million-weight network is 9.5 MB of `float32`, which is an awkward thing to keep in a
    repository. Halving the convolution kernels takes it to under 5 MB and costs about four
    decimal digits of a weight -- nothing a network whose inputs are 8-bit pixels can tell apart.
    Batch-norm statistics stay at full precision: a running variance can be small enough that
    half precision would round it towards zero, and they are only a few thousand numbers anyway.
    All arithmetic stays `float32`; this is a storage format, not a precision mode.
    """
    return {k: (v.half() if v.is_floating_point() and v.ndim >= 2 else v) for k, v in state.items()}


def unpack_weights(state: dict) -> dict:
    """Inverse of `pack_weights`: back to `float32` for the network to run in."""
    return {k: (v.float() if v.dtype == torch.float16 else v) for k, v in state.items()}
