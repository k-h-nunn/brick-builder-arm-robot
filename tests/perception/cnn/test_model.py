"""The network itself: shapes, size, and that it is a pure function of its input."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from brick_builder.perception.cnn.labels import CLASS_NAMES, N_CLASSES, class_map_from_instance
from brick_builder.perception.cnn.model import OFFSET_SCALE, SceneModelNet, count_macs


def test_forward_shapes_at_the_window_size():
    net = SceneModelNet().eval()
    with torch.no_grad():
        out = net(torch.zeros(2, 3, 256, 256))
    assert out.class_logits.shape == (2, N_CLASSES, 256, 256)
    assert out.centre_logits.shape == (2, 1, 256, 256)
    assert out.offset.shape == (2, 2, 256, 256)
    assert out.depth.shape == (2, 1, 256, 256)


def test_fully_convolutional_accepts_other_multiples_of_32():
    net = SceneModelNet().eval()
    with torch.no_grad():
        out = net(torch.zeros(1, 3, 192, 192))
    assert out.class_logits.shape[-2:] == (192, 192)


def test_parameter_count_is_in_the_intended_band():
    """A laptop model, not a microcontroller one and not a server one: a few million weights."""
    n = SceneModelNet().n_parameters()
    assert 1_000_000 <= n <= 3_000_000, n


def test_cost_is_measured_not_asserted():
    macs = count_macs(SceneModelNet(), 256)
    assert 0.5e9 < macs < 3.0e9, macs


def test_same_input_gives_the_same_output_bit_for_bit():
    torch.manual_seed(0)
    net = SceneModelNet().eval()
    x = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        a, b = net(x), net(x)
    torch.testing.assert_close(a.class_logits, b.class_logits, rtol=0, atol=0)
    torch.testing.assert_close(a.offset, b.offset, rtol=0, atol=0)


def test_offset_scale_is_a_sensible_unit_for_this_camera():
    """Roughly a brick's half-diagonal in a 256-pixel window: the regression target then lives
    near 1, not near 0.01 or near 100."""
    assert 16.0 <= OFFSET_SCALE <= 128.0


def test_class_map_translates_the_instance_convention():
    instance = np.array([[0, 1], [2, 7]], dtype=np.uint8)
    classes = class_map_from_instance(instance)
    assert CLASS_NAMES[classes[0, 0]] == "background"
    assert CLASS_NAMES[classes[0, 1]] == "arm"
    assert CLASS_NAMES[classes[1, 0]] == "baseplate"
    assert CLASS_NAMES[classes[1, 1]] == "brick"


def test_width_multiplier_makes_a_smaller_network():
    small = SceneModelNet(width=0.5).n_parameters()
    full = SceneModelNet(width=1.0).n_parameters()
    assert small < full
    with torch.no_grad():
        out = SceneModelNet(width=0.5).eval()(torch.zeros(1, 3, 64, 64))
    assert out.class_logits.shape == (1, N_CLASSES, 64, 64)


@pytest.mark.parametrize("size", [64, 128])
def test_small_inputs_still_run(size):
    with torch.no_grad():
        out = SceneModelNet().eval()(torch.zeros(1, 3, size, size))
    assert out.depth.shape == (1, 1, size, size)
