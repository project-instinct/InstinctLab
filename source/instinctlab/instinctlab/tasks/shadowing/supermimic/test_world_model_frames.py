"""Checks for the heading-frame world-model parameterization."""

import math
import torch

from isaaclab.utils.math import quat_apply, quat_from_angle_axis, quat_mul, yaw_quat

from instinctlab.tasks.shadowing.supermimic.supermimic_imagination import make_state_format
from instinctlab.tasks.shadowing.supermimic.world_model_frames import (
    HeadingFrameParameterization,
    _exp,
    _heading,
    _log,
    _mul,
    _rotate,
)

NUM_LINKS, NUM_JOINTS = 3, 5


def _quat(n, gen):
    q = torch.randn(n, 4, generator=gen, dtype=torch.float64)
    return q / q.norm(dim=-1, keepdim=True)


def _state(batch, gen):
    fmt = make_state_format(NUM_LINKS, NUM_JOINTS)
    parts = {}
    for name, shape in fmt.items():
        if name.endswith("quat_w"):
            parts[name] = _quat(batch * (shape[0] if len(shape) == 2 else 1), gen).reshape(batch, -1)
        else:
            parts[name] = torch.randn(batch, *shape, generator=gen, dtype=torch.float64).reshape(batch, -1)
    return fmt, torch.cat([parts[n] for n in fmt], dim=-1)


def test_quaternion_helpers_match_isaaclab():
    gen = torch.Generator().manual_seed(0)
    a, b = _quat(8, gen), _quat(8, gen)
    v = torch.randn(8, 3, generator=gen, dtype=torch.float64)
    assert torch.allclose(_mul(a, b), quat_mul(a, b), atol=1e-12)
    assert torch.allclose(_rotate(a, v), quat_apply(a, v), atol=1e-12)
    yaw = yaw_quat(a)
    mine = _heading(a)
    assert torch.allclose(torch.where(mine[:, 3:] * yaw[:, 3:] < 0, -mine, mine), yaw, atol=1e-9)
    # exp/log round trip, and agreement with an axis-angle construction
    r = torch.randn(8, 3, generator=gen, dtype=torch.float64)
    r = r / r.norm(dim=-1, keepdim=True) * torch.rand(8, 1, generator=gen, dtype=torch.float64) * 3.0
    assert torch.allclose(_log(_exp(r)), r, atol=1e-9)
    q = quat_from_angle_axis(r.norm(dim=-1), r / r.norm(dim=-1, keepdim=True))
    assert torch.allclose(_exp(r), q, atol=1e-9)


def test_targets_and_integrate_are_inverse():
    gen = torch.Generator().manual_seed(1)
    fmt, s0 = _state(6, gen)
    _, s1 = _state(6, gen)
    param = HeadingFrameParameterization(fmt)
    back = param.integrate(s0, param.targets(s0, s1))
    # quaternions may come back with the opposite sign
    for name, sl in _slices(fmt).items():
        a, b = back[:, sl], s1[:, sl]
        if name.endswith("quat_w"):
            a, b = a.reshape(6, -1, 4), b.reshape(6, -1, 4)
            a = torch.where((a * b).sum(-1, keepdim=True) < 0, -a, a)
        assert torch.allclose(a, b, atol=1e-8), name
    # the residual of the true next state against itself is zero
    assert torch.allclose(param.error(s1, s1), torch.zeros(6, param.target_size, dtype=s1.dtype), atol=1e-6)


def test_features_and_targets_ignore_position_and_heading():
    gen = torch.Generator().manual_seed(2)
    fmt, s0 = _state(5, gen)
    _, s1 = _state(5, gen)
    param = HeadingFrameParameterization(fmt)
    yaw = quat_from_angle_axis(
        torch.tensor([0.7] * 5, dtype=torch.float64), torch.tensor([[0.0, 0.0, 1.0]] * 5, dtype=torch.float64)
    )
    shift = torch.tensor([3.0, -2.0, 0.0], dtype=torch.float64)

    def move(state):
        out = state.clone()
        for name, sl in _slices(fmt).items():
            part = state[:, sl].reshape(
                5, -1, 4 if name.endswith("quat_w") else (3 if name.startswith(("root", "body")) else 1)
            )
            if name.startswith("joint"):
                continue
            y = yaw.unsqueeze(1).expand(-1, part.shape[1], -1)
            if name.endswith("quat_w"):
                moved = quat_mul(y, part)
            else:
                moved = quat_apply(y.reshape(-1, 4), part.reshape(-1, 3)).reshape(part.shape)
                if name.endswith("pos_w"):
                    moved = moved + shift
            out[:, sl] = moved.reshape(5, -1)
        return out

    assert torch.allclose(param.features(move(s0)), param.features(s0), atol=1e-9)
    assert torch.allclose(param.targets(move(s0), move(s1)), param.targets(s0, s1), atol=1e-9)


def test_integrate_is_differentiable():
    gen = torch.Generator().manual_seed(3)
    fmt, s0 = _state(4, gen)
    param = HeadingFrameParameterization(fmt)
    s0 = s0.float().requires_grad_(True)
    y = torch.zeros(4, param.target_size, requires_grad=True)
    out = param.integrate(s0, y)
    out.square().sum().backward()
    assert torch.isfinite(s0.grad).all() and torch.isfinite(y.grad).all()
    assert math.isclose(float((out - s0).abs().max()), 0.0, abs_tol=1e-5)


def _slices(fmt):
    out, offset = {}, 0
    for name, shape in fmt.items():
        size = 1
        for d in shape:
            size *= int(d)
        out[name] = slice(offset, offset + size)
        offset += size
    return out


def test_closed_form_tan_norm_matches_the_library():
    from instinctlab.tasks.shadowing.supermimic.supermimic_imagination import quat_to_tan_norm as closed_form
    from instinctlab.utils.math import quat_to_tan_norm as library

    q = _quat(16, torch.Generator().manual_seed(4))
    assert torch.allclose(closed_form(q), library(q), atol=1e-12)
