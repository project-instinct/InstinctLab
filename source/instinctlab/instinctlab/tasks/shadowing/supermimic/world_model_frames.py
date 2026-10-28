"""Heading-frame parameterization of the SuperMimic world-model state.

The world-model state (``make_state_format``) holds world-frame quantities, because the reward and
termination compare them with the world-frame reference. Fed to the dynamics network as they are, they
make it learn something the physics does not contain: the next step depends on neither where the robot
stands nor which way it faces, yet raw world coordinates differ for every environment origin and
heading. Worse, the robot's posture is then only visible as the small difference of two large
coordinates (a link's and the root's world position, normalised over the spread of all environment
origins), which the input normalisation squeezes to almost nothing.

This module gives the generic :class:`~instinct_rl.modules.world_model.WorldModel` a parameterization
that removes both: the network reads the root height and tilt, root velocities in the root frame,
joints, and every link relative to the root in the root's heading frame; it predicts frame-local
increments (root displacement in the heading frame, rotation increments in the body frame, velocity
changes in the heading or root frame), which are integrated back into the world-frame state.

Quaternions are ``(x, y, z, w)``, the convention of the state's ``*_quat_w`` components.
"""

from __future__ import annotations

import numpy as np
import torch
from collections import OrderedDict

from instinct_rl.modules.world_model import StateParameterization

# ---------------------------------------------------------------------------------------------------
# Quaternion helpers, (x, y, z, w), broadcasting over leading dimensions and differentiable everywhere
# ---------------------------------------------------------------------------------------------------


def _conj(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((-q[..., :3], q[..., 3:]), dim=-1)


def _mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    av, aw = a[..., :3], a[..., 3:]
    bv, bw = b[..., :3], b[..., 3:]
    vec = aw * bv + bw * av + torch.linalg.cross(av, bv)
    return torch.cat((vec, aw * bw - (av * bv).sum(-1, keepdim=True)), dim=-1)


def _rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """``q`` applied to ``v``."""
    u, w = q[..., :3], q[..., 3:]
    t = 2.0 * torch.linalg.cross(u, v)
    return v + w * t + torch.linalg.cross(u, t)


def _rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """``q^-1`` applied to ``v``."""
    return _rotate(_conj(q), v)


def _heading(q: torch.Tensor) -> torch.Tensor:
    """Yaw-only part of ``q``."""
    x, y, z, w = q.unbind(-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    zero = torch.zeros_like(yaw)
    return torch.stack((zero, zero, torch.sin(0.5 * yaw), torch.cos(0.5 * yaw)), dim=-1)


def _log(q: torch.Tensor) -> torch.Tensor:
    """Rotation vector (axis * angle) of ``q``, taking the shorter of ``q`` and ``-q``."""
    q = torch.where(q[..., 3:] < 0.0, -q, q)
    vec, w = q[..., :3], q[..., 3:]
    norm = torch.sqrt((vec * vec).sum(-1, keepdim=True) + 1e-12)
    return vec * (2.0 * torch.atan2(norm, w) / norm)


def _exp(r: torch.Tensor) -> torch.Tensor:
    """Unit quaternion of the rotation vector ``r``."""
    angle = torch.sqrt((r * r).sum(-1, keepdim=True) + 1e-12)
    return torch.cat((r * (torch.sin(0.5 * angle) / angle), torch.cos(0.5 * angle)), dim=-1)


def _tan_norm(q: torch.Tensor) -> torch.Tensor:
    """Sign-invariant 6-D rotation feature: the rotated x and z axes."""
    from instinctlab.tasks.shadowing.supermimic.supermimic_imagination import quat_to_tan_norm

    return quat_to_tan_norm(q)


def _normalize(q: torch.Tensor) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-6)


# ---------------------------------------------------------------------------------------------------
# Parameterization
# ---------------------------------------------------------------------------------------------------

ROOT_COMPONENTS = ("root_pos_w", "root_quat_w", "root_lin_vel_w", "root_ang_vel_w", "joint_pos", "joint_vel")
LINK_COMPONENTS = ("body_link_pos_w", "body_link_quat_w", "body_link_lin_vel_w", "body_link_ang_vel_w")


class HeadingFrameParameterization(StateParameterization):
    """Position- and heading-invariant features, frame-local increments.

    With ``p, q`` the root position and orientation, ``h`` its heading (yaw-only part of ``q``) and,
    per link, ``b_p, b_q, b_v, b_w``:

    * features: ``p_z``, tilt ``h^-1 q`` (6-D), root velocities in the root frame, joints, and per link
      ``h^-1 (b_p - p)``, ``h^-1 b_q`` (6-D), ``h^-1 b_v``, ``h^-1 b_w``;
    * targets: root displacement ``h^-1 (p' - p)``, root rotation increment ``log(q^-1 q')``, root
      velocity changes in the root frame, joint changes, and per link the change of its offset from the
      root ``h^-1 ((b_p' - p') - (b_p - p))``, its rotation increment ``log(b_q^-1 b_q')`` and its
      velocity changes in the heading frame.

    ``error`` uses the same expressions with the true state's frames, so it is the target-layout
    residual of a predicted state.
    """

    def __init__(self, state_format: OrderedDict[str, tuple]):
        self.state_format = OrderedDict(state_format)
        missing = [n for n in ROOT_COMPONENTS + LINK_COMPONENTS if n not in self.state_format]
        extra = [n for n in self.state_format if n not in ROOT_COMPONENTS + LINK_COMPONENTS]
        if missing or extra:
            raise ValueError(f"unsupported state layout: missing {missing}, unknown {extra}")
        self.num_joints = int(np.prod(self.state_format["joint_pos"]))
        self.num_links = int(self.state_format["body_link_pos_w"][0])
        self._slices = OrderedDict()
        offset = 0
        for name, shape in self.state_format.items():
            size = int(np.prod(shape))
            self._slices[name] = slice(offset, offset + size)
            offset += size
        self.state_size = offset
        j, n = self.num_joints, self.num_links
        self.feature_size = 1 + 6 + 3 + 3 + 2 * j + n * (3 + 6 + 3 + 3)
        self.target_segments = OrderedDict(
            (name, size)
            for name, size in (
                ("root_pos_w", 3),
                ("root_quat_w", 3),
                ("root_lin_vel_w", 3),
                ("root_ang_vel_w", 3),
                ("joint_pos", j),
                ("joint_vel", j),
                ("body_link_pos_w", 3 * n),
                ("body_link_quat_w", 3 * n),
                ("body_link_lin_vel_w", 3 * n),
                ("body_link_ang_vel_w", 3 * n),
            )
        )
        self.target_size = int(sum(self.target_segments.values()))

    # -- layout --------------------------------------------------------------------------------------

    def _unpack(self, state: torch.Tensor) -> dict[str, torch.Tensor]:
        lead = state.shape[:-1]
        out = {}
        for name, sl in self._slices.items():
            part = state[..., sl]
            out[name] = part.reshape(*lead, self.num_links, -1) if name in LINK_COMPONENTS else part
        return out

    def _pack(self, parts: dict[str, torch.Tensor]) -> torch.Tensor:
        lead = parts["root_pos_w"].shape[:-1]
        return torch.cat([parts[name].reshape(*lead, -1) for name in self.state_format], dim=-1)

    def _split_target(self, target: torch.Tensor) -> dict[str, torch.Tensor]:
        lead = target.shape[:-1]
        out, offset = {}, 0
        for name, size in self.target_segments.items():
            part = target[..., offset : offset + size]
            out[name] = part.reshape(*lead, self.num_links, 3) if name in LINK_COMPONENTS else part
            offset += size
        return out

    # -- StateParameterization ----------------------------------------------------------------------

    def features(self, state: torch.Tensor) -> torch.Tensor:
        s = self._unpack(state)
        q = s["root_quat_w"]
        h = _heading(q)
        h_l = h.unsqueeze(-2)  # broadcast over links
        lead = state.shape[:-1]
        return torch.cat(
            (
                s["root_pos_w"][..., 2:3],
                _tan_norm(_mul(_conj(h), q)),
                _rotate_inverse(q, s["root_lin_vel_w"]),
                _rotate_inverse(q, s["root_ang_vel_w"]),
                s["joint_pos"],
                s["joint_vel"],
                _rotate_inverse(h_l, s["body_link_pos_w"] - s["root_pos_w"].unsqueeze(-2)).reshape(*lead, -1),
                _tan_norm(_mul(_conj(h_l), s["body_link_quat_w"])).reshape(*lead, -1),
                _rotate_inverse(h_l, s["body_link_lin_vel_w"]).reshape(*lead, -1),
                _rotate_inverse(h_l, s["body_link_ang_vel_w"]).reshape(*lead, -1),
            ),
            dim=-1,
        )

    def _residual(self, ref: dict[str, torch.Tensor], other: dict[str, torch.Tensor]) -> torch.Tensor:
        """``other`` expressed against ``ref`` in the target layout, in ``ref``'s frames."""
        q = ref["root_quat_w"]
        h = _heading(q)
        h_l = h.unsqueeze(-2)
        lead = q.shape[:-1]
        offset_ref = ref["body_link_pos_w"] - ref["root_pos_w"].unsqueeze(-2)
        offset_other = other["body_link_pos_w"] - other["root_pos_w"].unsqueeze(-2)
        return torch.cat(
            (
                _rotate_inverse(h, other["root_pos_w"] - ref["root_pos_w"]),
                _log(_mul(_conj(q), other["root_quat_w"])),
                _rotate_inverse(q, other["root_lin_vel_w"] - ref["root_lin_vel_w"]),
                _rotate_inverse(q, other["root_ang_vel_w"] - ref["root_ang_vel_w"]),
                other["joint_pos"] - ref["joint_pos"],
                other["joint_vel"] - ref["joint_vel"],
                _rotate_inverse(h_l, offset_other - offset_ref).reshape(*lead, -1),
                _log(_mul(_conj(ref["body_link_quat_w"]), other["body_link_quat_w"])).reshape(*lead, -1),
                _rotate_inverse(h_l, other["body_link_lin_vel_w"] - ref["body_link_lin_vel_w"]).reshape(*lead, -1),
                _rotate_inverse(h_l, other["body_link_ang_vel_w"] - ref["body_link_ang_vel_w"]).reshape(*lead, -1),
            ),
            dim=-1,
        )

    def targets(self, state: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        return self._residual(self._unpack(state), self._unpack(next_state))

    def error(self, predicted: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
        return self._residual(self._unpack(actual), self._unpack(predicted))

    def integrate(self, state: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        s = self._unpack(state)
        d = self._split_target(target)
        q = s["root_quat_w"]
        h = _heading(q)
        h_l = h.unsqueeze(-2)
        root_pos = s["root_pos_w"] + _rotate(h, d["root_pos_w"])
        offset = s["body_link_pos_w"] - s["root_pos_w"].unsqueeze(-2) + _rotate(h_l, d["body_link_pos_w"])
        nxt = {
            "root_pos_w": root_pos,
            "root_quat_w": _normalize(_mul(q, _exp(d["root_quat_w"]))),
            "root_lin_vel_w": s["root_lin_vel_w"] + _rotate(q, d["root_lin_vel_w"]),
            "root_ang_vel_w": s["root_ang_vel_w"] + _rotate(q, d["root_ang_vel_w"]),
            "joint_pos": s["joint_pos"] + d["joint_pos"],
            "joint_vel": s["joint_vel"] + d["joint_vel"],
            "body_link_pos_w": root_pos.unsqueeze(-2) + offset,
            "body_link_quat_w": _normalize(_mul(s["body_link_quat_w"], _exp(d["body_link_quat_w"]))),
            "body_link_lin_vel_w": s["body_link_lin_vel_w"] + _rotate(h_l, d["body_link_lin_vel_w"]),
            "body_link_ang_vel_w": s["body_link_ang_vel_w"] + _rotate(h_l, d["body_link_ang_vel_w"]),
        }
        return self._pack(nxt)
