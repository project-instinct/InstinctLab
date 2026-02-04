import torch

import isaaclab.utils.math as math_utils


@torch.jit.script
def rotmat_to_euler_yzx(mat):
    """mat: shape (N, 3, 3) 3d rotation matrix"""
    # get the rotation parameters in x(q0)z(q1)y(q2) sequence
    x = torch.atan2(mat[:, 2, 1], mat[:, 1, 1])  # x
    z = torch.asin(-mat[:, 0, 1])  # z
    y = torch.atan2(mat[:, 0, 2], mat[:, 0, 0])  # y
    x = math_utils.wrap_to_pi(x)
    z = math_utils.wrap_to_pi(z)
    y = math_utils.wrap_to_pi(y)
    return y, z, x


@torch.jit.script
def rotmat_to_euler_xzy(mat):
    """mat: shape (N, 3, 3) 3d rotation matrix"""
    # get the rotation parameters in y(q0)z(q1)x(q2) sequence
    y = torch.atan2(-mat[:, 2, 0], mat[:, 0, 0])  # y
    z = torch.asin(mat[:, 1, 0])  # z
    x = torch.atan2(-mat[:, 1, 2], mat[:, 1, 1])  # x
    y = math_utils.wrap_to_pi(y)
    z = math_utils.wrap_to_pi(z)
    x = math_utils.wrap_to_pi(x)
    return x, z, y


def zxy_to_xyz(points):
    """Convert the points from y-up to z-up."""
    return points[..., [2, 0, 1]]


def xyz_to_zxy(points):
    """Convert the points from z-up to y-up."""
    return points[..., [1, 2, 0]]


@torch.jit.script
def quat_to_tan_norm(q: torch.Tensor) -> torch.Tensor:
    """Convert a quaternion to tangent and normal vectors. (Directly copied from MaskedMimic's ProtoMotions)
    Args:
        q: The quaternion in (w, x, y, z). Shape is (..., 4)

    Returns:
        The tangent and normal vectors in the quaternion representation. Shape is (..., 6)
    """
    # represents a rotation using the tangent and normal vectors
    ref_tan = torch.zeros_like(q[..., 0:3])
    ref_tan[..., 0] = 1
    tan = math_utils.quat_apply(q, ref_tan)

    ref_norm = torch.zeros_like(q[..., 0:3])
    ref_norm[..., -1] = 1
    norm = math_utils.quat_apply(q, ref_norm)

    tan_norm = torch.cat([tan, norm], dim=len(tan.shape) - 1)
    return tan_norm


@torch.jit.script
def tan_norm_to_quat(tannorm: torch.Tensor) -> torch.Tensor:
    """Convert tangent and normal vectors to a quaternion. NOTE: assuming both tangent and normal vectors are normalized.
    Args:
        tannorm: The tangent and normal vectors in the quaternion representation. Shape is (..., 6)

    Returns:
        The quaternion in (w, x, y, z). Shape is (..., 4)
    """
    tan = tannorm[..., 0:3]
    norm = tannorm[..., 3:6]
    conj_axis = torch.cross(norm, tan, dim=len(tan.shape) - 1)
    matrix = torch.stack([tan, conj_axis, norm], dim=-1)
    quat = math_utils.quat_from_matrix(matrix)
    return quat


@torch.jit.script
def quat_slerp_batch(q1: torch.Tensor, q2: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """Interpolate and sample across the quaternion (batch) based on the given t (batch).
    Args:
        q1, q2: The quaternion in (w, x, y, z). Shape is (B, 4)
        tau: The interpolation factor. Shape is (B,). Type is float where 0 <= t <= 1
    Return:
        The interpolated quaternion in (w, x, y, z). Shape is (B, 4)
    """
    # ensure the input is in the right shape
    assert q1.shape[-1] == 4, "The quaternion must be in (w, x, y, z) format."
    assert q2.shape[-1] == 4, "The quaternion must be in (w, x, y, z) format."
    assert tau.shape == q1.shape[:-1], "The batch size must be the same for all inputs."
    assert q1.shape[0] == q2.shape[0] == tau.shape[0], "The batch size must be the same for all inputs."
    assert (tau >= 0).all() and (tau <= 1).all(), "The interpolation factor must be in (0, 1) range."

    # if the dot product is negative, flip the quaternion
    dot_product = torch.sum(q1 * q2, dim=-1, keepdim=True).clip(-1, 1)  # shape (B, 1)
    q2 = torch.where(dot_product < 0, -q2, q2)
    dot_product = torch.where(dot_product < 0, -dot_product, dot_product)

    # calculate the angle between the two quaternions
    theta = torch.acos(dot_product)
    sin_theta = torch.sin(theta)
    q_too_similar = (dot_product > (1 - 1e-9)) | (torch.abs(theta) < (1e-9))

    # avoid division by zero
    sin_theta = torch.where(sin_theta == 0, torch.ones_like(sin_theta), sin_theta)

    # calculate the interpolation factor
    s1 = torch.sin((1 - tau).unsqueeze(-1) * theta) / sin_theta
    s2 = torch.sin(tau.unsqueeze(-1) * theta) / sin_theta

    # calculate the interpolated quaternion
    interpolated_quat = (s1 * q1 + s2 * q2) * (~q_too_similar) + q_too_similar * q1
    interpolated_quat = math_utils.normalize(interpolated_quat)

    return interpolated_quat


@torch.jit.script
def quat_angular_velocity(q_prev: torch.Tensor, q_next: torch.Tensor, dt: float) -> torch.Tensor:
    """Compute the angular velocity between two quaternions. (from q_prev to q_next in dt seconds)
    Args:
        q_prev, q_next: The quaternion in (w, x, y, z). Shape is (..., 4)
        dt: The time difference between the two quaternions.
    Returns:
        The angular velocity in the local frame. Shape is (..., 3)
    """
    # ensure the input is in the right shape
    assert q_prev.shape[-1] == 4, "The quaternion must be in (w, x, y, z) format."
    assert q_next.shape[-1] == 4, "The quaternion must be in (w, x, y, z) format."
    assert q_prev.shape == q_next.shape, "The shape of the two quaternions must be the same."
    assert dt > 0, "The time difference must be positive."

    # if the dot product is negative, flip the quaternion
    dot_product = torch.sum(q_prev * q_next, dim=-1, keepdim=True).clip(-1, 1)  # shape (..., 1)
    q_next = torch.where(dot_product < 0, -q_next, q_next)

    quat_diff = math_utils.quat_mul(q_next, math_utils.quat_conjugate(q_prev))  # q_next * q_prev^-1
    axis_angle_diff = math_utils.axis_angle_from_quat(quat_diff)
    angular_velocity = axis_angle_diff / dt

    return angular_velocity
