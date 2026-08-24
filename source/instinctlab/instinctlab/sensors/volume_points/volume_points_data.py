from __future__ import annotations

from dataclasses import MISSING, dataclass

import warp as wp

from isaaclab.utils.warp import ProxyArray


@dataclass
class VolumePointsData:
    """Warp-first data container for the volume-points sensor.

    Array properties are :class:`ProxyArray` objects. Use ``.torch`` for a
    cached zero-copy Torch view or ``.warp`` for the underlying Warp array.
    """

    pos_w: ProxyArray = MISSING
    """The position of the volume points sensor in the world frame.

    Shape: (N, B, 3), where N is the number of envs, B is the number of bodies in each env.
    """

    quat_w: ProxyArray = MISSING
    """The quaternion of the volume points sensor in the world frame.

    Shape: (N, B, 4), where N is the number of envs, B is the number of bodies in each env.
    The quaternion is in the format (x, y, z, w).
    """

    vel_w: ProxyArray = MISSING
    """The velocity of the volume points sensor in the world frame.

    Shape: (N, B, 3), where N is the number of envs, B is the number of bodies in each env.
    The velocity is in the format (vx, vy, vz).
    """

    ang_vel_w: ProxyArray = MISSING
    """The angular velocity of the volume points sensor in the world frame.

    Shape: (N, B, 3), where N is the number of envs, B is the number of bodies in each env.
    """

    point_num_each_body: int = MISSING
    """The number of volume points in each body.
    This is used to calculate the shape of the volume points data.
    """

    points_pos_w: ProxyArray = MISSING
    """The position of the volume points in the world frame.

    Shape is (N, B, point_num_each_body, 3),
    where N is the number of sensors and B is the number of bodies in each sensor.
    """

    points_vel_w: ProxyArray = MISSING
    """The velocity of the volume points in the world frame.

    Shape is (N, B, point_num_each_body, 3),
    where N is the number of sensors and B is the number of bodies in each sensor.
    """

    penetration_offset: ProxyArray = MISSING
    """The penetration offset of the volume points sensor.
    This is the offset from the surface of the body to the volume points.

    If the point moves along the penetration direction, the penetration depth increases.
    If the point has no penetration, the penetration depth is zero and the direction is undefined.

    Shape is (N, B, point_num_each_body, 3), where N is the number of envs, B is the number of bodies in each env.
    """

    @staticmethod
    def make_zero(
        num_envs: int,
        num_bodies: int,
        point_num_each_body: int,
        device: str = "cpu",
    ) -> VolumePointsData:
        """Create zero-initialized Warp buffers with cached dual access."""
        return VolumePointsData(
            pos_w=ProxyArray(wp.zeros((num_envs, num_bodies), dtype=wp.vec3f, device=device)),
            quat_w=ProxyArray(wp.zeros((num_envs, num_bodies), dtype=wp.quatf, device=device)),
            vel_w=ProxyArray(wp.zeros((num_envs, num_bodies), dtype=wp.vec3f, device=device)),
            ang_vel_w=ProxyArray(wp.zeros((num_envs, num_bodies), dtype=wp.vec3f, device=device)),
            point_num_each_body=point_num_each_body,
            points_pos_w=ProxyArray(
                wp.zeros((num_envs, num_bodies, point_num_each_body), dtype=wp.vec3f, device=device)
            ),
            points_vel_w=ProxyArray(
                wp.zeros((num_envs, num_bodies, point_num_each_body), dtype=wp.vec3f, device=device)
            ),
            penetration_offset=ProxyArray(
                wp.zeros((num_envs, num_bodies, point_num_each_body), dtype=wp.vec3f, device=device)
            ),
        )
