import math
from typing import Sequence, Tuple


def quaternion_to_euler_xyz(
    x: float, y: float, z: float, w: float
) -> Tuple[float, float, float]:
    # Intrinsic XYZ (roll, pitch, yaw) in radians.
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)
    t2 = max(-1.0, min(1.0, t2))
    pitch = math.asin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)

    return roll, pitch, yaw


def quaternion_multiply(
    a: Sequence[float], b: Sequence[float]
) -> Tuple[float, float, float, float]:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def quaternion_conjugate(q: Sequence[float]) -> Tuple[float, float, float, float]:
    x, y, z, w = q
    return (-x, -y, -z, w)


def quaternion_inverse(q: Sequence[float]) -> Tuple[float, float, float, float]:
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    if n == 0.0:
        return (0.0, 0.0, 0.0, 1.0)
    cx, cy, cz, cw = quaternion_conjugate(q)
    return (cx / n, cy / n, cz / n, cw / n)


def quaternion_to_matrix(q: Sequence[float]) -> Tuple[Tuple[float, float, float], ...]:
    x, y, z, w = q
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    return (
        (1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)),
        (2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)),
        (2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)),
    )


def matrix_to_quaternion(m: Sequence[Sequence[float]]) -> Tuple[float, float, float, float]:
    m00, m01, m02 = m[0]
    m10, m11, m12 = m[1]
    m20, m21, m22 = m[2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    return (x, y, z, w)


def transform_vr_to_robot_pose(
    position: Sequence[float], quaternion: Sequence[float]
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
    x, y, z = position
    robot_position = (z, -x, y)

    transform = (
        (0.0, 0.0, 1.0),
        (-1.0, 0.0, 0.0),
        (0.0, -1.0, 0.0),
    )
    vr_matrix = quaternion_to_matrix(quaternion)
    tmp = (
        (
            transform[0][0] * vr_matrix[0][0]
            + transform[0][1] * vr_matrix[1][0]
            + transform[0][2] * vr_matrix[2][0],
            transform[0][0] * vr_matrix[0][1]
            + transform[0][1] * vr_matrix[1][1]
            + transform[0][2] * vr_matrix[2][1],
            transform[0][0] * vr_matrix[0][2]
            + transform[0][1] * vr_matrix[1][2]
            + transform[0][2] * vr_matrix[2][2],
        ),
        (
            transform[1][0] * vr_matrix[0][0]
            + transform[1][1] * vr_matrix[1][0]
            + transform[1][2] * vr_matrix[2][0],
            transform[1][0] * vr_matrix[0][1]
            + transform[1][1] * vr_matrix[1][1]
            + transform[1][2] * vr_matrix[2][1],
            transform[1][0] * vr_matrix[0][2]
            + transform[1][1] * vr_matrix[1][2]
            + transform[1][2] * vr_matrix[2][2],
        ),
        (
            transform[2][0] * vr_matrix[0][0]
            + transform[2][1] * vr_matrix[1][0]
            + transform[2][2] * vr_matrix[2][0],
            transform[2][0] * vr_matrix[0][1]
            + transform[2][1] * vr_matrix[1][1]
            + transform[2][2] * vr_matrix[2][1],
            transform[2][0] * vr_matrix[0][2]
            + transform[2][1] * vr_matrix[1][2]
            + transform[2][2] * vr_matrix[2][2],
        ),
    )
    robot_matrix = (
        (
            tmp[0][0] * transform[0][0]
            + tmp[0][1] * transform[0][1]
            + tmp[0][2] * transform[0][2],
            tmp[0][0] * transform[1][0]
            + tmp[0][1] * transform[1][1]
            + tmp[0][2] * transform[1][2],
            tmp[0][0] * transform[2][0]
            + tmp[0][1] * transform[2][1]
            + tmp[0][2] * transform[2][2],
        ),
        (
            tmp[1][0] * transform[0][0]
            + tmp[1][1] * transform[0][1]
            + tmp[1][2] * transform[0][2],
            tmp[1][0] * transform[1][0]
            + tmp[1][1] * transform[1][1]
            + tmp[1][2] * transform[1][2],
            tmp[1][0] * transform[2][0]
            + tmp[1][1] * transform[2][1]
            + tmp[1][2] * transform[2][2],
        ),
        (
            tmp[2][0] * transform[0][0]
            + tmp[2][1] * transform[0][1]
            + tmp[2][2] * transform[0][2],
            tmp[2][0] * transform[1][0]
            + tmp[2][1] * transform[1][1]
            + tmp[2][2] * transform[1][2],
            tmp[2][0] * transform[2][0]
            + tmp[2][1] * transform[2][1]
            + tmp[2][2] * transform[2][2],
        ),
    )
    robot_quaternion = matrix_to_quaternion(robot_matrix)
    return robot_position, robot_quaternion
