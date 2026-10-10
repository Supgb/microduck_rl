from copy import deepcopy
import math
import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.sensor import CameraSensorCfg

from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    MicroduckRlCfg,
    make_microduck_velocity_env_cfg,
)

MicroduckVisionRlCfg = deepcopy(MicroduckRlCfg)
MicroduckVisionRlCfg.experiment_name = "vision"
MicroduckVisionRlCfg.run_name = "flat_microduck"
MicroduckVisionRlCfg.logger = "tensorboard"
MicroduckVisionRlCfg.upload_model = False

def make_microduck_vision_env_cfg(
    play: bool = True,
) -> ManagerBasedRlEnvCfg:

    # reuse locomotion env cfg
    cfg = make_microduck_velocity_env_cfg(play=play, rough=False)
    cfg.scene.num_envs = 1

    camera_cfg = CameraSensorCfg(
        name="front_rgbd",
        parent_body="robot/jaw_soft",
        pos=(
            0.0155,
            -9.13778e-05,
            -0.0733,
        ),
        quat=(
            0.7071068,
            0.0,
            0.0,
            -0.7071068,
        ),
        width=128,
        height=128,
        fovy=60.0,
        data_types=("rgb", "depth"),
    )
    
    cfg.scene.sensors = (
        *cfg.scene.sensors,
        camera_cfg,
    )



    return cfg

def quaternion_to_matrix(
    quat: torch.Tensor,
) -> torch.Tensor:
    """Convert a normalized wxyz quaternion to a rotation matrix."""

    quat = quat / quat.norm().clamp_min(1e-8)

    w, x, y, z = quat.unbind()

    return torch.stack([
        1 - 2 * (y*y + z*z),
        2 * (x*y - z*w),
        2 * (x*z + y*w),

        2 * (x*y + z*w),
        1 - 2 * (x*x + z*z),
        2 * (y*z - x*w),

        2 * (x*z - y*w),
        2 * (y*z + x*w),
        1 - 2 * (x*x + y*y),
    ]).reshape(3, 3)

def get_camera_intrinsics(
    width: int,
    height: int,
    fovy_deg: float,
):
    fovy_rad = math.radians(fovy_deg)
    fy = height / (2.0 * math.tan(fovy_rad / 2.0))
    fx = fy
    cx = width / 2.0
    cy = height / 2.0
    return fx, fy, cx, cy

def backproject_pixel(
    col: int,
    row: int,
    depth: torch.Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> torch.Tensor:
    u = col + 0.5
    v = row + 0.5

    x = (u - cx) * depth / fx
    y = -(v - cy) * depth / fy
    z = -depth

    return torch.stack((x, y, z))

def pixel_to_robot_frame(
    env,
    col: int,
    row: int,
    fovy_deg: float = 60,
):

    # camera sensor
    camera = env.scene["front_rgbd"]
    depth_img = camera.data.depth
    _, height, width, _ = depth_img.shape
    depth = depth_img[0, row, col, 0]

    if not torch.isfinite(depth).item() or depth.item() <= 0.0:
        raise ValueError(f"Invalid depth value at pixel ({col}, {row})")

    # camera intrinsics
    fx, fy, cx, cy = get_camera_intrinsics(width, height, fovy_deg)

    # pixel -> camera frame
    point_m = backproject_pixel(col, row, depth, fx, fy, cx, cy)

    # camera -> world
    cam_id = camera.camera_idx
    R_WM = env.sim.data.cam_xmat[0, cam_id]
    t_WM = env.sim.data.cam_xpos[0, cam_id]

    point_w = R_WM @ point_m + t_WM

    # world -> robot
    robot = env.scene["robot"]
    t_WB = robot.data.root_link_pos_w[0]
    q_WB = robot.data.root_link_quat_w[0]

    R_WB = quaternion_to_matrix(q_WB)
    point_b = R_WB @ (point_w - t_WB)

    return point_m, point_w, point_b
