import math
from pathlib import Path

import mujoco
import torch
import numpy as np

from PIL import Image, ImageDraw

from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor.sensor_context import SensorContext

from mjlab_microduck.tasks.microduck_calibration_env_cfg import (
    make_microduck_calibration_env_cfg,
)


def _get_segmentation_id_type(self, cam_idx: int) -> torch.Tensor:
    """Return segmentation as [num_envs, height, width, 2] (id, type)."""
    if cam_idx not in self._cam_idx_to_list_idx:
        available = list(self._cam_idx_to_list_idx.keys())
        raise KeyError(
            f"Camera ID {cam_idx} not found in SensorContext. "
            f"Available camera IDs: {available}"
        )

    list_idx = self._cam_idx_to_list_idx[cam_idx]

    assert self._seg_adr_np is not None
    assert self._seg_torch is not None
    seg_adr = self._seg_adr_np[list_idx]
    if seg_adr < 0:
        raise RuntimeError(
            f"Camera ID {cam_idx} does not have segmentation rendering enabled."
        )

    sensor = self.camera_sensors[list_idx]
    w, h = sensor.cfg.width, sensor.cfg.height
    num_pixels = w * h
    nworld = self._data.nworld

    cam_data = self._seg_torch[:, seg_adr : seg_adr + num_pixels, :]
    return cam_data.view(nworld, h, w, 2)


# shortcut: mjlab 1.3.0 views this buffer as one channel, drop when mjlab >= 1.4.0
SensorContext.get_segmentation = _get_segmentation_id_type


DEVICE="cuda:0"
OUT_DIR = Path("camera_calibration_results")

FOVY_DEG = 60.0
BOARD_DISTANCE = 0.70
BOARD_HALF_THICKNESS = 0.005

OUT_DIR.mkdir(parents=True, exist_ok=True)


def camera_intrinsics(width, height, fovy_deg):

    fovy_rad = math.radians(fovy_deg)

    fy = height / (2 * math.tan(fovy_rad / 2))
    fx = fy
    cx = width / 2
    cy = height / 2

    return fx, fy, cx, cy

def pixel_rays_mujoco(rows, cols, intrinsics):

    fx, fy, cx, cy = intrinsics

    u = cols.float() + 0.5
    v = rows.float() + 0.5

    return torch.stack([
        (u - cx) / fx,
        - (v - cy) / fy,
        -torch.ones_like(u),
    ], dim=-1)

def quat_to_rotmat(q):

    q = q / q.norm().clamp(min=1e-8)
    w, x, y, z = q.unbind()

    return torch.stack([
        1 - 2 * (y*y + z*z),
        2 * (x*y - z*w),
        2 * (x*z + y*w),

        2 * (x*y + z*w),
        1 - 2 * (x*x + z*z),
        2 * (y*z - x*w),

        2 * (x*z + y*w),
        2 * (y*z + x*w),
        1 - 2 * (x*x + y*y),
    ]).reshape(3, 3)

def rotmat_to_quat(R):

    q = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(q, R.detach().cpu().numpy().astype(np.float64).ravel())

    return torch.tensor(q, dtype=R.dtype, device=R.device)

def refresh_sensors(env):
    env.sim.forward()
    env.sim.sense()

def project_world_points(
    point_w,
    cam_pos_w,
    cam_rot_w,
    intrinsics,
):

    fx, fy, cx, cy = intrinsics

    point_m = cam_rot_w.T @ (point_w - cam_pos_w)

    D = -point_m[2]
    if D.item() <= 0:
        raise RuntimeError("Point is behind the camera")
    
    u = fx * point_m[0] / D + cx
    v = -fy * point_m[1] / D + cy

    return u.item(), v.item(), point_m

def place_board(env):

    camera = env.scene["front_rgbd"]
    board = env.scene["calib_board"]

    cam_id = camera.camera_idx

    cam_pos_w = env.sim.data.cam_xpos[0, cam_id].clone()
    cam_rot_w = env.sim.data.cam_xmat[0, cam_id].clone()

    offset_m = torch.tensor([0.0, 0.0, -BOARD_DISTANCE], device=env.device)
    board_pos_w = cam_pos_w + cam_rot_w @ offset_m

    board_rot_w = cam_rot_w.clone()
    board_quat_w = rotmat_to_quat(board_rot_w)

    board_pose = torch.cat([board_pos_w, board_quat_w]).unsqueeze(0)

    board.write_mocap_pose_to_sim(board_pose)
    refresh_sensors(env)
    
    normal_w = board_rot_w[:, 2]
    surface_center_w = board_pos_w + normal_w * BOARD_HALF_THICKNESS
    
    return surface_center_w, normal_w

def set_head_yaw(env, yaw_deg):

    robot = env.scene["robot"]

    joint_ids, joint_names = robot.find_joints("head_yaw")

    if len(joint_ids) != 1:
        raise RuntimeError(f"Expected one head_yaw joint, got {joint_names}")

    print(f"Setting head yaw to {yaw_deg} deg")

    joint_id = joint_ids[0]

    yaw_rad = math.radians(yaw_deg)

    position = torch.tensor([[yaw_rad]], dtype=torch.float32, device=env.device)
    velocity = torch.zeros_like(position)

    robot.write_joint_state_to_sim(
        position=position,
        velocity=velocity,
        joint_ids=torch.tensor(
            [joint_id],
            dtype=torch.long,
            device=env.device,
        ),
    )

    refresh_sensors(env)


def experiment_b(env, surface_center_w, normal_w):

    camera = env.scene["front_rgbd"]
    board = env.scene["calib_board"]

    rgb = camera.data.rgb[0].clone()
    depth = camera.data.depth[0, :, :, 0].clone()
    seg = camera.data.segmentation[0].clone()

    H, W = depth.shape

    K = camera_intrinsics(W, H, FOVY_DEG)

    board_geom_id = int(board.indexing.geom_ids[0].item())

    mask = (seg[..., 0] == board_geom_id) & (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
    # erode the mask to avoid side faces and silhouette edges
    mask_float = mask.float()[None, None]

    interior = torch.nn.functional.avg_pool2d(
        mask_float,
        kernel_size=9,
        stride=1,
        padding=4,
    )[0, 0] > 0.999

    valid = (
        interior
        & torch.isfinite(depth)
        & (depth > 0.0)
    )

    rows, cols = torch.where(valid)

    if rows.numel() < 100:
        raise RuntimeError("Not enough valid pixels on the board")

    measured_depth = depth[rows, cols]

    rays_m = pixel_rays_mujoco(rows, cols, K)
    # backproject the measured depth
    points_m = measured_depth[:, None] * rays_m

    cam_id = camera.camera_idx
    R_WM = env.sim.data.cam_xmat[0, cam_id].clone()
    t_WM = env.sim.data.cam_xpos[0, cam_id].clone()

    points_w = points_m @ R_WM.T + t_WM

    rays_w = rays_m @ R_WM.T
    numerator = torch.dot(normal_w, surface_center_w - t_WM)
    denominator = rays_w @ normal_w

    if torch.any(denominator.abs() < 1e-6):
        raise RuntimeError("Some rays are parallel to the plane")

    gt_depth = numerator / denominator
    gt_points_w = t_WM + gt_depth[:, None] * rays_w

    # errors
    plane_error = torch.abs((points_w - surface_center_w) * normal_w)
    point_error = torch.linalg.norm(points_w - gt_points_w, dim=-1)
    depth_error = torch.abs(measured_depth - gt_depth)

    print("\n===== Experiment B =====")
    print(f"Valid pixels: {valid.sum().item()}")
    print(f"Plane error: {plane_error.mean():.4f} m")
    print(f"Point error: {point_error.mean():.4f} m")
    print(f"Depth error: {depth_error.mean():.4f} m")
    print(f"95% Point error: {point_error.quantile(0.95).item():.4f} m")

    # save images
    depth_m = depth.detach().cpu().numpy()
    depth_valid = np.isfinite(depth_m) & (depth_m > 0)
    depth_vis = np.zeros_like(depth_m, dtype=np.uint8)
    depth_vis[depth_valid] = (
        255.0 * np.clip(
            depth_m[depth_valid] / 3.0,
            0.0,
            1.0,
        )
    ).astype(np.uint8)
    Image.fromarray(rgb.cpu().numpy().astype(np.uint8)).save(OUT_DIR / "rgb.png")
    Image.fromarray(depth_vis).save(OUT_DIR / "depth.png")
    Image.fromarray(mask.cpu().numpy().astype(np.uint8)).save(OUT_DIR / "mask.png")

    return {
        "plane_error": plane_error.mean().item(),
        "point_error": point_error.mean().item(),
        "depth_error": depth_error.mean().item(),
        "p95_point_error": point_error.quantile(0.95).item(),
    }

def experiment_c(env, reference_point_w):

    camera = env.scene["front_rgbd"]
    robot = env.scene["robot"]
    board = env.scene["calib_board"]

    results = {}

    print("\n===== Experiment C =====")

    for yaw_deg in [-15.0, 0.0, 15.0]:
        set_head_yaw(env, yaw_deg)

        cam_id = camera.camera_idx

        R_WM = env.sim.data.cam_xmat[0, cam_id].clone()
        t_WM = env.sim.data.cam_xpos[0, cam_id].clone()

        rgb = camera.data.rgb[0].clone()
        depth = camera.data.depth[0, :, :, 0].clone()
        seg = camera.data.segmentation[0].clone()

        H, W = depth.shape
        K = camera_intrinsics(W, H, FOVY_DEG)

        u, v, expected_point_m = project_world_points(
            reference_point_w,
            t_WM,
            R_WM,
            K,
        )

        col = round(u - 0.5)
        row = round(v - 0.5)

        if not (0 <= col < W) or not (0 <= row < H):
            raise RuntimeError(
                f"Reference point outside image at yaw {yaw_deg} deg"
            )

        board_geom_id = int(board.indexing.geom_ids[0].item())

        if not (
            int(seg[row, col, 0].item()) == board_geom_id
            and int(seg[row, col, 1].item()) == int(mujoco.mjtObj.mjOBJ_GEOM)
        ):
            raise RuntimeError(
                f"Reference point not on board at yaw {yaw_deg} deg"
            )

        measured_depth = depth[row, col]
        rays_m = pixel_rays_mujoco(
            torch.tensor([row], device=env.device), 
            torch.tensor([col], device=env.device), 
            K,
        )
        point_m = measured_depth * rays_m[0]

        point_w = R_WM @ point_m + t_WM

        # world -> robot
        root_pos_w = robot.data.root_link_pos_w[0].clone()
        root_quat_w = robot.data.root_link_quat_w[0].clone()

        R_WB = quat_to_rotmat(root_quat_w)

        point_b = R_WB.T @ (point_w - root_pos_w)
        
        gt_point_b = R_WB.T @ (reference_point_w - root_pos_w)

        # errors
        error_w = torch.linalg.norm(point_w - reference_point_w).item()
        error_b = torch.linalg.norm(point_b - gt_point_b).item()

        print(f"Yaw {yaw_deg} deg:")
        print(f"Pixel: ({col}, {row})")
        print(f"Camera point: {point_m.detach().cpu().numpy()}")
        print(f"World point: {point_w.detach().cpu().numpy()}")
        print(f"Robot point: {point_b.detach().cpu().numpy()}")
        print(f"GT world point: {reference_point_w.detach().cpu().numpy()}")
        print(f"GT robot point: {gt_point_b.detach().cpu().numpy()}")
        print(f"World error: {error_w:.4f} m")
        print(f"Body error: {error_b:.4f} m")

        # save images
        image = Image.fromarray(rgb.cpu().numpy().astype(np.uint8))
        draw = ImageDraw.Draw(image)
        draw.ellipse(
            (col - 3, row - 3, col + 3, row + 3),
            outline=(0, 255, 0),
            width=2,
        )
        image.save(OUT_DIR / f"rgb_{yaw_deg:.0f}deg.png")

        results[yaw_deg] = {
            "point_m": point_m.clone(),
            "point_w": point_w.clone(),
            "point_b": point_b.clone(),
            "error_w": error_w,
            "error_b": error_b,
        }

    # compare reconstructed positions across head rotations
    reference = results[0.0]["point_w"]
    print("\n--- Cross-yaw world consistency ---")
    for yaw_deg in [-15.0, 0.0, 15.0]:
        delta = torch.linalg.norm(
            results[yaw_deg]["point_w"] - reference,
        ).item()

        print(f"Yaw {yaw_deg} deg vs 0 deg: {delta:.4f} m")

    return results

def main():

    cfg = make_microduck_calibration_env_cfg()
    env = ManagerBasedRlEnv(cfg, device=DEVICE)

    try:
        env.reset()
        set_head_yaw(env, 0.0)

        surface_center_w, normal_w = place_board(env)
        results_b = experiment_b(env, surface_center_w, normal_w)

        results_c = experiment_c(env, surface_center_w)

        print("\n All Completed Successfully!")

    finally:
        env.close()

if __name__ == "__main__":
    main()