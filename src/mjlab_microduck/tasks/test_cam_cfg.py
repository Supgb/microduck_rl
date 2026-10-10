import numpy as np
from PIL import Image

from mjlab.envs import ManagerBasedRlEnv

from mjlab_microduck.tasks.microduck_vision_env_cfg import (
    make_microduck_vision_env_cfg,
    pixel_to_robot_frame,
)


def main():

    cfg = make_microduck_vision_env_cfg(play=True)
    env = ManagerBasedRlEnv(cfg, device="cuda:0")

    try:
        env.reset()

        env.sim.forward()

        point_m, point_w, point_b = pixel_to_robot_frame(env, 64, 64)

        print("Camera:", point_m)
        print("World:", point_w)
        print("Robot:", point_b)

        camera = env.scene["front_rgbd"]
        rgb = (
            camera.data.rgb[0]
            .detach()
            .cpu()
            .numpy()
        )

        depth = (
            camera.data.depth[0, :, :, 0]
            .detach()
            .cpu()
            .numpy()
        )

        print("RGB shape:", rgb.shape)
        print("RGB dtype:", rgb.dtype)

        print("Depth shape:", depth.shape)
        print("Depth dtype:", depth.dtype)

        Image.fromarray(rgb).save("rgb.png")

        valid = np.isfinite(depth) & (depth > 0)

        depth_vis = np.zeros_like(
            depth,
            dtype=np.uint8,
        )

        depth_vis[valid] = (
            255.0 * np.clip(
                depth[valid] / 3.0,
                0.0,
                1.0,
            )
        ).astype(np.uint8)

        Image.fromarray(depth_vis).save("depth.png")

    finally:
        env.close()

if __name__ == "__main__":
    main()