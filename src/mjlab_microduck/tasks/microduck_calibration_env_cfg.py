import mujoco

from mjlab.sensor import CameraSensorCfg
from mjlab.entity import EntityCfg

from mjlab_microduck.tasks.microduck_vision_env_cfg import (
    make_microduck_vision_env_cfg,
)


def make_board_spec() -> mujoco.MjSpec:
    """Create a thin, red rectangular calibration board."""

    return mujoco.MjSpec.from_string(
        """
        <mujoco model="calibration_board">
        <worldbody>
            <body name="board_body">
                <!-- Face the camera sees has local +Z normal. Warp shades
                     L = -light_dir, so this light hits that face head-on.
                     Emission is ignored by the warp camera shader. -->
                <light name="board_light" directional="true" dir="0 0 -1" diffuse="1 1 1"/>
                <geom
                    name="board_geom"
                    type="box"
                    size="0.28 0.18 0.005"
                    rgba="1 0 0 1"
                    contype="0"
                    conaffinity="0"
                    group="0"
                />
            </body>
        </worldbody>
        </mujoco>
        """
    )

def make_microduck_calibration_env_cfg():
    cfg = make_microduck_vision_env_cfg(play=True)
    cfg.scene.num_envs = 1

    cfg.scene.entities["calib_board"] = EntityCfg(
        spec_fn=make_board_spec,
    )

    for sensor in cfg.scene.sensors:
        if (
            isinstance(sensor, CameraSensorCfg)
            and sensor.name == "front_rgbd"
        ):
            sensor.data_types = (
                "rgb",
                "depth",
                "segmentation",
            )
            break
    else:
        raise RuntimeError("front_rgbd camera not found")

    return cfg
