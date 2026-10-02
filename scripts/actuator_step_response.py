import argparse
import csv
import time
from collections import deque
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from infer_policy import (
    BAM_KP_FW,
    BAM_MAX_CURRENT,
    BAM_VIN_MIN,
    load_bam_model,
    load_mujoco_with_bam,
)

XML_PATH = Path("src/mjlab_microduck/robot/microduck/scene.xml")

DT = 0.005  # 200 hz physics

VIN = 7.4  # fixed voltage
VIN_DROP_GAIN = 0.0  # disable voltage sag


def get_mass_matrix(model, data):
    M = np.zeros(
        (model.nv, model.nv),
        dtype=np.float64,
    )

    mujoco.mj_fullM(
        model,
        data,
        M,
    )

    return M


def get_joint_indices(model, joint_name):
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)

    if joint_id < 0:
        raise ValueError(f"Joint '{joint_name}' not found")

    qpos_idx = int(model.jnt_qposadr[joint_id])
    dof_idx = int(model.jnt_dofadr[joint_id])

    return joint_id, qpos_idx, dof_idx


def reset_to_stand(model, data):
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "STAND")

    if key_id < 0:
        raise RuntimeError("STAND keyframe not found")

    mujoco.mj_resetDataKeyframe(model, data, key_id)

    mujoco.mj_forward(model, data)


def main(render=True):
    # Create BAM M6 actuator model
    bam_model = load_bam_model(kp_fw=BAM_KP_FW, vin=VIN, max_current=BAM_MAX_CURRENT)

    # load MicroDuck scene, and transform position actuators to BAM-controlled torque motors
    model, data, bam_ctrl, actuator_names = load_mujoco_with_bam(
        xml_path=str(XML_PATH),
        bam_model=bam_model,
        timestep=DT,
        vin_drop_gain=VIN_DROP_GAIN,
        vin_min=BAM_VIN_MIN,
    )

    # start from the STAND keyframe
    reset_to_stand(model, data)

    bam_ctrl.reset(data.qpos)

    initial_targets = data.qpos[bam_ctrl.qpos_indexes].copy()

    bam_ctrl.q_target[:] = initial_targets

    # find left_knee
    knee_joint_id, knee_qpos_idx, knee_dof_idx = get_joint_indices(model, "left_knee")

    controller_joint_ids = np.asarray(bam_ctrl.joint_indexes)

    matches = np.flatnonzero(controller_joint_ids == knee_joint_id)
    if len(matches) != 1:
        raise RuntimeError("Could not uniquely map left_knee to BAM controller")

    knee_ctrl_idx = matches[0]

    q0 = initial_targets[knee_ctrl_idx]

    print(f"left_knee initial target: {q0:.4f} rad")
    print(f"step target: {q0 + args.delta_q:.4f} rad")

    log = []
    viewer = None

    if render:
        viewer = mujoco.viewer.launch_passive(model, data)

    n_steps = int(args.duration / DT)

    target_buffer = deque(
        [initial_targets.copy() for _ in range(args.delay_steps + 1)],
        maxlen=args.delay_steps + 1,
    )

    kd_used = args.kd
    kd_initialized = args.kd is not None
    tau_pd_raw = np.nan

    for step in range(n_steps):
        wall_start = time.perf_counter()

        t = float(data.time)

        if args.controller == "pd" and not kd_initialized and t >= 8.0:
            mujoco.mj_forward(model, data)
            M = get_mass_matrix(model, data)
            I_knee = float(M[knee_dof_idx, knee_dof_idx])
            kd_used = 2.0 * args.zeta * np.sqrt(I_knee * args.kp)
            kd_initialized = True
            print(f"I_knee = {I_knee:.6f}")
            print(f"Kp = {args.kp:.6f}")
            print(f"Kd = {kd_used:.6f}")
            print(f"zeta = {args.zeta:.3f}")

        command_target = initial_targets.copy()
        if t >= args.step_time:
            command_target[knee_ctrl_idx] += args.delta_q

        target_buffer.append(command_target)

        applied_target = target_buffer[0]
        bam_ctrl.q_target[:] = applied_target

        # q_target
        #   -> BAM firmware control
        #   -> voltage/current/friction
        #   -> torque written to data.ctrl
        bam_ctrl.update()

        # log state
        q = float(data.qpos[knee_qpos_idx])
        qvel = float(data.qvel[knee_dof_idx])

        # generalized actuator torque
        act_idx = int(bam_ctrl.act_indexes[knee_ctrl_idx])

        if args.controller == "pd" and kd_initialized:
            tau_pd_raw = args.kp * (applied_target[knee_ctrl_idx] - q) - kd_used * qvel
            force_range = model.actuator_forcerange[act_idx]
            tau_pd = float(np.clip(tau_pd_raw, force_range[0], force_range[1]))
            data.ctrl[act_idx] = tau_pd

        torque = float(data.ctrl[act_idx])

        ncon = int(data.ncon)
        constraint_torque = float(data.qfrc_constraint[knee_dof_idx])

        base_z = float(data.qpos[2])
        base_vz = float(data.qvel[2])
        base_omega = data.qvel[3:6].copy()

        log.append(
            (
                t,
                float(command_target[knee_ctrl_idx]),
                float(applied_target[knee_ctrl_idx]),
                q,
                qvel,
                torque,
                tau_pd_raw,
                ncon,
                constraint_torque,
                base_z,
                base_vz,
                base_omega[0],
                base_omega[1],
                base_omega[2],
            )
        )

        mujoco.mj_step(model, data)

        if viewer is not None:
            if not viewer.is_running():
                break

            viewer.sync()

            elapsed = time.perf_counter() - wall_start

            time.sleep(max(0.0, DT - elapsed))

    if viewer is not None:
        viewer.close()

    output = Path(args.output)

    with output.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "time",
                "command_target",
                "applied_target",
                "q",
                "qvel",
                "torque",
                "tau_pd_raw",
                "num_constraint",
                "constraint_torque",
                "base_z_pos",
                "base_z_velocity",
                "base_omega_x",
                "base_omega_y",
                "base_omega_z",
            ]
        )
        writer.writerows(log)

    print(f"Saved: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--delay-steps", type=int, default=0)
    parser.add_argument("--output", type=str, default="bam_step_response.csv")
    parser.add_argument("--delta-q", type=float, default=0.1)
    parser.add_argument("--step-time", type=float, default=8.5)
    parser.add_argument("--duration", type=float, default=10)
    parser.add_argument("--controller", choices=["bam", "pd"], default="bam")
    parser.add_argument("--kp", type=float, default=0.56)
    parser.add_argument("--kd", type=float, default=None)
    parser.add_argument("--zeta", type=float, default=1.0)

    args = parser.parse_args()

    main(render=not args.no_render)
