"""
Simulation trajectory tracking evaluation in Isaac Lab.
Mirrors eight_shape.py (sim2real) but runs entirely in simulation.

Observation (25-dim, same as sim2real):
    joint_pos(6), joint_vel(6), rel_pos(3), rel_quat(4), ee_vel(6)

Usage:
    ./isaaclab.sh -p scripts/sim_trajectory_eval.py \\
        --checkpoint PPO/model.zip --trajectory circle
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import numpy as np
from collections import deque

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Simulation trajectory tracking evaluation.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model .zip file.")
parser.add_argument("--trajectory", type=str, default="eight", choices=["eight", "circle", "square", "lemniscate", "lissajous", "lemniscate_orient", "lissajous_orient"], help="Trajectory type to track.")
parser.add_argument("--scale_a", type=float, default=1.0, help="Action scale (1.0 matches sim training; 0.7 matches real robot).")
parser.add_argument("--duration", type=float, default=100.0, help="Total duration (seconds).")
parser.add_argument("--init_time", type=float, default=20.0, help="Settling time before logging/tracking (seconds).")
parser.add_argument("--output_csv", type=str, default="sim_trajectory_eval_temp.csv")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import csv
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

from stable_baselines3 import PPO, SAC
from sb3_contrib import TRPO

from kinova_lite_inv.robots.kinova_lite import KINOVA_CONFIG


# ── Scene ──────────────────────────────────────────────────────────────────────

@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    robot = KINOVA_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Robot")


# ── Trajectory generators (identical to sim2real) ─────────────────────────────

def generate_eight_trajectory(t, r=0.07, center=np.array([0.3, -0.1, 0.2]), angular_vel=0.1):
    """Figure-8 from two circles of radius r (fits inside original circle r=0.15).
    Bottom circle: clockwise. Top circle: counterclockwise. Tangent at center.
    """
    cx, cy, cz = center[0], center[1], center[2]
    phase = t * angular_vel
    cycle = np.clip(phase, 0.0, 4 * np.pi)  # clamp → stops at end of figure-8

    if cycle < 2 * np.pi:
        # Bottom circle (clockwise)
        x = cx + r * np.sin(cycle)
        y = (cy - r) + r * np.cos(cycle)
    else:
        # Top circle (counterclockwise)
        theta = cycle - 2 * np.pi
        x = cx + r * np.sin(theta)
        y = (cy + r) - r * np.cos(theta)

    progress = cycle / (4 * np.pi)  # 0→1 per full figure-8
    return np.array([x, y, cz]), progress


def generate_circular_trajectory(t, radius=0.15, center=np.array([0.3, -0.1, 0.2]), angular_vel=0.1):
    """Single circle of given radius, stops after one full revolution."""
    theta = np.clip(t * angular_vel, 0.0, 2 * np.pi)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    progress = theta / (2 * np.pi)  # 0→1 per full circle
    return np.array([x, y, center[2]]), progress

def generate_lemniscate_trajectory(t, a=0.15, center=np.array([0.3, -0.1, 0.2]), angular_vel=0.1):
    """Lemniscate of Bernoulli: r = sqrt(a^2 * cos(2*theta)).
    Parametric form: x = a*cos(t)/(1+sin^2(t)), y = a*sin(t)*cos(t)/(1+sin^2(t)).
    Traces full figure-8 shape for t in [0, 2*pi], stops at end.
    """
    theta = np.clip(t * angular_vel, 0.0, 2 * np.pi)
    denom = 1.0 + np.sin(theta) ** 2
    x_local = a * np.cos(theta) / denom
    y_local = a * np.sin(theta) * np.cos(theta) / denom
    progress = theta / (2 * np.pi)  # 0→1 per full trace
    return np.array([center[0] + x_local, center[1] + y_local, center[2]]), progress

def generate_lissajous_trajectory(t, a=0.15, center=np.array([0.3, -0.1, 0.2]), angular_vel=0.1):
    """Lemniscate of Bernoulli: r = sqrt(a^2 * cos(2*theta)).
    Parametric form: x = a*cos(t)/(1+sin^2(t)), y = a*sin(t)*cos(t)/(1+sin^2(t)).
    Traces full figure-8 shape for t in [0, 2*pi], stops at end.
    """
    theta = np.clip(t * angular_vel, 0.0, 2 * np.pi)
    x_local = a * np.cos(theta)
    y_local = a * np.sin(theta)
    z_local = a * np.sin(2*theta)
    progress = theta / (2 * np.pi)  # 0→1 per full trace
    return np.array([center[0] + x_local, center[1] + y_local, center[2] + z_local]), progress

def generate_lemniscate_orient_trajectory(t, a=0.15, center=np.array([0.3, -0.1, 0.2]),
                                           angular_vel=0.1, orient_amp=0.52):
    """Lemniscate trajectory with sinusoidal orientation change about the x-axis.

    φ = orient_amp * sin(2π * progress) → exactly one sine cycle over the full path.
    Base 180° around X composed with X-axis delta gives (w,x,y,z) = [-sin(φ/2), cos(φ/2), 0, 0].
    """
    theta = np.clip(t * angular_vel, 0.0, 2 * np.pi)
    denom = 1.0 + np.sin(theta) ** 2
    x_local = a * np.cos(theta) / denom
    y_local = a * np.sin(theta) * np.cos(theta) / denom
    progress = theta / (2 * np.pi)
    pos = np.array([center[0] + x_local, center[1] + y_local, center[2]])

    phi = orient_amp * np.sin(2 * np.pi * progress)
    orn = np.array([-np.sin(phi / 2), np.cos(phi / 2), 0.0, 0.0], dtype=np.float32)

    return pos, orn, progress


def generate_lissajous_orient_trajectory(t, a=0.15, center=np.array([0.3, -0.1, 0.2]),
                                          angular_vel=0.1, orient_amp=0.3):
    """Lissajous (3-D) trajectory with sinusoidal orientation change about the x-axis.

    φ = orient_amp * sin(2π * progress) → exactly one sine cycle over the full path.
    Base 180° around X composed with X-axis delta gives (w,x,y,z) = [-sin(φ/2), cos(φ/2), 0, 0].
    """
    theta = np.clip(t * angular_vel, 0.0, 2 * np.pi)
    x_local = a * np.cos(theta)
    y_local = a * np.sin(theta)
    z_local = a * np.sin(2 * theta)
    progress = theta / (2 * np.pi)
    pos = np.array([center[0] + x_local, center[1] + y_local, center[2] + z_local])

    phi = orient_amp * np.sin(2 * np.pi * progress)
    orn = np.array([-np.sin(phi / 2), np.cos(phi / 2), 0.0, 0.0], dtype=np.float32)

    return pos, orn, progress


def generate_square_trajectory(t, side_length=0.2, center=np.array([0.3, -0.1, 0.2]), speed=0.1/(2*np.pi)):
    """Generate square trajectory waypoint"""
    # Total perimeter
    perimeter = 4 * side_length
    distance = t * speed
    distance = np.clip(distance, 0.0, perimeter)
    half_side = side_length / 2
    
    # Four corners
    corner1 = center + np.array([-half_side, -half_side, 0])
    corner2 = center + np.array([half_side, -half_side, 0])
    corner3 = center + np.array([half_side, half_side, 0])
    corner4 = center + np.array([-half_side, half_side, 0])
    
    if distance < side_length:  # Bottom edge (corner1 -> corner2)
        progress = distance / side_length
        pos = corner1 + progress * (corner2 - corner1)
    elif distance < 2 * side_length:  # Right edge (corner2 -> corner3)
        progress = (distance - side_length) / side_length
        pos = corner2 + progress * (corner3 - corner2)
    elif distance < 3 * side_length:  # Top edge (corner3 -> corner4)
        progress = (distance - 2 * side_length) / side_length
        pos = corner3 + progress * (corner4 - corner3)
    else:  # Left edge (corner4 -> corner1)
        progress = (distance - 3 * side_length) / side_length
        pos = corner4 + progress * (corner1 - corner4)
    
    completion = distance / perimeter
    return pos, completion


# ── Quaternion helpers (identical to sim2real) ────────────────────────────────

def quat_conj(q):
    return np.concatenate((q[:1], -q[1:]))


def quat_mul_np(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    return np.array([
        qq - ww + (z1 - y1) * (y2 - z2),
        qq - xx + (x1 + w1) * (x2 + w2),
        qq - yy + (w1 - x1) * (y2 + z2),
        qq - zz + (z1 + y1) * (w2 - x2),
    ])


def quat_distance(q1, q2):
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    return 2 * np.arccos(np.clip(np.abs(np.dot(q1, q2)), 0.0, 1.0))


# ── Main evaluation ────────────────────────────────────────────────────────────

def run_eval(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    robot = scene["robot"]
    device = args_cli.device
    sim_dt = sim.get_physics_dt()
    robot.update(dt=sim_dt)

    # Joint indices
    all_joint_names = [
        "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6",
        "left_finger_bottom_joint", "right_finger_bottom_joint",
        "left_finger_tip_joint", "right_finger_tip_joint",
    ]
    all_joint_ids = robot.find_joints(all_joint_names)[0]
    arm_joint_ids  = all_joint_ids[:6]   # velocity targets
    finger_joint_ids = all_joint_ids[6:] # position targets

    ee_idx = robot.find_bodies("tool_frame")[0][0]

    # Finger open targets (shape: 1×4)
    rbo = torch.tensor(0.8, device=device)
    open_targets = torch.stack([
        torch.clamp(-1.0 * rbo, -0.96, 0.09),
        rbo,
        torch.clamp(-0.676 * rbo + 0.149, -0.50, 0.21),
        torch.clamp(-0.676 * rbo + 0.149, -0.50, 0.21),
    ]).unsqueeze(0)  # (1, 4)

    # Robot base pose (fixed throughout)
    base_pos  = robot.data.root_pos_w.clone()   # (1, 3)
    base_quat = robot.data.root_quat_w.clone()  # (1, 4)

    # Move to default (home) position before starting
    print("[INFO]: Moving to home position...")
    default_pos = robot.data.default_joint_pos.clone()  # (1, n_joints)
    default_vel = torch.zeros_like(default_pos)
    robot.write_joint_state_to_sim(default_pos, default_vel)
    robot.write_data_to_sim()
    robot.reset()
    for _ in range(200):
        robot.set_joint_position_target(default_pos, joint_ids=all_joint_ids)
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(sim_dt)
    print("[INFO]: Home position reached.")

    # Load policy
    print(f"Loading SAC from: {args_cli.checkpoint}")
    model = SAC.load(args_cli.checkpoint, device=device)


    # Select trajectory function and target orientation
    traj_returns_orn = False  # True for trajectories that also yield a per-step orientation
    if args_cli.trajectory == "circle":
        traj_fn    = generate_circular_trajectory
        target_orn = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)  # 180° around X
    elif args_cli.trajectory == "square":
        traj_fn    = generate_square_trajectory
        target_orn = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)  # 180° around X
    elif args_cli.trajectory == "lemniscate":
        traj_fn    = generate_lemniscate_trajectory
        target_orn = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)  # 180° around X
    elif args_cli.trajectory == "lissajous":
        traj_fn    = generate_lissajous_trajectory
        target_orn = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)  # 180° around X
    elif args_cli.trajectory == "lemniscate_orient":
        traj_fn    = generate_lemniscate_orient_trajectory
        target_orn = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)  # updated each step
        traj_returns_orn = True
    elif args_cli.trajectory == "lissajous_orient":
        traj_fn    = generate_lissajous_orient_trajectory
        target_orn = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)  # updated each step
        traj_returns_orn = True
    else:
        traj_fn    = generate_eight_trajectory
        target_orn = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)  # 180° around X

    # Logging buffers
    time_data, ee_pos_data, target_pos_data = [], [], []
    pos_err_data, orn_err_data = [], []
    action_data, jvel_data, progress_data = [], [], []

    total_steps = int(args_cli.duration / sim_dt)
    init_steps  = int(args_cli.init_time / sim_dt)

    print(f"\nTrajectory : {args_cli.trajectory}")
    print(f"Duration   : {args_cli.duration}s  ({total_steps} steps at {sim_dt*1000:.1f}ms/step)")
    print(f"Init time  : {args_cli.init_time}s  ({init_steps} steps)")
    print(f"Scale_a    : {args_cli.scale_a}\n")

    obs_history = deque([np.zeros(25, dtype=np.float32)] * 5, maxlen=5)

    for step in range(total_steps):
        if not simulation_app.is_running():
            break

        t_sim = step * sim_dt

        # ── Read robot state ──────────────────────────────────────────────
        joint_pos_t = robot.data.joint_pos[:, :6].clone()          # (1,6)
        joint_vel_t = robot.data.joint_vel[:, :6].clone()          # (1,6)
        ee_pos_w    = robot.data.body_pos_w[:, ee_idx].clone()     # (1,3)
        ee_quat_w   = robot.data.body_quat_w[:, ee_idx].clone()    # (1,4)
        ee_vel_t    = robot.data.body_com_vel_w[:, ee_idx].clone() # (1,6) lin+ang

        ee_pos_base, ee_quat_base = subtract_frame_transforms(
            base_pos, base_quat, ee_pos_w, ee_quat_w
        )

        joint_pos_np = joint_pos_t[0].cpu().numpy()
        joint_vel_np = joint_vel_t[0].cpu().numpy()
        ee_pos_np    = ee_pos_base[0].cpu().numpy()
        ee_quat_np   = ee_quat_base[0].cpu().numpy()
        ee_vel_np    = ee_vel_t[0].cpu().numpy()

        # ── Trajectory waypoint ───────────────────────────────────────────
        if t_sim > args_cli.init_time:
            if traj_returns_orn:
                target_pos_np, target_orn, progress = traj_fn(t_sim - args_cli.init_time)
            else:
                target_pos_np, progress = traj_fn(t_sim - args_cli.init_time)
        else:
            if traj_returns_orn:
                target_pos_np, target_orn, _ = traj_fn(0.0)
            else:
                target_pos_np, _ = traj_fn(0.0)
            progress = 0.0

        # ── Build 25-dim obs ──────────────────────────────────────────────
        rel_pos  = target_pos_np - ee_pos_np
        rel_quat = quat_mul_np(ee_quat_np, quat_conj(target_orn))
        obs_c = np.concatenate([
            joint_pos_np,   # 6
            joint_vel_np,   # 6
            rel_pos,        # 3
            rel_quat,       # 4
            ee_vel_np,      # 6
        ]).astype(np.float32)
        obs_history.append(obs_c)
        obs = np.concatenate(list(obs_history)).reshape(1, -1)

        # ── Policy inference ──────────────────────────────────────────────
        
        action, _ = model.predict(obs, deterministic=True)
        action = np.clip(action[0], -1.0, 1.0)          # (6,)
        action_scaled = action * args_cli.scale_a

        # ── Apply action ──────────────────────────────────────────────────
        action_t = torch.tensor(action_scaled, dtype=torch.float32, device=device).unsqueeze(0)
        robot.set_joint_velocity_target(action_t, joint_ids=arm_joint_ids)
        robot.set_joint_position_target(open_targets, joint_ids=finger_joint_ids)
        robot.write_data_to_sim()
        sim.step(render=True)
        robot.update(sim_dt)
        scene.update(sim_dt)

        # ── Errors ────────────────────────────────────────────────────────
        pos_error = np.linalg.norm(target_pos_np - ee_pos_np)
        orn_error = quat_distance(ee_quat_np, target_orn)

        # ── Log (only after init period, same as sim2real) ────────────────
        if t_sim > args_cli.init_time:
            time_data.append(t_sim - args_cli.init_time)
            ee_pos_data.append(ee_pos_np.copy())
            target_pos_data.append(target_pos_np.copy())
            pos_err_data.append(pos_error)
            orn_err_data.append(orn_error)
            action_data.append(action_scaled.copy())
            jvel_data.append(joint_vel_np.copy())
            progress_data.append(progress)

        if step % 60 == 0:
            print(f"t={t_sim:.2f}s | pos_err={pos_error:.4f}m | "
                  f"orn_err={orn_error:.4f}rad | progress={progress:.3f}")

    # ── Save CSV (identical columns to sim2real) ──────────────────────────────
    with open(args_cli.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "time",
            "ee_x", "ee_y", "ee_z",
            "target_x", "target_y", "target_z",
            "pos_error", "orn_error",
            "a1", "a2", "a3", "a4", "a5", "a6",
            "jv1", "jv2", "jv3", "jv4", "jv5", "jv6",
            "progress",
        ])
        for i in range(len(time_data)):
            writer.writerow([
                time_data[i],
                *ee_pos_data[i],
                *target_pos_data[i],
                pos_err_data[i], orn_err_data[i],
                *action_data[i],
                *jvel_data[i],
                progress_data[i],
            ])

    print(f"\nSaved {len(time_data)} samples to: {args_cli.output_csv}")


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=1 / 60, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    scene_cfg = SceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[INFO]: Setup complete...")
    run_eval(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
