# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to sweep through joint angle space, check collisions, and record
collision-free joint configurations with their end-effector poses.

For each collision-free configuration, saves:
    config_id, j1..j6, ee_x, ee_y, ee_z, ee_qw, ee_qx, ee_qy, ee_qz

Usage:
    ./isaaclab.sh -p scripts/joint_reachability_sweep.py --num_envs 4096 --steps_per_joint 12
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Joint angle sweep with collision check for Kinova.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments.")
parser.add_argument("--steps_per_joint", type=int, default=12, help="Number of discrete steps per joint.")
parser.add_argument("--settle_steps", type=int, default=500, help="Physics steps to settle after teleporting joints.")
parser.add_argument("--collision_threshold", type=float, default=0.0, help="Contact force threshold (N).")
parser.add_argument("--output_csv", type=str, default="joint_reachability_sweep_scale.csv",
                    help="Output CSV file path.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import csv

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

##
# Pre-defined configs
##
from kinova_lite_inv.robots.kinova_lite import KINOVA_CONFIG


@configclass
class SceneCfg(InteractiveSceneCfg):
    """Configuration for joint sweep scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    robot = KINOVA_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Contact sensor for collision detection
    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
    )


def generate_joint_configs(joint_limits_low, joint_limits_high, steps_per_joint, device):
    """Generate all joint configurations by sweeping each joint incrementally.

    Args:
        joint_limits_low: Tensor of shape (6,) with lower joint limits.
        joint_limits_high: Tensor of shape (6,) with upper joint limits.
        steps_per_joint: Number of discrete steps per joint.
        device: Torch device.

    Returns:
        joint_configs: Tensor of shape (N, 6) with all combinations.
    """
    # Create linspace for each joint
    joint_ranges = []
    for i in range(6):
        low = joint_limits_low[i].item()
        high = joint_limits_high[i].item()
        joint_ranges.append(torch.linspace(low, high, steps_per_joint, device=device))
        print(f"  Joint {i+1}: [{low:.4f}, {high:.4f}] rad, step size = {(high - low) / (steps_per_joint - 1):.4f} rad")

    # Create meshgrid of all combinations
    grids = torch.meshgrid(*joint_ranges, indexing='ij')
    # Flatten and stack: each grid is (S, S, S, S, S, S) -> flatten to (N,)
    joint_configs = torch.stack([g.reshape(-1) for g in grids], dim=1)

    return joint_configs


def run_sweep(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Run the joint angle sweep with collision checking."""
    robot = scene["robot"]
    contact_sensor = scene["contact_sensor"]
    num_envs = args_cli.num_envs

    # Joint names
    arm_joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6",
                       "left_finger_bottom_joint", "right_finger_bottom_joint",
                       "left_finger_tip_joint", "right_finger_tip_joint"]
    arm_joint_ids = robot.find_joints(arm_joint_names)[0]

    # Find end-effector body index
    ee_idx = robot.find_bodies("tool_frame")[0][0]

    # Finger configuration (open)
    right_bottom_open = torch.tensor(0.8, device=args_cli.device)
    finger_targets = torch.stack([
        torch.clamp(-1.0 * right_bottom_open + 0.0, -0.96, 0.09),
        right_bottom_open,
        torch.clamp(-0.676 * right_bottom_open + 0.149, -0.50, 0.21),
        torch.clamp(-0.676 * right_bottom_open + 0.149, -0.50, 0.21)
    ]).expand(num_envs, -1)

    sim_dt = sim.get_physics_dt()
    robot.update(dt=sim_dt)

    # Get joint limits from the robot
    joint_limits_low = robot.data.soft_joint_pos_limits[0, :6, 0]
    joint_limits_high = robot.data.soft_joint_pos_limits[0, :6, 1]

    print(f"\nJoint limits (from robot):")
    joint_configs = generate_joint_configs(
        joint_limits_low, joint_limits_high,
        args_cli.steps_per_joint, args_cli.device
    )
    total_configs = joint_configs.shape[0]

    # Store robot base pose for computing ee pose in base frame
    base_pos = robot.data.root_pos_w.clone()  # (1, 3)
    base_quat = robot.data.root_quat_w.clone()  # (1, 4)

    # Parameters
    settle_steps = args_cli.settle_steps
    collision_threshold = args_cli.collision_threshold

    print(f"\nTotal configurations to check: {total_configs}")
    print(f"Steps per joint: {args_cli.steps_per_joint}")
    print(f"Using {num_envs} parallel environments")
    print(f"Settle steps: {settle_steps}")
    print(f"Collision force threshold: {collision_threshold} N")
    print(f"Output file: {args_cli.output_csv}\n")

    # Open CSV file for writing results incrementally
    csv_file = open(args_cli.output_csv, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['config_id', 'j1', 'j2', 'j3', 'j4', 'j5', 'j6',
                         'ee_x', 'ee_y', 'ee_z', 'ee_qw', 'ee_qx', 'ee_qy', 'ee_qz'])

    batch_start_idx = 0
    total_collision_free = 0
    total_collisions = 0

    while simulation_app.is_running() and batch_start_idx < total_configs:
        batch_end_idx = min(batch_start_idx + num_envs, total_configs)
        batch_size = batch_end_idx - batch_start_idx

        # Get batch of joint configurations
        # target_joints = torch.tensor([0, -1, 1, 0, 1, 0], device=args_cli.device).expand(batch_size, -1)
        target_joints = joint_configs[batch_start_idx:batch_end_idx]

        # Pad if batch is smaller than num_envs
        if batch_size < num_envs:
            padding = torch.zeros((num_envs - batch_size, 6), device=args_cli.device)
            target_joints_padded = torch.cat([target_joints, padding], dim=0)
        else:
            target_joints_padded = target_joints

        # Build full joint state (arm + fingers)
        full_joint_pos = robot.data.default_joint_pos.clone()
        full_joint_vel = torch.zeros_like(full_joint_pos)
        robot.write_joint_state_to_sim(full_joint_pos, full_joint_vel)
        robot.write_data_to_sim()
        robot.reset()
        robot.update(sim_dt)
        full_joint_pos[:, :6] = target_joints_padded
        full_joint_pos[:, 6:] = finger_targets

        # Teleport robot to target joint configuration

        # Track collision during settling
        collision_detected = torch.zeros(batch_size, dtype=torch.bool, device=args_cli.device)

        # Step physics to settle and detect collisions
        for step in range(settle_steps):
            # Keep the target position
            robot.set_joint_position_target(full_joint_pos, joint_ids=arm_joint_ids)
            robot.write_data_to_sim()
            sim.step(render=False)
            robot.update(sim_dt)
            scene.update(sim_dt)

            # Check for collisions
            contact_forces = contact_sensor.data.net_forces_w[:batch_size]
            contact_force_magnitudes = torch.norm(contact_forces, dim=-1)
            total_contact_force = torch.sum(contact_force_magnitudes, dim=-1)

            step_collision = total_contact_force > collision_threshold
            collision_detected = collision_detected | step_collision

        # Read end-effector poses for the batch
        # print(full_joint_pos[:, :6]-robot.data.joint_pos[:, :6])
        
        ee_pos_w = robot.data.body_pos_w[:batch_size, ee_idx].clone()  # (batch, 3)
        ee_quat_w = robot.data.body_quat_w[:batch_size, ee_idx].clone()  # (batch, 4) [w, x, y, z]
        # print(collision_detected)
        # Convert ee pose to base frame
        base_pos_batch = base_pos[:batch_size]
        base_quat_batch = base_quat[:batch_size]
        ee_pos_base, ee_quat_base = subtract_frame_transforms(
            base_pos_batch, base_quat_batch,
            ee_pos_w, ee_quat_w
        )
        # print(ee_pos_base, ee_quat_base)

        # Save collision-free configurations
        batch_collision_free = 0
        for i in range(batch_size):
            if not collision_detected[i].item():
                config_id = batch_start_idx + i
                joints = target_joints[i].cpu().tolist()
                ee_p = ee_pos_base[i].cpu().tolist()
                ee_q = ee_quat_base[i].cpu().tolist()
                csv_writer.writerow([config_id] + joints + ee_p + ee_q)
                batch_collision_free += 1

        total_collision_free += batch_collision_free
        total_collisions += (batch_size - batch_collision_free)

        # Progress update
        checked = batch_end_idx
        print(f"Progress: {checked}/{total_configs} | "
              f"Collision-free: {total_collision_free} | "
              f"Collisions: {total_collisions} | "
              f"Free rate: {100*total_collision_free/checked:.1f}%")

        batch_start_idx = batch_end_idx

    csv_file.close()

    # Print summary
    print(f"\n{'='*60}")
    print(f"JOINT REACHABILITY SWEEP COMPLETE")
    print(f"{'='*60}")
    print(f"Total configurations checked: {total_configs}")
    print(f"Collision-free: {total_collision_free} ({100*total_collision_free/total_configs:.2f}%)")
    print(f"Collisions: {total_collisions} ({100*total_collisions/total_configs:.2f}%)")
    print(f"\nResults saved to: {args_cli.output_csv}")
    print(f"{'='*60}\n")


def main():
    """Main function."""
    sim_cfg = sim_utils.SimulationCfg(dt=1/60, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    scene_cfg = SceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[INFO]: Setup complete...")
    run_sweep(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
