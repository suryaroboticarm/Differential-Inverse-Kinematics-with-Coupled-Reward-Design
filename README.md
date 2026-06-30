# Differential Inverse Kinematics with Coupled Reward Design

This repository contains PPO-based RL controllers that map instantaneous pose error to joint velocity for reactive tracking on non-spherical-wrist manipulators. A coupled reward (trained only on static poses) yields smooth, baseline-outperforming motion and generalizes zero-shot to complex 3-D trajectories on hardware without sim-to-real degradation.

Key scripts and contents
- sim_trajectory_eval.py  — Run a trained SB3 PPO policy in Isaac Sim for open-loop trajectory tracking evaluation (25-D observation, 6-D action).
- sim_trajectory_eval_sac.py — Same as above but loads a Stable-Baselines3 SAC policy (example for SAC models).
- sim_trajectory_eval_trpo.py — Evaluation script targeting TRPO (sb3-contrib) policies.
- sim_traj_dls.py — Simulation-only trajectory follower using numerical differential kinematics (Damped Least Squares / QP controllers) for baseline comparisons.
- ppo_model.zip, sac_model.zip, trpo_model.zip — Example trained model archives included for convenience (large binary files).
- sb3/ — Directory for Stable Baselines 3 artifacts (models, tensorboard logs). See sb3/README.md.
- weights/ — Directory for raw model weight files and checkpoints. See weights/README.md.

Requirements
- Python 3.8+ (tested with 3.8–3.10)
- PyTorch
- Stable Baselines 3, and optionally sb3-contrib for TRPO
- numpy, scipy, osqp
- Isaac Sim / Isaac Lab runtime (the evaluation scripts launch and drive Isaac Sim scenes via isaaclab)
- kinova_lite_inv (robot kinematics and robot config used by the scenes)

Install (example)

1. Create virtual environment and install core dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch numpy scipy stable-baselines3 sb3-contrib osqp
```

2. Install Isaac Lab / Isaac Sim following NVIDIA's instructions and ensure `isaaclab` is importable in your Python environment. Also install or make available the kinova_lite_inv package used by the scene config.

Notes on large files
- This repository includes pre-trained model archives (ppo_model.zip, sac_model.zip, trpo_model.zip). These are large binaries — consider enabling Git LFS and tracking common model extensions (e.g., `*.zip`, `*.pt`, `*.pth`) to avoid inflating the Git history.

Examples — running evaluation in Isaac Sim

Before running any evaluation scripts, launch Isaac Sim / Isaac Lab as required by your installation. Many setups use isaaclab helper scripts; the example usage in the scripts expects an AppLauncher helper.

Run PPO policy evaluation:

```bash
# run inside Isaac Lab environment (see your isaaclab setup)
python sim_trajectory_eval.py --checkpoint ppo_model.zip --trajectory circle --duration 120 --init_time 20
```

Run SAC policy evaluation:

```bash
python sim_trajectory_eval_sac.py --checkpoint sac_model.zip --trajectory lemniscate --duration 100 --init_time 15
```

Run the DLS / QP baseline (simulation-only):

```bash
python sim_traj_dls.py --trajectory circle --duration 60
```

Script notes
- Observation vector: 25-D (joint_pos(6), joint_vel(6), rel_pos(3), rel_quat(4), ee_vel(6)).
- Actions: 6-D joint velocities (applied as velocity targets). Scripts scale actions by `--scale_a` to map policy outputs to robot commands.
- Several trajectory generators are included: `eight`, `circle`, `square`, `lemniscate`, `lissajous`, and orientation-variant variants.

Recommended repository housekeeping
- Add Git LFS for model files: `git lfs track "*.zip" "*.pt" "*.pth"` and commit a .gitattributes file.
- Add a .gitignore entry to avoid committing local logs and temporary files (e.g., `*.csv`, `__pycache__/`, `.venv/`).
- Move large model files into `weights/` or `sb3/` and keep lightweight manifests in the repository (e.g., `weights/manifest.json`) to describe available models.

Citing and contact
If you use this code in academic work, please cite the project and contact the maintainer for questions or collaborations.

---

If you want, I can:
- Commit this README.md to the repository now.
- Add a .gitattributes + .gitignore and a weights/manifest.json template.
- Move the included model zip files into the `weights/` directory and replace them with small manifest entries (requires confirming you want the files moved).

Tell me which of the above you want me to do next, and I'll proceed.