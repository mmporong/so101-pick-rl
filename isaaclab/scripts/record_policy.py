#!/usr/bin/env python3
"""Record a fixed Pick & Place policy, real RGB and aligned states/actions; never train."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "isaaclab"))
from runtime_metrics import ResourceSampler, enforce_resource_guard, write_json
from so101_pick_rl.capture_data import VideoWriter, aligned_arrays, sha256_file
from so101_pick_rl.run_contract import PICK_PLACE_TASK_ID, contract_sha256, load_resume_binding, run_binding

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output_dir", type=Path, required=True)
parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
parser.add_argument("--max_steps", type=int, default=600)
parser.add_argument("--width", type=int, default=960)
parser.add_argument("--height", type=int, default=720)
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.headless or not args.enable_cameras:
    parser.error("Capture requires --headless --enable_cameras; no GUI is opened")
if args.max_steps < 1 or min(args.width, args.height) < 64 or args.width % 2 or args.height % 2:
    parser.error("Positive max_steps and even image dimensions >=64 are required")
if len(set(args.seeds)) != len(args.seeds):
    parser.error("Seeds must be unique")

checkpoint = args.checkpoint.expanduser().resolve()
if not checkpoint.is_file():
    raise FileNotFoundError(checkpoint)
output = args.output_dir.expanduser().resolve()
output.mkdir(parents=True, exist_ok=False)
report_path = output / "manifest.json"
report = {
    "schema": "so101_pick_rl.policy_capture.v1", "status": "not_run",
    "purpose": "baseline_diagnostic_not_expert_demonstration_or_formal_success_evaluation",
    "started_at_utc": datetime.now(timezone.utc).isoformat(),
    "command": subprocess.list2cmdline(sys.argv), "task": PICK_PLACE_TASK_ID,
    "checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint)},
    "contract_sha256": contract_sha256(PICK_PLACE_TASK_ID),
    "git_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
    "git_dirty": bool(subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True).strip()),
    "requested_seeds": args.seeds, "num_envs": 1, "episodes": [],
    "alignment": "frame/state/observation i -> action i -> state/observation/frame i+1; terminal captured before auto-reset",
    "sampling": "policy-rate only; no claim of complete 120Hz contact history",
    "video_clock": "simulation time; includes initial frame; T transitions produce T+1 frames",
    "same_seed_caveat": "Seed plus recorded initial states define this one-env protocol; not a replay of previous 256-env evaluation",
}
write_json(report_path, report)
sampler = ResourceSampler()
app = env = wrapped = context = None


def main():
    global app, env, wrapped, context
    report["preflight_resources"] = enforce_resource_guard(40, 4096, 70)
    sampler.start()
    report["status"] = "launching"
    write_json(report_path, report)
    app = AppLauncher(args).app

    import gymnasium as gym
    import numpy as np
    import torch
    from PIL import Image
    from rsl_rl.runners import OnPolicyRunner
    import isaaclab_tasks  # noqa: F401
    import so101_pick_rl.tasks  # noqa: F401
    from isaaclab.managers import RecorderTermCfg
    from isaaclab.managers.recorder_manager import RecorderTerm, RecorderManagerBaseCfg, DatasetExportMode
    from isaaclab.utils import configclass
    from isaaclab.utils.io import load_yaml
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from so101_pick_rl.tasks.pick_place import mdp
    from so101_pick_rl.policy_evaluation import _phase_failure_masks

    class CaptureContext:
        def __init__(self, directory, fps):
            self.directory, self.fps = directory, fps
            self.states, self.transitions = [], []
            self.writer = VideoWriter(directory / "rollout.mp4", args.width, args.height, fps)

        def state(self, raw):
            def cpu(value):
                return value[0].detach().cpu().numpy().copy()
            cube, robot = raw.scene["cube"], raw.scene["robot"]
            snapshot = {
                "observations": cpu(raw.obs_buf["policy"]),
                "joint_position_rad": cpu(robot.data.joint_pos),
                "joint_velocity_rad_s": cpu(robot.data.joint_vel),
                "cube_pose_w": cpu(cube.data.root_pose_w),
                "cube_velocity_w": cpu(cube.data.root_vel_w),
                "target_position_w": cpu(raw.target_pos_w),
                "ee_position_w": cpu(raw.scene.sensors["ee_frame"].data.target_pos_w[:, 0]),
                "finger_contact_force_n": cpu(mdp.contact_forces(raw)),
                "phase_state": cpu(raw.pick_place_state.observation()),
                "gripper_open_fraction": cpu(mdp.gripper_open_fraction(raw)),
            }
            # Render without a physics step. Warm-up and post-step frames use the same path.
            raw.sim.render()
            pixels = raw.render()
            if pixels is None or not np.isfinite(pixels).all() or pixels.std() < 1:
                raise RuntimeError("Renderer returned a missing or blank frame")
            self.writer.write(pixels)
            if len(self.states) % 150 == 0 or bool(raw.reset_buf[0]):
                Image.fromarray(pixels).save(self.directory / f"frame_{len(self.states):04d}.png")
            self.states.append(snapshot)

        def transition(self, raw):
            def cpu(value):
                return value[0].detach().cpu().numpy().copy()
            term = raw.action_manager.get_term("joint_position_delta")
            self.transitions.append({
                "applied_policy_actions": cpu(raw.action_manager.action),
                "processed_joint_delta_rad": cpu(term.processed_actions),
                "commanded_joint_target_rad": cpu(term._target),
                "rewards": cpu(raw.reward_buf),
                "terminated": cpu(raw.reset_terminated), "time_out": cpu(raw.reset_time_outs),
            })
            self.state(raw)

    class CaptureTerm(RecorderTerm):
        def record_post_step(self):
            capture = getattr(self._env, "capture_context", None)
            if capture is not None:
                capture.transition(self._env)
            return None, None

    @configclass
    class CaptureCfg(RecorderManagerBaseCfg):
        dataset_export_mode = DatasetExportMode.EXPORT_NONE
        export_in_record_pre_reset = False
        capture = RecorderTermCfg(class_type=CaptureTerm)

    cfg = parse_env_cfg(PICK_PLACE_TASK_ID, device=args.device, num_envs=1, use_fabric=True)
    cfg.seed = args.seeds[0]
    cfg.viewer.eye = (0.60, 0.70, 0.52)
    cfg.viewer.lookat = (0.0, 0.23, 0.12)
    cfg.viewer.resolution = (args.width, args.height)
    cfg.recorders = CaptureCfg()
    env = gym.make(PICK_PLACE_TASK_ID, cfg=cfg, render_mode="rgb_array")
    wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
    raw = wrapped.unwrapped
    report["sidecar_binding"] = load_resume_binding(checkpoint, run_binding(
        PICK_PLACE_TASK_ID, int(wrapped.num_obs), int(wrapped.num_actions)))
    agent_path = checkpoint.parent / "params" / "agent.yaml"
    agent_cfg = load_yaml(str(agent_path))
    report["agent_config_sha256"] = sha256_file(agent_path)
    wrapped.clip_actions = agent_cfg.get("clip_actions")
    runner = OnPolicyRunner(wrapped, agent_cfg, log_dir=None, device=agent_cfg["device"])
    runner.load(str(checkpoint), load_optimizer=False)
    policy = runner.get_inference_policy(device=raw.device)
    fps = raw.pick_place_spec["control"]["policy_hz"]
    report.update({"status": "recording", "fps": fps, "resolution": [args.width, args.height],
                   "camera_eye": list(cfg.viewer.eye), "camera_lookat": list(cfg.viewer.lookat),
                   "joint_names": raw.scene["robot"].joint_names,
                   "observation_spec": raw.pick_place_spec["observation"],
                   "environment_device": str(raw.device),
                   "policy_device": str(next(runner.alg.policy.parameters()).device)})
    write_json(report_path, report)
    with torch.inference_mode():
        for seed in args.seeds:
            raw.capture_context = None
            torch.manual_seed(seed)
            raw.reset(seed=seed)
            observations, _ = wrapped.get_observations()
            # Initialize the RGB annotator, then warm up only the renderer (not physics).
            raw.render()
            for _ in range(12):
                raw.sim.render()
            directory = output / f"seed_{seed}"
            directory.mkdir()
            context = CaptureContext(directory, fps)
            raw.capture_context = context
            context.state(raw)
            proposed_actions = []
            completed = False
            for step in range(min(args.max_steps, int(raw.max_episode_length))):
                actions = policy(observations)
                proposed_actions.append(actions[0].cpu().numpy().copy())
                observations, _, dones, _ = wrapped.step(actions)
                if bool(dones[0]):
                    completed = True
                    break
                if (step + 1) % 150 == 0:
                    print(f"CAPTURE seed={seed} step={step+1}", flush=True)
            raw.capture_context = None
            context.writer.close()
            arrays = aligned_arrays(context.states, context.transitions, fps)
            arrays["proposed_policy_actions"] = np.stack(proposed_actions)
            np.savez_compressed(directory / "trajectory.npz", **arrays)
            outcome = "capture_truncated_not_episode_result"
            if completed:
                manager = raw.termination_manager
                if bool(manager.get_term("non_finite")[0]):
                    outcome = "non_finite"
                elif bool(manager.get_term("workspace_exit")[0]):
                    outcome = "workspace_exit"
                elif bool(manager.get_term("pick_place_success")[0]):
                    outcome = "success"
                else:
                    outcome = next(name for name, mask in _phase_failure_masks(raw.last_episode_metrics).items() if bool(mask[0]))
            summary = {
                "seed": seed, "completed_episode": completed, "outcome": outcome,
                "transitions": len(context.transitions), "frames": context.writer.frames,
                "simulation_duration_seconds": len(context.transitions) / fps,
                "initial_cube_pose_w": arrays["cube_pose_w"][0].tolist(),
                "initial_joint_position_rad": arrays["joint_position_rad"][0].tolist(),
                "target_position_w": arrays["target_position_w"][0].tolist(),
                "minimum_xy_error_m": float(np.linalg.norm(arrays["target_position_w"][:, :2] - arrays["cube_pose_w"][:, :2], axis=1).min()),
                "files": {name: {"path": str(directory / name), "sha256": sha256_file(directory / name)}
                          for name in ("rollout.mp4", "trajectory.npz")},
            }
            if summary["frames"] != summary["transitions"] + 1:
                raise RuntimeError("Frame/transition alignment mismatch")
            report["episodes"].append(summary)
            write_json(directory / "episode.json", summary)
            write_json(report_path, report)
            context = None
            print(f"CAPTURE_EPISODE seed={seed} outcome={outcome}", flush=True)
    report["status"] = "captured_diagnostic"


try:
    main()
except BaseException as exc:
    report.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    raise
finally:
    try:
        if context is not None:
            context.writer.close()
    finally:
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        report["resources"] = sampler.stop()
        write_json(report_path, report)
        if wrapped is not None:
            wrapped.close()
        elif env is not None:
            env.close()
        if app is not None:
            app.close()
        print(f"CAPTURE_MANIFEST={report_path}", flush=True)
