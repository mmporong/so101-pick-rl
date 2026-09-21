#!/usr/bin/env python3
"""Fixed 64-reset closed-loop BC diagnostic; not full contact-sequence certification."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "isaaclab"))
from so101_pick_rl.demo_bc import build_policy, file_sha256, load_contract


def diagnostic_flags(observation, initial_z):
    """Batched source predicates, without claiming physical grasp validity."""
    import torch
    if observation.ndim != 2 or observation.shape[1] != 34 or initial_z.shape != (len(observation),):
        raise ValueError("expected (N,34) observations and (N,) initial heights")
    if not torch.isfinite(observation).all() or not torch.isfinite(initial_z).all():
        raise ValueError("diagnostic state must be finite")
    lift = observation[:, 20] - initial_z
    relative = observation[:, 18:21] - observation[:, 31:34]
    release = ((relative[:, :2].abs() < .045).all(dim=1) & (relative[:, 2] > .012)
               & (relative[:, 2] < .075) & (observation[:, 5] > .26)
               & (observation[:, 25:28].norm(dim=1) <= .03)
               & (observation[:, 28:31].norm(dim=1) <= .5))
    return lift, release


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=ROOT / "common/demo_box_spec.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    contract, digest = load_contract(args.contract)
    training = json.loads(args.training_report.read_text(encoding="utf-8"))
    checkpoint_sha = file_sha256(args.checkpoint)
    if training["status"] != "passed" or training["checkpoint"]["sha256"] != checkpoint_sha or training["contract"]["sha256"] != digest:
        raise ValueError("checkpoint/training report/contract provenance mismatch")
    if file_sha256(args.dataset) != contract["dataset"]["expected_sha256"]:
        raise ValueError("dataset SHA mismatch")
    # Fixed before evaluation; source-shard validation is not independent demo lineage.
    ids = list(range(32)) + list(range(400, 432))
    steps, n = 900, len(ids)
    states, targets, sources = [], [], []
    with h5py.File(args.dataset, "r") as dataset:
        for episode_id in ids:
            group = dataset[f"data/demo_{episode_id}"]
            states.append({kind: {name: {field: ds[:] for field, ds in entity.items()}
                                 for name, entity in entities.items()}
                           for kind, entities in group["initial_state"].items()})
            targets.append(group["obs/joint_pos_target"][0])
            sources.append(int(group.attrs["source_index"]))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "so101_pick_rl.demo_bc_rollout.v1", "status": "running",
        "checkpoint_sha256": checkpoint_sha, "task_contract_sha256": digest,
        "training_report_sha256": file_sha256(args.training_report),
        "dataset_sha256": contract["dataset"]["expected_sha256"],
        "num_envs": n, "seed": 0, "control_steps": steps, "episodes": ids,
        "source_indices": sources, "training": False, "ppo_updates": 0,
        "full_sequence_audit": "not_run", "hardware_validated": False,
        "independent_heldout_evaluation": "not_available_original_demo_lineage_unknown",
        "metric_scope": "8cm lift held0.2s and source box release held0.5s; no contact validity certificate",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)),
        "code_sha256": {p: file_sha256(ROOT / p) for p in
                        ("isaaclab/scripts/evaluate_demo_bc.py", "isaaclab/so101_pick_rl/demo_bc.py", "isaaclab/so101_pick_rl/demo_box_runtime.py")},
    }
    if contract["policy"].get("variant") == "previous_target_residual":
        report["code_sha256"]["isaaclab/so101_pick_rl/demo_bc_residual.py"] = file_sha256(ROOT / "isaaclab/so101_pick_rl/demo_bc_residual.py")
    def save():
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    save()
    app = runtime = None
    try:
        app = AppLauncher(args).app
        import torch
        from so101_pick_rl.demo_box_runtime import DemoBoxRuntime
        torch.manual_seed(0)
        runtime = DemoBoxRuntime(args.snapshot, n, args.device, contract_path=args.contract)
        checkpoint = torch.load(args.checkpoint, map_location=runtime.device, weights_only=True)
        if checkpoint["contract_sha256"] != digest or checkpoint["dataset_sha256"] != report["dataset_sha256"]:
            raise ValueError("checkpoint embedded hashes do not match inputs")
        mean, std = checkpoint["observation_mean"], checkpoint["observation_std"]
        if mean.shape != (34,) or std.shape != (34,) or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or not (std > 0).all():
            raise ValueError("invalid checkpoint observation normalizer")
        policy = build_policy(contract, mean, std).to(runtime.device).eval()
        policy.load_state_dict(checkpoint["model_state_dict"])
        state = {kind: {name: {field: torch.as_tensor(np.concatenate([s[kind][name][field] for s in states]), device=runtime.device)
                               for field in entity}
                        for name, entity in entities.items()} for kind, entities in states[0].items()}
        obs = runtime.reset(state, torch.as_tensor(np.stack(targets), device=runtime.device))
        start_z = obs[:, 20].clone()
        peak_lift = torch.zeros(n, device=runtime.device)
        lift_run = torch.zeros(n, dtype=torch.int64, device=runtime.device)
        release_run = lift_run.clone()
        longest_lift = lift_run.clone()
        longest_release = lift_run.clone()
        max_normalized_obs = torch.zeros(n, device=runtime.device)
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.inference_mode():
            for _ in range(steps):
                normalized = (obs - mean) / std
                max_normalized_obs = torch.maximum(max_normalized_obs, normalized.abs().max(dim=1).values)
                obs = runtime.step(policy(normalized))
                lift, release = diagnostic_flags(obs, start_z)
                peak_lift = torch.maximum(peak_lift, lift)
                lift_run = torch.where(lift >= .08, lift_run + 1, 0)
                longest_lift = torch.maximum(longest_lift, lift_run)
                release_run = torch.where(release, release_run + 1, 0)
                longest_release = torch.maximum(longest_release, release_run)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        report["results"] = [
            {"episode": episode_id, "source_index": sources[i],
             "split": "train_shard_reset" if sources[i] < 16 else "validation_shard_reset",
             "maximum_lift_m": float(peak_lift[i].item()),
             "lift_hold_steps": int(longest_lift[i].item()),
             "source_release_hold_steps": int(longest_release[i].item()),
             "lift_8cm_0_2s": bool(longest_lift[i] >= 12),
             "source_stable_release": bool(longest_release[i] >= 30),
             "maximum_abs_normalized_observation": float(max_normalized_obs[i].item())}
            for i, episode_id in enumerate(ids)]
        report.update(status="completed", wall_seconds=elapsed, transitions=n * steps,
                      simulation_steps_per_second=n * steps / elapsed,
                      peak_torch_vram_bytes=torch.cuda.max_memory_allocated(),
                      gpu=torch.cuda.get_device_name(), torch_version=torch.__version__)
        report["summary"] = {split: {"episodes": len(rows),
                                     "lift_8cm_0_2s": sum(row["lift_8cm_0_2s"] for row in rows),
                                     "source_stable_release": sum(row["source_stable_release"] for row in rows)}
                             for split in ("train_shard_reset", "validation_shard_reset")
                             for rows in ([row for row in report["results"] if row["split"] == split],)}
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save()
        if runtime is not None:
            runtime.close()
        if app is not None:
            app.close(wait_for_replicator=False)
    print(json.dumps(report["summary"]))


if __name__ == "__main__":
    main()
