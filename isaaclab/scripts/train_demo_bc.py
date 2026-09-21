#!/usr/bin/env python3
"""Train the bounded DemoBox approach/grasp/carry-prefix BC baseline."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
ISAACLAB_DIR = SCRIPT_DIR.parent
REPOSITORY_ROOT = ISAACLAB_DIR.parent
sys.path.insert(0, str(ISAACLAB_DIR))

from so101_pick_rl.demo_bc import (  # noqa: E402
    build_policy,
    file_sha256,
    fit_observation_normalizer,
    load_bc_dataset,
    load_contract,
    normalize_observations,
    verify_smoke_reports,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--expected-contract-sha256", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--smoke-report", type=Path, nargs=2, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("epochs, batch-size, and learning-rate must be positive")
    return args


def git_metadata() -> dict[str, object]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(REPOSITORY_ROOT), *args], check=True,
            capture_output=True, text=True,
        ).stdout.strip()

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(status),
        "dirty_paths": status.splitlines(),
    }


def evaluate_loss(model, observations, actions, device: torch.device, batch_size: int) -> float:
    model.eval()
    total_squared_error = 0.0
    values = 0
    with torch.no_grad():
        for start in range(0, len(observations), batch_size):
            obs = torch.from_numpy(observations[start:start + batch_size]).to(device)
            target = torch.from_numpy(actions[start:start + batch_size]).to(device)
            prediction = model(obs)
            total_squared_error += torch.nn.functional.mse_loss(
                prediction, target, reduction="sum",
            ).item()
            values += target.numel()
    return total_squared_error / values


def tensors_finite(value) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, dict):
        return all(tensors_finite(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return all(tensors_finite(child) for child in value)
    return True


def main() -> int:
    args = parse_args()
    started_at = datetime.now(timezone.utc)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse BC output directory: {output_dir}")

    contract, contract_sha = load_contract(args.contract, args.expected_contract_sha256)
    dataset_path = args.dataset.expanduser().resolve()
    dataset_sha = file_sha256(dataset_path)
    expected_dataset_sha = contract["dataset"]["expected_sha256"]
    if dataset_sha != expected_dataset_sha:
        raise ValueError(f"dataset SHA-256 mismatch: expected {expected_dataset_sha}, got {dataset_sha}")
    smoke_reports = verify_smoke_reports(args.smoke_report, contract_sha)
    dataset = load_bc_dataset(dataset_path, contract)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA BC was requested but CUDA is unavailable")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    mean, std = fit_observation_normalizer(dataset.train_observations)
    train_obs = normalize_observations(dataset.train_observations, mean, std)
    validation_obs = normalize_observations(dataset.validation_observations, mean, std)
    model = build_policy().to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    history = []
    train_started = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(train_obs), generator=generator).numpy()
        squared_error = 0.0
        values = 0
        for start in range(0, len(order), args.batch_size):
            indices = order[start:start + args.batch_size]
            obs = torch.from_numpy(train_obs[indices]).to(device)
            target = torch.from_numpy(dataset.train_actions[indices]).to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(obs)
            loss = torch.nn.functional.mse_loss(prediction, target)
            loss.backward()
            optimizer.step()
            squared_error += torch.nn.functional.mse_loss(
                prediction.detach(), target, reduction="sum",
            ).item()
            values += target.numel()
        history.append({
            "epoch": epoch + 1,
            "train_mse": squared_error / values,
            "validation_mse": evaluate_loss(
                model, validation_obs, dataset.validation_actions, device, args.batch_size,
            ),
        })
        if not np.isfinite(history[-1]["train_mse"]) or not np.isfinite(history[-1]["validation_mse"]):
            raise RuntimeError("BC produced a non-finite train or validation loss")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - train_started

    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = output_dir / "demo_bc.pt"
    checkpoint = {
        "schema": "so101_pick_rl.demo_bc_checkpoint.v1",
        "model_state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "observation_mean": torch.from_numpy(mean),
        "observation_std": torch.from_numpy(std),
        "contract_sha256": contract_sha,
        "dataset_sha256": dataset_sha,
        "seed": args.seed,
        "num_envs": 0,
        "epochs": args.epochs,
        "architecture": [34, 128, 128, 6],
        "output_activation": "tanh",
        "purpose": "original_source_task_approach_grasp_carry_prefix_initialization",
        "supported_placement_training_eligible": False,
    }
    if not tensors_finite(checkpoint):
        raise RuntimeError("BC checkpoint contains a non-finite tensor")
    torch.save(checkpoint, checkpoint_path)
    saved_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not tensors_finite(saved_checkpoint):
        raise RuntimeError("serialized BC checkpoint contains a non-finite tensor")
    checkpoint_sha = file_sha256(checkpoint_path)
    git = git_metadata()
    gpu = None
    peak_vram = None
    if device.type == "cuda":
        gpu = {
            "name": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
            "torch_cuda_version": torch.version.cuda,
        }
        peak_vram = int(torch.cuda.max_memory_allocated(device))
    report = {
        "schema": "so101_pick_rl.demo_bc_run.v1",
        "status": "passed",
        "classification": "measured_loss_only",
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "original_source_task_approach_grasp_carry_prefix_initialization",
        "supported_placement_training_eligible": False,
        "full_task_policy_success_claimed": False,
        "validation_loss_is_policy_success": False,
        "lineage_independence_known": False,
        "independent_heldout_success_claimed": False,
        "contract": {"path": str(Path(args.contract).expanduser().resolve()), "sha256": contract_sha},
        "dataset": {"path": str(dataset_path), "sha256": dataset_sha, "audit": dataset.audit},
        "smoke_reports": [str(Path(path).resolve()) for path in args.smoke_report],
        "smoke_num_envs": sorted(report["num_envs"] for report in smoke_reports),
        "git": git,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "device": str(device),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "gpu": gpu,
            "training_wall_clock_seconds": training_seconds,
            "peak_vram_bytes": peak_vram,
            "simulation_steps_per_second": None,
        },
        "transitions": {
            "train": dataset.audit["train_transitions"],
            "validation": dataset.audit["validation_transitions"],
            "optimizer_training_transition_visits": dataset.audit["train_transitions"] * args.epochs,
        },
        "evaluation": {
            "id": {"status": "not_run", "success_rate": None},
            "held_out": {"status": "not_run", "success_rate": None},
            "policy_success_claimed": False,
        },
        "validation_checks": {
            "finite_train_loss": all(np.isfinite(row["train_mse"]) for row in history),
            "finite_validation_loss": all(np.isfinite(row["validation_mse"]) for row in history),
            "checkpoint_tensors_finite": True,
            "one_and_64_environment_smokes_passed": True,
            "source_timing_all_shards_validated": set(
                dataset.audit["source_timing_validated_indices"]
            ) == set(dataset.audit["train_source_indices"] + dataset.audit["validation_source_indices"]),
        },
        "loss_history": history,
        "final_train_mse": history[-1]["train_mse"],
        "final_validation_mse": history[-1]["validation_mse"],
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha,
            "size_bytes": checkpoint_path.stat().st_size,
        },
        "implementation": {
            "module": {
                "path": str((ISAACLAB_DIR / "so101_pick_rl" / "demo_bc.py").resolve()),
                "sha256": file_sha256(ISAACLAB_DIR / "so101_pick_rl" / "demo_bc.py"),
            },
            "script": {
                "path": str(Path(__file__).resolve()),
                "sha256": file_sha256(Path(__file__).resolve()),
            },
        },
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"BC_REPORT={report_path}")
    print(f"BC_CHECKPOINT={checkpoint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
