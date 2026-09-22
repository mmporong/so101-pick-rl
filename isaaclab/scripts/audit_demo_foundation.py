#!/usr/bin/env python3
"""Measure source reset discontinuity and opening geometry without editing data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import h5py

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "isaaclab"))
from so101_pick_rl.demo_bc import file_sha256, load_contract
from so101_pick_rl.demo_action_contract import validate_source_timing
from so101_pick_rl.demo_foundation_audit import audit_episode, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=ROOT / "common/demo_box_spec.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract, contract_sha = load_contract(args.contract)
    dataset_sha = file_sha256(args.dataset)
    if dataset_sha != contract["dataset"]["expected_sha256"]:
        raise ValueError("source dataset hash mismatch")
    if args.output.exists():
        raise FileExistsError(args.output)
    episodes = {}
    with h5py.File(args.dataset, "r") as source:
        for metadata in source["source_metadata"].values():
            validate_source_timing(json.loads(metadata["data_attrs"].attrs["env_args"]))
        for name, group in source["data"].items():
            episodes[name] = audit_episode(
                group["obs/joint_pos"][:], group["obs/joint_pos_target"][:],
                group["states/articulation/robot/joint_position"][:],
                group["states/rigid_object/cube/root_pose"][:],
                group["states/rigid_object/box_target/root_pose"][:],
                group["states/rigid_object/cube/root_velocity"][:],
                float(group["initial_state/rigid_object/cube/root_pose"][0, 2]),
            )
    report = {
        "schema": "so101_pick_rl.demo_foundation_audit.v1", "status": "completed",
        "dataset_sha256": dataset_sha, "contract_sha256": contract_sha,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)),
        "code_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): file_sha256(p) for p in
                        (Path(__file__), ROOT / "isaaclab/so101_pick_rl/demo_foundation_audit.py")},
        "predicate": "first measured gripper q crossing above 0.26 rad after a recorded 8 cm lift; same poststate geometry and velocity",
        "limitations": ["Joint opening is not contact-verified object release", "No contact data: support and full success remain unmeasured",
                        "Reset discontinuity indicates a missing training region, not proof of the sole rollout failure cause"],
        "summary": summarize(episodes), "episodes": episodes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2, allow_nan=False)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
