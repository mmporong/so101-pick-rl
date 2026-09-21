#!/usr/bin/env python3
"""Audit carry fragments without approving BC, PPO, or full-task success."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "isaaclab"))
from so101_pick_rl.demo_action_contract import normalize_targets
from so101_pick_rl.demo_segments import carry_ranges, state_action_pair
from so101_pick_rl.demo_sequence import sequence_features
import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_trace_digest(path, expected, *, allow_unbound=False):
    if expected is None and not allow_unbound:
        raise ValueError("trace digest missing from replay report")
    digest = sha256(path)
    if expected is not None and expected != digest:
        raise ValueError("trace digest mismatch")
    return digest


def audit(report_path, *, allow_unbound_trace=False):
    report_path = Path(report_path).resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["status"] != "replay_complete_sequence_audited":
        raise ValueError("completed replay report required")
    if report["physics_substeps"] != 1 or report["control_dt_s"] != 1/60:
        raise ValueError("candidate adapter requires consecutive 60 Hz states")
    required_code = {"replay_demo_box.py", "demo_sequence.py", "demo_replay_metrics.py",
                     "grasp_audit.py", "demo_action_contract.py", "demo_box_pad_geometry.json"}
    if not required_code.issubset(report["code_sha256"]):
        raise ValueError("incomplete replay provenance")
    for name, digest in report["code_sha256"].items():
        if sha256(report_path.parent / "code" / name) != digest:
            raise ValueError("replay code snapshot hash mismatch: " + name)
    geometry_path = report_path.parent / "code/demo_box_pad_geometry.json"
    if sha256(geometry_path) != report["geometry_sha256"]:
        raise ValueError("geometry hash mismatch")
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    cfg = report["effective_sequence_gate"]
    if cfg.get("schema") != "so101_pick_rl.demo_box_sequence_gate.v2" or cfg.get("orientation_mode") != "world_approach_axis":
        raise ValueError("v2 world-axis audit required")
    if hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest() != report["effective_sequence_gate_sha256"]:
        raise ValueError("effective gate hash mismatch")
    limits = np.asarray(report["limits_rad"], dtype=float)
    if limits.shape != (6,2):
        raise ValueError("six joint limits required")
    results = []
    for result in report["results"]:
        trace = report_path.parent / (result["episode"] + "_trace.jsonl")
        expected_digest = result.get("trace_sha256")
        trace_digest = verify_trace_digest(trace, expected_digest, allow_unbound=allow_unbound_trace)
        records = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
        if not records or len(records) != result["total_executed_steps"]:
            raise ValueError("trace length mismatch")
        features = [sequence_features(record, geometry, cfg) for record in records]
        ranges = carry_ranges(features, cfg)
        count = 0
        for start, end in ranges:
            for index in range(start+1, end+1):
                _, target = state_action_pair(records[index-1], records[index])
                normalize_targets(target[None,:], limits[:,0], limits[:,1])
                count += 1
        results.append({"episode": result["episode"], "source_episode": result["source_episode"],
                        "source_path": result["source_path"], "outcomes": result["outcomes"],
                        "sequence_failures": result["sequence_gate"]["failures"],
                        "trace_path": str(trace), "trace_sha256": trace_digest,
                        "trace_bound_to_replay_report": expected_digest is not None,
                        "carry_state_ranges_1based_inclusive": [[a+1,b+1] for a,b in ranges],
                        "aligned_carry_transitions": count})
    return {"schema": "so101_pick_rl.carry_candidate_audit.v1", "report_path": str(report_path),
            "report_sha256": sha256(report_path), "dataset_sha256": report["dataset_sha256"],
            "gate_sha256": report["effective_sequence_gate_sha256"],
            "audit_code_sha256": {p.name: sha256(p) for p in
                (Path(__file__), ROOT / "isaaclab/so101_pick_rl/demo_segments.py",
                 ROOT / "isaaclab/so101_pick_rl/demo_sequence.py",
                 ROOT / "isaaclab/so101_pick_rl/demo_action_contract.py",
                 ROOT / "isaaclab/so101_pick_rl/demo_replay_metrics.py",
                 ROOT / "isaaclab/so101_pick_rl/grasp_audit.py")},
            "scope": "carry fragments, possibly after an episode failure; no episode is rehabilitated",
            "asset_hash_scope": "reported runtime hashes; original asset and dataset files not rehashed by this audit",
            "training_ready": False, "full_task_success": False, "hardware_validated": False,
            "limitations": ["Approach, release, and recovery coverage not established.",
                            "No train/held-out lineage split or DemoBox training contract approved."],
            "results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-unbound-trace", action="store_true",
                        help="Legacy diagnostics only; explicitly mark traces lacking a report-bound digest")
    args = parser.parse_args()
    result = audit(args.report, allow_unbound_trace=args.allow_unbound_trace)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({"episodes": len(result["results"]), "carry_transitions":
        sum(r["aligned_carry_transitions"] for r in result["results"]), "training_ready": False}))
