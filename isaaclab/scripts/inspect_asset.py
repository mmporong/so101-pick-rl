#!/usr/bin/env python3
"""Inspect the pinned robot USD through the installed Isaac Sim runtime."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher


def default_asset_path() -> Path:
    override = os.environ.get("SO101_RL_ASSET_PATH")
    if override:
        return Path(override).expanduser().resolve()
    return Path(os.environ["LOCALAPPDATA"]) / "so101-pick-rl" / "assets" / "leisaac" / "so101_follower.usd"


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--asset", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402


def main() -> int:
    asset_path = (args.asset or default_asset_path()).expanduser().resolve()
    if not asset_path.is_file():
        raise FileNotFoundError(asset_path)

    stage = Usd.Stage.Open(str(asset_path))
    if stage is None:
        raise RuntimeError(f"Unable to open USD: {asset_path}")

    summary = {
        "asset": str(asset_path),
        "default_prim": stage.GetDefaultPrim().GetPath().pathString,
        "meters_per_unit": stage.GetMetadata("metersPerUnit"),
        "up_axis": stage.GetMetadata("upAxis"),
        "prims": [],
    }
    for prim in stage.Traverse():
        is_joint = prim.IsA(UsdPhysics.Joint)
        is_rigid = prim.HasAPI(UsdPhysics.RigidBodyAPI)
        if not (is_joint or is_rigid):
            continue
        row = {
            "path": prim.GetPath().pathString,
            "type": prim.GetTypeName(),
            "rigid_body": is_rigid,
            "applied_schemas": list(prim.GetAppliedSchemas()),
        }
        if prim.IsA(UsdGeom.Xformable):
            transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            translation = transform.ExtractTranslation()
            row["translation"] = [translation[0], translation[1], translation[2]]
        if is_joint:
            joint = UsdPhysics.Joint(prim)
            row["body0"] = [str(path) for path in joint.GetBody0Rel().GetTargets()]
            row["body1"] = [str(path) for path in joint.GetBody1Rel().GetTargets()]
            for attribute_name in (
                "physics:axis",
                "physics:localPos0",
                "physics:localPos1",
                "physics:lowerLimit",
                "physics:upperLimit",
            ):
                attribute = prim.GetAttribute(attribute_name)
                if attribute and attribute.HasAuthoredValueOpinion():
                    value = attribute.Get()
                    row[attribute_name] = list(value) if hasattr(value, "__len__") and not isinstance(value, str) else value
        summary["prims"].append(row)

    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


try:
    raise SystemExit(main())
finally:
    simulation_app.close()
