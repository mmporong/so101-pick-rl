#!/usr/bin/env python3
"""Validate run manifests against the repository's flat JSON schema without extra packages."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPOSITORY_ROOT / "common" / "schemas" / "run_manifest.schema.json"


def matches_type(value: Any, expected: str) -> bool:
    if expected == "null":
        return value is None
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "string":
        return isinstance(value, str)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    return False


def validate(payload: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = set(schema.get("required", []))
    properties = schema.get("properties", {})
    missing = sorted(required - payload.keys())
    if missing:
        errors.append(f"missing required fields: {missing}")
    if schema.get("additionalProperties") is False:
        extra = sorted(payload.keys() - properties.keys())
        if extra:
            errors.append(f"unexpected fields: {extra}")

    for name, value in payload.items():
        rules = properties.get(name)
        if rules is None:
            continue
        expected_types = rules.get("type")
        if expected_types:
            if isinstance(expected_types, str):
                expected_types = [expected_types]
            if not any(matches_type(value, expected) for expected in expected_types):
                errors.append(f"{name}: expected type {expected_types}, got {type(value).__name__}")
                continue
        if "const" in rules and value != rules["const"]:
            errors.append(f"{name}: expected {rules['const']!r}, got {value!r}")
        if "enum" in rules and value not in rules["enum"]:
            errors.append(f"{name}: value {value!r} not in {rules['enum']!r}")
        if isinstance(value, str):
            if len(value) < rules.get("minLength", 0):
                errors.append(f"{name}: string is too short")
            if "pattern" in rules and re.fullmatch(rules["pattern"], value) is None:
                errors.append(f"{name}: does not match {rules['pattern']!r}")
            if rules.get("format") == "date-time":
                try:
                    datetime.fromisoformat(value.replace("Z", "+00:00"))
                except ValueError:
                    errors.append(f"{name}: invalid date-time {value!r}")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if "minimum" in rules and value < rules["minimum"]:
                errors.append(f"{name}: {value} is below {rules['minimum']}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", nargs="+", type=Path)
    args = parser.parse_args()
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    failed = False
    for path in args.manifests:
        payload = json.loads(path.read_text(encoding="utf-8"))
        errors = validate(payload, schema)
        if errors:
            failed = True
            for error in errors:
                print(f"MANIFEST_FAIL {path}: {error}", file=sys.stderr)
        else:
            print(f"MANIFEST_PASS {path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
