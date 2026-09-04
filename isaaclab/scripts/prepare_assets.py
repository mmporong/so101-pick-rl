#!/usr/bin/env python3
"""Download the pinned LeIsaac SO-101 USD into a local, untracked cache."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path


ASSET_URL = "https://github.com/LightwheelAI/leisaac/releases/download/v0.1.0/so101_follower.usd"
ASSET_SHA256 = "64a877c3b82cdc4a48ab8a1f321a2dd3ef7c55d4b10bce222b58c530d978ae58"


def default_asset_path() -> Path:
    local_app_data = os.environ.get("LOCALAPPDATA")
    if not local_app_data:
        raise RuntimeError("LOCALAPPDATA is unavailable; pass --output explicitly.")
    return Path(local_app_data) / "so101-pick-rl" / "assets" / "leisaac" / "so101_follower.usd"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output = (args.output or default_asset_path()).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.exists() and not args.force:
        actual = sha256(output)
        if actual == ASSET_SHA256:
            print(f"ASSET_READY path={output}")
            print(f"sha256={actual}")
            return 0
        print(f"ASSET_HASH_MISMATCH path={output} actual={actual}", file=sys.stderr)
        return 2

    temporary = output.with_suffix(output.suffix + ".part")
    try:
        with urllib.request.urlopen(ASSET_URL, timeout=60) as response, temporary.open("wb") as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
        actual = sha256(temporary)
        if actual != ASSET_SHA256:
            print(f"ASSET_HASH_MISMATCH expected={ASSET_SHA256} actual={actual}", file=sys.stderr)
            temporary.unlink(missing_ok=True)
            return 2
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)

    print(f"ASSET_READY path={output}")
    print(f"sha256={ASSET_SHA256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
