#!/usr/bin/env python3
"""Export validated numeric trajectories from successful Mimic HDF5 shards.

RGB datasets are identified from metadata and are never decoded. The optional
full-file hash reads all file bytes, including encoded RGB storage. The output is a
new HDF5 file containing renumbered episodes plus an embedded provenance
manifest.  Source files are opened read-only and checked for changes before the
atomic no-clobber publication.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np


SCHEMA = "so101_pick_rl.mimic_numeric_export.v1"
DEFAULT_ALIGNMENT_ATOL = 1.0e-5
READ_TARGET_BYTES = 16 * 1024 * 1024


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _stat(path: Path) -> dict[str, int]:
    value = path.stat()
    return {
        "size": value.st_size,
        "mtime_ns": value.st_mtime_ns,
        "device": value.st_dev,
        "inode": value.st_ino,
    }


def _is_failed_file(path: Path) -> bool:
    return path.stem.lower().endswith("_failed")


def resolve_inputs(values: Iterable[str | Path]) -> list[Path]:
    """Expand repeated files, directories, and explicit glob expressions."""
    candidates: list[Path] = []
    for raw_value in values:
        raw = os.fspath(raw_value)
        if glob.has_magic(raw):
            matches = [Path(item) for item in glob.glob(raw)]
            if not matches:
                raise ValueError(f"input glob matched no files: {raw}")
            candidates.extend(matches)
            continue
        path = Path(raw).expanduser()
        if path.is_dir():
            matches = sorted(path.glob("shard_*.hdf5"))
            if not matches:
                raise ValueError(f"input directory has no shard_*.hdf5 files: {path}")
            candidates.extend(matches)
        else:
            candidates.append(path)

    resolved: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        path = candidate.resolve()
        if _is_failed_file(path):
            continue
        if path.suffix.lower() not in {".h5", ".hdf5"}:
            raise ValueError(f"input is not an HDF5 file: {path}")
        if not path.is_file():
            raise ValueError(f"input file does not exist: {path}")
        if path not in seen:
            resolved.append(path)
            seen.add(path)
    if not resolved:
        raise ValueError("no successful HDF5 inputs remain after excluding *_failed files")
    return sorted(resolved)


def _copy_attrs(source: h5py.AttributeManager, destination: h5py.AttributeManager) -> None:
    for key in source:
        destination[key] = source[key]


def _validate_numeric_attrs(attrs: h5py.AttributeManager, location: str) -> None:
    for key in attrs:
        value = np.asarray(attrs[key])
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError(f"non-finite numeric attribute: {location}@{key}")


def _hash_attrs(digest: Any, attrs: h5py.AttributeManager, location: str) -> None:
    for key in sorted(attrs.keys()):
        value = _json_value(attrs[key])
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        digest.update(f"attr\0{location}\0{key}\0{payload}\0".encode("utf-8"))


def _is_rgb_dataset(path: str, dataset: h5py.Dataset) -> bool:
    if dataset.dtype.kind != "u" or dataset.dtype.itemsize != 1 or dataset.ndim < 3:
        return False
    if dataset.shape[-1] not in (3, 4):
        return False
    tokens = {part.lower() for part in path.replace("-", "_").split("/")}
    named_image = any(
        token == "rgb" or "image" in token or "camera" in token or "wrist" in token or "front" in token
        for token in tokens
    )
    return dataset.ndim >= 4 or named_image


def _dataset_slices(dataset: h5py.Dataset) -> Iterable[tuple[slice, ...]]:
    if dataset.shape == ():
        yield ()
        return
    if dataset.shape[0] == 0:
        return
    frame_items = int(np.prod(dataset.shape[1:], dtype=np.int64)) or 1
    rows = max(1, READ_TARGET_BYTES // max(1, frame_items * dataset.dtype.itemsize))
    for start in range(0, dataset.shape[0], rows):
        yield (slice(start, min(dataset.shape[0], start + rows)),) + (slice(None),) * (dataset.ndim - 1)


def _create_output_dataset(destination: h5py.Group, name: str, source: h5py.Dataset) -> h5py.Dataset:
    kwargs: dict[str, Any] = {}
    if source.shape and all(size > 0 for size in source.shape):
        # Let h5py choose bounded chunks.  Recorder datasets can retain a chunk
        # length larger than a short episode because their source maxshape is
        # extensible; reusing that chunk on this fixed-size export can fail.
        kwargs["chunks"] = True
    return destination.create_dataset(name, shape=source.shape, dtype=source.dtype, **kwargs)


def _copy_numeric_tree(
    source: h5py.Group,
    destination: h5py.Group,
    *,
    source_prefix: str,
    digest: Any,
    rgb_metadata: list[dict[str, Any]],
    skipped_metadata: list[dict[str, Any]],
) -> None:
    _validate_numeric_attrs(source.attrs, source_prefix)
    _copy_attrs(source.attrs, destination.attrs)
    _hash_attrs(digest, source.attrs, source_prefix)
    for name in sorted(source.keys()):
        item = source[name]
        path = f"{source_prefix}/{name}"
        if isinstance(item, h5py.Group):
            child = destination.create_group(name)
            _copy_numeric_tree(
                item,
                child,
                source_prefix=path,
                digest=digest,
                rgb_metadata=rgb_metadata,
                skipped_metadata=skipped_metadata,
            )
            continue
        metadata = {"path": path, "shape": list(item.shape), "dtype": str(item.dtype)}
        if _is_rgb_dataset(path, item):
            rgb_metadata.append(metadata)
            continue
        if not (np.issubdtype(item.dtype, np.number) or np.issubdtype(item.dtype, np.bool_)):
            skipped_metadata.append(metadata)
            continue

        output = _create_output_dataset(destination, name, item)
        _copy_attrs(item.attrs, output.attrs)
        _validate_numeric_attrs(item.attrs, path)
        _hash_attrs(digest, item.attrs, path)
        digest.update(f"dataset\0{path}\0{item.dtype.str}\0{item.shape}\0".encode("utf-8"))
        for selection in _dataset_slices(item):
            values = np.asarray(item[selection] if selection else item[()])
            if not np.isfinite(values).all():
                raise ValueError(f"non-finite numeric dataset: {path}")
            if selection:
                output[selection] = values
            else:
                output[()] = values
            digest.update(np.ascontiguousarray(values).tobytes())


def _explicit_success(episode: h5py.Group, location: str) -> bool:
    if "success" not in episode.attrs:
        raise ValueError(f"episode has no explicit success attribute: {location}")
    value = np.asarray(episode.attrs["success"])
    if value.shape != () or value.dtype.kind not in "biu":
        raise ValueError(f"episode success attribute is not a scalar boolean/0/1: {location}")
    integer = int(value)
    if integer not in (0, 1):
        raise ValueError(f"episode success attribute is not boolean/0/1: {location}")
    return integer == 1


def _required_dataset(episode: h5py.Group, relative_path: str, location: str) -> h5py.Dataset:
    if relative_path not in episode or not isinstance(episode[relative_path], h5py.Dataset):
        raise ValueError(f"required dataset missing: {location}/{relative_path}")
    return episode[relative_path]


def _alignment_check(episode: h5py.Group, location: str, atol: float) -> dict[str, Any]:
    target = _required_dataset(episode, "obs/joint_pos_target", location)
    observed = _required_dataset(episode, "obs/joint_pos", location)
    state = _required_dataset(episode, "states/articulation/robot/joint_position", location)
    actions = _required_dataset(episode, "actions", location)
    if "num_samples" not in episode.attrs:
        raise ValueError(f"episode has no num_samples attribute: {location}")
    num_samples_value = np.asarray(episode.attrs["num_samples"])
    if num_samples_value.shape != () or num_samples_value.dtype.kind not in "iu":
        raise ValueError(f"episode num_samples is not a scalar integer: {location}")
    num_samples = int(num_samples_value)
    if num_samples < 0:
        raise ValueError(f"episode num_samples is negative: {location}")
    if actions.ndim < 1 or actions.shape[0] != num_samples:
        raise ValueError(
            f"actions first axis does not match num_samples at {location}: "
            f"actions={actions.shape}, num_samples={num_samples}"
        )
    if target.ndim < 2:
        raise ValueError(f"joint_pos_target must include time and joint axes: {location}")
    if target.shape != observed.shape:
        raise ValueError(
            f"joint_pos_target and joint_pos shapes differ at {location}: "
            f"target={target.shape}, observed={observed.shape}"
        )
    if observed.shape != state.shape or observed.ndim < 2 or observed.shape[0] < 2:
        raise ValueError(
            f"alignment arrays require equal [time, ...] shapes with at least two frames: "
            f"{location} observed={observed.shape} state={state.shape}"
        )
    if observed.shape[0] != num_samples:
        raise ValueError(f"joint observation time axis does not match num_samples: {location}")
    max_abs_error = 0.0
    rows = max(1, READ_TARGET_BYTES // max(1, int(np.prod(observed.shape[1:])) * observed.dtype.itemsize))
    for start in range(1, observed.shape[0], rows):
        stop = min(observed.shape[0], start + rows)
        lhs = np.asarray(observed[start:stop], dtype=np.float64)
        rhs = np.asarray(state[start - 1 : stop - 1], dtype=np.float64)
        if lhs.size:
            max_abs_error = max(max_abs_error, float(np.max(np.abs(lhs - rhs))))
    result = {
        "comparison": "obs/joint_pos[1:] == states/articulation/robot/joint_position[:-1]",
        "atol": atol,
        "rtol": 0.0,
        "max_abs_error": max_abs_error,
        "passed": max_abs_error <= atol,
        "frames_compared": int(observed.shape[0] - 1),
    }
    if not result["passed"]:
        raise ValueError(
            f"pre/post alignment failed at {location}: max_abs_error={max_abs_error:.9g}, atol={atol:.9g}"
        )
    return result


def _full_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(READ_TARGET_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def export_numeric(
    inputs: Iterable[str | Path],
    output: str | Path,
    *,
    alignment_atol: float = DEFAULT_ALIGNMENT_ATOL,
    full_source_sha256: bool = False,
    skip_failed_episodes: bool = False,
) -> dict[str, Any]:
    if alignment_atol < 0 or not np.isfinite(alignment_atol):
        raise ValueError("alignment_atol must be finite and non-negative")
    sources = resolve_inputs(inputs)
    output_path = Path(output).expanduser().resolve()
    if output_path in sources:
        raise ValueError("input and output paths must differ")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.partial")

    before = {path: _stat(path) for path in sources}
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "output": str(output_path),
        "rgb_policy": "datasets_not_decoded",
        "full_file_bytes_read_for_sha256": full_source_sha256,
        "training_ready": False,
        "label_alignment": "not_applied_original_recording_preserved",
        "source_timing_validation": "pending_training_adapter",
        "physics_replay_validated": False,
        "alignment_contract": {
            "comparison": "obs/joint_pos[1:] == states/articulation/robot/joint_position[:-1]",
            "atol": alignment_atol,
            "rtol": 0.0,
            "failure_policy": "reject_export",
        },
        "sources": [],
        "episodes": [],
        "skipped_episodes": [],
    }

    try:
        with h5py.File(temporary, "w") as destination:
            destination.attrs["schema"] = SCHEMA
            data_output = destination.create_group("data")
            source_metadata = destination.create_group("source_metadata")
            output_episode_index = 0
            total_samples = 0

            for source_index, source_path in enumerate(sources):
                digest = hashlib.sha256()
                digest.update(f"{SCHEMA}\0numeric-content\0".encode("utf-8"))
                rgb_metadata: list[dict[str, Any]] = []
                skipped_metadata: list[dict[str, Any]] = []
                source_record: dict[str, Any] = {
                    "index": source_index,
                    "path": str(source_path),
                    "stat_before": before[source_path],
                    "full_source_sha256": _full_sha256(source_path) if full_source_sha256 else None,
                    "episodes": 0,
                }
                with h5py.File(source_path, "r") as source:
                    _validate_numeric_attrs(source.attrs, "/")
                    source_group = source_metadata.create_group(f"source_{source_index:04d}")
                    source_group.attrs["path"] = str(source_path)
                    _copy_attrs(source.attrs, source_group.attrs)
                    _hash_attrs(digest, source.attrs, "/")
                    if "data" not in source or not isinstance(source["data"], h5py.Group):
                        raise ValueError(f"source has no HDF5 data group: {source_path}")
                    data_source = source["data"]
                    _validate_numeric_attrs(data_source.attrs, "/data")
                    data_attrs = source_group.create_group("data_attrs")
                    _copy_attrs(data_source.attrs, data_attrs.attrs)
                    _hash_attrs(digest, data_source.attrs, "/data")
                    if len(data_source) == 0:
                        raise ValueError(f"source data group contains no episodes: {source_path}")
                    for episode_name in sorted(data_source.keys()):
                        episode = data_source[episode_name]
                        location = f"{source_path}:/data/{episode_name}"
                        if not isinstance(episode, h5py.Group):
                            raise ValueError(f"data member is not an episode group: {location}")
                        succeeded = _explicit_success(episode, location)
                        if not succeeded:
                            if not skip_failed_episodes:
                                raise ValueError(f"failed episode is not accepted by numeric exporter: {location}")
                            skipped = {
                                "source_index": source_index,
                                "source_episode": episode_name,
                                "reason": "success=false",
                            }
                            manifest["skipped_episodes"].append(skipped)
                            source_record.setdefault("skipped_episodes", []).append(skipped)
                            continue
                        alignment = _alignment_check(episode, location, alignment_atol)
                        output_name = f"demo_{output_episode_index}"
                        output_episode = data_output.create_group(output_name)
                        _copy_numeric_tree(
                            episode,
                            output_episode,
                            source_prefix=f"/data/{episode_name}",
                            digest=digest,
                            rgb_metadata=rgb_metadata,
                            skipped_metadata=skipped_metadata,
                        )
                        output_episode.attrs["source_index"] = source_index
                        output_episode.attrs["source_path"] = str(source_path)
                        output_episode.attrs["source_episode"] = episode_name
                        samples = int(episode.attrs.get("num_samples", episode["actions"].shape[0]))
                        total_samples += samples
                        manifest["episodes"].append({
                            "output_episode": output_name,
                            "source_index": source_index,
                            "source_episode": episode_name,
                            "num_samples": samples,
                            "alignment": alignment,
                        })
                        output_episode_index += 1
                        source_record["episodes"] += 1

                source_record["numeric_content_sha256"] = digest.hexdigest()
                source_record["rgb_datasets"] = rgb_metadata
                source_record["skipped_nonnumeric_datasets"] = skipped_metadata
                manifest["sources"].append(source_record)

            if output_episode_index == 0:
                raise ValueError("numeric export contains zero successful episodes")
            data_output.attrs["total"] = total_samples
            data_output.attrs["episode_count"] = output_episode_index
            manifest["episode_count"] = output_episode_index
            manifest["total_samples"] = total_samples

            after = {path: _stat(path) for path in sources}
            changed = [str(path) for path in sources if after[path] != before[path]]
            if changed:
                raise RuntimeError(f"input files changed during export: {changed}")
            for record, path in zip(manifest["sources"], sources, strict=True):
                record["stat_after"] = after[path]
                record["unchanged_during_export"] = True

            manifest_group = destination.create_group("manifest")
            manifest_json = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False)
            manifest_group.create_dataset("json", data=manifest_json, dtype=h5py.string_dtype("utf-8"))
            destination.flush()
        # A hard-link publication is atomic and fails if another process wins
        # the output path race.  Unlike os.replace, it can never overwrite an
        # output created after the initial existence check.
        os.link(temporary, output_path)
        temporary.unlink()
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Input HDF5 file, directory (shard_*.hdf5), or glob; repeatable.",
    )
    parser.add_argument("--output", required=True, help="New numeric-only HDF5 output path.")
    parser.add_argument("--alignment-atol", type=float, default=DEFAULT_ALIGNMENT_ATOL)
    parser.add_argument(
        "--full-source-sha256",
        action="store_true",
        help="Also hash every source byte; expensive for RGB-heavy shards.",
    )
    parser.add_argument(
        "--skip-failed-episodes",
        action="store_true",
        help="Skip episodes with an explicit success=false; malformed or missing success still fails.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = export_numeric(
        args.input,
        args.output,
        alignment_atol=args.alignment_atol,
        full_source_sha256=args.full_source_sha256,
        skip_failed_episodes=args.skip_failed_episodes,
    )
    print(f"output={Path(args.output).expanduser().resolve()}")
    print(f"episodes={manifest['episode_count']}")
    print(f"total_samples={manifest['total_samples']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
