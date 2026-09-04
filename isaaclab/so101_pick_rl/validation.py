"""Pure validation helpers shared by runtime scripts and unit tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def count_reset_events(terminated_flags: Sequence[bool], truncated_flags: Sequence[bool]) -> int:
    """Count unique reset events without double-counting overlapping done flags."""
    if len(terminated_flags) != len(truncated_flags):
        raise ValueError("terminated and truncated flags must have the same length")
    return sum(bool(terminated) or bool(truncated) for terminated, truncated in zip(terminated_flags, truncated_flags))


def summarize_reset_failures(reason_flags: Mapping[str, Sequence[bool]]) -> dict[str, object]:
    """Summarize per-reset validation failures by reason and unique reset event."""
    if not reason_flags:
        return {"failure_count": 0, "reason_counts": {}}

    lengths = {len(flags) for flags in reason_flags.values()}
    if len(lengths) != 1:
        raise ValueError("all reset validation reason flags must have the same length")

    event_count = lengths.pop()
    failed_events = [False] * event_count
    reason_counts: dict[str, int] = {}
    for reason, flags in reason_flags.items():
        normalized = [bool(flag) for flag in flags]
        reason_counts[reason] = sum(normalized)
        failed_events = [failed or current for failed, current in zip(failed_events, normalized)]
    return {"failure_count": sum(failed_events), "reason_counts": reason_counts}


def finalize_smoke_checks(
    base_checks: Mapping[str, bool],
    termination_counts: Mapping[str, int],
    *,
    is_so101_task: bool,
    is_g3_run: bool,
    automatic_reset_count: int,
    reset_failure_count: int,
    resource_sample_count: int,
    simulator_log_path: str | None,
    simulator_error_count: int | None,
) -> dict[str, bool]:
    """Return all checks that must pass before a smoke run can be accepted."""
    checks = dict(base_checks)
    checks["resource_samples_present"] = resource_sample_count > 0
    checks["simulator_log_captured"] = bool(simulator_log_path)
    checks["simulator_error_free"] = simulator_error_count == 0
    if is_so101_task:
        checks["no_non_finite_termination"] = termination_counts.get("non_finite", 0) == 0
        checks["no_workspace_exit"] = termination_counts.get("workspace_exit", 0) == 0
    if is_g3_run:
        checks["at_least_100_automatic_resets"] = automatic_reset_count >= 100
        checks["reset_failure_count_zero"] = reset_failure_count == 0
    return checks
