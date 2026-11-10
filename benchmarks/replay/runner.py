"""Paired and persistent-sequence geometry replay with honest timing boundaries."""

import hashlib
import json
import subprocess
import tempfile
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from benchmarks.replay.backends import (
    CONTRACTS,
    Output,
    UnsupportedState,
    canonical_rows,
    create_backend,
)
from benchmarks.replay.fixture import digest, motion_features
from benchmarks.run import cpu_quota_delta, cpu_quota_snapshot


def output_hash(rows):
    return hashlib.sha256(np.asarray(rows, dtype="<i8").tobytes()).hexdigest()


def topology_churn(previous, current):
    if len(previous) == 0 or len(current) == 0:
        return None
    dtype = np.dtype((np.void, previous.shape[1] * np.dtype(np.int64).itemsize))
    a, b = [
        np.ascontiguousarray(rows, dtype=np.int64).view(dtype).ravel()
        for rows in [previous, current]
    ]
    shared = int(np.intersect1d(a, b, assume_unique=True).size)
    return dict(
        previous_rows=len(previous),
        current_rows=len(current),
        retained_rows=shared,
        removed_fraction=(len(previous) - shared) / len(previous),
        added_fraction=(len(current) - shared) / len(current),
        scope="offline set comparison; not a cheap routing signal",
    )


def exact_check(frame, rows, binary):
    """Run the existing exhaustive CGAL checker on a nonmatching tetrahedron set.

    Orientation preprocessing is only a convenience. Acceptance still requires
    the checker to verify every physical orientation and global sphere exactly.
    """
    points, ids = frame.canonical()
    local = np.searchsorted(ids, rows)
    if np.any(local >= len(ids)) or not np.array_equal(ids[local], rows):
        return dict(valid=False, error="output references an inactive/unknown site ID")
    xyz = points.astype(np.float64)[local]
    negative = np.linalg.det(xyz[:, 1:] - xyz[:, :1]) < 0
    local[negative, :2] = local[negative, 1::-1]
    with tempfile.TemporaryDirectory(prefix="warpower-exact-check-") as directory:
        root = Path(directory)
        source, target = root / "points.txt", root / "tets.txt"
        with source.open("w") as stream:
            stream.write(f"{len(points)}\n")
            np.savetxt(stream, points, fmt="%.17g")
        with target.open("w") as stream:
            stream.write(f"{len(local)}\n")
            np.savetxt(stream, local, fmt="%d")
        result = subprocess.run(
            [str(Path(binary).resolve()), str(source), str(target)], text=True, capture_output=True
        )
    return dict(
        valid=result.returncode == 0,
        exit_code=result.returncode,
        stdout=result.stdout.strip(),
        stderr=result.stderr.strip(),
        scope="exact physical orientation, global empty sphere, pairing, coverage, supporting hull, and incidence checks; accepts geometric alternatives, does not assert identical symbolic tie policy",
    )


def reference(frame, kind="scipy", cgal_binary=None):
    from scipy.spatial import Delaunay, QhullError

    points, ids = frame.canonical()
    if not np.isfinite(points).all():
        return None, "nonfinite reference input", {}
    # Both exact and floating references use canonical first-ID duplicate
    # ownership, matching the site's identity convention rather than insertion
    # order chosen internally by a reference library.
    _, first = np.unique(points, axis=0, return_index=True)
    first.sort()
    points, ids = points[first], ids[first]
    if kind == "cgal":
        if cgal_binary is None:
            raise ValueError("CGAL validation requires --cgal-binary")
        with tempfile.TemporaryDirectory(prefix="warpower-reference-") as directory:
            root = Path(directory)
            source, target = root / "points.txt", root / "tets.txt"
            with source.open("w") as stream:
                stream.write(f"{len(points)}\n")
                np.savetxt(stream, points, fmt="%.17g")
            process = subprocess.run(
                [str(Path(cgal_binary).resolve()), str(source), str(target)],
                capture_output=True,
                text=True,
            )
            if process.returncode:
                return (
                    None,
                    f"CGAL reference exited {process.returncode}: {process.stderr.strip()}",
                    {},
                )
            metadata = json.loads(process.stderr)
            if int(target.read_text().splitlines()[0]) == 0:
                return None, "reference has no full-dimensional tetrahedra", metadata
            values = np.loadtxt(target, skiprows=1, dtype=np.int64).reshape(-1, 4)
            return canonical_rows(ids[values]), None, metadata
    try:
        rows = canonical_rows(ids[Delaunay(points.astype(np.float64)).simplices])
        return rows, None, {}
    except (QhullError, ValueError) as error:
        return None, f"{type(error).__name__}: {str(error).splitlines()[0]}", {}


def _execute(backend, frame):
    prepare_started = time.perf_counter()
    prepared = backend.prepare(frame)
    backend.synchronize()
    prepare_seconds = time.perf_counter() - prepare_started
    started = time.perf_counter()
    failure_kind = None
    try:
        backend.update(prepared)
        backend.synchronize()
    except (RuntimeError, ValueError) as error:
        backend.synchronize()
        failure_kind = "unsupported-state" if isinstance(error, UnsupportedState) else "exception"
        output = Output(np.empty((0, 4), np.int64), f"{type(error).__name__}: {error}", {})
        backend.close()
        backend.ids = None
    seconds = time.perf_counter() - started
    if failure_kind is None:
        output = backend.readback()
        if output.error is not None:
            failure_kind = "reported-construction-failure"
    row = dict(
        seconds=seconds,
        prepare_seconds=prepare_seconds,
        failure_kind=failure_kind,
        error=output.error,
        output_rows=len(output.rows),
        output_sha256=output_hash(output.rows),
        diagnostics=output.diagnostics,
    )
    return row, output


def supervise(command, *, timeout, output_path):
    """Bound a whole optional-reference process, including native calls holding the GIL.

    Python signal/thread timeouts cannot reliably interrupt a stuck extension.
    Keep partial replay evidence and let the OS terminate the entire child.
    """
    try:
        return subprocess.run(command, timeout=timeout, check=False).returncode
    except subprocess.TimeoutExpired:
        path = Path(output_path)
        report = json.loads(path.read_text()) if path.exists() else {}
        report.update(
            complete=False,
            process_timeout_seconds=timeout,
            failure="The replay worker exceeded the total wall-clock limit and was terminated; partial samples are retained. This is not a measured per-update timeout.",
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n")
        return 124


def run(
    fixture,
    *,
    names,
    mode="sequence",
    transitions=None,
    repeats=5,
    warmups=1,
    seed=2026,
    validate="scipy",
    cgal_binary=None,
    exact_checker=None,
    options=None,
    output_path=None,
):
    if mode not in ["pairs", "sequence"]:
        raise ValueError("mode must be pairs or sequence")
    if repeats < 1 or warmups < 1:
        raise ValueError("at least one measured repetition and one warm-up are required")
    if not names or len(set(names)) != len(names):
        raise ValueError("backend names must be nonempty and unique")
    contracts = {CONTRACTS[name] for name in names}
    if len(contracts) != 1:
        raise ValueError("full-hull tetrahedra and bounded adjacency require separate replay runs")
    contract = contracts.pop()
    if contract != "full-hull-tetrahedra" and validate != "none":
        raise ValueError("full-hull reference validation is not a bounded-adjacency certificate")
    fixture.validate()
    indices = list(range(1, len(fixture))) if transitions is None else sorted(set(transitions))
    if not indices or indices[0] < 1 or indices[-1] >= len(fixture):
        raise ValueError("transitions must name existing current frames after frame zero")
    if mode == "sequence" and indices != list(range(1, len(fixture))):
        raise ValueError("sequence replay must traverse every frame to retain legitimate state")
    frames = [fixture.frame(i) for i in range(len(fixture))]
    expected = {}
    if validate != "none":
        # Includes each previous input, so timing cannot start from an unvalidated
        # previous mesh merely because the current result happens to look right.
        for i in sorted(set(indices + [i - 1 for i in indices])):
            expected[i] = reference(frames[i], validate, cgal_binary)
    report = dict(
        fixture_manifest_sha256=fixture.sha256,
        fixture_name=fixture.document["name"],
        fixture_limitations=fixture.document["limitations"],
        contract=contract,
        mode=mode,
        repeats=repeats,
        warmups=warmups,
        randomized_order_seed=seed,
        options={
            key: str(value) if isinstance(value, Path) else value
            for key, value in (options or {}).items()
        },
        source_sha256={
            str(path): digest(path)
            for path in [
                *sorted(Path(__file__).parent.glob("*.py")),
                Path(__file__).parents[1] / "incremental_replay.py",
            ]
        },
        warpower_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        reference=validate,
        reference_binary_sha256=digest(cgal_binary) if validate == "cgal" else None,
        exact_checker_sha256=digest(exact_checker) if exact_checker is not None else None,
        timing_scope="Original-coordinate canonicalization and host-to-device input staging occur before each timed update. Device copies, internal point reordering, workspace allocation on count changes, geometry work, synchronization, and state switching occur inside timing. Readback, output hashing, independent reference checking and offline motion features are outside. Sequence trials retain each backend's state across every frame; paired trials rebuild and check the previous input before timing each transition.",
        validation_scope="SciPy is a floating-point reference; the optional CGAL executable constructs and validates an exact-predicate reference. Both comparisons require tetrahedron-set equality after canonical duplicate ownership. Cospherical alternatives may legitimately differ and are reported as mismatches, not accepted as equivalent. VTM's complete runtime exact/symbolic certificate stays enabled. No public API or default changes are implied.",
        cold_initializations=[],
        order=[],
        records={
            str(i): dict(
                frame=i,
                metadata=frames[i].record,
                motion=motion_features(frames[i - 1], frames[i]),
                variants={name: [] for name in names},
            )
            for i in indices
        },
    )
    report["versions"] = {}
    for package in ["numpy", "scipy", "warp-lang", "torch"]:
        try:
            report["versions"][package] = version(package)
        except PackageNotFoundError:
            pass
    rng = np.random.default_rng(seed)
    exact_cache = {}

    def check(row, result, index):
        if validate != "none":
            ref, error, metadata = expected[index]
            row["reference_error"] = error
            row["reference_metadata"] = metadata
            row["matches_reference"] = (
                bool(np.array_equal(ref, result.rows))
                if ref is not None and result.error is None
                else None
            )
            if row["matches_reference"] is False and exact_checker is not None:
                key = (index, row["output_sha256"])
                if key not in exact_cache:
                    exact_cache[key] = exact_check(frames[index], result.rows, exact_checker)
                row["exact_geometric_check"] = exact_cache[key]

    def save():
        if output_path is not None:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    for repeat in range(-warmups, repeats):
        order = list(rng.permutation(names))
        report["order"].append(dict(repeat=repeat, backends=order))
        for name in order:
            started = time.perf_counter()
            backend = create_backend(name, **(options or {}))
            backend.synchronize()
            initialization = dict(
                repeat=repeat, backend=name, constructor_seconds=time.perf_counter() - started
            )
            if mode == "sequence":
                initial, result = _execute(backend, frames[0])
                check(initial, result, 0)
                initialization["previous_input"] = initial
                previous_verified = initial.get("matches_reference")
                prior_output = result
            report["cold_initializations"].append(initialization)
            for index in indices:
                previous = None
                if mode == "pairs":
                    backend.close()
                    backend.ids = None
                    previous, previous_result = _execute(backend, frames[index - 1])
                    check(previous, previous_result, index - 1)
                    previous_verified = previous.get("matches_reference")
                    prior_output = previous_result
                quota_before = cpu_quota_snapshot()
                row, result = _execute(backend, frames[index])
                quota = cpu_quota_delta(quota_before, cpu_quota_snapshot())
                check(row, result, index)
                if (
                    prior_output.error is None
                    and result.error is None
                    and frames[index - 1].record["identity_epoch"]
                    == frames[index].record["identity_epoch"]
                ):
                    row["topology_churn"] = topology_churn(prior_output.rows, result.rows)
                else:
                    row["topology_churn"] = None
                row.update(
                    repeat=repeat,
                    previous_input_matches_reference=previous_verified,
                    previous_input=previous,
                    cpu_quota=quota,
                    cpu_throttled=any(x["throttled_periods"] for x in quota.values()),
                    retained_state=(mode == "sequence"),
                )
                previous_verified = row.get("matches_reference")
                prior_output = result
                report["records"][str(index)]["variants"][name].append(row)
                save()
            backend.close()
    for record in report["records"].values():
        record["summary"] = {}
        for name, samples in record["variants"].items():
            measured = [sample for sample in samples if sample["repeat"] >= 0]
            passed = [sample for sample in measured if sample["error"] is None]
            seconds = [sample["seconds"] for sample in passed]
            comparable = [
                sample for sample in passed if sample.get("matches_reference") is not None
            ]
            verified = [
                sample["seconds"]
                for sample in passed
                if sample.get("matches_reference") is True
                and sample.get("previous_input_matches_reference") is True
            ]
            record["summary"][name] = dict(
                measured_samples=len(measured),
                successful_samples=len(passed),
                median_seconds=float(np.median(seconds)) if seconds else None,
                p95_seconds=float(np.percentile(seconds, 95)) if seconds else None,
                max_seconds=float(np.max(seconds)) if seconds else None,
                failures=[
                    dict(repeat=s["repeat"], kind=s["failure_kind"], error=s["error"])
                    for s in measured
                    if s["error"] is not None
                ],
                all_comparable_outputs_match_reference=all(
                    s["matches_reference"] for s in comparable
                )
                if comparable
                else None,
                comparable_outputs=len(comparable),
                previous_inputs_validated=sum(
                    s.get("previous_input_matches_reference") is True for s in measured
                ),
                independently_comparable_samples=len(verified),
                reference_validated_median_seconds=float(np.median(verified)) if verified else None,
            )
    report["complete"] = True
    save()
    return report
