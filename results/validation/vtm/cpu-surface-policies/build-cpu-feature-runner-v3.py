"""Prepare explicit topology controls and failure-preserving offline benchmark records."""
from pathlib import Path
import ast

w=Path('.work/vtm')
src=w/'cpu_mesher_feature_policy_v2.py'
dest=w/'cpu_mesher_feature_policy_v3.py'
assert not dest.exists()
s=src.read_text().replace('import subprocess\n','import subprocess\nimport time\n')
s=s.replace('    parser.add_argument("--feature-angle", type=float, default=45.0)','    parser.add_argument("--feature-angle", type=float, default=45.0)\n    parser.add_argument("--topology", choices=["non_manifold", "manifold"], default="non_manifold")')
s=s.replace('    args.output.mkdir(parents=True, exist_ok=True)\n    records = []','''    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        parser.error("--output must be an empty directory; existing evidence is never overwritten")
    records = []
    checkpoint = args.output / "runs.json"

    def save_runs(stage):
        checkpoint.write_text(json.dumps(dict(stage=stage, runs=records), indent=2) + "\\n")
''')
s=s.replace('        if args.feature_angle != 45.0:', '        if args.feature_angle != 45.0 or args.topology != "non_manifold":')
s=s.replace('            command.append(str(args.feature_angle))\n        before =', '            command.append(str(args.feature_angle))\n        if args.topology != "non_manifold":\n            command.append(args.topology)\n        start = time.perf_counter()\n        before =')
begin=s.index('        result = subprocess.run(command')
end=s.index('    cpu_model = next(',begin)
s=s[:begin]+'''        result = subprocess.run(command, text=True, capture_output=True)
        quota = cpu_quota_delta(before, cpu_quota_snapshot())
        wall_seconds = time.perf_counter() - start
        (args.output / f"run-{i}.stdout").write_text(result.stdout)
        (args.output / f"run-{i}.stderr").write_text(result.stderr)
        record = dict(command=command, exit_code=result.returncode,
                      wall_seconds=wall_seconds, cpu_quota_delta=quota,
                      cpu_throttled=any(x["throttled_periods"] or x["throttled_usec"] for x in quota.values()),
                      native_valid=False, quality_valid=False)
        if mesh_path.exists():
            record["mesh_sha256"] = hashlib.sha256(mesh_path.read_bytes()).hexdigest()
        records.append(record)
        save_runs("native")
        try:
            if result.returncode != 0:
                raise RuntimeError(f"Native process exited {result.returncode}")
            native = json.loads(result.stdout)
            if native["feature_angle"] != args.feature_angle:
                raise ValueError("Native feature angle does not match requested policy")
            if native.get("topology", "non_manifold") != args.topology:
                raise ValueError("Native topology does not match requested policy")
            if "mesh_sha256" not in record:
                raise ValueError("Native process did not produce a mesh")
            record.update(native)
            record["native_valid"] = True
            record["topology_native_reported"] = "topology" in native
        except Exception as error:
            record["native_error"] = dict(type=type(error).__name__, message=str(error))
        save_runs("native")
    # All timed native executions finish before independent validation starts.
    save_runs("validation")
    oracle = NativeSurfaceDistance(args.distance_binary, args.output) if args.distance_binary else None
    quality = {}
    errors = {}
    for i, record in enumerate(records):
        if not record["native_valid"]:
            continue
        digest = record["mesh_sha256"]
        if digest not in quality and digest not in errors:
            try:
                p, t = read_mesh(args.output / f"mesh-{i}.txt")
                quality[digest] = measure(p, t, points, faces, distance_query=oracle)
                if oracle is not None:
                    quality[digest]["distance_oracle"] = oracle.metadata
            except Exception as error:
                errors[digest] = dict(type=type(error).__name__, message=str(error),
                                     stage="independent_material_quality_and_topology")
        if digest in quality:
            record["independent_quality"] = quality[digest]
            record["quality_valid"] = True
        else:
            record["quality_error"] = errors[digest]
        save_runs("validation")
    valid = [x for x in records if x["native_valid"] and x["quality_valid"]]
    clean = [x["complete_seconds"] for x in valid if not x["cpu_throttled"]]
    native_valid = [x for x in records if x["native_valid"]]
''' + s[end:]
s=s.replace('        median_complete_seconds=float(np.median([x["complete_seconds"] for x in records])),','''        complete=True,
        valid=len(valid) == args.repeats,
        successful_samples=len(valid),
        native_successful_samples=len(native_valid),
        median_native_complete_seconds=float(np.median([x["complete_seconds"] for x in native_valid])) if native_valid else None,
        native_unthrottled_samples=sum(not x["cpu_throttled"] for x in native_valid),
        median_complete_seconds=float(np.median([x["complete_seconds"] for x in valid])) if valid else None,''')
s=s.replace('    print(json.dumps(report), flush=True)','    save_runs("complete")\n    print(json.dumps(report), flush=True)\n    if not report["valid"]:\n        raise SystemExit(1)')
ast.parse(s)
dest.write_text(s)
print(dest)
