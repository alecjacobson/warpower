"""Fresh-process policy instrumentation after retained combined-process timeout."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

root = Path.cwd()
w = root / ".work/vtm"
python = root / ".venv/bin/python"
dest = w / "tangential-production-memory-v3-commands.json"
assert not dest.exists()
prior = json.loads((w / "tangential-production-memory-v2-commands.json").read_text())
assert prior[-1]["exit_code"] == 124
isolation = json.loads((w / "tangential-production-policy-isolation-v1.json").read_text())
for path_key, hash_key in [("source", "source_sha256"), ("generated", "generated_sha256")]:
    assert hashlib.sha256((root / isolation[path_key]).read_bytes()).hexdigest() == isolation[hash_key]
assert "1326 passed" in (w / "tangential-production-full-suite-v1.log").read_text()
policy = str(w / "test_tangential_policy_isolated_v1.py")
groups = [("kernels", ["tests/test_topology_recovery.py", "-k", "not test_recovery_policy_scopes_storage_and_healthy_solve_skips_search"], 8, 600)]
groups += [("policy-" + name, [policy + "::test_isolated_policy_" + name], 1, 600) for name in ["healthy", "disabled", "implied"]]
groups += [("loop", [str(w / "test_tangential_production_loop_v2.py")], 1, 1200)]
rows = []
for name, tests, count, timeout in groups:
    for mode in ["padded", "lifetime"]:
        label = f"tangential-production-memory-v3-{name}-{mode}"
        env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="4", PYTHONPATH=".", WARPOWER_TOPOLOGY_LOOP_OUTPUT=str(w / (label + "-frames")))
        options = []
        if mode == "lifetime":
            env["WARPOWER_AUDIT_DISABLE_POOL"] = "1"
            options = ["--check-api-memory-access", "yes", "--track-stream-ordered-races", "use-after-free"]
        cmd = ["/usr/local/cuda-12.9/bin/compute-sanitizer", "--tool", "memcheck", "--padding", "32", "--error-exitcode", "99", *options, str(python), str(w / "large-reuse-isolated-test.py"), *tests]
        log = w / (label + ".log")
        assert not log.exists()
        start = time.time()
        with log.open("w") as stream:
            p = subprocess.Popen(cmd, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = p.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGTERM)
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait()
                code = 124
        text = log.read_text()
        row = dict(name=name, mode=mode, command=cmd, exit_code=code, wall_seconds=time.time()-start, timeout_seconds=timeout, disable_pool=mode=="lifetime", zero_memcheck_errors="ERROR SUMMARY: 0 errors" in text, expected_tests=count)
        rows.append(row)
        dest.write_text(json.dumps(rows, indent=2) + "\n")
        print(row, flush=True)
        assert code == 0 and row["zero_memcheck_errors"] and f"{count} passed" in text, row
assert len(rows) == 10
