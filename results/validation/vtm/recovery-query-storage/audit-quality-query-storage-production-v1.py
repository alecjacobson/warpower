"""Bind constructor-only production changes to the qualified private storage probe."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import textwrap

root = Path.cwd()
w = root / ".work/vtm"
baseline = "5035fa5895a3bdcc9542bbeebabd75f3d955450b"
dest = w / "quality-query-storage-production-source-v1.json"
assert not dest.exists()
assert not subprocess.check_output(["git", "status", "--porcelain"], text=True)
runtime = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
assert runtime != baseline
checks = {}
for path, owner in [("warpower/_quality_moves.py", "QualityMove"), ("warpower/_topology_recovery.py", "TopologyRecovery")]:
    before = subprocess.check_output(["git", "show", baseline+":"+path], text=True)
    after = (root / path).read_text()
    old, new = ast.parse(before), ast.parse(after)
    old_class = next(n for n in old.body if isinstance(n, ast.ClassDef) and n.name == owner)
    new_class = next(n for n in new.body if isinstance(n, ast.ClassDef) and n.name == owner)
    old_init = next(n for n in old_class.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    new_init = next(n for n in new_class.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    normalized = copy.deepcopy(new_init)
    if owner == "QualityMove":
        index = next(i for i, arg in enumerate(normalized.args.kwonlyargs) if arg.arg == "_approximate_queries")
        assert normalized.args.kw_defaults.pop(index).value is True
        normalized.args.kwonlyargs.pop(index)

        class NormalizePolicy(ast.NodeTransformer):
            def visit_Name(self, node):
                if node.id == "_approximate_queries":
                    return ast.parse('getattr(self, "_use_approximate_queries", True)', mode="eval").body
                return node

        normalized = NormalizePolicy().visit(normalized)
        expected = textwrap.dedent("".join(before.splitlines(keepends=True)[old_init.lineno-1:old_init.end_lineno]))
        old_if = 'if getattr(core, "_cache_surface_queries", False):'
        assert expected.count(old_if) == 1
        expected = expected.replace(old_if, 'if getattr(self, "_use_approximate_queries", True) and getattr(core, "_cache_surface_queries", False):')
        for name in ["error", "count", "minimum", "keep", "quality"]:
            lines = expected.splitlines()
            indices = [i for i, line in enumerate(lines) if line.strip().startswith("self._query_"+name+" = ")]
            assert len(indices) == 1
            lines[indices[0]] += ' if getattr(self, "_use_approximate_queries", True) else None'
            expected = "\n".join(lines) + "\n"
        assert ast.dump(normalized) == ast.dump(ast.parse(expected).body[0])
    else:
        matches = []
        for node in ast.walk(normalized):
            if isinstance(node, ast.Call):
                for keyword in node.keywords:
                    if keyword.arg == "_approximate_queries":
                        assert keyword.value.value is False
                        matches.append((node, keyword))
        assert len(matches) == 1
        matches[0][0].keywords.remove(matches[0][1])
        assert ast.dump(normalized) == ast.dump(old_init)
    # Entire remaining module must retain the original AST, not only kernels.
    new_class.body[new_class.body.index(new_init)] = old_init
    assert ast.dump(new) == ast.dump(old)
    checks[path] = dict(only_constructor_changed=True, all_other_module_AST_unchanged=True, constructor_matches_qualified_policy=True)
prior = json.loads((w / "tangential-production-source-audit-v1.json").read_text())
paths = sorted(set(prior["source_sha256"]) | {"tests/test_recovery_query_storage.py"})
changed, unchanged, hashes = [], [], {}
for path in paths:
    data = (root / path).read_bytes()
    hashes[path] = hashlib.sha256(data).hexdigest()
    if path == "tests/test_recovery_query_storage.py":
        changed.append(path)
    else:
        old = subprocess.check_output(["git", "show", baseline+":"+path])
        (unchanged if old == data else changed).append(path)
assert set(changed) == {"warpower/_quality_moves.py", "warpower/_topology_recovery.py", "tests/test_recovery_query_storage.py"}
assert len(hashes) == 262 and len(unchanged) == 259
assert "README.md" in unchanged
report = dict(scope=__doc__, baseline=baseline, runtime=runtime, source_sha256=hashes, changed_files=changed, unchanged_files=unchanged, constructor_checks=checks, private_prototype_sha256=hashlib.sha256((w / "quality_query_storage_v1.py").read_bytes()).hexdigest(), readme_unchanged=True, independent_diagram_and_delaunay_sources_unchanged=True)
dest.write_text(json.dumps(report, indent=2) + "\n")
print("262 sources pinned;259 unchanged; only two constructors and one focused test differ.")
