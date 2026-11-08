"""Preserve every policy assertion while separating independent constructors for instrumentation."""
import ast
import copy
import hashlib
import json
from pathlib import Path

w = Path(".work/vtm")
source = Path("tests/test_topology_recovery.py")
tree = ast.parse(source.read_text())
name = "test_recovery_policy_scopes_storage_and_healthy_solve_skips_search"
original = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
body = original.body
assert ast.unparse(body[0]).startswith("d = warpower.SurfaceDomain(")
boundaries = []
for label in ["disabled", "implied"]:
    matches = [i for i, node in enumerate(body) if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == label]
    assert len(matches) == 1
    boundaries.append(matches[0])
a, b = boundaries
groups = [("healthy", body[1:a]), ("disabled", body[a:b]), ("implied", body[b:])]
recovered = [body[0]] + [node for _, nodes in groups for node in nodes]
assert [ast.dump(n) for n in recovered] == [ast.dump(n) for n in body]
code = '"""Diagnostic decomposition of the unchanged multi-workspace policy test.\n\nEach named case must run in a fresh instrumented process. This does not claim\ncombined-process instrumentation passed; all original assertions are preserved.\n"""\nfrom tests.test_topology_recovery import *\n\n'
checks = {}
for label, nodes in groups:
    function = copy.deepcopy(original)
    function.name = "test_isolated_policy_" + label
    function.body = copy.deepcopy([body[0]] + nodes)
    assert ast.dump(function.body[0]) == ast.dump(body[0])
    checks[label] = dict(function=function.name, original_statement_indices=list(range(body.index(nodes[0]), body.index(nodes[-1])+1)), original_statements_AST_equal=True)
    code += ast.unparse(ast.fix_missing_locations(function)) + "\n\n"
target = w / "test_tangential_policy_isolated_v1.py"
assert not target.exists()
ast.parse(code)
target.write_text(code)
report = dict(scope=__doc__, source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), generated=str(target), generated_sha256=hashlib.sha256(target.read_bytes()).hexdigest(), all_original_statements_preserved_once_except_repeated_domain_setup=True, groups=checks, limitation="Isolation changes workspace coexistence during instrumentation. The unchanged original combined test passed the ordinary full suite; its incomplete sanitizer runs remain failures to complete, not passes.")
dest = w / "tangential-production-policy-isolation-v1.json"
assert not dest.exists()
dest.write_text(json.dumps(report, indent=2) + "\n")
print(report)
