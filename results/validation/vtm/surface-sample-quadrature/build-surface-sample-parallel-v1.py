"""Generate a private per-sample kernel directly from current extraction arithmetic."""
import ast
import copy
import hashlib
import json
import subprocess
from pathlib import Path

w = Path(__file__).parent
source = Path('warpower/_extraction.py')
original = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'surface_error')
assert isinstance(original.body[-1], ast.For)
assert ast.unparse(original.body[-1].iter) == 'range(3)'
candidate = copy.deepcopy(original)
candidate.name = 'surface_error_samples'
candidate.body = ast.parse('i = wp.tid() // 3\nk = wp.tid() % 3').body + candidate.body[1:-1] + candidate.body[-1].body
functions = [candidate]
for name, base in [('reference_observed', original), ('samples_observed', candidate)]:
    node = copy.deepcopy(base)
    node.name = name
    node.args.args += ast.parse('def stub(hits: wp.array[Closest], weighted: wp.array[wp.float64]): pass').body[0].args.args
    body = node.body[-1].body if name == 'reference_observed' else node.body
    at = next(i for i, s in enumerate(body) if isinstance(s, ast.Assign) and ast.unparse(s.targets[0]) == 'hit')
    body[at+1:at+1] = ast.parse('hits[3 * i + k] = hit\nweighted[3 * i + k] = weight * hit.distance2').body
    functions.append(node)
text = '"""Private per-sample quadrature and instrumented arithmetic references; not adopted."""\nimport warp as wp\nfrom warpower._surface_bvh import SurfaceTree, Closest, closest_point\nwp.set_module_options({"enable_backward": False, "fast_math": False})\n\n'
text += '\n\n'.join(ast.unparse(ast.fix_missing_locations(n)) for n in functions) + '\n'
target = w/'surface_sample_parallel_v1.py'
assert not target.exists()
target.write_text(text)
record = dict(baseline=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(), original=str(source), original_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), candidate=str(target), candidate_sha256=hashlib.sha256(target.read_bytes()).hexdigest(), generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), change='Flatten original three-query face loop to one (face, sample) per thread. All original arithmetic and reductions retained; atomic sum order may differ.', adopted=False)
out = w/'surface-sample-parallel-source-v1.json'
assert not out.exists();out.write_text(json.dumps(record,indent=2)+'\n')
print(record)
