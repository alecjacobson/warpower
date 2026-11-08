"""Private setup-route extension; existing original-coordinate proof kernels unchanged."""
import ast
import hashlib
from pathlib import Path

import warpower._active_delaunay as active

path = Path(active.__file__)
source = path.read_text()
needle = 'self._full_reuses_cells = 2048 < len(points) <= 4096'
assert source.count(needle) == 1
candidate = source.replace(needle, 'self._full_reuses_cells = 2048 < len(points) <= 16384')
original_tree = ast.parse(source)
candidate_tree = ast.parse(candidate)
original_class = next(n for n in original_tree.body if isinstance(n, ast.ClassDef) and n.name == 'ActiveDelaunayWorkspace')
candidate_class = next(n for n in candidate_tree.body if isinstance(n, ast.ClassDef) and n.name == 'ActiveDelaunayWorkspace')
recovered = ast.parse(candidate.replace('self._full_reuses_cells = 2048 < len(points) <= 16384', needle))
assert ast.dump(original_tree) == ast.dump(recovered)
# Reuse the original module globals, including all existing kernels. Only the
# Python setup class is recompiled; it records existing device operations.
namespace = dict(vars(active))
exec(compile(ast.Module(body=[candidate_class],type_ignores=[]), str(path), 'exec'), namespace)
ActiveDelaunayWorkspace = namespace['ActiveDelaunayWorkspace']
source_sha256 = hashlib.sha256(source.encode()).hexdigest()
candidate_sha256 = hashlib.sha256(candidate.encode()).hexdigest()


def install():
    active.ActiveDelaunayWorkspace = ActiveDelaunayWorkspace

