"""Create an isolated larger-prefix setup prototype, preserving all device algorithms."""
import ast,hashlib,json,subprocess,textwrap
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';source=root/'warpower/_active_delaunay.py';text=source.read_text();tree=ast.parse(text)
c=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ActiveDelaunayWorkspace')
f=next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
s='\n'.join(text.splitlines()[f.lineno-1:f.end_lineno])+'\n'
s=s.replace('prefix_capacities=(1024, 2048)','prefix_capacities=None')
s=s.replace('        levels = tuple(operator.index(n) for n in prefix_capacities)','        if prefix_capacities is None:\n            prefix_capacities = tuple(n for n in (1024, 2048, 4096, 8192) if n < len(points))\n        levels = tuple(operator.index(n) for n in prefix_capacities)')
s=s.replace('n > 2048','n > 8192').replace('in [4, 2048]','in [4, 8192]').replace('wp.load_module(module=__name__,','wp.load_module(module="warpower._active_delaunay",')
header='''"""Private setup-only prototype: existing constructors plus larger prefix choices.

The production plan keeps the direct private router's old default; VTM requests
these explicit levels. This prototype installs that policy before VTM setup.
No device kernel is changed. The full constructor remains output owner/fallback.
"""
import operator
import warp as wp
from warpower._active_delaunay import ActiveDelaunayWorkspace as Base
from warpower._delaunay_recovery import DelaunayWorkspace
from warpower._wall_reuse import WallReuseDelaunayWorkspace as ReuseDelaunayWorkspace

class LargerPrefixDelaunayWorkspace(Base):
'''
suffix='''
def install():
    import warpower._active_delaunay as active
    assert active.ActiveDelaunayWorkspace is Base
    active.ActiveDelaunayWorkspace = LargerPrefixDelaunayWorkspace
'''
p=w/'larger_prefix_v1.py';assert not p.exists();p.write_text(header+s+suffix);ast.parse(p.read_text())
for kind,module in [('mesher','benchmarks.mesher'),('gallery','examples.vtm_gallery')]:
 p=w/f'run-larger-prefix-{kind}-v1.py';assert not p.exists();p.write_text('''"""Install private setup before the unchanged benchmark."""
import sys
if '--candidate' in sys.argv:
    sys.argv.remove('--candidate')
    from larger_prefix_v1 import install
    install()
from '''+module+''' import main
main()
''')
r=dict(baseline=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),baseline_router_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),prototype_sha256=hashlib.sha256((w/'larger_prefix_v1.py').read_bytes()).hexdigest(),scope='Setup-only explicit4096/8192children, all device algorithms and publish/reset/fallback methods inherited unchanged. Prototype default expands only to emulate planned VTM-owned explicit levels. No performance result.')
p=w/'larger-prefix-prototype-source-v1.json';assert not p.exists();p.write_text(json.dumps(r,indent=2)+'\n');print(r)
