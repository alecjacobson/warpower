"""Private issue241 setup-only probe; omit unreachable approximate-query storage."""
import inspect,textwrap
from warpower import _quality_moves as qm
from warpower import _topology_recovery as tr


def install():
    source=textwrap.dedent(inspect.getsource(qm.QualityMove.__init__))
    old='if getattr(core, "_cache_surface_queries", False):'
    assert source.count(old)==1
    source=source.replace(old,'if getattr(self, "_use_approximate_queries", True) and getattr(core, "_cache_surface_queries", False):')
    for name in ['error','count','minimum','keep','quality']:
        lines=source.splitlines()
        matches=[(i,l) for i,l in enumerate(lines) if l.strip().startswith('self._query_'+name+' = ')]
        assert len(matches)==1
        i,line=matches[0]
        lines[i]=line+' if getattr(self, "_use_approximate_queries", True) else None'
        source='\n'.join(lines)+'\n'
    namespace=dict(qm.__dict__)
    exec(compile(source,__file__,'exec'),namespace)
    qm.QualityMove.__init__=namespace['__init__']
    tr.TopologyRecovery._use_approximate_queries=False
