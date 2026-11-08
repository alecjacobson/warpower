"""Run unchanged complete-mesher graph/failure contracts with private tracing."""
import json
import os
from pathlib import Path
import pytest
import step_observer_v1
from warpower import _mesher

instances=[]
observations=[]
step_observer_v1.install()
original=_mesher.VariationalMeshingWorkspace.__init__


def init(self,*args,**kwargs):
    original(self,*args,**kwargs)
    instances.append(self)


_mesher.VariationalMeshingWorkspace.__init__=init


@pytest.fixture(autouse=True)
def retain_observation(request):
    start=len(instances)
    yield
    for m in instances[start:]:
        o=m.core.step_observer
        count=int(o.count.numpy()[0]);overflow=int(o.overflow.numpy()[0])
        assert overflow==0 and 0<=count<=o.capacity
        ints=o.ints.numpy()[:count]
        events=ints[:,0]
        observations.append(dict(test=request.node.nodeid,count=count,overflow=overflow,
            core_begins=int((events==1).sum()),trials=int((events==13).sum()),
            local_retries=int((events==15).sum()),error=int(m.error.numpy()[0]),
            valid=int(m.valid.numpy()[0])))
    del instances[start:]


def pytest_sessionfinish(session,exitstatus):
    p=Path(os.environ["WARPOWER_STEP_CONTRACT_REPORT"])
    assert not p.exists()
    p.write_text(json.dumps(dict(exitstatus=exitstatus,observations=observations),indent=2)+"\n")
