"""Integrate the independently validated v2 policy, sharing unchanged scalar logic."""
import ast,hashlib,json,subprocess
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm'
assert not subprocess.check_output(['git','status','--porcelain'])
for filename in ['selective-step-history-pilot-v2-independent-v3.json','selective-step-history-gallery-v2-independent-v2.json']:
    assert json.loads((w/filename).read_text())['complete']
source=w/'selective_step_history_v2.py';tree=ast.parse(source.read_text())
path=root/'warpower/_vtm.py';original=path.read_text();text=original
for name in ['start_attempt','finish_attempt']:
    node=next(n for n in ast.parse(original).body if isinstance(n,ast.FunctionDef) and n.name==name)
    private=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='original_'+name)
    assert ast.dump(ast.Module(body=node.body,type_ignores=[]))==ast.dump(ast.Module(body=private.body,type_ignores=[]))
    lines=original.splitlines(keepends=True);old=''.join(lines[node.decorator_list[0].lineno-1:node.end_lineno])
    func=old.replace('@wp.kernel','@wp.func',1).replace('def '+name+'(','def _'+name+'(',1)
    wrapper=old[:old.index('):\n')+3]+'    _'+name+'('+', '.join(a.arg for a in node.args.args)+')\n'
    assert text.count(old)==1;text=text.replace(old,func+'\n\n'+wrapper)
text=text.replace('        self.local_backtracking = None\n','        self.local_backtracking = None\n        self.step_history = None\n        self._start_attempt_kernel, self._start_attempt_args = start_attempt, []\n        self._finish_attempt_kernel, self._finish_attempt_args = finish_attempt, []\n',1)
needle='                self.controls = DeviceControls(self, iteration_limit, selective_enabled)\n'
assert text.count(needle)==1
text=text.replace(needle,needle+'            if selective_enabled is not None and local_quality_retries == 0:\n                from ._selective_step_history import SelectiveStepHistory\n\n                self.step_history = SelectiveStepHistory(self)\n')
text=text.replace('    def _record(self):\n        self._interior_known.zero_()','    def _record(self):\n        if self.step_history is not None:\n            self.step_history.reset()\n        self._interior_known.zero_()',1)
needle='''            start_attempt,
            1,''';assert text.count(needle)==1;text=text.replace(needle,needle.replace('start_attempt,','self._start_attempt_kernel,'))
needle='''                self._running,
            ],
        )
        wp.capture_while(self._attempt''';assert text.count(needle)==1;text=text.replace(needle,needle.replace('            ],','            ] + self._start_attempt_args,'))
needle='''            finish_attempt,
            1,''';assert text.count(needle)==1;text=text.replace(needle,needle.replace('finish_attempt,','self._finish_attempt_kernel,'))
needle='''                self.energy_history,
            ],
        )

    def _accept''';assert text.count(needle)==1;text=text.replace(needle,needle.replace('            ],','            ] + self._finish_attempt_args,'))
ast.parse(text);path.write_text(text)
# The candidate kernels call the shared original functions, with no policy changes.
functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['start_attempt','finish_attempt']]
module='''"""Bounded selective step history with full-search convergence and failure fallbacks.

Only device-controlled selective cores without local vertex rollback opt in.
History resets at every core invocation; statistics cover a complete mesher run.
Every actual proposal retains the ordinary geometry, objective, and quality guards.
"""

import warp as wp
from ._vtm import _start_attempt, _finish_attempt

wp.set_module_options({"enable_backward": False, "fast_math": False})

'''
lines=source.read_text().splitlines(keepends=True)
for n in functions:
    s=''.join(lines[n.decorator_list[0].lineno-1:n.end_lineno]);module+=s.replace('original_start_attempt','_start_attempt').replace('original_finish_attempt','_finish_attempt')+'\n\n'
module+='''class SelectiveStepHistory:
    def __init__(self, core):
        # valid, previous absolute halving level, default retried, warm start,
        # quality-only rejection history, previous selective rejection count.
        self.history = wp.zeros(6, dtype=int, device=core.device)
        # Warm starts, convergence retries, exhaustion retries, warm accepts,
        # and immediate default retries after non-quality failures.
        self.statistics = wp.zeros(5, dtype=int, device=core.device)
        core._start_attempt_kernel = start_attempt
        core._start_attempt_args = [core.controls.mode, self.history, self.statistics,
                                   core.max_backtracks, core.selection.rejections]
        core._finish_attempt_kernel = finish_attempt
        core._finish_attempt_args = [core.controls.mode, self.history, self.statistics,
                                    core.damping, core.selection.rejections,
                                    core.selection.last_error]
        wp.load_module(module=__name__, device=core.device)

    def reset(self):
        self.history.zero_()
'''
path=root/'warpower/_selective_step_history.py';assert not path.exists();ast.parse(module);path.write_text(module)
path=root/'warpower/_mesher.py';s=path.read_text();needle='    def _record(self):\n';assert s.count(needle)==1
s=s.replace(needle,needle+'        if self.core.step_history is not None:\n            self.core.step_history.statistics.zero_()\n');path.write_text(s)
path=root/'tests/test_selective_step_history.py';assert not path.exists()
s=(w/'test_selective_step_history_v2.py').read_text().replace('Private state-machine checks','State-machine checks').replace('import selective_step_history_v2 as candidate','from warpower import _selective_step_history as candidate')
path.write_text(s)
report=dict(scope=__doc__,baseline=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),private_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),private_tests_sha256=hashlib.sha256((w/'test_selective_step_history_v2.py').read_bytes()).hexdigest(),files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [root/'warpower/_vtm.py',root/'warpower/_selective_step_history.py',root/'warpower/_mesher.py',root/'tests/test_selective_step_history.py']})
(w/'selective-step-history-integration-v1.json').write_text(json.dumps(report,indent=2)+'\n')
