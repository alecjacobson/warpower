from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';prototype=(w/'tangential_topology_recovery_v2.py').read_text()
# Isolate the existing bounded transaction; production owns its setup and capture.
s=prototype[:prototype.index('class Mesher(OriginalMesher):')]
a=s.index('OriginalBand=');b=s.index('@wp.kernel\ndef eligible(')
s=s[:a]+s[b:]
s=s.replace('import warpower\n','').replace('from warpower import _exterior_band as band, _mesher as mesher, _quality_moves as qm','from . import _mesher as mesher, _quality_moves as qm').replace('from warpower._mesh_topology import previous_edge\n','').replace('from warpower._surface_bvh import','from ._surface_bvh import')
s=s.replace('Private #228 prototype: quality-bounded moves after a rejected link transaction.','Bounded quality recovery after a rejected boundary-link transaction.')
s=s.replace('witness:wp.array[int],roles:wp.array[int]', 'witness:wp.array[int],band_rejected:wp.array[int],roles:wp.array[int]')
s=s.replace('if state.error[0]==0 and state.extraction_reason[0]!=0 and core_error[0]==0 and extraction_error[0]==0:', 'if state.error[0]==0 and state.extraction_reason[0]!=0 and core_error[0]==0 and extraction_error[0]==0 and band_rejected[0]!=0:')
s=s.replace('max_attempts=32,extraction_tolerance=owner.extraction.tolerance)', 'max_attempts=32,extraction_tolerance=owner.extraction.tolerance,_proven_exterior_recovery=False)')
s=s.replace('        # A rejected candidate does not need the expensive additional exterior\n        # proof search. All required candidate validity guards remain in place.\n        self.extraction._proven_exterior=None\n','')
s=s.replace('e._exterior.pinched_site,\n                                m.core.boundary.roles','e._exterior.pinched_site,e._exterior.rejected,\n                                m.core.boundary.roles')
s=s.replace('def restored(core_error:wp.array[int],mesh_error:wp.array[int],invalid:wp.array[int],boundary_error:wp.array[int]):\n    core_error[0]=int(mesh_error[0]!=0 or invalid[0]!=0 or boundary_error[0]!=0)', '''def restored(state:qm.MoveState,core_error:wp.array[int],mesh_error:wp.array[int],invalid:wp.array[int],boundary_error:wp.array[int]):
    core_error[0]=0
    if mesh_error[0]!=0:core_error[0]|=1
    if boundary_error[0]!=0:core_error[0]|=2
    if invalid[0]!=0:core_error[0]|=4
    if core_error[0]!=0:
        state.error[0]|=8;state.source_count[0]=0''')
s=s.replace('self._launch(restored,1,[c.error,c.mesh.error,c._invalid,c.boundary.error])','self._launch(restored,1,[self._state,c.error,c.mesh.error,c._invalid,c.boundary.error])')
p=root/'warpower/_topology_recovery.py';assert not p.exists();p.write_text(s)
# Only an explicitly supplied witness adds work to a band transaction.
p=root/'warpower/_exterior_band.py';s=p.read_text().replace('from ._containment import bad_edge','from ._containment import bad_edge\nfrom ._mesh_topology import previous_edge')
witness=prototype[prototype.index('@wp.kernel\ndef remember_pinched'):prototype.index('class WitnessBand')]
s=s.replace('class ExteriorBandRecovery:',witness+'class ExteriorBandRecovery:')
s=s.replace('def __init__(self, extraction, roles, enabled):','def __init__(self, extraction, roles, enabled, *, pinched_site=None):')
s=s.replace('        self.device = extraction.device','        self.device = extraction.device\n        self.pinched_site = pinched_site')
s=s.replace('    def _attempt(self):\n        e = self.extraction','    def _attempt(self):\n        if self.pinched_site is not None:\n            self.pinched_site.fill_(2**31 - 1)\n        e = self.extraction')
s=s.replace('    def _restore(self):\n        e = self.extraction','''    def _restore(self):
        e = self.extraction
        if self.pinched_site is not None:
            t = e._topology
            self._launch(remember_pinched,len(e.points),[t._first,t._degree,t._twins,
                         e.error,e.diagnostics,self.roles,self.pinched_site])''')
p.write_text(s)
p=root/'warpower/_extraction.py';s=p.read_text().replace('        _cache_surface_queries=False,','        _cache_surface_queries=False,\n        _rejected_link_witness=None,')
s=s.replace('        self._query_cache = None','''        self._rejected_link_witness = (
            None if _rejected_link_witness is None
            else _int_array(_rejected_link_witness, self.device, (1,))
        )
        self._query_cache = None''')
s=s.replace('                    self, self.feature_roles, self.exterior_enabled','                    self, self.feature_roles, self.exterior_enabled,\n                    pinched_site=self._rejected_link_witness')
p.write_text(s)
# Avoid allocating/capturing an unused proof workspace in the new recovery.
p=root/'warpower/_quality_moves.py';s=p.read_text().replace('        extraction_tolerance=0.4,','        extraction_tolerance=0.4,\n        _proven_exterior_recovery=None,')
s=s.replace('                core.material_recovery is not None and core.proven_exterior_recovery\n            ),','''                (core.material_recovery is not None and core.proven_exterior_recovery)
                if _proven_exterior_recovery is None else bool(_proven_exterior_recovery)
            ),''')
p.write_text(s)
p=root/'warpower/_mesher.py';s=p.read_text().replace('        proven_exterior_recovery=None,','        proven_exterior_recovery=None,\n        tangential_recovery=None,')
needle='        self._uses_recovery = (';assert s.count(needle)==1
s=s.replace(needle,'''        self.tangential_recovery = (
            self.exterior_recovery if tangential_recovery is None else bool(tangential_recovery)
        )
        if self.tangential_recovery and not self.exterior_recovery:
            raise ValueError("tangential_recovery requires exterior_recovery")
        self.topology_recovery = None
'''+needle)
needle='            self.extraction = ExtractionWorkspace(';assert s.count(needle)==1
s=s.replace(needle,'''            self._pinched_site = (
                wp.full(1, 2**31 - 1, dtype=int) if self.tangential_recovery else None
            )
'''+needle)
a=s.index('            self.extraction = ExtractionWorkspace(');b=s.index('            self.full_tetrahedra =',a)
s=s[:a]+s[a:b].replace('                _cache_surface_queries=True,','                _cache_surface_queries=True,\n                _rejected_link_witness=self._pinched_site,')+s[b:]
needle='            wp.load_module(module=__name__, device=self.device)\n            with wp.ScopedCapture() as cap:';assert s.count(needle)==1
s=s.replace(needle,'''            if self.tangential_recovery:
                from ._topology_recovery import TopologyRecovery
                self.topology_recovery = TopologyRecovery(self, threshold=quality_threshold)
'''+needle)
s=s.replace('    def _record(self):\n        self.topology.zero_()','    def _record(self):\n        if self.topology_recovery is not None:\n            self.topology_recovery.reset()\n        self.topology.zero_()')
needle='        self._launch(require_extraction, 1, [self._state])';assert s.count(needle)==1
s=s.replace(needle,'''        if self.topology_recovery is not None:
            self.topology_recovery.run()
            wp.capture_if(self.topology_recovery.invoked, self._extract)
'''+needle)
p.write_text(s)
