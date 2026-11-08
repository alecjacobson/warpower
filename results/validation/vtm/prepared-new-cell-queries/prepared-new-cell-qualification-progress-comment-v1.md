Production qualification update for runtime `3b68a6043860209d3c507951b0e51309ed8bb8c5`:

- All 1,279 tests pass; 119 focused cases also pass. The measured simple prototype's two kernel ASTs match production, and 141 other runtime/benchmark/example sources are unchanged.
- The 26-job primary timing cohort is complete with clean quota windows. All fourteen saved full meshes independently validate, with bitwise coordinates, canonical full/material cells, oriented boundary, roles and solver decisions matching each control.
- One initial graded candidate recorded a CPU-quota throttling period. Its valid mesh and all samples are retained separately; the complete graded pair was rerun. The earlier three clean pairs were retained without repeating them.
- Complete-solve and changing-input-loop native audits pass. The loop has valid, deformed, invalid-count, collapsed and recovered phases; all three valid meshes independently check, invalid outputs are zero, recovery is exact, and pointers remain stable. The first loop launch includes deferred graph instantiation/upload before its single launch.
- Isolated padded/lifetime memory checks are still running, with no errors so far.

The standalone hand timing investigation is tracked in #235. Primary and ABBA samples remain visible, including the slower measurements. Native traces localize variation to the existing local-sphere checker in both versions. An actual dependency audit finds the same 40 project modules loaded, with neither modified runtime module imported by that standalone workload. Thus this change adds no dependency/work there; the detailed cause of the preexisting timing variance remains open rather than being called resolved noise.

Final memory gates and the remaining archived-pair validation must finish before publication and merge. README stays unchanged; no wide comparable-quality CPU advantage is claimed.
