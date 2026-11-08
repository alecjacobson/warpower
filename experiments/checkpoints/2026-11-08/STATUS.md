# Discussion checkpoint — 2026-11-08

Work is paused at the user's request. No background experiments should be resumed
until the user asks to continue. PR #260 remains a draft and has not been merged.

The branch contains the selective-step-history runtime commit `b343f703` plus
this compact checkpoint. All 1,407 production tests, 60 targeted tests, the full
five-phase 8K recovery loop, 32 clean timing jobs, and independent checks of all
22 final meshes passed. The final 16K torus improves from 4.378 to 2.898 seconds
(1.51×). Standalone Delaunay timings change by −0.31% to +0.09%; power diagrams
by −3.29%, with their implementations unchanged. All three native trace jobs
completed, but final combined qualification is unfinished.

The first mixed-workspace memory audit hit its 600-second deadline. Its isolated
replacement preserves all 37 contracts. Completed groups and the user-stopped
job are recorded in `summary.json`; remaining audits and frame comparisons are
required before merging. A user-stopped job is not a passing qualification.

The optional CGAL perturbation/exudation benchmark code is a private prototype,
not integrated into the public benchmark. Both native builds and functional
contracts passed. An initial checker incorrectly expected exudation to change
the global maximum vertex weight; its corrected version checks the actual
changed weighted vertices and mesh quality and retains the failed invocation.

The repeated scan pilot has six independently valid, unthrottled coarse outputs:
CGAL sequential median 0.616 s and four-thread median 0.427 s, at roughly6,000
sites. They have stronger worst-element quality and tighter surface fit but lower
mean radius ratio than the 8K Warp output. All six finer ODT10 outputs failed
independent material topology checks and remain retained. These are not a
matched-quality CPU speedup result; the requested wide GPU advantage is unmet.

Raw experiment snapshots, profiler traces, meshes and native executables are
excluded from this checkpoint. Their paths and hashes are inventoried; originals
remain on persistent local storage. Forward publication is prepared to use
checksum-verified release bundles, with only compact manifests/summaries in Git.
No release bundle has been published and no historical Git migration performed.

The `sources/` directory preserves the current private scripts with original
`.work/vtm` path mappings in `source-manifest.json`. Some are superseded or
unexecuted preparations, not finished public APIs. They depend on retained inputs
and helpers identified by their source and the raw inventory. In particular,
release publication/checking scripts still need adaptation to the isolated memory
report; do not execute them as a completed qualification pipeline.

Next steps after discussion: finish memory/frame qualification and update the
release publication scripts before merging #260; then continue the CGAL quality/
size/fit comparison (#244/#261/#40). Keep raw evidence out of ordinary Git (#262).
