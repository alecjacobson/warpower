# Issue241 constructor-only hypothesis

This is a separate unexecuted follow-up to runtime6178890. The active issue228
qualification must finish before GPU work begins. Do not change the measured
production sources while its gates are running.

The local call-site audit finds `_query_cache`, `_query_error`, `_query_count`,
`_query_minimum`, `_query_keep` and `_query_quality` used only by QualityMove's
constructor and `_approximate_quality`. TopologyRecovery overrides `_quality`
to call `_complete_quality` unconditionally; its `_initial` is also specialized.
Inherited `_try` dispatches through that override. `_record`, `_build`, `_metrics`,
`_commit` and the subclass restoration do not read the approximate buffers. The
candidate ExtractionWorkspace's own `_query_cache` is separate and remains active.
The `_old_points/_old_tets/_old_count` checkpoints are NOT part of this change.

Private `quality_query_storage_v1.py` copies only QualityMove.__init__ and adds a
setup predicate. All ordinary subclasses retain the original default. Only the
TopologyRecovery class opts out. The constructor does not allocate the unused
cache/table or five query arrays. No device kernel or runtime method changes.
Before adoption, prefer an explicit private constructor policy over the probe's
class attribute and bind the final constructor to this tested version.

The skipped cache has two QueryStates, each84bytes per slot, an int32 table of
capacity next_pow2(4*m), and an int64 version. The five arrays add12*m+16bytes.
Predicted payload is180*m+4*next_pow2(4*m)+24bytes. Actual allocator/persistent
memory must be measured; this formula does not include allocator/capture overhead
and is not the entire feature cost (11,832,209/47,869,935/95,729,647/191,449,087bytes
at1K/4K/8K/16K in the recorded controls).

Required checks: constructor policy/storage identity for ordinary and recovery
subclasses, graph recording with recovery._approximate_quality deliberately
forbidden, actual normal/invalid/forced-exhaustion recovery, exact source/hash
comparisons, full independent geometry, setup/pool memory and clean full solves,
native/lifetime diagnostics, independent diagram/Delaunay no-regression. Preserve
the current feature's complete-input/reconstruction/acceptance contracts.
