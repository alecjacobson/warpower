# Production clipping workspace evidence

See [the report](../../docs/clipping-workspace.md). This compares runtime
`8a575f1` with `bfd5448`; the isolated allocator-hook pilot has separate evidence.
All raw files, including failed instrumentation/setup attempts, are checksum-pinned
in `raw-manifest.json`. Full allocation-order instrumentation is explicitly not
green; unchanged baseline controls and scoped passing checks are retained.

The [published evidence bundle](https://github.com/alecjacobson/warpower/releases/tag/evidence-clipping-production-bfd5448) contains the raw outputs, traces, source snapshots and drivers. All 1376 members were verified after an independent download. Archive and manifest SHA-256 values are pinned in `release.json`.
