# Cell-continuation evidence

See [the report](../../docs/cell-continuation.md). All four variants are negative
performance results; no production path changes.

- `pilots.json`: pinned source hashes, three functional tests per version, setup,
  memory, nine alternating pairs per cloud, input hashes and CPU-quota records.
- `memory.json`: six isolated checks on v4, including allocation lifetime.
- `kernel-properties-v2.json`: Warp-reported register/local-memory use.
- `runtime.json`: proof that the branch changes no production runtime files.
- `raw-manifest.json`: retained local log/report hashes, not public download URLs.

Raw logs and mesh arrays are excluded from Git. The original and simplified
continuation layouts remain accessible through the source commits in the report.
