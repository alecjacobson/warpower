# Local material-rollback pilot evidence

See [the report](../../docs/selective-topology-rollback.md). The benchmark improves
uniform replay time while slightly reducing worst-element quality; no production
policy is adopted. Raw outputs and unsuccessful attempts are checksum-pinned in
`raw-manifest.json` and published as evidence assets outside ordinary Git.

The [published evidence bundle](https://github.com/alecjacobson/warpower/releases/tag/evidence-clipping-production-bfd5448) contains the raw outputs, traces, source snapshots and drivers. All 1376 members were verified after an independent download. Archive and manifest SHA-256 values are pinned in `release.json`.
