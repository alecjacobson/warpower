# Experiment artifacts

Keep source, compact reports, manifests and reproduction commands in Git. Store raw
meshes, input snapshots, profiler traces, compiled benchmark executables and large
logs in checksum-pinned release bundles. Small deterministic fixtures under
`tests/data/` may stay in the checkout. Use LFS for large maintained inputs that
must behave like checkout files; a benchmark run is normally a release artifact.

New or changed Git blobs are limited to 512 KiB. The artifact-policy command checks
PR changes, rejects raw/binary artifacts outside the small-fixture exception, and
accepts LFS pointers. A [workflow template](ci/artifact-policy.yml) is ready to copy
to `.github/workflows/artifact-policy.yml`; it is not active because the current
GitHub connection lacks the `workflow` permission needed to install it. Until
activation, run the local check before merging. It does not remove existing history. This threshold includes
text reports: summarize large reports and publish the full records in the bundle.

Keep local raw output under ignored `results/local/` or `.work/`. Keep unique data,
source and Git metadata on persistent storage. Publish an archive only after its
member hashes match the pinned inventory; download it again and verify the remote
copy before considering publication complete. Preserve failures and interrupted
runs, identifying them separately from successful qualification.

## Bundle format and tools

A JSON inventory contains `files`, each with a repository-relative POSIX `path`,
byte count `bytes` and lowercase SHA-256 `sha256`. Include pinned source revisions,
commands, input provenance and the measurement scope as metadata. Packing refuses
stale source files and existing destination files:

```bash
python -m benchmarks.artifacts pack --root . --inventory inventory.json \
  --archive results/local/run-evidence.tar.gz \
  --manifest results/local/run-manifest.json
```

The generated manifest adds the archive's name, bytes and SHA-256. Both archive
identity and every member are checked; unexpected entries, duplicates, links,
unsafe paths, missing files and mismatched content fail verification:

```bash
python -m benchmarks.artifacts verify \
  --archive results/local/run-evidence.tar.gz --manifest run-manifest.json
```

Download a release with `gh release download TAG --repo alecjacobson/warpower
--pattern ASSET --dir results/local/download`. Verify it against the committed
manifest. Extract verified bundles into a **new empty directory**, retaining the
original relative paths, before copying selected files into an existing checkout.
They can contain experimental or superseded source; publication does not qualify
that code or imply the reported goal was achieved. Native benchmark executables
are platform-specific evidence; prefer rebuilding from the pinned sources.

Run the forward policy locally with:

```bash
python -m benchmarks.check_artifact_policy --base origin/main
python -m unittest tests.test_artifacts
```

The check inspects committed changes, so run it after committing. Unchanged
historical raw data is not a reason to keep adding more.

## Historical migration

The original size inventory found approximately 2.21 GB of tracked files, almost
all under `results/`. Publishing new release bundles does not shrink those old
objects. No history rewrite or deletion of historical evidence is part of this
forward policy.

A separate migration must inventory every affected blob and its referencing
commits/tags, preserve original checksums and commit-to-artifact mappings, verify
all remote bundles, and provide a tested recovery copy of refs and objects. Then
compare a normal forward deletion (old clones retain history) with a coordinated
history rewrite (commit IDs change and collaborators must resynchronize).
[#262](https://github.com/alecjacobson/warpower/issues/262) remains open for that
inventory, CI activation and a concrete migration/recovery plan. Any rewrite requires explicit
user authorization after that plan is reviewable.

## Published checkpoint bundle

The [selective-step-history evidence release](https://github.com/alecjacobson/warpower/releases/tag/evidence-selective-step-history-b343f70)
preserves the November 8 discussion checkpoint for draft PR #260. Its 150,350,767-byte
archive contains 609 original raw records, 59 pinned source files and ten checkpoint
documents. The downloaded archive and all 678 members match the
[committed manifest](../experiments/artifact-bundles/selective-step-history/manifest.json);
[remote verification](../experiments/artifact-bundles/selective-step-history/remote-verification.json)
records the independent download check. Original persistent files remain intact.

This is a prerelease of experimental evidence. Interrupted qualification and failed
CPU comparisons remain included; neither PR #260 nor the wide CPU-speedup goal is
marked complete by this publication.

## Precision, CPU baseline and continuation bundle

The [VTM precision and baseline evidence release](https://github.com/alecjacobson/warpower/releases/tag/evidence-vtm-precision-and-baselines-f3b35c9)
contains raw outputs for mixed boundary queries (#279), CGAL optimizer controls
(#281), initial corner budgets (#277), and the negative cell-continuation trials
(#282). All 1,047 members of the 106,229,911-byte archive were verified again after
an independent download. The [manifest](../experiments/artifact-bundles/vtm-precision-and-baselines/manifest.json)
and [verification record](../experiments/artifact-bundles/vtm-precision-and-baselines/remote-verification.json)
pin both the complete archive and each member.

The archive preserves failed trials, commands, source snapshots and one-off
drivers as well as accepted results. It does not include subsequent selective
step-history qualification. Each experiment identifies its actual runtime; the
included `f3b35c9` main snapshot is for convenience. This is not a repository or
environment backup, resolution of the default Armadillo boundary problem, or a
quality-matched CPU speedup claim. Original persistent evidence and historical
Git objects remain intact.
