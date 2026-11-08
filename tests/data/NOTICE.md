# Dataset attribution and terms

`exterior-band-patch.json` derives from the Stanford Armadillo model, distributed by
libigl tutorial-data. Credit: Stanford University Computer Graphics Laboratory.
The original source commit, SHA256 and our coordinate normalization are recorded
in [the uniform example's provenance](../../results/examples/vtm/armadillo-uniform/input-provenance.json).
The input surface connectivity was not repaired or simplified. Armadillo is an additional scanned benchmark, not a model
shown in the Alliez VTM paper.

The Stanford terms permit research use and free redistribution with source
acknowledgment. Commercial use requires Stanford permission. This fixture is
not relicensed under the Warpower code license. See the [Stanford terms](https://graphics.stanford.edu/data/3Dscanrep/) and
[pinned libigl attribution](https://raw.githubusercontent.com/libigl/libigl-tutorial-data/644dd4104843b6d736745d9dafbd70bf8d175648/README.md).

The fixture is a reduced eleven-cell patch of the intentionally preserved
failed material candidate in [issue #142](https://github.com/alecjacobson/warpower/issues/142).
Its initial boundary is nonmanifold. It tests a post-classification recovery
transaction; it is not a complete mesher output. This notice applies only to
`exterior-band-patch.json`, not to other test fixtures in this directory.

# Proven exterior recovery fixtures

`proven-exterior-original.npz` and `proven-exterior-continuation.npz` are reduced
field exports from Warpower's own procedural thin-hand example, seed 7, with
1,000 sites. They contain no third-party scan. The original snapshot is the
failed material extraction after the ordinary fitting schedule at revision
`a34910d64a27dbb8e0fb03f0813e76e00e799744`. The continuation snapshot follows one
certified, energy-guarded midpoint-aware fitting step on that same failed state.
Its supplied connectivity is the complete Delaunay hull, before material
extraction. Tests use the snapshots to check successful proof removal, complete
rollback, and captured producer/consumer failure recovery independently of fresh
sampling. Fresh-solve tests separately generate their inputs on device.

Source snapshots and independent geometric checks are retained in the
[proven exterior recovery evidence](../../results/validation/vtm/proven-exterior-recovery/manifest.json).
