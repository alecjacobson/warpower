# Exact-input reconstruction opportunities

An observation-only study finds too little opportunity to justify an all-input
equality cache for the large fixed-site trajectory. No cache or runtime change
is adopted. This is a bounded result under [#130](https://github.com/alecjacobson/warpower/issues/130);
it does not settle reuse when only a subset of sites moves.

On current runtime `c2843f6`, NVIDIA L40 / Warp 1.19.0, each private constructor
compares all original float32 coordinate bits with its internally saved input
from the last successful construction. Signed zeros remain distinct. Invalid
construction and fresh-solve reset invalidate the checkpoint. Every original
construction and certificate still executes; no result is accepted using the
observer. All routes are observed, including inactive reserved constructors.

| Complete trajectory | Reconstructions per solve | Exactly unchanged | Fraction of calls |
| --- | ---: | ---: | ---: |
| torus1000 | 141 | 8 | 5.67% |
| torus4096 | 211 | 9 | 4.27% |
| torus8192-progressive | 249 | 13 | 5.22% |
| torus16384-fixed32 | 227 | 3 | 1.32% |
| graded-torus16384 | 247 | 15 | 6.07% |
| reserved16384-target4096 | 211 | 9 | 4.27% |
| thin-hand | 203 | 10 | 4.93% |

Each case runs three fresh solves. Exact equality decisions and geometry/error
counts repeat across all three. Small changed-site-count differences occur during
the thin-hand trajectory and remain in the raw records. Counting calls alone
does not measure time saved.

## Native cost attribution

One warmed fixed 16K solve records 70,696 kernel activities
over a 5,094.265 ms GPU timeline. Its three unchanged
reconstruction windows total **21.587 ms
(0.424%)**. These windows exclude the observer's
comparison/record/copy work. Even removing all work in those windows gives only
that optimistic avoided-span estimate before cache overhead. It is an
instrumented trace, not a clean performance sample or an implemented speedup.

The trace has one graph launch and an external final synchronization. Captured
nodes are kernels, conditional work, memset and DtoD copies; no host transfers,
callbacks or explicit allocation/free occur inside the profiled work. The
[per-call attribution](../results/validation/vtm/exact-input-observation/unchanged-input-fixed16k-native-v1-attribution.json)
and lossless native trace retain every window and marker match.

## Output and observer validation

The observer's captured test checks unchanged input, a signed-zero bit change,
invalid input, recovery and fresh reset. All seven observation runs succeed.
Six cases have bitwise-identical coordinates, canonical full/material cells,
oriented boundary, roles and work compared with the independently validated
[current controls](larger-active-prefixes.md). The thin-hand control and observer
differ only at point 743's z coordinate, by about **8.0e-19**, while cells,
boundary, roles and work match. Both hand outputs independently pass full-hull,
orientation/empty-sphere, material quality/topology, vertex-link and boundary
self-intersection checks; both VTU exports round-trip. No bitwise-equivalent
hand claim is made.

Two overly strict analysis attempts are retained: v1 required the unused count
in an invalid checkpoint and all numerical change counts to repeat exactly;
v2 additionally required bitwise equality of the hand output. The v3 analysis
checks the actual equality decisions, preserves all numerical differences and
independently validates both hand meshes. No failed runtime invocation is
counted as a successful one.

The [manifest](../results/validation/vtm/exact-input-observation/manifest.json)
pins sources, commands, raw outputs, checker failures, corrected validation,
native trace and shared validators. Reproduce by restoring each artifact's
original `.work/vtm` path and the shared references, with the pinned runtime;
adapt the recorded absolute workspace root as needed. The study driver refuses
overwrites. Use `PYTHONDONTWRITEBYTECODE=1`, the usual regenerable Warp cache,
`PYTHONPATH=.:.work/vtm`, `OPENBLAS_NUM_THREADS=1`, and `OMP_NUM_THREADS=4`.
Observer/native times are diagnostic. Standalone APIs, defaults and README's
earned Paragram result are unchanged; the wide comparable-quality CPU goal
remains unmet.
