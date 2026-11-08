# Issue 233 adoption decision

Adopt the simpler shared-geometry/current-segment prototype (v1), with unchanged
original point predicates and error/query/proof counters. Keep the split v2
experiment as measured evidence; it adds no useful full-solve benefit here.

V1 fresh pairs: fixed16k 5233.149 -> 4895.015 ms, graded16k 7311.472 ->
6980.263 ms. All four outputs independently valid; bitwise coordinates and
canonical cells/material/oriented boundary/roles/work match.

V2 fresh pairs: fixed16k 5252.001 -> 4896.842 ms, graded16k 7321.416 ->
6992.625 ms, 8k progressive3230.791 ->3096.101 ms, 8k fixed1595.624 ->1503.609 ms.
Grid retained replays2.687680 ->2.702912 ms. All timing windows quota-clean.
Independent v2 geometry checks are running at decision time; no merge yet.

Additional v1 candidates, compared to those retained fresh controls (not new
interleaved pairs): 8k progressive3087.348 ms, fixed1500.224 ms, retained grid
2.690048 ms. All quota-clean; independent checks pending. The first grid harness
attempt followed the venv interpreter symlink to system Python and failed at
NumPy import before GPU setup. Its log/driver are retained; a new explicit-venv
retry succeeded. No meshing failure is hidden.

Native fixed16k v1: prepare6.744 ms +checker464.820 ms; v2: prepare6.757 +
strict434.386 +exact40.593 ms, each554calls. Strict uses80registers vs combined
255, but its30.434 ms saving is smaller than the added40.593 ms exact pass.
Totals4889.125 vs4899.521 ms are diagnostic profiles, not clean timing evidence.

Production qualification must still cover full suite, private scratch lifetime,
full/loop native captures, fresh matched production controls including unchanged
public APIs, and independent final output checks. Wide native CPU advantage and
Armadillo boundary failure remain open. README stays unchanged.
