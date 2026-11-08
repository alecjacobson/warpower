# Larger-prefix production candidate

Baseline df74493c97633324147cb39ef4c41a7d34c7b90a, NVIDIA L40, Warp1.19.0.
The isolated prototype is selected for production qualification, not yet merged.

Four fresh interleaved control/candidate pairs, three complete timed replays each,
passed the zero-CPU-quota-throttling gate. Compilation, setup, upload and offline
validation are excluded from replay; first replay and setup are recorded.

| Complete solve | Before ms | Candidate ms |
| --- | ---: | ---: |
| target4K, reserved16K | 7228.489258 | 1556.588135 |
| progressive8K | 3085.531250 | 2899.556152 |
| fixed16K, 32 iterations | 4894.558105 | 4884.342285 |
| graded16K | 6972.486816 | 5733.368164 |

All eight saved meshes independently pass full-hull/orientation/empty-sphere,
material quality/topology, boundary vertex links and self-intersection checks;
graded VTU exports round-trip. Every pair has bitwise-identical points, canonical
full/material cells, oriented boundary, roles and solver work. Routing counts
change intentionally and are recorded separately in the original benchmark JSON.
The strict checker is larger-prefix-pilot-v1-independent.json; no failed timing,
geometry or strict comparison attempt occurred in this prototype batch.

The tradeoff is substantial additional persistent storage. At reservation16K,
two children add 1,853,169,562 bytes (target4K: 2,551,578,988 → 4,404,748,550;
fixed16K: 2,538,128,443 → 4,391,298,005). At reservation8K, the 4K child adds
894,484,877 bytes (2,044,662,012 → 2,939,146,889). These are workspace pool bytes,
not process peak memory. Setup increases from 4.001 to5.034s for the first case,
3.942 to4.481s for progressive8K, and3.327 to4.184s for fixed16K. Do not hide
these costs or describe this change as scratch sharing; issue122 remains open.

Twelve private correctness cases pass: nine main captured/failure cases (29.34s)
and three sparse-first-owner/one-ULP edge cases (5.12s), including full capacity
16385 where the existing full constructor does not retain fragments.

Production keeps the direct private router default unchanged, expands only its
explicit level validation to8192, and has VTM supply fitting levels at setup.
No device kernel, point predicate, certificate, publish/reset/fallback method or
standalone public construction algorithm changes. Production tests add the twelve
private cases and an oversized-child setup rejection check.

Merge still requires production full suite, fresh production/baseline pairs,
independent outputs, standalone guards, known Armadillo failure guards, captured
producer/mesher/consumer validation, native audits and isolated memory checks.
The comparable-quality wide CPU goal and Armadillo issue228 remain unmet.
