# Schedule study for #37

The study is a comparison of existing public scheduling options, with no runtime
or default changes. Every graph starts from fresh samples, includes surface refit,
and finishes extraction and diagnostics. All requested sites are referenced in
the torus outputs. Allocation, compilation, transfers and serialization stay
outside replay and are separately reported. Independent checks follow timing.

Twelve exploratory torus configurations are complete on 9228a3e, seed 2026.
The fixed-32 schedule improves minimum, p1, p5, median and p95 radius ratios,
sampled two-sided fit and absolute volume error over the progressive default,
with 628 ms versus 821 ms. These are sample-specific results. A 40-iteration
cap worsens minimum quality; no monotonicity claim is warranted.

Before publication, confirm the selected fixed-32 option in a new five-replay
batch against the default, on torus seeds 2026 and 2027, cavity seed 2026 and
disconnected components seed 2026. Use 20 surface subdivisions for the last two
(both schedules use identical sampling). Keep validity failures and every quality
metric, even if the fixed schedule does not generalize. Do not change defaults.
Full-hull exact checks, independent quality and boundary links for all valid
outputs are mandatory. A data-only PR can close #37's remaining comparison
requirement; #40's CPU target remains open.

Publish the parameter table, raw timing/mesh/validation records, full commands,
source/native-reference hashes, and a scientific quality/time plot. README stays
unchanged. CGAL baseline differs in counts and fit; no CPU speedup claim.
