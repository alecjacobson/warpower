# Issue 230: next experiment after issue 233 qualification

Current private ActiveDelaunayWorkspace reserves only 1024/2048 child routes.
Above2048 active sites, it pays for all reserved sites, including repeated site0
padding. The exact prefix proof already applies at4096/8192: prepared inactive
sites equal original site0, every distinct input/canonical first owner remains
in the selected prefix, and original IDs are not renumbered. Existing child
failure ->full constructor recovery must remain intact.

First experiment should add existing complete4096/8192 private constructors,
without introducing new geometry algorithms. Keep the full workspace as output
owner. All children/views/conditions are created before capture; choose on the
device count, copy accepted connectivity/scalar diagnostics DtoD, and preserve
full-route fallback. Reset every child's retained fragment/preference at fresh
solve start. Deformation or shrinkage never grants a stale proof: reuse still
rebuilds current owners/hierarchy and certifies retained cells.

Two setup choices to compare before production selection:
- Extend automatic private default prefix capacities, filtering below the full
  reserved size, while keeping explicit caller levels validated and ordered.
- Keep the router's old default and let the VTM owner request larger children.
The latter minimizes unrelated direct-private changes; the former gives the
router a consistent automatic policy. Neither changes public standalone APIs.

Storage is the main tradeoff. Independent children duplicate scratch and graphs;
record persistent workspace bytes, setup/first execution, graph size, and all
level transitions. Do not label this scratch sharing; issue122 remains separate.
Measure the unused-child cost on fixed full16k solves and smaller/default cases.

Acceptance cases:4095/4096/4097,8191/8192/8193 and16384; growth then shrink;
unrelated inputs, first-owner duplicates, ties, nonfinite/flat data, invalid
counts and recovery; forced child capacity/strategy failure and full fallback;
non-default captured streams and stable output pointers. The full constructor
must remain globally certified. No view/allocation/readback in capture.

Performance: reserved16k target4k is currently~7.3s vs standalone-capacity4k~1.5s
in the same VTM schedule. Progressive8k/graded16k should benefit at early levels;
full16k fixed32 is a guard against setup/unused-branch overhead, not an expected
large gain. Use fresh control/candidate full solves, no CPU validation overlap
with clean timing, and independent saved full/material/boundary/quality checks.

Prefix route changes can change floating reduction order even if exact symbolic
Delaunay connectivity is unique. Preserve any failed bitwise comparison rather
than silently relaxing it; report independent validity, work, surface fit and
quality if trajectories differ. Do not infer CPU advantage from padding savings.

This file is design only: no routing implementation or performance result yet.
Keep its eventual PR separate from233. Larger remaining16k fixed costs include
retained geometry, facet search and hull support; local updates/flips (211/212)
remain relevant to the much larger comparable-quality CPU target.
