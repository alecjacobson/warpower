# Boundary nearest-site seeding: unadopted probe

The private probe in [issue #221](https://github.com/alecjacobson/warpower/issues/221)
does not change production. It improves a complete torus4096 solve by only
0.79% and slows hand7 by 4.25%. The hand material/boundary comparison also fails,
so this is not a quality-equivalent speedup result.

The preceding [query-cache profile](surface-query-cache.md) attributes about
151 ms to 458 boundary-sample accumulation calls on torus4096. Each sample
already has an owner slot. The prototype evaluates that previous owner's
**current** position against the current sample to initialize a search bound.
It does not reuse an old distance or accept an old nearest result.

Only an owner within the current active prefix and array length is eligible.
Free-site queries also check occupancy. Nonfinite distances and values outside
the original finite search limit retain the ordinary initialization. Current
hierarchy traversal, pruning, leaf arithmetic and lowest-ID tie handling remain.
Both existing owner arrays are initialized before any graph executes; there
are no extra persistent arrays or replay copies/launches.

Sixty original/seeded comparisons cover 12,000 queries across random sites,
duplicates, ties, occupied sites, stale/invalid owners, changing active prefixes,
extreme scales and nonfinite queries. Owner IDs and distance bits match.
That focused result alone does not establish complete trajectory equivalence.

The clean pilot uses main `74601fc1670f66bf46298a3b6a78e5b91df86230`, the L40,
Warp 1.19.0 and three additional fresh complete replays per job. All four jobs
have clean quota windows, with independent CPU geometry validation afterward.
Setup, compilation, transfers and exports are outside replay timing.

| Complete solve | Original ms | Seeded ms | Change |
|---|---:|---:|---:|
| Torus, 4,096 sites | 1,512.812 | 1,500.886 | −0.79% |
| Hand seed7, 1,000 sites | 1,297.702 | 1,352.870 | +4.25% |

All four meshes independently pass exact original-coordinate full-hull,
orientation and empty-sphere checks, material quality/topology, boundary vertex
links, self-intersection and export checks. Recorded counts and work match.
Torus positions and canonical full/material/oriented-boundary connectivity match.
Hand coordinates differ by at most 1.0408340855860843e-17 and its material cells
and boundary differ. The strict checker failure is retained; this probe does
not claim that variation has the same cause or assessment as issue #220.

The performance regression is sufficient to reject adoption. No broader
production memory/capture gates were run for this unadopted variant. Further
work should measure the accumulation kernel's shared reductions before assuming
nearest-site traversal is its dominant cost.

The [manifest](../results/validation/vtm/boundary-owner-seed/manifest.json)
preserves the prototype, proof scope, query comparisons, all four meshes,
timings and original failed strict comparison. Independent diagram/Delaunay,
boundary/core routines and the complete mesher retain their previous behavior;
the README flagship is unchanged.
