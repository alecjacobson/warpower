# Shared transient clipping storage: isolated pilot

Each full/prefix Delaunay constructor currently reserves separate FP64 clipping
scratch for original and normalized coordinates. These builders execute serially
inside one mesher graph, and cell scratch is overwritten between chunks. The
benchmark-only prototype shares that temporary storage within one constructor
family, retaining separate hierarchies, output connectivity, fragments, strategy
state and certificates. The largest constructor allocates first.

This pilot uses a setup-only allocator hook and an explicit retained owner; it
restores the allocator after construction. Production adoption uses explicit
ownership rather than this process-global hook, and requires separate qualification.
The historical prototype must be run at its pinned revision, before production
storage ownership changes; running it against newer defaults is not this experiment.

## Complete solves

NVIDIA L40 / Warp 1.19.0, median five additional fresh captured complete solves.
Every timing window has full CPU quota. Setup, transfers, export and independent
CPU validation are separate. The storage columns measure active Warp memory-pool
bytes added by the complete workspace constructor, excluding its existing domain,
compiled modules and CUDA graph/driver storage. GB denotes 10^9 bytes.

| Case | Separate (ms) | Shared (ms) | Separate storage (GB) | Shared storage (GB) | Saved bytes |
|---|---:|---:|---:|---:|---:|
| torus4096 | 1286.80 | 1285.82 | 1.825 | 0.776 | 1,048,576,000 |
| torus8192 | 1972.95 | 1974.82 | 3.031 | 1.144 | 1,887,436,800 |
| torus16384-prefix4096 | 3113.39 | 3107.63 | 4.592 | 1.866 | 2,726,297,600 |
| torus16384-fixed32 | 2905.45 | 2905.23 | 4.579 | 1.853 | 2,726,297,600 |
| uniform | 2323.03 | 2326.33 | 2.956 | 1.069 | 1,887,436,800 |
| graded | 2299.94 | 2305.56 | 2.956 | 1.069 | 1,887,436,800 |

The uniform/graded cases are 8K Armadillo with native surface queries, mixed
nearest-site queries and trusted input. The torus uses the reference backend.
Both sides include the merged selective step-history policy. This is an isolated
storage experiment, not a first-pass capacity or CPU speed comparison.

All 12 saved meshes pass independent full-hull original-coordinate CGAL checks,
material incidence, boundary vertex links and exact boundary self-intersection
checks. Gallery VTU round trips are exact. Sites are bitwise equal; roles,
canonical full/material connectivity, oriented boundary faces and accepted work
counts match in every pair. All 24 existing active/larger-prefix contracts also
pass with the prototype applied, including changed inputs, failure/fallback and
recovery. This pilot does not constitute the production memory/lifetime gate.

The byte savings exactly match the removed duplicate scratch allocations.
An 8K workspace saves 1,887,436,800 bytes; a 16K workspace saves 2,726,297,600.
Replay time is effectively unchanged. Source pins, commands, every timing sample,
allocation requests, full independent reports and raw-file hashes are in
[experiments/shared-clipping-scratch](../experiments/shared-clipping-scratch/).
No raw meshes or traces are added to ordinary Git. Production work continues in #122.
