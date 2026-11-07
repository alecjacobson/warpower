# Possible #43 fast candidate construction (not implemented)

The complete default torus1000 uses141successful base reconstructions, noaux/symbolic.
Native3solveprofilecf15:423facechecks5584ms,423ordinarybuilds5136ms,423spherechecks2273ms.
Issue84targetsfacecompaction/normalreuse. WideCPUgoal probablyalso needs fast ordinary
construction: current100krandomF64fullhull~473ms vsbasicF32diagram~11ms, beforecertcost.

Potential architecture: F32 candidate cells first; exact original-coordinate
certificate accepts only full valid mesh. Interior bounded F32 cells can emit
finite dual vertices in their existing clip pass; cells touching clipping-box
planes, explicit status failures and near-degenerate suspects enter a compact
F64 repair list. Hull cells are a small fraction at large randomN. Keep existing
F64 and auxiliary/symbolic full-recovery paths as a final bounded fallback.
Avoid globalF32thenalwaysglobalF64 if one defect occurs; local repair is key.

Each candidate tet's owner is its minimum siteID. Certificate rejection can mark
its generating owner for F64 reconstruction. A missing tet might yield only an
unsupported exposed face. For every site on the outside of that oriented face,
mark the site's cell plus all face vertices. The true missing adjacent tet's
opposite vertex is one such witness; if it has smallerIDthanallfacevertices, that
marks its generating owner too. Repair a layer of missing geometry, certify
again; after a small finite repair-round budget, use the complete existing F64
constructor ratherthanassuming convergence. A strict final original-coordinate
certificate remains the correctness gate regardless of candidate quality.

Repair regenerates candidates for marked owners and retains only candidates of
unmarked owners; append alone would retain wrong cells/duplicates. Preserve
canonical duplicate ownership and consistent symbolic tie certification. A
per-site repair mask and device compact worklist can use existing prefix/chunk
infrastructure. Add no optional work to defaultdiagram/uncertifiedDelaunaypaths
unless opted in; prefer a private candidate layer owned by DelaunayWorkspace.
No native CUDA snippets or CPU geometry, fixedbuffers/devicecounts/captureloops.

This is design only. Need inspect current _graph F32boundedclipper scratch
lifetime and _tetrahedra candidate emission before choosing reuse versus a
separate F32 full-hull kernel. Shared kernels should compile away optional
emission flags for ordinary calls. Validate wrong sphere/negative face/missing
cells, repeated repair/maskedprefix/invalidinputs, harddegeneracies, first/warm
performance atlargeN. Still need sphere-cert cheap conservative radial bounds
and facecompaction: even 11mscandidate+290mscertificate wouldremainfarfromCPU10x.
