The current 4k torus trace spends 736.293 ms in private prepared seed queries. #169's initial exact-rejection stop saves only 1.5–2.1% of complete 1k/4k solves, with bitwise outputs; broader adoption validation is still pending. Investigate the arithmetic cost paid for points inside a candidate's conservative sphere box.

Two isolated FP64 variants are worth comparing:
1. Specialize the current prepared cofactor interval expression to a point. Replace interval(p,p) with the exactly promoted point scalar before the existing bounded subtraction/products. A point needs no AABB midpoint/radius uncertainty. Keep the current prepared Sphere error bounds, strict-negative acceptance and exact/symbolic fallback.
2. After the cheap sphere box, use #166's certified analytic physical-sphere filter on original coordinates. A strict negative sign passes that query; a strict positive sign rejects the candidate. Uncertainty must first use the existing prepared interval test, then the unchanged symbolic fallback. This avoids forcing poorly conditioned queries into exact arithmetic when the prepared cofactor expression can already certify them.

Measure these separately from #169's rejection stop, without adding storage or changing live/error/reset policy. Keep ordinary seed/standalone paths untouched during exploration. Do not remove the prepared interval fallback merely because an analytic filter is usually fast; their numerical conditioning differs.

Acceptance: independently check full point-expression error enclosures and strict accepted signs against rational original-coordinate sphere determinants; exercise valid/invalid candidates, exact ties, extreme float32 ranges and changed captured producers. Compare candidate masks and complete output/work with the current checker. Require fresh complete timing pilots before broad examples, then full independent geometry, memory/capture and standalone gates before adoption. Preserve all failed/rejected variants. Pure Warp/device execution and graph capture remain mandatory.

Tracked under #29/#40/#43/#87. No implementation or gain is claimed by this issue; #169 and #167 remain separate experiments and the CPU goal is unmet.
