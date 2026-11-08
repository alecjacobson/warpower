The #233 standalone-path independence question is resolved; the detailed cause of existing timing variability remains open here.

Two whole-benchmark native traces show the same143 executed kernel configurations (names, counts, grids, blocks and registers) in all18 measured full replays. No new private query kernel executes. Crucially, an actual dependency audit of both complete benchmark runs finds40 byte-identical project modules loaded, and **neither modified runtime module is imported**. The hand fixture bytes also match. Thus #233 introduces no code/work dependency in this standalone call.

The slow/fast variation is present in the existing `check_local_spheres_4b35eafc` kernel in both source versions: baseline0.503–0.912ms, candidate0.512–0.894ms in these diagnostic traces. The rest of the dominant clipping work stays stable. This localizes the observed variation; it does not establish a low-level cause. Row ordering/divergence or scheduling are hypotheses, not proven explanations.

All original and ABBA timing samples are retained, including the primary+2.23% and the mixed repeats. Profiler numbers are diagnostic and will not replace clean benchmark samples. Keep this issue open for explaining/reducing the preexisting local-certificate variance. The source-dependency and native-signature evidence will be published with #233 qualification.
