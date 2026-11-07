"""Count proof reuse per complete trajectory; diagnostic timings are excluded."""
import json
import os
from pathlib import Path

import warp as wp
import temporal_seed_queries as module


@wp.kernel
def record(proved: wp.array[int], count: wp.array[int], changed_count: wp.array[int],
           error: wp.array[int], records: wp.array2d[int], calls: wp.array[int],
           overflow: wp.array[int]):
    slot = calls[0]
    calls[0] += 1
    if slot >= records.shape[0]:
        overflow[0] = 1
        return
    cached = int(0)
    for i in range(wp.min(count[0], proved.shape[0])):
        cached += int(proved[i] != 0)
    records[slot, 0] = count[0]
    records[slot, 1] = cached
    records[slot, 2] = changed_count[0]
    records[slot, 3] = error[0]


instances = []


class CountedQueries(module.TemporalSeedQueries):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = wp.zeros((1024, 4), dtype=int, device=self.device)
        self.calls = wp.zeros(1, dtype=int, device=self.device)
        self.overflow = wp.zeros(1, dtype=int, device=self.device)
        wp.load_module(module=__name__, device=self.device)
        instances.append(self)

    def run(self):
        super().run()
        wp.launch(record, 1, inputs=[self._temporal_inputs[-3], self._check_inputs[2],
            self._temporal_inputs[-1], self._check_inputs[5], self.records, self.calls,
            self.overflow], device=self.device)


if __name__ == '__main__':
    module.TemporalSeedQueries = CountedQueries
    module.install()
    from benchmarks.mesher import main
    main()
    rows = []
    for query in instances:
        count = int(query.calls.numpy()[0])
        assert query.overflow.numpy()[0] == 0 and count <= len(query.records)
        history = query.records.numpy()[:count]
        rows.append(dict(site_capacity=len(query._check_inputs[0]), cell_capacity=query.capacity,
                         calls=count, cells=int(history[:, 0].sum()),
                         unchanged_proved_cells=int(history[:, 1].sum()),
                         changed_queries=int(history[:, 2].sum()), records=history.tolist()))
    Path(os.environ['WARPOWER_TEMPORAL_DIAGNOSTIC']).write_text(json.dumps(dict(
        scope='Instrumented complete solves, including all graph executions; no diagnostic timing claim.',
        columns=['cells', 'unchanged_proved_cells', 'changed_current_canonical_queries', 'error'],
        instances=rows), indent=2)+'\n')
