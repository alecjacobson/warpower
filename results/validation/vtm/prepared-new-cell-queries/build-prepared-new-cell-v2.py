from pathlib import Path
w=Path('.work/vtm')
s=(w/'prepared_new_cell_v1.py').read_text()
s=s.replace('Private #233 primitive: unchanged predicates/counters, shared current geometry.', 'Private #233 split prototype: preserve query counts, cache exact fallback.')
s=s.replace('from warpower import _sphere_bounds as sb','from warpower import _sphere_bounds as sb\nfrom warpower import _bounded_predicates as low\nfrom warpower._analytic_predicates import strict\nfrom warpower._prepared_seeds import point_value')
start=s.index('@wp.kernel\ndef check_prepared_unproved(')
end=s.index('\n\nclass PreparedUnprovedQueries:')
old=s[start:end]
sig=old[:old.index('    tid = wp.tid()')].replace('check_prepared_unproved','strict_unproved').replace('    segment: Tree,\n','    segment: Tree,\n    pending: wp.array[wp.uint32],\n    unresolved: wp.array2d[int],\n')
body=old[old.index('    tid = wp.tid()'):]
body=body.replace('    shape = shapes[i]', '    a, b, c, d = points[ids[0]], points[ids[1]], points[ids[2]], points[ids[3]]\n    first = int(-1)\n    shape = shapes[i]')
exact='''        result = symbolic_sphere(points, ids, q)
        if result.valid == 0:
            wp.atomic_or(error, 0, 2)
        elif result.sign >= 0:
            wp.atomic_or(error, 0, 8)'''
filters='''        fast = low.sphere_filter(a, b, c, d, p)
        if strict(fast):
            if fast.value > wp.float64(0.0):
                wp.atomic_or(error, 0, 8)
            continue
        value = point_value(shape, p)
        if strict(value):
            if value.value > wp.float64(0.0):
                wp.atomic_or(error, 0, 8)
            continue
'''
body=body.replace(exact, filters+'''        if first == -1:
            first = q
        else:
            first = -2''')
body += '''
    if first != -1:
        unresolved[i, lane] = first
        wp.atomic_or(pending, i, wp.uint32(1) << wp.uint32(lane))
'''
strictkernel=sig+body
exactbody=body
proof='''    if proved[i] != 0:
        if lane == 0:
            wp.atomic_add(proof_statistics, 0, 1)
        return
    if lane == 0:
        wp.atomic_add(proof_statistics, 1, 1)
'''
assert proof in exactbody
exactbody=exactbody.replace(proof,'''    if proved[i] != 0 or (pending[i] & (wp.uint32(1) << wp.uint32(lane))) == wp.uint32(0):
        return
''')
needle='    a, b, c, d = points[ids[0]], points[ids[1]], points[ids[2]], points[ids[3]]'
exactbody=exactbody.replace(needle,'''    saved = unresolved[i, lane]
    if saved >= 0:
        if saved >= points.shape[0]:
            wp.atomic_or(error, 0, 2)
            return
        result = symbolic_sphere(points, ids, saved)
        if result.valid == 0:
            wp.atomic_or(error, 0, 2)
        elif result.sign >= 0:
            wp.atomic_or(error, 0, 8)
        return
    if saved != -2:
        wp.atomic_or(error, 0, 2)
        return
'''+needle)
exactbody=exactbody.replace('    first = int(-1)\n','').replace('    queries = int(0)\n','').replace('        queries += 1\n','')
exactbody=exactbody.replace('''        if first == -1:
            first = q
        else:
            first = -2''',exact)
exactbody=exactbody[:exactbody.index('    wp.atomic_add(statistics, 0, queries)')]
s=s[:start]+strictkernel+'\n\n'+sig.replace('strict_unproved','exact_unproved')+exactbody+s[end:]
s=s.replace('        self.prepare_inputs =', '        self.pending = wp.empty(self.capacity, dtype=wp.uint32, device=self.device)\n        self.unresolved = wp.empty((self.capacity, 32), dtype=int, device=self.device)\n        self.prepare_inputs =')
s=s.replace('            segment,\n        ]','            segment,\n            self.pending,\n            self.unresolved,\n        ]')
s=s.replace('            check_prepared_unproved,','            strict_unproved,')
last=s[s.rindex('        wp.launch('):].replace('strict_unproved','exact_unproved')
s=s+last
# Separate the query passes for the box-disabled comparison fixture.
s=s.replace('        wp.launch(\n            strict_unproved,','        self.query()\n\n    def query(self):\n        self.pending.zero_()\n        wp.launch(\n            strict_unproved,',1)
p=w/'prepared_new_cell_v2.py';assert not p.exists();p.write_text(s)
s=(w/'prepared_new_cell_integration_v1.py').read_text().replace('prepared_new_cell_v1','prepared_new_cell_v2')
s=s.replace('        self.segment = seed_queries.segment','        self.segment = seed_queries.segment\n        self.pending = seed_queries.pending\n        self.unresolved = seed_queries.unresolved\n        if len(self.pending) < self.capacity or self.unresolved.shape[0] < self.capacity or self.unresolved.shape[1] != 32:\n            raise ValueError("Seed fallback scratch must cover completion cells")')
s=s.replace('            self.segment,\n        ]','            self.segment,\n            self.pending,\n            self.unresolved,\n        ]')
s=s.replace('        wp.launch(\n            primitive.check_prepared_unproved,','        self.pending.zero_()\n        wp.launch(\n            primitive.strict_unproved,')
pos=s.index('\n\nclass PreparedProvedCertificate')
last=s[s.rindex('        wp.launch(',0,pos):pos].replace('primitive.strict_unproved','primitive.exact_unproved')
s=s[:pos]+last+s[pos:]
s=s.replace('and hasattr(seed, "segment")','and hasattr(seed, "segment")\n            and hasattr(seed, "pending")\n            and hasattr(seed, "unresolved")')
s=s.replace('            self._certificate._shared_unproved = shared','            assert shared.pending is seed.pending and shared.unresolved is seed.unresolved\n            self._certificate._shared_unproved = shared')
p=w/'prepared_new_cell_integration_v2.py';assert not p.exists();p.write_text(s)
s=(w/'test_prepared_new_cell_v1.py').read_text().replace('from prepared_new_cell_v1 import PreparedUnprovedQueries, check_prepared_unproved, prepare_unproved','from prepared_new_cell_v2 import PreparedUnprovedQueries, prepare_unproved')
s=s.replace('''                            wp.launch(
                                check_prepared_unproved,
                                32 * capacity,
                                inputs=prepared.query_inputs,
                                block_dim=block,
                            )''','''                            prepared.query()''')
s=s.replace('''                        if mode == 1:
                            prepared.run()''','''                        prepared.pending.fill_(wp.uint32(0xFFFFFFFF))
                        prepared.unresolved.fill_(-233)
                        if mode == 1:
                            prepared.run()''')
p=w/'test_prepared_new_cell_v2.py';assert not p.exists();p.write_text(s)
