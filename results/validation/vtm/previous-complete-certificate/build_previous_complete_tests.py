from pathlib import Path
w=Path('.work/vtm');s=Path('tests/test_large_reuse.py').read_text()
pos=s.index('@wp.kernel\ndef remove_last_cell')
s=s[:pos]+'''@wp.kernel
def consume_whole(phase:wp.array[int],status:wp.array[int],history:wp.array2d[int]):
    i=wp.tid()
    history[phase[0],i]=status[i]


'''+s[pos:]
s=s.replace('    pointers = w.tetrahedra.ptr, w.tetrahedra_count.ptr, w.points.ptr','    whole = wp.empty((len(counts),3),dtype=int,device=device)\n    pointers = w.tetrahedra.ptr, w.tetrahedra_count.ptr, w.points.ptr')
s=s.replace('            wp.launch(advance, 1, inputs=[phase])','            wp.launch(consume_whole,3,inputs=[phase,full._whole_status,whole])\n            wp.launch(advance, 1, inputs=[phase])')
s=s.replace('    assert reused[1, 2] > 0  # The large completion consumes current-input proofs.','    assert whole.numpy()[1].tolist() == [1,1,0]  # Fresh whole-mesh certificate.\n    assert reused[1,2] == 0  # No fragment proof-consumption count on that branch.')
(w/'test_previous_complete_large.py').write_text(s)
(w/'test-previous-complete-existing.py').write_text('''import pytest
from whole_previous_certificate import install
install()
raise SystemExit(pytest.main(['-q','tests/test_facet_seeds.py','tests/test_proved_completion.py','tests/test_reuse_delaunay.py','.work/vtm/test_previous_complete_large.py','tests/test_support_reuse.py']))
''')
(w/'run-previous-complete.py').write_text('''from whole_previous_certificate import install
install()
from benchmarks.mesher import main
main()
''')
