from pathlib import Path
w=Path('.work/vtm');s=Path('tests/test_stackless_dispatch.py').read_text()
s=s[:s.index('        live = wp.empty(5, dtype=int)')]
s=s.replace('from warpower._facet_seeds import reset_live\n','').replace('from warpower._grouped_seeds import GroupedSeedQueries\n','').replace('from warpower._stackless_seeds import StacklessSeedQueries','from prepared_new_cell_v2 import PreparedUnprovedQueries\nfrom warpower._proved_completion import check_unproved_spheres')
s=s.replace('def test_masks_reset_and_exact_dispatch(mode, block):','@pytest.mark.parametrize("initial_error", [0, 64])\ndef test_masks_reset_and_exact_dispatch(mode, block, initial_error):')
s+='''        error = wp.empty(1, dtype=int)
        statistics = wp.empty(1, dtype=int)
        proof_statistics = wp.empty(2, dtype=int)
        proved = wp.zeros(5, dtype=int)
        original = [points, tets, count, tree, error, statistics, proved, proof_statistics]
        split = PreparedUnprovedQueries(*original, segment=segment, block_dim=block)
        first_error = wp.empty_like(error)
        first_statistics = wp.empty_like(statistics)
        first_proofs = wp.empty_like(proof_statistics)
        wp.load_module(module="warpower._proved_completion", device="cuda:0", block_dim=block)
        with wp.ScopedCapture() as cap:
            error.fill_(initial_error)
            statistics.fill_(7)
            proof_statistics.fill_(11)
            wp.launch(check_unproved_spheres, 32 * 5, inputs=original, block_dim=block)
            wp.copy(first_error, error)
            wp.copy(first_statistics, statistics)
            wp.copy(first_proofs, proof_statistics)
            error.fill_(initial_error)
            statistics.fill_(7)
            proof_statistics.fill_(11)
            split.pending.fill_(wp.uint32(0xFFFFFFFF))
            split.unresolved.fill_(-233)
            split.run()
        for active in [1, 0, 1]:
            count.fill_(active)
            wp.capture_launch(cap.graph)
            np.testing.assert_array_equal(error.numpy(), first_error.numpy())
            np.testing.assert_array_equal(statistics.numpy(), first_statistics.numpy())
            np.testing.assert_array_equal(proof_statistics.numpy(), first_proofs.numpy())
            np.testing.assert_array_equal(split.pending.numpy()[active:], 0)
            if active:
                assert error.numpy()[0] == initial_error | (0 if expected[0] else 8)
                assert split.pending.numpy()[0] == want
                saved = split.unresolved.numpy()[0]
                if mode == "lane31-rejected":
                    assert saved[31] == 31
                elif mode == "all":
                    np.testing.assert_array_equal(saved, -2)
                elif mode.startswith("overflow"):
                    assert saved[4] == (4 if mode == "overflow-first-only" else -2)
            else:
                assert error.numpy()[0] == initial_error
                assert statistics.numpy()[0] == 7
                np.testing.assert_array_equal(proof_statistics.numpy(), 11)
'''
p=w/'test_prepared_new_cell_dispatch_v2.py';assert not p.exists();p.write_text(s)
s=(w/'test-prepared-new-cell-existing-v1.py').read_text().replace('integration_v1','integration_v2')
p=w/'test-prepared-new-cell-existing-v2.py';assert not p.exists();p.write_text(s)
