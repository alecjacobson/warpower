"""Run existing support/proof/reuse regressions with the private kernel installed."""
import sys
import warp as wp
from bounded_hull_box_v2 import install
wp.init()
for dim in [32,256]:wp.load_module(module='bounded_hull_box_v2',device='cuda:0',block_dim=dim)
install()
import pytest
raise SystemExit(pytest.main(['-q','tests/test_axis_hull_support.py','tests/test_bounded_hull_support.py','tests/test_support_reuse.py','tests/test_proved_completion.py','tests/test_reuse_delaunay.py','tests/test_wall_reuse.py',*sys.argv[1:]]))
