import pytest
import warpower._reuse_delaunay as reuse
import tests.test_facet_seeds as seed_tests
from seed_filter_staged import StagedSeeds

reuse.CertifiedSeeds = StagedSeeds
seed_tests.CertifiedSeeds = StagedSeeds
raise SystemExit(pytest.main(['-q', '.work/vtm/test_seed_staged_bounds.py', 'tests/test_facet_seeds.py', 'tests/test_proved_completion.py', 'tests/test_reuse_delaunay.py', 'tests/test_support_reuse.py']))
