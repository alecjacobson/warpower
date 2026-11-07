"""Run existing independent predicate/fragment/capture checks with private preparation."""
import pytest
import warpower._reuse_delaunay as reuse
import tests.test_facet_seeds as seed_tests
from seed_filter_prepared import PreparedSeeds

reuse.CertifiedSeeds=PreparedSeeds
seed_tests.CertifiedSeeds=PreparedSeeds
raise SystemExit(pytest.main(['-q','tests/test_facet_seeds.py','tests/test_proved_completion.py',
                             'tests/test_reuse_delaunay.py','tests/test_support_reuse.py']))
