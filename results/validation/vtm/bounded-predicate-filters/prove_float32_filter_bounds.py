"""Exact rational envelope for the original first-stage arithmetic DAG.

This accompanies the algebraic proof; it is not a substitute for the proof.
No sampled geometry enters the bound. Every original coordinate has |x|<=16.
The executable grouping comes directly from the current float64 filter source.
"""
import ast
from fractions import Fraction as Q
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

root = Path.cwd()
source = root / 'warpower/_analytic_predicates.py'
unit = Q(1, 2**23)  # Twice the actual binary32 round-to-nearest unit roundoff.
tiny = Q(1, 2**125)  # Twice the smallest positive normal binary32 value.
floor = Q(1, 2**64)
nodes = []


class Envelope:
    def __init__(self, magnitude=Q(16), error=Q(0), rounds=0):
        self.magnitude, self.error, self.rounds = magnitude, error, rounds
        nodes.append(self)

    def __add__(self, other):
        return Envelope((self.magnitude + other.magnitude) * (1 + unit),
                        (self.error + other.error + 2 * tiny) * (1 + unit) + tiny,
                        max(self.rounds, other.rounds) + 1)

    __sub__ = __add__

    def __mul__(self, other):
        # Allow each actual operand to flush a subnormal before the operation,
        # plus an absolute error for output underflow/flush after rounding.
        a, b = self.error + tiny, other.error + tiny
        return Envelope(self.magnitude * other.magnitude * (1 + unit),
                        (self.magnitude * b + other.magnitude * a + a * b) * (1 + unit) + tiny,
                        self.rounds + other.rounds + 1)

    def __neg__(self):
        return Envelope(self.magnitude, self.error + tiny, self.rounds)

    __abs__ = __neg__


module = ast.parse(source.read_text())
functions = [n for n in module.body if isinstance(n, ast.FunctionDef)
             and n.name in {'orientation_filter', 'sphere_filter'}]
assert len(functions) == 2
for function in functions:
    function.decorator_list = []
    function.returns = None
    for argument in function.args.args:
        argument.annotation = None
    found = 0
    for node in ast.walk(function):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Attribute) and node.targets[0].attr == 'error':
            assert isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.Mult)
            assert isinstance(node.value.left, ast.Name) and node.value.left.id in {'ORIENT_BOUND', 'SPHERE_BOUND'}
            node.value = node.value.right  # Return the unscaled permanent envelope.
            found += 1
    assert found == 1
code = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
namespace = dict(wp=SimpleNamespace(float64=lambda x: x, abs=abs), fp=SimpleNamespace(Ball=SimpleNamespace))
exec(compile(code, str(source), 'exec'), namespace)


def pow2_upper(q):
    power = q.numerator.bit_length() - q.denominator.bit_length()
    while Q(2)**power < q:
        power += 1
    return power


class Polynomial:
    """Exact expansion in the independent coordinate-difference variables."""
    def __init__(self, terms=None):
        self.terms = {m: c for m, c in (terms or {}).items() if c}

    def __add__(self, other):
        terms = dict(self.terms)
        for monomial, coefficient in other.terms.items():
            terms[monomial] = terms.get(monomial, 0) + coefficient
        return Polynomial(terms)

    def __neg__(self):
        return Polynomial({m: -c for m, c in self.terms.items()})

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        terms = {}
        for a, ca in self.terms.items():
            for b, cb in other.terms.items():
                monomial = tuple(sorted(a + b))
                terms[monomial] = terms.get(monomial, 0) + ca * cb
        return Polynomial(terms)

    def __abs__(self):
        # In these source expressions abs only encloses a difference variable
        # or its single product; the relative-only factors stay positive.
        assert len(self.terms) <= 1
        return Polynomial({m: abs(c) for m, c in self.terms.items()})


result = dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
              coordinate_magnitude_limit=16, relative_model_unit=str(unit),
              absolute_per_operation_allowance=str(tiny), threshold_floor=str(floor), filters={})
for name, n, coefficient in [('orientation_filter', 4, Q(1, 2**19)), ('sphere_filter', 5, Q(1, 2**17))]:
    nodes.clear()
    answer = namespace[name](*[[Envelope() for _ in range(3)] for _ in range(n)])
    determinant, permanent = answer.value, answer.error
    # Expand the grouped determinant into signed monomials of exact coordinate
    # differences. A monomial has at most N relative-rounding factors. The
    # permanent contains those same absolute monomials, each with at most M
    # factors. All factors lie in [1-unit,1+unit], including shared factors.
    N, M = determinant.rounds, permanent.rounds
    generic_coefficient = ((1 + unit)**N - 1) / (1 - unit)**M
    absolute_error = determinant.error + generic_coefficient * permanent.error
    # Lower bound on the rounded coefficient*actual_permanent+floor, allowing
    # two rounded operations plus conservative operand/output flush allowances.
    coefficient_lower = coefficient * (1 - unit)**2
    floor_lower = floor * (1 - unit) - 8 * tiny
    assert coefficient_lower > generic_coefficient
    assert floor_lower > absolute_error
    peak = max(x.magnitude + x.error for x in nodes)
    assert peak < Q(2)**40  # All intermediates are very far from binary32 overflow.
    expected = (8, 8) if n == 4 else (16, 17)
    assert (N, M) == expected, (name, N, M)
    vectors = [[Polynomial({(f'{i}:{j}',): 1}) for j in range(3)] for i in range(n)]
    # Work in exact difference coordinates: orientation's origin is a, and
    # sphere's origin is q. This is an algebraic variable choice, not a runtime
    # translation of the floating-point input.
    vectors[0 if n == 4 else 4] = [Polynomial() for _ in range(3)]
    expanded = namespace[name](*vectors)
    absolute_monomials = {m: abs(c) for m, c in expanded.value.terms.items()}
    assert expanded.error.terms == absolute_monomials
    assert len(absolute_monomials) == (6 if n == 4 else 72)
    result['filters'][name] = dict(
        determinant_round_factors=N, permanent_round_factors=M,
        generic_relative_coefficient=str(generic_coefficient),
        threshold_coefficient=str(coefficient), coefficient_inequality=True,
        absolute_error_bound=str(absolute_error), absolute_error_pow2_upper=pow2_upper(absolute_error),
        floor_inequality=True, intermediate_magnitude_pow2_upper=pow2_upper(peak),
        operations_or_leaves=len(nodes), permanent_matches_absolute_monomials=True,
        monomials=len(absolute_monomials))
out = root / '.work/vtm/float32-filter-bound-proof.json'
out.write_text(json.dumps(result, indent=2) + '\n')
for name, record in result['filters'].items():
    print(name, {k: record[k] for k in ['determinant_round_factors', 'permanent_round_factors',
          'absolute_error_pow2_upper', 'intermediate_magnitude_pow2_upper', 'coefficient_inequality', 'floor_inequality']})
