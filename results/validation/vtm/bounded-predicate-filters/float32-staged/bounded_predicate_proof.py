"""Exact rational envelope for the original first-stage arithmetic DAG.

This accompanies the algebraic proof; it is not a substitute for the proof.
No sampled geometry enters the bound. Every original coordinate has |x|<=16.
The executable grouping comes directly from the current float64 filter source.
"""

import argparse
import ast
import copy
import hashlib
import json
from fractions import Fraction as Q
from pathlib import Path
from types import SimpleNamespace

root = Path(__file__).resolve().parents[1]
source = root / "warpower/_analytic_predicates.py"
unit = Q(1, 2**23)  # Twice the actual binary32 round-to-nearest unit roundoff.
tiny = Q(1, 2**125)  # Twice the smallest positive normal binary32 value.
floor = Q(1, 2**64)
nodes = []

# Couple the algebraic proof to the executable binary32 expression tree. Only
# the range gate, final exact promotions and separately bounded threshold floor
# differ from the reference grouping analyzed below.
bounded_source = root / "warpower/_bounded_predicates.py"
bounded_module = ast.parse(bounded_source.read_text())

# The conservative factor count specifies unfused, non-fast-math expressions.
for filename in ["_bounded_predicates.py", "_proved_completion.py", "_grouped_seeds.py"]:
    tree = ast.parse((root / "warpower" / filename).read_text())
    options = [
        ast.literal_eval(n.args[0])
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and ast.unparse(n.func) == "wp.set_module_options"
    ]
    assert len(options) == 1
    assert options[0].get("fast_math") is False and options[0].get("fuse_fp") is False


def function_in(module, name):
    return next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == name)


class RemovePromotions(ast.NodeTransformer):
    def visit_Call(self, node):
        node = self.generic_visit(node)
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "wp"
            and node.func.attr == "float64"
        ):
            assert len(node.args) == 1 and not node.keywords
            return node.args[0]
        return node


def normalized(module, name, binary32):
    f = copy.deepcopy(function_in(module, name))
    if binary32:
        gate = f.body.pop(0)
        args = [a.arg for a in f.args.args]
        expected = ast.parse(
            "if not ("
            + " and ".join("bounded(" + a + ")" for a in args)
            + "):\n    return unavailable()"
        ).body[0]
        assert ast.dump(gate) == ast.dump(expected)
        for node in ast.walk(f):
            if (
                isinstance(node, ast.Assign)
                and isinstance(node.targets[0], ast.Attribute)
                and node.targets[0].attr == "error"
            ):
                value = node.value
                assert isinstance(value, ast.Call) and ast.unparse(value.func) == "wp.float64"
                term = value.args[0]
                assert (
                    isinstance(term, ast.BinOp)
                    and isinstance(term.op, ast.Add)
                    and ast.unparse(term.right) == "ABS_FLOOR"
                )
                node.value = term.left
    f = RemovePromotions().visit(f)
    return ast.dump(f)


reference_module = ast.parse(source.read_text())
for name in ["orientation_filter", "sphere_filter"]:
    assert normalized(reference_module, name, False) == normalized(bounded_module, name, True), name
expected_range = ast.parse(
    "return "
    + " and ".join("wp.isfinite(p[" + str(i) + "])" for i in range(3))
    + " and "
    + " and ".join("wp.abs(p[" + str(i) + "]) <= 16.0" for i in range(3))
).body[0]
assert ast.dump(function_in(bounded_module, "bounded").body[0]) == ast.dump(expected_range)
constants = {
    node.targets[0].id: node.value
    for node in bounded_module.body
    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
}
for name, exponent in [("ORIENT_BOUND", -19), ("SPHERE_BOUND", -17), ("ABS_FLOOR", -64)]:
    expected = ast.parse("wp.constant(wp.float32(2.0**" + str(exponent) + "))", mode="eval").body
    assert ast.dump(constants[name]) == ast.dump(expected), name


class Envelope:
    def __init__(self, magnitude=Q(16), error=Q(0), rounds=0):
        self.magnitude, self.error, self.rounds = magnitude, error, rounds
        nodes.append(self)

    def __add__(self, other):
        return Envelope(
            (self.magnitude + other.magnitude) * (1 + unit),
            (self.error + other.error + 2 * tiny) * (1 + unit) + tiny,
            max(self.rounds, other.rounds) + 1,
        )

    __sub__ = __add__

    def __mul__(self, other):
        # Allow each actual operand to flush a subnormal before the operation,
        # plus an absolute error for output underflow/flush after rounding.
        a, b = self.error + tiny, other.error + tiny
        return Envelope(
            self.magnitude * other.magnitude * (1 + unit),
            (self.magnitude * b + other.magnitude * a + a * b) * (1 + unit) + tiny,
            self.rounds + other.rounds + 1,
        )

    def __neg__(self):
        return Envelope(self.magnitude, self.error + tiny, self.rounds)

    __abs__ = __neg__


module = ast.parse(source.read_text())
functions = [
    n
    for n in module.body
    if isinstance(n, ast.FunctionDef) and n.name in {"orientation_filter", "sphere_filter"}
]
assert len(functions) == 2
for function in functions:
    function.decorator_list = []
    function.returns = None
    for argument in function.args.args:
        argument.annotation = None
    found = 0
    for node in ast.walk(function):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Attribute)
            and node.targets[0].attr == "error"
        ):
            assert isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.Mult)
            assert isinstance(node.value.left, ast.Name) and node.value.left.id in {
                "ORIENT_BOUND",
                "SPHERE_BOUND",
            }
            node.value = node.value.right  # Return the unscaled permanent envelope.
            found += 1
    assert found == 1
code = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
namespace = dict(
    wp=SimpleNamespace(float64=lambda x: x, abs=abs), fp=SimpleNamespace(Ball=SimpleNamespace)
)
exec(compile(code, str(source), "exec"), namespace)


def pow2_upper(q):
    power = q.numerator.bit_length() - q.denominator.bit_length()
    while Q(2) ** power < q:
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


result = dict(
    bounded_source_sha256=hashlib.sha256(bounded_source.read_bytes()).hexdigest(),
    formula_range_constants_verified=True,
    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    coordinate_magnitude_limit=16,
    relative_model_unit=str(unit),
    absolute_per_operation_allowance=str(tiny),
    threshold_floor=str(floor),
    filters={},
)
for name, n, coefficient in [
    ("orientation_filter", 4, Q(1, 2**19)),
    ("sphere_filter", 5, Q(1, 2**17)),
]:
    nodes.clear()
    answer = namespace[name](*[[Envelope() for _ in range(3)] for _ in range(n)])
    determinant, permanent = answer.value, answer.error
    # Expand the grouped determinant into signed monomials of exact coordinate
    # differences. A monomial has at most N relative-rounding factors. The
    # permanent contains those same absolute monomials, each with at most M
    # factors. All factors lie in [1-unit,1+unit], including shared factors.
    N, M = determinant.rounds, permanent.rounds
    generic_coefficient = ((1 + unit) ** N - 1) / (1 - unit) ** M
    absolute_error = determinant.error + generic_coefficient * permanent.error
    # Lower bound on the rounded coefficient*actual_permanent+floor, allowing
    # two rounded operations plus conservative operand/output flush allowances.
    coefficient_lower = coefficient * (1 - unit) ** 2
    floor_lower = floor * (1 - unit) - 8 * tiny
    assert coefficient_lower > generic_coefficient
    assert floor_lower > absolute_error
    peak = max(x.magnitude + x.error for x in nodes)
    assert peak < Q(2) ** 40  # All intermediates are very far from binary32 overflow.
    expected = (8, 8) if n == 4 else (16, 17)
    assert (N, M) == expected, (name, N, M)
    vectors = [[Polynomial({(f"{i}:{j}",): 1}) for j in range(3)] for i in range(n)]
    # Work in exact difference coordinates: orientation's origin is a, and
    # sphere's origin is q. This is an algebraic variable choice, not a runtime
    # translation of the floating-point input.
    vectors[0 if n == 4 else 4] = [Polynomial() for _ in range(3)]
    expanded = namespace[name](*vectors)
    absolute_monomials = {m: abs(c) for m, c in expanded.value.terms.items()}
    assert expanded.error.terms == absolute_monomials
    assert len(absolute_monomials) == (6 if n == 4 else 72)
    result["filters"][name] = dict(
        determinant_round_factors=N,
        permanent_round_factors=M,
        generic_relative_coefficient=str(generic_coefficient),
        threshold_coefficient=str(coefficient),
        coefficient_inequality=True,
        absolute_error_bound=str(absolute_error),
        absolute_error_pow2_upper=pow2_upper(absolute_error),
        floor_inequality=True,
        intermediate_magnitude_pow2_upper=pow2_upper(peak),
        operations_or_leaves=len(nodes),
        permanent_matches_absolute_monomials=True,
        monomials=len(absolute_monomials),
    )
parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path)
args = parser.parse_args()
if args.output:
    args.output.write_text(json.dumps(result, indent=2) + "\n")
for name, record in result["filters"].items():
    print(
        name,
        {
            k: record[k]
            for k in [
                "determinant_round_factors",
                "permanent_round_factors",
                "absolute_error_pow2_upper",
                "intermediate_magnitude_pow2_upper",
                "coefficient_inequality",
                "floor_inequality",
            ]
        },
    )
