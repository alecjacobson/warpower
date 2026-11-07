"""Rational roundoff budget for bounded box-distance predicate filters.

For finite float64 p and float32 lo/hi, all abs <=16. Let B be the
max(abs(float32(p)), abs(lo), abs(hi), 2^-30). Each float32 operation obeys
abs(fl(x)-x)<=u*abs(x)+eta, using eta=2^-126 even if subnormals flush.
Double uses v=2^-53, eta64=2^-1022. Three products/two sums, including any
FMA contraction, are bounded by gamma5 and a conservative five-op eta term.
All quantities below are normalized by B or B^2; maxima use B_min.
"""
from fractions import Fraction as F
import json
from pathlib import Path

u=F(1,2**24);v=F(1,2**53);bmin=F(1,2**30)
eta=F(1,2**126);eta64=F(1,2**1022)
g=lambda e:5*e/(1-5*e)
q_conversion=(u+eta/bmin)/(1-u)
subtract32=2*u+eta/bmin
subtract64=v*(2+q_conversion)+eta64/bmin
delta=q_conversion+subtract32+subtract64
d32=2+subtract32
d64=2+q_conversion+subtract64
coordinate_square_difference=3*delta*(d32+d64)
dot32=g(u)*3*d32*d32+5*eta/bmin**2*(1+u)**5
dot64=g(v)*3*d64*d64+5*eta64/bmin**2*(1+v)**5
distance_difference=coordinate_square_difference+dot32+dot64
# B^2 is normal. Multiplication by2^-16 is exact normal scaling. The final
# float64 interval endpoints round by <=v*(s32+error)+eta64.
stored_error_min=F(1,2**16)*(1-u)
stored_error_max=F(1,2**16)*(1+u)
interval_roundoff=v*(3*d32*d32+dot32+stored_error_max)+eta64/bmin**2
assert distance_difference+interval_roundoff < F(1,2**17) < stored_error_min
values=dict(q_conversion=q_conversion,subtract32=subtract32,subtract64=subtract64,
            delta=delta,d32=d32,d64=d64,coordinate_square_difference=coordinate_square_difference,
            dot32=dot32,dot64=dot64,distance_difference=distance_difference,
            interval_roundoff=interval_roundoff,stored_error_min=stored_error_min,
            stored_error_max=stored_error_max,margin=stored_error_min-distance_difference-interval_roundoff)
result=dict(scope='Arithmetic bound under the documented IEEE rounding/FMA model; not a compiler/runtime verification or a whole-algorithm proof.',
            proved=True,normalized_values={k:dict(decimal=float(x),exact=str(x)) for k,x in values.items()},
            normalized_difference_upper=2.0**-17,stored_error_coefficient=2.0**-16,
            coordinate_limit=16,b_min=2.0**-30)
path=Path('.work/vtm/surface-box-filter-proof-v1.json');assert not path.exists()
path.write_text(json.dumps(result,indent=2)+'\n')
print('Bounded difference/B^2:',float(distance_difference),'stored error/B^2 >=',float(stored_error_min))
