"""Reconcile current architecture while keeping historical measurements explicit."""
from pathlib import Path
assert Path('docs/larger-active-prefixes.md').exists()
def replace(name,old,new):
 p=Path('docs')/name;s=p.read_text();assert s.count(old)==1,(name,old);p.write_text(s.replace(old,new))
replace('architecture.md','''Preallocated 1,024- and 2,048-site constructors handle the smallest fitting prefix;
the full constructor handles larger inputs and prefix failures.''','''Preallocated 1,024/2,048/4,096/8,192-site constructors below the reservation
handle the smallest fitting prefix; the full constructor handles larger inputs
and prefix failures. The [larger-prefix report](larger-active-prefixes.md) measures
the speed gains and additional persistent storage/setup cost.''')
replace('vtm.md','''[route reconstruction through smaller preallocated prefixes](active-reconstruction.md)
until the sites outgrow them.''','''[route reconstruction through smaller preallocated prefixes](active-reconstruction.md)
until the sites outgrow them. [Additional 4,096/8,192-site branches](larger-active-prefixes.md)
reduce large-reservation costs while adding persistent storage and setup work.''')
replace('active-reconstruction.md','''The internal VTM core now preallocates constructors for 1,024 and 2,048 sites
alongside the full constructor when a device active count is present
and reserved capacity exceeds 2,048. A device kernel selects the smallest fitting
prefix. Each small constructor can recertify and complete previous fragments;
the full route now also [uses fragment reuse at reserved capacities of
2,049–4,096 sites](large-fragment-reuse.md), with current-input proofs retained
through completion. Larger reserved capacities use ordinary reconstruction.
A failed small attempt also falls
back to the full constructor, preserving its larger output and strategy budgets.''','''The internal VTM core preallocates constructors at fitting capacities from
1,024/2,048/4,096/8,192, alongside the full constructor, when a device active
count is present and reserved capacity exceeds 2,048. The later
[larger-prefix extension](larger-active-prefixes.md) adds the 4,096/8,192 branches
and reports their memory/setup cost. A device kernel selects the smallest
fitting prefix. Each child can recertify and complete previous fragments;
the full route also [uses fragment reuse at reserved capacities of
2,049–16,384 sites](extended-fragment-reuse.md), with current-input proofs
retained through completion. The full route above 16,384 uses ordinary
reconstruction. A failed child also falls back to the full constructor,
preserving its larger output and strategy budgets.''')
replace('active-reconstruction.md','''[later full-route extension](extended-fragment-reuse.md) provides current larger-case
timings, its additional scratch cost and new validation; it preserves the routing
capacities and the ordinary standalone APIs.''','''[later full-route extension](extended-fragment-reuse.md) measures larger-case
reuse and its scratch cost. The subsequent [larger-prefix extension](larger-active-prefixes.md)
adds two routing capacities with separate timing, storage and validation evidence.
The ordinary standalone APIs retain their paths.''')
replace('extended-fragment-reuse.md','''The timings in this report predate that follow-up. Larger active prefixes still pay for the full
reservation after 2,048 sites; [#230](https://github.com/alecjacobson/warpower/issues/230)
tracks additional routes, with scratch sharing in
[#122](https://github.com/alecjacobson/warpower/issues/122). These are separate
follow-ups so their timing and storage costs can be measured independently.''','''The timings in this report predate that follow-up. A later
[larger-prefix extension](larger-active-prefixes.md) adds device-selected 4,096/8,192
routes and measures their additional memory/setup cost separately. Scratch sharing
remains [#122](https://github.com/alecjacobson/warpower/issues/122).''')
replace('prepared-new-cell-queries.md','''Tracked in [#233](https://github.com/alecjacobson/warpower/issues/233). Larger
active-prefix routing [#230](https://github.com/alecjacobson/warpower/issues/230),
scratch sharing [#122](https://github.com/alecjacobson/warpower/issues/122), and
Armadillo extraction failures [#228](https://github.com/alecjacobson/warpower/issues/228)
remain open.''','''Tracked in [#233](https://github.com/alecjacobson/warpower/issues/233). A later
[larger-prefix extension](larger-active-prefixes.md) implements #230 with separately
measured storage and setup costs. Scratch sharing
[#122](https://github.com/alecjacobson/warpower/issues/122) and Armadillo extraction
failures [#228](https://github.com/alecjacobson/warpower/issues/228) remain open.''')
print('Updated five current architecture/follow-up documents; historical timings retained.')
