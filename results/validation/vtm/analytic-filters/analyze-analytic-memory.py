"""Report actual instrumentation scope; never discard or relabel strict errors."""
import hashlib,json,re
from pathlib import Path
w=Path('.work/vtm');rows=json.loads((w/'analytic-audit-commands.json').read_text())
assert len(rows)==19
assert all(r['exit_code']==0 and r['zero_memcheck_errors'] for r in rows[:18])
strict=rows[-1];path=w/'analytic-route-5-lifetime-all.log';text=path.read_text()
counts=[int(n) for n in re.findall(r'ERROR SUMMARY: (\d+) errors',text)]
blocks=re.findall(r'========= Invalid .*?(?=\n========= Invalid |\n========= ERROR SUMMARY:|\Z)',text,re.S)
matching=[all(v in b for v in ['DeviceRadixSortHistogramKernel','+0x1a40','before it is allocated','size 10751 bytes']) for b in blocks]
record=dict(padded_memory_cases=51,new_predicate_and_query_full_api_lifetime_cases=44,routed_api_after_free_cases=7,all_those_scopes_zero_error=True,strict_route_control=dict(command=strict['command'],exit_code=strict['exit_code'],zero_errors=strict['zero_memcheck_errors'],error_summaries=counts,printed_invalid_diagnostics=len(blocks),all_printed_invalid_diagnostics_match_prior_cub_histogram_allocation_order=bool(blocks) and all(matching),source_log=str(path),source_log_sha256=hashlib.sha256(path.read_bytes()).hexdigest()),prior_control='results/validation/vtm/large-fragment-reuse/large-reuse-lifetime-analysis.json',scope='Full allocation-order tracking of every routed case is not claimed. The one strict routed control remains a failure if it reports errors. Pattern matching classifies only printed diagnostics, without suppressions or assertions about unprinted errors.')
(w/'analytic-memory-coverage.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
if strict['exit_code']!=0:
    assert strict['exit_code']!=124 and bool(blocks) and all(matching) and counts and counts[-1]>0,record
else:
    assert strict['zero_memcheck_errors'] and '1 passed' in text,record
