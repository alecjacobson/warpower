"""Summarize retained strict failures and verify actual memory/lifetime coverage."""
import hashlib,json,re
from pathlib import Path
w=Path('.work/vtm')
first=json.loads((w/'large-reuse-isolated-audit-commands.json').read_text())
later=json.loads((w/'large-reuse-lifetime-controls.json').read_text())
coverage=json.loads((w/'large-reuse-lifetime-coverage.json').read_text())
r=dict(tests={},controls={},strict_failures={},scope='No error suppression. Full allocation-order failures remain failed. Zero-error use-after-free/API coverage does not establish full allocation-order validity.')
for i in range(7):
 normal=w/f'large-reuse-memcheck-isolated-{i}.log';strict=w/f'large-reuse-lifetime-isolated-{i}.log'
 row=next(x for x in first+later if x['name']==f'lifetime-isolated-{i}')
 checked=w/f'large-reuse-afterfree-isolated-{i}.log' if row['exit_code'] else strict
 for p in [normal,checked]:assert '1 passed' in p.read_text() and 'ERROR SUMMARY: 0 errors' in p.read_text(),p
 r['tests'][str(i)]=dict(padded_memory_pass=True,api_memory_and_afterfree_pass=True,full_tracking_pass=row['exit_code']==0,checked_log=checked.name,checked_sha256=hashlib.sha256(checked.read_bytes()).hexdigest())
def failure(name):
 p=w/f'large-reuse-{name}.log';s=p.read_text()
 kernels=sorted(set(re.findall(r'^=========     at (.*)',s,re.M)))
 kinds=sorted(set(re.findall(r'^========= (Invalid .*|Uninitialized .*|Race .*|Invalid.*)',s,re.M)))
 storage=sorted(set(map(int,re.findall(r'nearest allocation at .* of size (\d+) bytes',s))))
 summaries=re.findall(r'^========= ERROR SUMMARY: (\d+) errors$',s,re.M)
 assert len(kernels)==1 and 'DeviceRadixSortHistogramKernel' in kernels[0] and kernels[0].endswith('+0x1a40'),(name,kernels)
 assert kinds==['Invalid __global__ atomic of size 4 bytes'],(name,kinds)
 assert storage==[10751],(name,storage)
 assert 'potentially accessed before it is allocated' in s
 return dict(log=p.name,log_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),total_reported_errors=int(summaries[-1]),printed_kernel=kernels[0],printed_kinds=kinds,printed_native_scratch_sizes=storage,printed_reports=s.count('Invalid __global__ atomic'),application_fatal_python_error='Fatal Python error' in s,scope='Only printed diagnostics are classified; the raw log records any print-limit truncation.')
for i in coverage['full_tracking_failed_indices']:r['strict_failures'][f'lifetime-isolated-{i}']=failure(f'lifetime-isolated-{i}')
for name in ['radix4096-conditional-all','route-before-all','route-after-all']:
 row=next(x for x in later if x['name']==name);assert row['exit_code']==99,row
 r['strict_failures'][name]=failure(name)
for name in ['radix4096-top-all','radix4096-conditional-normal','radix4096-conditional-afterfree']:
 row=next(x for x in later if x['name']==name);s=(w/f'large-reuse-{name}.log').read_text()
 assert row['exit_code']==0 and 'ERROR SUMMARY: 0 errors' in s and '"exact_keys_and_payloads_match": true' in s
 r['controls'][name]=dict(exit_code=0,exact_keys_and_payloads_match=True,zero_errors=True)
r['coverage']=coverage
(w/'large-reuse-lifetime-analysis.json').write_text(json.dumps(r,indent=2)+'\n')
print('verified coverage',len(r['tests']),'passing sort controls',len(r['controls']),'strict failures retained',len(r['strict_failures']),flush=True)
