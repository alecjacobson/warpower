"""Point unexecuted qualification/report drivers at the resumed primary cohort."""
from pathlib import Path
w=Path('.work/vtm')
p=w/'check-prepared-new-cell-final-v2.py';assert not p.exists();p.write_text((w/'check-prepared-new-cell-final-v1.py').read_text().replace('prepared-new-cell-final-v1','prepared-new-cell-final-v2'))
p=w/'run-prepared-new-cell-final-independent-v2.py';assert not p.exists();s=(w/'run-prepared-new-cell-final-independent-v1.py').read_text().replace('prepared-new-cell-final-v1','prepared-new-cell-final-v2').replace('prepared-new-cell-final-independent-v1','prepared-new-cell-final-independent-v2');p.write_text(s)
for name in ['check-prepared-new-cell-grid-v1.py','run-prepared-new-cell-qualification-v1.py','profile-prepared-new-cell-production-v1.py','write-prepared-new-cell-doc-v1.py']:
 p=w/name;s=p.read_text().replace('prepared-new-cell-final-v1','prepared-new-cell-final-v2')
 if name=='write-prepared-new-cell-doc-v1.py':
  old='''are retained in the [comparison](../results/validation/vtm/prepared-new-cell-queries/prepared-new-cell-final-v2-independent.json).
'''
  new=old+'''
The first batch stopped when the graded candidate recorded one CPU-quota
throttling period (about54ms). Its valid mesh and raw timings are retained as
diagnostic evidence. A fresh graded control/candidate pair passed the same
zero-throttling gate; three earlier clean pairs were retained without repeating
them. The primary cohort therefore contains seven complete clean pairs. The
[affected pair](../results/validation/vtm/prepared-new-cell-queries/prepared-new-cell-throttled-pair-independent-v1.json)
is independently validated as well and is excluded from the primary timing table.
'''
  assert old in s;s=s.replace(old,new.replace('about54ms','about 54 ms'))
 p.write_text(s)
p=w/'publish-prepared-new-cell-v1.py';s=p.read_text().replace('prepared-new-cell-final-v1-independent.json','prepared-new-cell-final-v2-independent.json')
s=s.replace("'prepared-new-cell-simple-guards-v1','prepared-new-cell-final-v1']","'prepared-new-cell-simple-guards-v1','prepared-new-cell-final-v1','prepared-new-cell-final-v2']")
s=s.replace("'audit-prepared-new-cell','new-cell-query'","'audit-prepared-new-cell','prepare-prepared-new-cell','update-prepared-new-cell','new-cell-query'")
s=s.replace("final='prepared-new-cell-final-v1'","final='prepared-new-cell-final-v2',prior_stopped_batch='prepared-new-cell-final-v1',retained_clean_jobs=6")
s=s.replace("checks=read('prepared-new-cell-final-v2-independent.json');", "assert read('prepared-new-cell-throttled-pair-independent-v1.json')['complete']\nchecks=read('prepared-new-cell-final-v2-independent.json');")
s=s.replace("'The private strict/exact v2 experiment", "'The initial production graded candidate had one CPU quota-throttled period; its mesh/timings are retained separately and a fresh complete pair was required. Three preceding clean pairs are retained in the primary cohort.', 'The private strict/exact v2 experiment")
s=s.replace("for label in ['prepared-new-cell-profile-v1'", "for row in read('prepared-new-cell-final-v1-throttled-archives.json'):\n packed=Path(row['path']);assert sha(packed)==row['sha256'];assert hashlib.sha256(gzip.decompress(packed.read_bytes())).hexdigest()==row['raw_sha256']\nfor label in ['prepared-new-cell-profile-v1'")
p.write_text(s)
