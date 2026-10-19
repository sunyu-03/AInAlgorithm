"""Read-only verification of a frozen local submission using the official validator."""
import bootstrap
from pathlib import Path
import argparse,json,hashlib
from waam_validator import run_validation
def main():
 p=argparse.ArgumentParser();p.add_argument('--release',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--out',type=Path,required=True);a=p.parse_args();release=a.release.resolve();out=a.out.resolve()
 if release==out or release in out.parents:raise ValueError('Verification output must stay outside frozen release')
 out.mkdir(parents=True,exist_ok=False);manifest=json.loads((release/'FROZEN_VERSION.json').read_text(encoding='utf-8'));benchmark=json.loads((release/'benchmark.json').read_text(encoding='utf-8'));checks={n:hashlib.sha256((release/n).read_bytes()).hexdigest()==v for n,v in manifest['files_sha256'].items()};assert all(checks.values()),'Frozen files changed'
 records=[]
 for m in benchmark['models']:
  result=run_validation(release/f'model{m["model"]}',out/f'model{m["model"]}');s=json.loads((out/f'model{m["model"]}'/'summary.json').read_text());ok=result.status=='PASS' and abs(result.schedule.makespan_s-m['selected_s'])<1e-6 and s['collision']['collision_event_count']==0 and s['reach']['passed'];records.append(dict(model=m['model'],status=result.status,verified=ok,makespan_s=result.schedule.makespan_s));print(records[-1],flush=True)
 report=dict(status='PASS' if all(v['verified'] for v in records) else 'FAIL',release=str(release),file_hashes_verified=len(checks),models=records);(out/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8');assert report['status']=='PASS'
if __name__=='__main__':main()
