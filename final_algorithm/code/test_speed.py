import sys,unittest,tempfile,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap
from speed_tasks import choose_budget,acceptable
from optimize import incumbents,sha
from types import SimpleNamespace

class BudgetTests(unittest.TestCase):
 def setUp(self):self.sc=SimpleNamespace(minimum_overall_iou=.9,minimum_overall_coverage=.8,maximum_overall_overfill_ratio=.3)
 def option(self,i,length):return dict(m=dict(target=100,intersection=i,union=100,overfill=0,iou=i/100,length=length))
 def test_budget_allows_one_sparse_layer_but_not_two(self):
  o=[[self.option(98,100),self.option(83,50)] for _ in range(2)]
  chosen,total=choose_budget(o,self.sc)
  self.assertEqual(sum(chosen),1);self.assertTrue(acceptable(total,self.sc));self.assertAlmostEqual(total['intersection']/total['union'],.905)
 def test_impossible_global_budget_is_rejected(self):
  with self.assertRaises(ValueError):choose_budget([[self.option(83,50)]],self.sc)
 def test_shorter_equal_shape_candidate_is_preferred(self):
  chosen,_=choose_budget([[self.option(98,100),self.option(98,80)]],self.sc);self.assertEqual(chosen,[1])
 def test_archive_hash_mismatch_is_rejected(self):
  with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
   d=Path(tmp);job=d/'job';job.mkdir();arch=d/'archive';arch.mkdir()
   for n in ('target.stl','config.yaml','trajectory.csv'):(job/n).write_text(n)
   (arch/'summary.json').write_text(json.dumps(dict(status='PASS',input=dict(directory=str(job)),schedule=dict(makespan_s=1))))
   data={n:dict(sha256=sha(job/n)) for n in ('target.stl','config.yaml','trajectory.csv')}
   (arch/'validation_inputs.json').write_text(json.dumps(dict(inputs=data)))
   self.assertEqual(len(incumbents(job,arch)),1)
   (job/'trajectory.csv').write_text('tampered');self.assertEqual(incumbents(job,arch),[])
if __name__=='__main__':unittest.main()
