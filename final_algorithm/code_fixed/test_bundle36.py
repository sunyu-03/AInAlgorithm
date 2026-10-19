import sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent
if not (ROOT/'bootstrap.py').exists():ROOT=Path(r'C:\Users\User\Desktop\AInAlgorithm\1014')
sys.path.insert(0,str(ROOT));import bootstrap
from test_bundle_finite_stay import Tests as Fixtures
from bundle_reassign36 import bundle_candidates
class Tests(unittest.TestCase):
 def test_sizes_3_6_preserve_tasks_order_direction(self):
  e=Fixtures().env();seq={0:[(i,i%2==1) for i in range(6)]};got=bundle_candidates(e,seq,{}, {(1,0):100},1)
  self.assertEqual([c['bundle_size'] for g,c in got],[3,6])
  for groups,c in got:
   self.assertEqual(sorted(v for s in groups.values() for v in s),seq[0]);target=groups[c['to_robot']-1];bundle=[v for v in seq[0] if v[0] in c['tasks']];k=target.index(bundle[0]);self.assertEqual(target[k:k+len(bundle)],bundle)
 def test_short_route_has_no_candidate(self):
  e=Fixtures().env();self.assertEqual(bundle_candidates(e,{0:[(0,False),(1,False)]},{},{(1,0):100},1),[])
 def test_unreachable_candidates_excluded(self):
  e=Fixtures().env();e.robots[1].xy_reach_radius_mm=1.;e.robots[2].xy_reach_radius_mm=1.;self.assertEqual(bundle_candidates(e,{0:[(i,False) for i in range(6)]},{},{(1,0):100},1),[])
 def test_matched_control_changes_only_bundle_sizes(self):
  old=(ROOT/'bundle_reassign.py').read_text(encoding='utf-8');new=(ROOT/'bundle_reassign36.py').read_text(encoding='utf-8');self.assertEqual(new,old.replace('sizes=(2,4)','sizes=(3,6)').replace('bundle_sizes=[2,4]','bundle_sizes=[3,6]'))
if __name__=='__main__':unittest.main()
