from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap
from timed_wait_parallel import wait_conflict_end,leg_start,schedule_robot
import numpy as np
class Tests(unittest.TestCase):
 def model(self,unsafe):
  rows=[[0.,0.,0.,0.,'T'],[1.,1.,0.,0.,'T'],[2.,0.,0.,0.,'W']]
  empty=np.empty((0,2),int)
  return dict(routes={0:(rows,[])},geometry={0:([],[],np.array([[0.,1.],[1.,2.]]))},leg_parts={0:{0:np.array([0]),1:np.array([1])}},leg_mm={(0,1,0):empty,(0,1,1):np.array([[1,0]])},leg_mv={(0,1,0):empty,(0,1,1):empty},vertex_vm={(0,1,0):np.array([],int),(0,1,1):np.array([0],int) if unsafe else np.array([],int)},vertex_vv={(0,1,0):np.array([],int),(0,1,1):np.array([],int)})
 def reserved(self,start=1.5):return {1:dict(times=np.array([[start,4.]]),waits=[])}
 def test_wait_is_checked_only_during_its_actual_window(self):
  m=self.model(True);self.assertIsNone(wait_conflict_end(m,0,1,1.,2.,self.reserved(3.)));self.assertGreater(wait_conflict_end(m,0,1,2.,3.,self.reserved(3.)),4.)
 def test_safe_wait_is_inserted_at_work_vertex(self):
  r=schedule_robot(self.model(False),0,False,self.reserved())
  self.assertEqual(r['restarts'],0);self.assertEqual(len(r['waits']),1);self.assertEqual(r['waits'][0][0],1)
  self.assertTrue(any(a[4]=='W' and a[1:4]==b[1:4] for a,b in zip(r['rows'],r['rows'][1:])))
 def test_unsafe_wait_delays_entry_from_home(self):
  r=schedule_robot(self.model(True),0,False,self.reserved());self.assertGreater(r['restarts'],0);self.assertEqual(r['waits'],[])
 def test_other_robots_reserved_wait_blocks_a_move(self):
  m=self.model(False);m['leg_mv'][0,1,0]=np.array([[0,2]])
  old={1:dict(times=np.array([[0.,1.]]),waits=[(2,2.,4.)])}
  self.assertGreater(leg_start(m,0,0,np.array([0]),np.array([[0.,1.],[1.,2.]]),0.,old,2.),4.)
if __name__=='__main__':unittest.main()
