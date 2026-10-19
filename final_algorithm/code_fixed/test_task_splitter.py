from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap
from task_splitter import split_records
import numpy as np
class Tests(unittest.TestCase):
 def sample(self):return [dict(task_id=9,layer=2,start_xyz_mm=[-80.,0.,5.],end_xyz_mm=[80.,0.,5.])]
 def test_connected_collinear_pieces_and_parent_lineage(self):
  rows,lineage=split_records(self.sample(),80.);self.assertEqual(len(rows),2);self.assertEqual(rows[0]['end_xyz_mm'],rows[1]['start_xyz_mm']);self.assertEqual([r['task_id'] for r in rows],[0,1]);self.assertEqual([r['parent_task'] for r in lineage],[9,9]);self.assertEqual(sum(np.linalg.norm(np.asarray(r['end_xyz_mm'])-r['start_xyz_mm']) for r in rows),160.)
 def test_invalid_limit_is_rejected(self):
  with self.assertRaises(ValueError):split_records(self.sample(),0.)
 def test_short_task_is_retained(self):self.assertEqual(len(split_records(self.sample(),200.)[0]),1)
if __name__=='__main__':unittest.main()
