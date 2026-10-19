import unittest,tempfile,json,hashlib
from pathlib import Path
from optimize_schedule import build_archive
class ArchiveTests(unittest.TestCase):
 def test_only_successful_spatial_rules_are_retained_without_changing_sources(self):
  root=Path(__file__).resolve().parent
  with tempfile.TemporaryDirectory(dir=str(root)) as tmp:
   tmp=Path(tmp).resolve();assert tmp.is_relative_to(root)
   p=tmp/'a.json';p.write_text(json.dumps(dict(layers=[dict(accepted=True,change=dict(kind='spatial_joint'),saved_s=3,z=1),dict(accepted=False,change=dict(kind='spatial_joint')),dict(accepted=True,change=dict(kind='routing_only'))])))
   before=hashlib.sha256(p.read_bytes()).hexdigest();dest=tmp/'memory.json';self.assertEqual(build_archive([p],dest),1);self.assertEqual(json.loads(dest.read_text())['layers'][0]['saved_s'],3);self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),before)
if __name__=='__main__':unittest.main()
