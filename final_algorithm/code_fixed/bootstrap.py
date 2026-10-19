from pathlib import Path
import sys,os
ROOT=Path(__file__).resolve().parent
deps=Path(os.environ.get('WAAM_DEPS',str(ROOT.parent/'waam_generalization/integration/.deps')))
sys.path.insert(0,str(deps));sys.path.insert(0,str(ROOT/'backend'))
# Import the bundled version gate before any validator imports.
from environment import env as checks
