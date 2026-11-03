import os
import tempfile
from pathlib import Path

# Isolate every test run from the real data directory before dlo.config is imported.
_tmp = Path(tempfile.mkdtemp(prefix="dlo-test-"))
os.environ["DLO_DATA_DIR"] = str(_tmp)
