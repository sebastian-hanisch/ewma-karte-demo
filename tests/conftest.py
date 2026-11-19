"""Gemeinsame Einstellungen: Wurzelordner im Importpfad, BLAS-Threads auf 1 (kleine dichte Löser sind sonst 100-mal langsamer)."""

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
