"""Repo-root pytest bootstrap.

Ensures the repository root is on sys.path so ``import degradiq`` works regardless of
how pytest is invoked (the ``pytest`` console script does not add the cwd, unlike
``python -m pytest``). Without this, CI collection fails with
``ModuleNotFoundError: No module named 'degradiq'``.
"""
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
