"""NUTS fit of the semi-parametric choice model (python -m cropchoice fit-semipar).

Thin wrapper kept at its historical location; all logic lives in cropchoice.fit_semipar.
"""
import sys
from pathlib import Path

try:
    import cropchoice  # noqa: F401  (pip install -e <repo root>)
except ImportError:                      # fall back to the repository layout
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cropchoice.fit_semipar import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]) and 0)
