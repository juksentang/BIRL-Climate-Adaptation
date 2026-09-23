"""NUTS fit by country x asset tercile (python -m cropchoice fit-semipar-assets).

Thin wrapper kept at its historical location; all logic lives in cropchoice.fit_semipar_assets.
"""
import sys
from pathlib import Path

try:
    import cropchoice  # noqa: F401  (pip install -e <repo root>)
except ImportError:                      # fall back to the repository layout
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cropchoice.fit_semipar_assets import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]) and 0)
