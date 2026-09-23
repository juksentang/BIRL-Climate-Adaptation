"""Simulation-recovery test of the Stone-Geary variants (08_BIRL_v2/simulate_recover.py),
run in-process so `python -m cropchoice recover --mode svi|nuts ...` works from anywhere."""
import runpy, sys

from cropchoice.config import BASE_DIR


def main(argv=None):
    script = BASE_DIR / "simulate_recover.py"
    sys.argv = [str(script)] + list(sys.argv[2:] if argv is None else argv)
    runpy.run_path(str(script), run_name="__main__")
