"""Stone-Geary/CRRA variants of the redesign (`v2_country`, `v2_country_gfix`), kept
for the record: they showed that rho and the subsistence share are not identified
from crop choice.  The driver lives in 08_BIRL_v2/run_v2.py; this module runs it
in-process so `python -m cropchoice fit-v2 ...` works from anywhere.
"""
import runpy, sys

from cropchoice.config import BASE_DIR


def main(argv=None):
    script = BASE_DIR / "run_v2.py"
    sys.argv = [str(script)] + list(sys.argv[2:] if argv is None else argv)
    runpy.run_path(str(script), run_name="__main__")
