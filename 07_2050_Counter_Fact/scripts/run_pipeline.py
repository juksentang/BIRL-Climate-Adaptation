"""
End-to-end pipeline orchestrator for 2050 Counterfactual Analysis.

Usage:
  python scripts/run_pipeline.py              # Run stages 1-3 sequentially
  python scripts/run_pipeline.py --from 3     # Start from stage 3
  python scripts/run_pipeline.py --only 2     # Run only stage 2

Stage 0: Export GPS points (prerequisite for GEE)
Stage 1: Process CMIP6 climate data
Stage 2: Generate counterfactual matrices
Stage 3: Compute welfare (CE) for all scenarios
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent

STAGES = {
    0: ("00_export_gps_points.py", "Export GPS points for GEE"),
    1: ("01_process_climate.py", "Process CMIP6 climate data"),
    2: ("02_generate_cf_matrices.py", "Generate counterfactual matrices"),
    3: ("03_compute_welfare.py", "Compute CE for all scenarios"),
    # Stage 4 (04_make_figures.py) archived to _archived_plots/
}


def run_stage(stage_num: int):
    script, desc = STAGES[stage_num]
    script_path = SCRIPTS_DIR / script

    print(f"\n{'='*60}")
    print(f"Stage {stage_num}: {desc}")
    print(f"Script: {script}")
    print(f"{'='*60}\n")

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(SCRIPTS_DIR.parent),
    )

    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\nERROR: Stage {stage_num} failed (exit code {result.returncode})")
        sys.exit(result.returncode)

    print(f"\nStage {stage_num} completed in {elapsed:.0f}s")


def main():
    parser = argparse.ArgumentParser(description="2050 Counterfactual Pipeline")
    parser.add_argument("--from", type=int, dest="from_stage", default=1,
                       help="Start from this stage (default: 1)")
    parser.add_argument("--only", type=int, default=None,
                       help="Run only this stage")
    args = parser.parse_args()

    t_start = time.time()

    if args.only is not None:
        run_stage(args.only)
    else:
        for stage in sorted(STAGES.keys()):
            if stage >= args.from_stage:
                run_stage(stage)

    total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Pipeline complete. Total time: {total:.0f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
