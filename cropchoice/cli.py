"""`python -m cropchoice <command> [args]`

  fit-semipar          NUTS fit of the country-level semi-parametric choice model
  fit-semipar-assets   same, by country x household-asset tercile
  counterfactual       Step 07 stage 3: climate x policy scenarios on the fitted model
  report               Step 07 stage 4: tables and figures from the stage-3 outputs
  sobol                Step 07 stage 5: Saltelli sweep over the policy parameters
  fit-v2               Stone-Geary/CRRA variants of the redesign (kept for the record)
  recover              simulation-recovery test of the Stone-Geary variants

Every command accepts --help; `--toy` (or EXP_TOY=1 for the fits) runs on
synthetic data without the restricted files.
"""
import sys

COMMANDS = {
    "fit-semipar": ("cropchoice.fit_semipar", "main"),
    "fit-semipar-assets": ("cropchoice.fit_semipar_assets", "main"),
    "counterfactual": ("cropchoice.counterfactual", "main"),
    "report": ("cropchoice.report", "main"),
    "sobol": ("cropchoice.sobol", "main"),
    "fit-v2": ("cropchoice.fit_v2", "main"),
    "recover": ("cropchoice.recover", "main"),
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__); return 0
    cmd, rest = argv[0], argv[1:]
    if cmd not in COMMANDS:
        print(f"unknown command {cmd!r}; one of {', '.join(COMMANDS)}", file=sys.stderr); return 2
    import importlib
    mod, fn = COMMANDS[cmd]
    return getattr(importlib.import_module(mod), fn)(rest)


if __name__ == "__main__":
    sys.exit(main() or 0)
