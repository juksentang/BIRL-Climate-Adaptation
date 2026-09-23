# Superseded (2026-09)

This step estimated the hierarchical CRRA / Stone-Geary model (`hier_noalpha`) used in paper v1.1. Its results were superseded by Step 08 (`08_BIRL_v2/`): the parameters ρ and γ it estimates are not identified from crop choice (ρ at its bounds, γ at its bound in three countries, 10.7% divergences when re-run at the country level). See `docs/08_birl_v2/STATUS_2026-09-20.md`.

The code and outputs are kept unchanged so that paper v1.1 can be reproduced (`run_birl.py`, `outputs/hier_noalpha/`). Do not build new work on it. The data files under `data/` are still the inputs of Steps 07 and 08.
