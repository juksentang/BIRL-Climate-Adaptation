"""Compatibility shim: the choice engine moved into the cropchoice package
(cropchoice.policies, cropchoice.counterfactual, cropchoice.quantiles)."""
from cropchoice.policies import POLICIES, DEFAULT_PARAMS, cost_context, apply_policy, inverse_threshold  # noqa: F401
from cropchoice.counterfactual import (load_posterior_thinned, scenario_metrics, observed_shares,  # noqa: F401
                                       _batch_metrics, _softmax_feasible)
from cropchoice.quantiles import Z, W5, mu_sigma, five_nodes, e5  # noqa: F401
from cropchoice.models_v2 import INFEASIBLE_LOGIT as INFEASIBLE  # noqa: F401
