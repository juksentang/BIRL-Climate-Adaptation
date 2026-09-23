"""Real-data SVI preview: hierarchical priors, AutoMultivariateNormal, point
estimates of (rho_c, s_c, gamma_c, beta_c) per country.  A quick look while
the NUTS jobs run; NOT a substitute for 02_main.  Also refits with s fixed at
0.3 (the gfix variant) so the s-rho trade-off is visible immediately.
Run from 08_BIRL_v2/:  python3 slurm/svi_preview.py [steps]
"""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("BIRL_HOST_DEVICES", "1")

from src.config import log, DATA_DIR, OUT_DIR  # noqa: E402
import jax, jax.numpy as jnp, numpy as np  # noqa: E402
import numpyro  # noqa: E402
from numpyro.infer import SVI, Trace_ELBO  # noqa: E402
from numpyro.infer.autoguide import AutoMultivariateNormal  # noqa: E402
from src.data_loader import load_data, model_kwargs  # noqa: E402
from src.models import v2_country, v2_country_gfix, derive_country_params, log_likelihood  # noqa: E402

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 6000


def say(msg):
    print(f"[svi {time.strftime('%H:%M:%S')}] {msg}", flush=True)


data = load_data(DATA_DIR)
countries = data["countries"] if "countries" in data else ["Ethiopia", "Malawi", "Mali", "Nigeria", "Tanzania", "Uganda"]
m_c = np.asarray(data["m_c"])
mk = {"data": model_kwargs(data), "s_max": 0.6, "flat_priors": False, "noncentered": False}
say(f"devices {jax.devices()}  N_obs {int(mk['data']['obs_action'].shape[0])}  steps {STEPS}")

results = {}
for name, model, extra in (("v2_country", v2_country, {}),
                           ("v2_country_gfix_s030", v2_country_gfix, {"s_fixed": 0.3})):
    guide = AutoMultivariateNormal(model, init_loc_fn=numpyro.infer.init_to_median())
    svi = SVI(model, guide, numpyro.optim.Adam(1e-2), Trace_ELBO())
    t0 = time.time()
    res = svi.run(jax.random.PRNGKey(0), STEPS, progress_bar=False, **mk, **extra)
    losses = np.asarray(res.losses)
    med = {k: np.asarray(v) for k, v in guide.median(res.params).items()}
    est = derive_country_params(med, m_c, s_max=0.6, s_fixed=extra.get("s_fixed"))
    if "s_fixed" in extra:
        est["s_c"] = np.full(len(m_c), extra["s_fixed"]); est["gamma_c"] = est["s_c"] * m_c
    ll = float(log_likelihood(est["rho_c"], est["s_c"], est["beta_c"], mk["data"]))
    say(f"{name}: {STEPS} steps in {time.time()-t0:.0f}s, loss {losses[0]:.0f} -> {losses[-1]:.0f} "
        f"(last-500 mean {losses[-500:].mean():.0f}), log-lik at median {ll:.1f}")
    say(f"{'country':10s} {'rho':>6s} {'s':>6s} {'gamma$':>7s} {'beta':>6s}   m_c")
    for i, c in enumerate(countries):
        say(f"{c:10s} {est['rho_c'][i]:6.2f} {est['s_c'][i]:6.3f} {est['gamma_c'][i]:7.2f} {est['beta_c'][i]:6.2f}   {m_c[i]:.1f}")
    results[name] = {"loss_last500": float(losses[-500:].mean()), "loglik": ll,
                     **{k: np.asarray(v).tolist() for k, v in est.items()}}
    # Uniform-choice baseline for scale: log-lik of a model with beta=0
    if name == "v2_country":
        ll0 = float(log_likelihood(est["rho_c"], est["s_c"], np.zeros_like(est["beta_c"]), mk["data"]))
        say(f"  baseline log-lik with beta=0 (uniform over feasible): {ll0:.1f}; gain per obs = "
            f"{(ll-ll0)/mk['data']['obs_action'].shape[0]:.4f} nat")
        results["loglik_beta0"] = ll0

out = OUT_DIR / "svi_preview"; out.mkdir(parents=True, exist_ok=True)
json.dump(results, open(out / "svi_preview.json", "w"), indent=2)
say(f"written {out/'svi_preview.json'}")
