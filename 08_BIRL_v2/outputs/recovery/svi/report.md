# BIRL v2 simulation-recovery report (svi)

Date: 2026-09-20T02:30:18.324542  
Platform: gpu x 1  
Countries (order): ['Ethiopia', 'Malawi', 'Mali', 'Nigeria', 'Tanzania', 'Uganda']  
m_c (USD): Ethiopia=19.86, Malawi=19.12, Mali=129.96, Nigeria=102.74, Tanzania=26.03, Uganda=25.46  
Priors: flat (independent wide per-country), s_max=0.6  
SVI: AutoMultivariateNormal, Adam lr=0.01, 3000 steps, point estimates (guide median)  
Tolerances: |d rho| <= 0.25, |d s| <= 0.05, |d log beta| <= 0.2; PASS (NUTS only) additionally needs >= 80% of the 36 truths inside their 89% HPDI  

**Verdict: SMOKE** - SVI point estimates: 31/36 tolerances met (smoke check only; the PASS verdict is judged on --mode nuts)

## Truth set A

rho=[1.5, 3.0, 2.0, 1.0, 2.5, 3.5], s=[0.2, 0.4, 0.3, 0.15, 0.5, 0.35], beta=[3, 8, 5, 2, 6, 4]; simulated actions agree with observed on 4.9% of obs; log-lik(truth) = -502633.1

### SVI point estimate (17/18 tolerances met; 43s, ELBO loss 568062 -> 502733; log-lik truth -502633.1 vs estimate -502650.2)

| param | country | truth | estimate | abs err | tol | ok |
|---|---|---|---|---|---|---|
| rho_c | Ethiopia | 1.5 | 1.49 | 0.0099 | 0.25 | ok |
| rho_c | Malawi | 3 | 3.03 | 0.0302 | 0.25 | ok |
| rho_c | Mali | 2 | 2.025 | 0.0251 | 0.25 | ok |
| rho_c | Nigeria | 1 | 1.114 | 0.114 | 0.25 | ok |
| rho_c | Tanzania | 2.5 | 2.495 | 0.00499 | 0.25 | ok |
| rho_c | Uganda | 3.5 | 3.457 | 0.0429 | 0.25 | ok |
| s_c | Ethiopia | 0.2 | 0.2032 | 0.00325 | 0.05 | ok |
| s_c | Malawi | 0.4 | 0.3974 | 0.00256 | 0.05 | ok |
| s_c | Mali | 0.3 | 0.3016 | 0.00161 | 0.05 | ok |
| s_c | Nigeria | 0.15 | 0.03609 | 0.114 | 0.05 | FAIL |
| s_c | Tanzania | 0.5 | 0.4976 | 0.00235 | 0.05 | ok |
| s_c | Uganda | 0.35 | 0.3462 | 0.00375 | 0.05 | ok |
| beta_c | Ethiopia | 3 | 2.997 | 0.000974 | 0.20 | ok |
| beta_c | Malawi | 8 | 7.921 | 0.00992 | 0.20 | ok |
| beta_c | Mali | 5 | 5.059 | 0.0117 | 0.20 | ok |
| beta_c | Nigeria | 2 | 2.018 | 0.00877 | 0.20 | ok |
| beta_c | Tanzania | 6 | 6.058 | 0.00959 | 0.20 | ok |
| beta_c | Uganda | 4 | 3.928 | 0.0181 | 0.20 | ok |
| gamma_c | Ethiopia | 3.971 | 4.036 | 0.0644 | - | - |
| gamma_c | Malawi | 7.649 | 7.6 | 0.049 | - | - |
| gamma_c | Mali | 38.99 | 39.2 | 0.209 | - | - |
| gamma_c | Nigeria | 15.41 | 3.707 | 11.7 | - | - |
| gamma_c | Tanzania | 13.01 | 12.95 | 0.0612 | - | - |
| gamma_c | Uganda | 8.911 | 8.815 | 0.0955 | - | - |

## Truth set B

rho=[0.5, 4.5, 1.0, 3.0, 2.0, 1.5], s=[0.55, 0.1, 0.58, 0.05, 0.3, 0.45], beta=[10, 1.5, 4, 6, 2, 8]; simulated actions agree with observed on 5.2% of obs; log-lik(truth) = -396961.6

### SVI point estimate (14/18 tolerances met; 30s, ELBO loss 528536 -> 402989; log-lik truth -396961.6 vs estimate -402897.6)

| param | country | truth | estimate | abs err | tol | ok |
|---|---|---|---|---|---|---|
| rho_c | Ethiopia | 0.5 | 0.5011 | 0.00108 | 0.25 | ok |
| rho_c | Malawi | 4.5 | 4.229 | 0.271 | 0.25 | FAIL |
| rho_c | Mali | 1 | 0.9731 | 0.0269 | 0.25 | ok |
| rho_c | Nigeria | 3 | 2.986 | 0.0136 | 0.25 | ok |
| rho_c | Tanzania | 2 | 1.997 | 0.00254 | 0.25 | ok |
| rho_c | Uganda | 1.5 | 2.231 | 0.731 | 0.25 | FAIL |
| s_c | Ethiopia | 0.55 | 0.548 | 0.00202 | 0.05 | ok |
| s_c | Malawi | 0.1 | 0.0883 | 0.0117 | 0.05 | ok |
| s_c | Mali | 0.58 | 0.5837 | 0.00368 | 0.05 | ok |
| s_c | Nigeria | 0.05 | 0.05345 | 0.00345 | 0.05 | ok |
| s_c | Tanzania | 0.3 | 0.3083 | 0.00829 | 0.05 | ok |
| s_c | Uganda | 0.45 | 0.001178 | 0.449 | 0.05 | FAIL |
| beta_c | Ethiopia | 10 | 9.995 | 0.000468 | 0.20 | ok |
| beta_c | Malawi | 1.5 | 1.443 | 0.0389 | 0.20 | ok |
| beta_c | Mali | 4 | 3.936 | 0.0162 | 0.20 | ok |
| beta_c | Nigeria | 6 | 5.988 | 0.00197 | 0.20 | ok |
| beta_c | Tanzania | 2 | 2.002 | 0.0012 | 0.20 | ok |
| beta_c | Uganda | 8 | 6.304 | 0.238 | 0.20 | FAIL |
| gamma_c | Ethiopia | 10.92 | 10.88 | 0.0401 | - | - |
| gamma_c | Malawi | 1.912 | 1.689 | 0.224 | - | - |
| gamma_c | Mali | 75.37 | 75.85 | 0.478 | - | - |
| gamma_c | Nigeria | 5.137 | 5.492 | 0.355 | - | - |
| gamma_c | Tanzania | 7.808 | 8.024 | 0.216 | - | - |
| gamma_c | Uganda | 11.46 | 0.02998 | 11.4 | - | - |
