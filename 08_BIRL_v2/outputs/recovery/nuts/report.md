# BIRL v2 simulation-recovery report (nuts)

Date: 2026-09-20T02:34:30.505058  
Platform: gpu x 1  
Countries (order): ['Ethiopia', 'Malawi', 'Mali', 'Nigeria', 'Tanzania', 'Uganda']  
m_c (USD): Ethiopia=19.86, Malawi=19.12, Mali=129.96, Nigeria=102.74, Tanzania=26.03, Uganda=25.46  
Priors: flat (independent wide per-country), s_max=0.6  
NUTS: 2 x (500+500), chain_method=vectorized  
Tolerances: |d rho| <= 0.25, |d s| <= 0.05, |d log beta| <= 0.2; PASS (NUTS only) additionally needs >= 80% of the 36 truths inside their 89% HPDI  

**Verdict: FAIL** - 33/36 tolerances met (need all), 33/36 = 92% truths inside 89% HPDI (need >= 80%)

## Truth set A

rho=[1.5, 3.0, 2.0, 1.0, 2.5, 3.5], s=[0.2, 0.4, 0.3, 0.15, 0.5, 0.35], beta=[3, 8, 5, 2, 6, 4]; simulated actions agree with observed on 4.9% of obs; log-lik(truth) = -502633.1

### NUTS posterior median (18/18 tolerances met; 5.6 min, divergences 0/1000, 17/18 truths in HPDI; log-lik truth -502633.1 vs estimate -502626.1)

| param | country | truth | median | sd | z | 89% HPDI | in HPDI | abs err | tol | ok |
|---|---|---|---|---|---|---|---|---|---|---|
| rho_c | Ethiopia | 1.5 | 1.489 | 0.0071 | -1.54 | [1.478, 1.501] | yes | 0.0109 | 0.25 | ok |
| rho_c | Malawi | 3 | 3.03 | 0.0426 | +0.70 | [2.959, 3.097] | yes | 0.03 | 0.25 | ok |
| rho_c | Mali | 2 | 2.021 | 0.0303 | +0.71 | [1.975, 2.07] | yes | 0.0215 | 0.25 | ok |
| rho_c | Nigeria | 1 | 1.029 | 0.0173 | +1.70 | [1, 1.055] | no | 0.0293 | 0.25 | ok |
| rho_c | Tanzania | 2.5 | 2.506 | 0.0464 | +0.13 | [2.432, 2.576] | yes | 0.00608 | 0.25 | ok |
| rho_c | Uganda | 3.5 | 3.443 | 0.0817 | -0.69 | [3.306, 3.562] | yes | 0.0565 | 0.25 | ok |
| s_c | Ethiopia | 0.2 | 0.2017 | 0.00231 | +0.74 | [0.1977, 0.2051] | yes | 0.00171 | 0.05 | ok |
| s_c | Malawi | 0.4 | 0.3992 | 0.00454 | -0.19 | [0.3911, 0.4057] | yes | 0.000849 | 0.05 | ok |
| s_c | Mali | 0.3 | 0.3014 | 0.00366 | +0.39 | [0.2956, 0.3071] | yes | 0.00142 | 0.05 | ok |
| s_c | Nigeria | 0.15 | 0.155 | 0.0117 | +0.43 | [0.1358, 0.1719] | yes | 0.005 | 0.05 | ok |
| s_c | Tanzania | 0.5 | 0.4986 | 0.00458 | -0.30 | [0.4907, 0.5053] | yes | 0.00138 | 0.05 | ok |
| s_c | Uganda | 0.35 | 0.3467 | 0.00277 | -1.21 | [0.3424, 0.3512] | yes | 0.00333 | 0.05 | ok |
| beta_c | Ethiopia | 3 | 3.009 | 0.0141 | +0.64 | [2.985, 3.03] | yes | 0.00303 | 0.20 | ok |
| beta_c | Malawi | 8 | 7.963 | 0.257 | -0.14 | [7.531, 8.354] | yes | 0.00464 | 0.20 | ok |
| beta_c | Mali | 5 | 5.034 | 0.0504 | +0.67 | [4.951, 5.11] | yes | 0.00677 | 0.20 | ok |
| beta_c | Nigeria | 2 | 1.999 | 0.0164 | -0.05 | [1.973, 2.024] | yes | 0.00039 | 0.20 | ok |
| beta_c | Tanzania | 6 | 6.039 | 0.181 | +0.22 | [5.743, 6.323] | yes | 0.00656 | 0.20 | ok |
| beta_c | Uganda | 4 | 3.922 | 0.0785 | -0.99 | [3.794, 4.038] | yes | 0.0196 | 0.20 | ok |
| gamma_c | Ethiopia | 3.971 | 4.005 | 0.0459 | +0.74 | [3.926, 4.072] | yes | 0.034 | - | - |
| gamma_c | Malawi | 7.649 | 7.633 | 0.0869 | -0.19 | [7.48, 7.758] | yes | 0.0162 | - | - |
| gamma_c | Mali | 38.99 | 39.17 | 0.476 | +0.39 | [38.42, 39.91] | yes | 0.185 | - | - |
| gamma_c | Nigeria | 15.41 | 15.92 | 1.2 | +0.43 | [13.95, 17.66] | yes | 0.514 | - | - |
| gamma_c | Tanzania | 13.01 | 12.98 | 0.119 | -0.30 | [12.77, 13.15] | yes | 0.036 | - | - |
| gamma_c | Uganda | 8.911 | 8.826 | 0.0704 | -1.21 | [8.716, 8.941] | yes | 0.0849 | - | - |

## Truth set B

rho=[0.5, 4.5, 1.0, 3.0, 2.0, 1.5], s=[0.55, 0.1, 0.58, 0.05, 0.3, 0.45], beta=[10, 1.5, 4, 6, 2, 8]; simulated actions agree with observed on 5.2% of obs; log-lik(truth) = -396961.6

### NUTS posterior median (15/18 tolerances met; 7.8 min, divergences 0/1000, 16/18 truths in HPDI; log-lik truth -396961.6 vs estimate -405938.7)

| param | country | truth | median | sd | z | 89% HPDI | in HPDI | abs err | tol | ok |
|---|---|---|---|---|---|---|---|---|---|---|
| rho_c | Ethiopia | 0.5 | 0.5015 | 0.00335 | +0.45 | [0.4962, 0.5068] | yes | 0.0015 | 0.25 | ok |
| rho_c | Malawi | 4.5 | 4.562 | 0.176 | +0.35 | [4.305, 4.864] | yes | 0.0616 | 0.25 | ok |
| rho_c | Mali | 1 | 0.9775 | 0.0182 | -1.23 | [0.9503, 1.006] | yes | 0.0225 | 0.25 | ok |
| rho_c | Nigeria | 3 | 2.985 | 0.028 | -0.53 | [2.945, 3.032] | yes | 0.0149 | 0.25 | ok |
| rho_c | Tanzania | 2 | 1.987 | 0.0388 | -0.34 | [1.924, 2.045] | yes | 0.0132 | 0.25 | ok |
| rho_c | Uganda | 1.5 | 1.848 | 0.359 | +0.97 | [1.498, 2.234] | yes | 0.348 | 0.25 | FAIL |
| s_c | Ethiopia | 0.55 | 0.5477 | 0.00385 | -0.59 | [0.5422, 0.5544] | yes | 0.00229 | 0.05 | ok |
| s_c | Malawi | 0.1 | 0.03798 | 0.0274 | -2.26 | [0.0006119, 0.07937] | no | 0.062 | 0.05 | FAIL |
| s_c | Mali | 0.58 | 0.5864 | 0.00674 | +0.95 | [0.5757, 0.5964] | yes | 0.00638 | 0.05 | ok |
| s_c | Nigeria | 0.05 | 0.05353 | 0.00297 | +1.19 | [0.04876, 0.0579] | yes | 0.00353 | 0.05 | ok |
| s_c | Tanzania | 0.3 | 0.3063 | 0.00588 | +1.07 | [0.2969, 0.316] | yes | 0.00628 | 0.05 | ok |
| s_c | Uganda | 0.45 | 0.2237 | 0.224 | -1.01 | [3.336e-05, 0.4497] | no | 0.226 | 0.05 | FAIL |
| beta_c | Ethiopia | 10 | 10.03 | 0.0628 | +0.54 | [9.934, 10.13] | yes | 0.0034 | 0.20 | ok |
| beta_c | Malawi | 1.5 | 1.493 | 0.0388 | -0.19 | [1.43, 1.556] | yes | 0.00499 | 0.20 | ok |
| beta_c | Mali | 4 | 3.954 | 0.0425 | -1.09 | [3.886, 4.024] | yes | 0.0116 | 0.20 | ok |
| beta_c | Nigeria | 6 | 5.986 | 0.042 | -0.33 | [5.917, 6.052] | yes | 0.00231 | 0.20 | ok |
| beta_c | Tanzania | 2 | 1.987 | 0.0357 | -0.37 | [1.934, 2.047] | yes | 0.00668 | 0.20 | ok |
| beta_c | Uganda | 8 | 7.162 | 0.831 | -1.01 | [6.29, 8.057] | yes | 0.111 | 0.20 | ok |
| gamma_c | Ethiopia | 10.92 | 10.88 | 0.0764 | -0.59 | [10.77, 11.01] | yes | 0.0454 | - | - |
| gamma_c | Malawi | 1.912 | 0.7263 | 0.524 | -2.26 | [0.0117, 1.518] | no | 1.19 | - | - |
| gamma_c | Mali | 75.37 | 76.2 | 0.876 | +0.95 | [74.81, 77.51] | yes | 0.83 | - | - |
| gamma_c | Nigeria | 5.137 | 5.499 | 0.305 | +1.19 | [5.01, 5.949] | yes | 0.362 | - | - |
| gamma_c | Tanzania | 7.808 | 7.972 | 0.153 | +1.07 | [7.729, 8.223] | yes | 0.164 | - | - |
| gamma_c | Uganda | 11.46 | 5.696 | 5.71 | -1.01 | [0.0008495, 11.45] | no | 5.76 | - | - |
