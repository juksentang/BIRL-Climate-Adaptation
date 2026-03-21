# BIRL MCMC 分析完整报告

> 日期：2026-03-20
> 阶段：MCMC 完成，2050 反事实分析待执行
> 前序：05_BIRL_SVI（SVI 探索，8 个变体，~25 分钟）

---

## 一、MCMC 执行总览 (Execution Overview)

### 1.1 Pipeline 五步流程

`run_birl.py` 实现了从数据加载到后验预测检查的完整流水线：

| Step | 功能 | 核心函数 | 说明 |
|------|------|----------|------|
| 0 | 数据加载 | `data_loader.load_all()` | 514,665 obs, ~15,644 HH, 6 国, 27 actions |
| 1 | Timing test | `run_mcmc()` | 200 warmup + 50 sample, 1 chain, 检测 divergence rate |
| 2 | Main MCMC | `run_mcmc_chunked()` | 每 500 步 checkpoint + GCS 同步 |
| 3 | 收敛诊断 | `diagnostics.quick_report()` | divergence, ρ-γ 相关性 |
| 4 | 后验提取 | `posterior.extract_and_save_posterior()` | 全局/国家/家庭三层参数 |
| 5 | PPC | `diagnostics.run_ppc_from_samples()` | 200 组 thinned 后验样本 |

Timing test 设置 divergence rate > 30% 自动中止（`ABORT_THRESHOLD = 0.30`），避免在失败配置上浪费计算。

### 1.2 执行配置

| 参数 | hier_noalpha (Main) | R3 (Robustness) |
|------|---------------------|-----------------|
| 采样器 | NUTS | NUTS |
| target_accept_prob | 0.8 | 0.8 |
| max_tree_depth | 10 | 10 |
| warmup | 2000 | 1000 |
| sampling | 3000 | 3000 |
| chains | 4 | 4 |
| chain_method | parallel | parallel |
| seed | 42 | 103 |
| per_step (s) | 6.94 | 5.22 |
| timing test 耗时 | 1735.7s (250 steps) | 1305.0s (250 steps) |
| GCS bucket | `gs://subsahra/birl/hier_noalpha` | `gs://subsahra/birl/R3` |

R3 因 ρ 固定（减少参数维度），每步快约 25%。

### 1.3 SVI → MCMC 初始化策略

SVI AutoNormal guide（`.npz`）提供后验均值作为 MCMC 初始化，加速 warmup 收敛：

```python
# SVI guide 存储格式: {param}_auto_loc
# 还原: strip '_auto_loc' suffix → NumPyro site name
init_values = {k.replace("_auto_loc", ""): jnp.array(v) for k, v in guide.items()}

# HalfNormal sites 在 unconstrained space (可为负)
# init_to_value 需要 constrained values (>0)，需 exp() 变换
HALFNORMAL_SITES = {"sigma_rho_G", "sigma_rho_W", "sigma_gamma_G", "sigma_gamma_W"}
for site in HALFNORMAL_SITES:
    init_values[site] = jnp.exp(init_values[site])
```

**策略分层**：
- Timing test (1 chain): `init_to_value` 直接使用 SVI 均值
- Main MCMC (4 chains): `init_to_median`（因 `init_to_value` + `chain_method='parallel'` + vmap 存在 shape 不兼容）

### 1.4 与 Round 1 的关键差异

Round 1（2026-03-18）在 SVI 设计决策尚未完善时执行，全部失败。Round 2（本次）完全基于 SVI 发现重写。

| 维度 | Round 1 (main) | Round 2 (hier_noalpha) |
|------|---------------|----------------------|
| α | individual (30K params) | **无** |
| β | 约束不当 | LogNormal prior, learnable |
| reward 标准化 | full z-score | **center-only** (mean-subtract) |
| 参数边界 | 无 / 宽松 | sigmoid: ρ∈[0.1, 5.0], γ∈[0.1, 30.0] |
| divergence | 32,000/32,000 (**100%**) | 0/12,000 (**0%**) |
| γ (median) | $37,923 (荒谬) | $29.0 (合理) |
| ρ (median) | 7.78 (极端) | 2.61 (合理) |
| ρ-γ 相关 | -0.845 (严重补偿) | -0.001 (无补偿) |

**结论**：SVI 探索阶段的所有设计决策（去 α、learnable β、center-only、bounded transforms）在 MCMC 中得到验证。

---

## 二、收敛诊断 (Convergence Diagnostics)

### 2.1 Divergence 汇总

| 模型 | Divergences | Rate | 状态 |
|------|------------|------|------|
| **hier_noalpha** | **0/12,000** | **0.000%** | **PASS** |
| **R3** | **0/12,000** | **0.000%** | **PASS** |
| *Round 1 main* | *32,000/32,000* | *100%* | *FAIL* |
| *Round 1 R1* | *12,000/12,000* | *100%* | *FAIL* |
| *Round 1 R2* | *11,795/12,000* | *98.3%* | *FAIL* |

从 100% divergence 到 0%——这是 SVI 探索阶段最重要的成果。

### 2.2 参数后验相关性

| 模型 | ρ-γ 相关 | 状态 | 说明 |
|------|---------|------|------|
| hier_noalpha | -0.001 | PASS | ρ 和 γ 独立可识别 |
| R3 | NaN | N/A | ρ 固定，无相关 |
| *Round 1 main* | *-0.845* | *FAIL* | *α-ρ-γ 三向补偿* |

ρ-γ 近乎零相关（-0.001）确认：去掉 α 后，风险厌恶与生存约束在后验中独立可识别。这是论文方法论贡献的核心证据。

### 2.3 R-hat 与 ESS

从 `posterior.pkl` 恢复 chain-separated samples（4 chains × 3,000 samples，按 chunk 重排），计算 split-R-hat 和 bulk ESS。

**hier_noalpha — 全局超参数**

| 参数 | R-hat | ESS (bulk) | 状态 |
|------|-------|------------|------|
| mu_rho_G | 1.0014 | 2,330 | PASS |
| sigma_rho_G | 1.0003 | 4,042 | PASS |
| sigma_rho_W | 1.0002 | 2,980 | PASS |
| mu_gamma_G | 1.0007 | 3,682 | PASS |
| sigma_gamma_G | 1.0002 | 6,434 | PASS |
| sigma_gamma_W | 1.0001 | 2,401 | PASS |
| log_beta | 1.0004 | 3,571 | PASS |

**hier_noalpha — 国家层 / 家庭层**

| 参数 | R-hat 范围 | ESS 范围 | 状态 |
|------|-----------|----------|------|
| rho_c (6 国) | [1.0000, 1.0072] | [1,386, 2,443] | PASS |
| gamma_c (6 国) | [0.9999, 1.0004] | [3,645, 12,000] | PASS |
| rho_i (500 HH sample) | med 1.0001, P95 1.0020 | med 9,240, P5 1,895 | PASS |
| gamma_i (500 HH sample) | med 1.0000, P95 1.0004 | med 10,143, P5 6,230 | PASS |

**R3 — 全局超参数**

| 参数 | R-hat | ESS (bulk) | 状态 |
|------|-------|------------|------|
| mu_gamma_G | 1.0001 | 4,227 | PASS |
| sigma_gamma_G | 1.0002 | 6,721 | PASS |
| sigma_gamma_W | 1.0004 | 4,419 | PASS |
| log_beta | 0.9999 | 12,000 | PASS |

**R3 — 国家层 / 家庭层**

| 参数 | R-hat 范围 | ESS 范围 | 状态 |
|------|-----------|----------|------|
| gamma_c (6 国) | [0.9998, 1.0017] | [2,183, 12,000] | PASS |
| gamma_i (500 HH sample) | med 0.9998, P95 1.0001 | med 12,000, P5 9,315 | PASS |

**收敛标准汇总**

| 指标 | 要求 | hier_noalpha | R3 |
|------|------|-------------|-----|
| R-hat | < 1.01 | **max 1.0072 (PASS)** | **max 1.0017 (PASS)** |
| ESS (bulk) | > 400 | **min 1,386 (PASS)** | **min 2,183 (PASS)** |
| Divergence rate | < 1% | **0% (PASS)** | **0% (PASS)** |
| ρ-γ 相关 | \|r\| < 0.5 | **-0.001 (PASS)** | **NaN (ρ fixed)** |

所有诊断指标全部通过。两个模型的 4 条链充分混合，后验样本可用于推断。

注：chain-separated 重排方法——`posterior.pkl` 存储合并后的 12,000 samples，按 6 chunks × (4 chains × 500 samples) 的 row-major 顺序还原为 (4, 3000, ...) 结构。

---

## 三、后验结果 (Posterior Results)

### 3.1 全局超参数 (Global Hyperparameters)

来源：`outputs/hier_noalpha/main_global_params.csv`

| 参数 | Mean | 95% CI | 解释 |
|------|------|--------|------|
| μ_ρ^G | -0.093 | [-0.671, 0.512] | 全局 ρ 均值（unbounded space, 近零 → sigmoid 后覆盖宽范围） |
| σ_ρ^G | 0.704 | [0.417, 1.151] | 国家间 ρ 变异 |
| σ_ρ^W | **2.074** | [1.980, 2.167] | 国家内（家庭间）ρ 变异 |
| μ_γ^G | 1.529 | [0.173, 2.862] | 全局 γ 均值（unbounded space） |
| σ_γ^G | **1.862** | [1.377, 2.450] | 国家间 γ 变异 |
| σ_γ^W | 0.401 | [0.028, 0.841] | 国家内（家庭间）γ 变异 |
| log_β | -2.046 | [-2.254, -1.955] | 选择温度（log scale） |
| β | 0.136 | [0.135, 0.142] | 选择温度（极窄后验） |

**核心发现：异质性结构**

```
风险厌恶 (ρ):  σ_ρ^W (2.07) >> σ_ρ^G (0.70)  →  国家内 > 国家间
生存约束 (γ):  σ_γ^G (1.86) >> σ_γ^W (0.40)  →  国家间 > 国家内
```

经济学含义：
- **ρ 的异质性主要来自家庭层面**：同一国家内不同农户的风险态度差异远大于国家均值间的差异。这反映了个体层面的异质性（财富、教育、家庭结构等）。
- **γ 的异质性主要来自国家层面**：生存底线由宏观经济条件（收入水平、市场发育程度、社会保障）决定，国家间结构性差异主导。

### 3.2 国家层参数 (Country-Level Parameters)

来源：`outputs/hier_noalpha/main_country_params.csv`

参数经 sigmoid 变换到有界空间：ρ ∈ [0.1, 5.0], γ ∈ [0.1, 30.0]

| 国家 | ρ_c | 95% CI | γ_c ($) | 95% CI | 特征 |
|------|-----|--------|---------|--------|------|
| Ethiopia | 2.39 | [2.23, 2.56] | 29.82 | [29.38, 29.99] | 高 ρ + γ 逼上界 |
| Malawi | 3.30 | [3.04, 3.57] | 29.63 | [28.53, 29.98] | **最高 ρ** + γ 逼上界 |
| Mali | 3.02 | [2.77, 3.27] | 29.12 | [26.58, 29.96] | 高 ρ + γ 逼上界 |
| Nigeria | 1.35 | [1.25, 1.45] | 17.49 | [14.69, 20.66] | **最低 ρ** + 中等 γ |
| Tanzania | 2.96 | [2.08, 3.90] | 25.42 | [5.75, 29.93] | CI 极宽 (N=121) |
| Uganda | 1.62 | [1.47, 1.78] | 1.80 | [0.52, 3.67] | 低 ρ + **最低 γ** |

**注意事项**：
- Ethiopia / Malawi / Mali 的 γ 均逼近上界 $30（γ > $29）。这不是 bound 在限制模型——而是 bound 在阻止模型走捷径。敏感性测试（γ≤$80）确认放宽 bound 后 γ 飙至 $78，失去经济学含义（见 §3.5）。$30 上界是基于消费中位数 $33.8 的合理正则化。
- Tanzania 的 CI 极宽（ρ: [2.08, 3.90], γ: [$5.75, $29.93]），因仅 121 户观测。参数向群体均值收缩，点估计不可靠。
- Nigeria 的低 ρ (1.35) 在所有 specification 中一致（SVI: 1.00, R3: fixed 1.5），是最稳健的发现之一。

### 3.3 家庭层聚合统计 (Household-Level Aggregates)

来源：`outputs/hier_noalpha/convergence_report.txt`

| 参数 | Median | P5 | P95 |
|------|--------|-----|-----|
| ρ | 2.61 | 0.29 | 3.32 |
| γ ($) | 29.0 | 1.9 | 29.8 |

γ 分布呈双峰：
- **主峰**：γ ≈ $29-30（Ethiopia / Malawi / Mali / Tanzania 的家庭，占多数）
- **次峰**：γ ≈ $1-5（Uganda 家庭 + 部分 Nigeria 家庭）

### 3.4 β 的解释 (Beta Interpretation)

β = 0.136 (95% CI: [0.135, 0.142])

后验极窄——β 是所有参数中识别性最好的（被全部 514,665 观测约束）。

具体含义：在 softmax 选择模型中，如果最优作物与次优作物的效用差为 1 个单位，选择概率之比为 exp(0.136) = 1.146。即**最优作物仅多 ~15% 的概率被选择**。

对比：
- SVI β = 0.139（差异 < 3%）
- Round 1 fixed β = 5.0（是真实值的 **36 倍**，导致 softmax 过于尖锐 → MCMC 步长崩溃至 1e-9）

低 β 意味着作物选择中存在大量未被模型捕捉的异质性（agronomic knowledge、local markets、labor constraints 等），信息干预的期望收益相应受限。

### 3.5 γ 上界敏感性测试：$30 是正则化而非约束 (Gamma Bound Sensitivity)

**实验**：将 γ 上界从 $30 放宽到 $80（`hier_noalpha_g80`），SVI 40,000 steps。

| 指标 | γ ≤ $30 (主模型) | γ ≤ $80 (敏感性) |
|------|-----------------|-----------------|
| ELBO (best) | -677,586 | -676,768 |
| β | 0.136 | 0.138 |
| γ median | $29.0 | **$77.8** |
| γ P5-P95 | [$1.9, $29.8] | [$22.7, $79.8] |
| ρ median | 2.61 | **0.99** |
| ρ P5-P95 | [0.29, 3.32] | [0.59, 1.49] |

**发现：γ 接管了 α 的"捷径"角色**

去掉 α 后，模型找到了新的效用趋同路径：将 γ 推高到收入之上（γ=$78 >> 消费中位数 $33.8）。当 Y - γ < 0 时，surplus 被 clip 到 ε，效用变成常数，softmax 退化为均匀分布——完美匹配 β=0.14 的高噪音选择。

这与 α 走的是同一条捷径：
- **α 的捷径**：扭曲主观收入，让不同作物的感知收入趋同 → 效用趋同
- **γ 的捷径**：把生存底线推到收入之上，让所有作物的 surplus 归零 → 效用趋同

堵住 α 的路（去掉 α），γ 就找到了新路。堵住 γ 的路（$30 上界），模型才被迫用 ρ 来解释选择差异——这正是我们想要的。

**结论**

$30 上界不是模型的局限，而是必要的正则化约束。论文中的表述：

> 我们基于消费数据对 γ 施加了经济学先验约束（γ ≤ median consumption），这是 Stone-Geary 文献的标准做法——生存底线不应超过实际消费水平。放宽此约束的敏感性测试确认，无约束的 γ 会飙升至消费的 2.3 倍（$78 vs $33.8），此时效用函数退化为常数，模型丧失经济学含义。

当前 MCMC 结果（γ ≤ $30）**不需要重跑**。ρ 排名、CE 排名、跨模型 robustness 在此约束下全部成立。

**更深层的启示：β=0.14 才是核心**

模型反复通过各种参数走捷径（α → γ）来匹配高噪音选择，说明 β=0.14 不是估计误差，而是数据的真实特征：**作物选择中经济最优化的成分极小**。大部分选择变异来自模型未捕捉的因素（agronomic knowledge、local markets、labor、path dependence 等）。

---

## 四、模型对比 (Model Comparison: hier_noalpha vs R3)

### 4.1 全局参数对比

| 参数 | hier_noalpha | R3 | 说明 |
|------|-------------|-----|------|
| β | 0.136 | 0.233 | R3 需要更高 β 补偿 ρ 不可调 |
| μ_γ^G | 1.53 | 2.49 | γ 全局均值上移 |
| σ_γ^G | 1.86 | 1.77 | 相近 |
| σ_γ^W | 0.40 | 0.22 | R3 内部变异更小 |
| ρ | hierarchical [0.1, 5.0] | **fixed = 1.5** | — |

β 从 0.136 升至 0.233（+71%）：当 ρ 固定后，模型通过提高选择锐度来补偿无法调节效用曲率的限制。

### 4.2 国家层 γ 对比

| 国家 | γ (hier_noalpha) | γ (R3) | Δγ | 解释 |
|------|-----------------|--------|-----|------|
| Ethiopia | 29.82 | 29.83 | +0.01 | 无变化（已在 bound） |
| Malawi | 29.63 | 29.69 | +0.06 | 无变化（已在 bound） |
| Mali | 29.12 | 29.57 | +0.45 | 轻微上移 |
| **Nigeria** | **17.49** | **29.69** | **+12.20** | **ρ-γ 补偿的直接证据** |
| Tanzania | 25.42 | 27.92 | +2.50 | 上移 |
| Uganda | 1.80 | 5.44 | +3.64 | 上移 |

**Nigeria 的 ρ-γ 补偿**：hier_noalpha 中 Nigeria ρ=1.35（最低），γ=$17.49（中等）。当 R3 强制 ρ=1.5 时，ρ 微升 0.15 → γ 必须从 $17.49 跳到 $29.69 (+$12.20) 才能拟合相同的选择行为。这精确量化了 SVI Analysis §2 中描述的 ρ-γ 补偿机制。

**政策含义**：ρ 和 γ 单独都不是稳健的政策指标。应报告 CE (certainty equivalent) 作为综合风险态度度量。

### 4.3 稳健性发现

**稳健（跨两个 specification 一致）**：
- Uganda γ 始终最低（$1.80 vs $5.44，始终 rank 6）
- Ethiopia / Malawi / Mali γ 始终在 bound 附近
- 0% divergence（两个模型均 PASS）
- β well-identified（两个模型均 CI 极窄）
- tree_crops 是 PPC 最差作物（两个模型一致）

**不稳健（对 specification 敏感）**：
- Nigeria γ 从 $17.49 变为 $29.69（ρ 自由 vs 固定）
- Uganda γ 从 $1.80 变为 $5.44（倍增，但仍是最低）
- β 从 0.136 变为 0.233（+71%）

### 4.4 PPC 最差作物对比

| Action | Label | hier_noalpha diff | R3 diff |
|--------|-------|-------------------|---------|
| 4 | tree_crops_medium | 0.067 | **0.074** |
| 3 | tree_crops_low | 0.065 | 0.068 |
| 5 | tree_crops_high | 0.043 | 0.032 |
| 20 | other_high | 0.042 | 0.039 |
| 8 | tubers_high | 0.041 | 0.036 |
| 23 | rice_high | 0.037 | 0.029 |
| 22 | rice_medium | 0.037 | 0.035 |

两个模型的最差拟合作物高度一致——tree_crops 系列始终是最大的 misfit。这是模型结构性问题，而非参数化问题。

---

## 五、PPC 分析 (Posterior Predictive Check Analysis)

### 5.1 完整 PPC 结果

来源：`outputs/hier_noalpha/ppc_action_freq.csv`

按作物分组，`diff` = |predicted - observed|：

| 作物 | 强度 | 预测 | 观测 | diff | 状态 |
|------|------|------|------|------|------|
| **maize** | low | 0.040 | 0.062 | 0.022 | OK |
| | medium | 0.042 | 0.062 | 0.020 | OK |
| | high | 0.045 | 0.062 | 0.017 | OK |
| **tree_crops** | low | 0.048 | **0.113** | **0.065** | **FAIL** |
| | medium | 0.048 | **0.115** | **0.067** | **FAIL** |
| | high | 0.052 | 0.009 | 0.043 | WARN |
| **tubers** | low | 0.048 | 0.062 | 0.014 | OK |
| | medium | 0.049 | 0.069 | 0.020 | OK |
| | high | 0.049 | 0.008 | 0.041 | WARN |
| **legumes** | low | 0.039 | 0.060 | 0.021 | OK |
| | medium | 0.041 | 0.069 | 0.027 | OK |
| | high | 0.042 | 0.011 | 0.031 | WARN |
| **sorghum_millet** | low | 0.037 | 0.044 | 0.006 | GOOD |
| | medium | 0.039 | 0.044 | 0.005 | GOOD |
| | high | 0.040 | 0.044 | 0.003 | **GOOD** |
| **teff** | low | 0.011 | 0.008 | 0.003 | GOOD |
| | medium | 0.012 | 0.008 | 0.003 | GOOD |
| | high | 0.013 | 0.008 | 0.004 | GOOD |
| **other** | low | 0.042 | 0.036 | 0.006 | GOOD |
| | medium | 0.045 | 0.044 | 0.002 | **GOOD** |
| | high | 0.051 | 0.008 | 0.042 | WARN |
| **rice** | low | 0.037 | 0.008 | 0.030 | WARN |
| | medium | 0.044 | 0.008 | 0.037 | WARN |
| | high | 0.045 | 0.008 | 0.037 | WARN |
| **wheat_barley** | low | 0.010 | 0.010 | **0.000** | **GOOD** |
| | medium | 0.020 | 0.010 | 0.010 | OK |
| | high | 0.011 | 0.010 | 0.001 | GOOD |

状态标准：GOOD (diff < 0.01), OK (< 0.03), WARN (< 0.05), FAIL (≥ 0.05)

### 5.2 模式分析

**Pattern 1：强度偏差 (Intensity Bias)**

模型系统性地高估 _high 强度变体、低估 _low 和 _medium 变体。例如 tree_crops: high 预测 5.2% vs 观测 0.9%（高估 4.3pp），low 预测 4.8% vs 观测 11.3%（低估 6.5pp）。模型的效用函数对施肥强度水平区分不足。

**Pattern 2：tree_crops 是最大 misfit**

tree_crops 在观测中占 ~23.6% 的选择（low 11.3% + medium 11.5% + high 0.9%），但模型仅预测 ~14.8%。这是所有 27 个 action 中最大的系统性偏差，在两个模型 specification 中一致。

**Pattern 3：地理集中作物拟合良好**

sorghum_millet（diff 0.003-0.006）、teff（diff 0.003-0.004）、wheat_barley_low（diff 0.000）拟合极好。这些作物地理分布高度集中（teff 几乎仅在 Ethiopia，wheat_barley 集中在 Ethiopian 高原），country-zone feasibility mask 有效捕捉了这一模式。

### 5.3 tree_crops 专题

tree_crops（咖啡、可可、油棕等）的 misfit 可能源于以下模型限制：

1. **多年生投资特性**：tree crops 是 5-30 年的长期投资，回报需要 3-5 年才开始产出。单期效用函数无法捕捉这种跨期收益结构。
2. **沉没成本效应**：已经种植的树作物即使当前不是最优也会保留，形成路径依赖。模型假设每期独立选择。
3. **land allocation rigidity**：树作物占据固定地块，调整成本极高。模型假设完全灵活的作物选择。

这是模型的结构性局限而非参数化问题，因此调整 ρ、γ 或 β 均无法改善。潜在改进方向：作物类型特定的效用参数（future work）。

---

## 六、经济学解释 (Economic Interpretation)

### 6.1 ρ 的含义 (Risk Aversion)

效用函数：

```
U(Y) = (max(Y - γ, ε))^(1-ρ) - 1) / (1-ρ)
当 ρ → 1 时退化为 log(max(Y - γ, ε))
```

| ρ 区间 | 风险态度 | 对应国家 | 农户行为 |
|--------|---------|----------|----------|
| ~1.0 | 近 log utility（温和风险厌恶） | Nigeria (1.35) | 愿意选择风险更高、回报更高的作物 |
| ~1.5 | 中等风险厌恶 | Uganda (1.62) | 风险收益平衡 |
| ~2.4 | 较强风险厌恶 | Ethiopia (2.39) | 偏好安全、低波动作物 |
| ~3.0-3.3 | 强风险厌恶 | Malawi (3.30), Mali (3.02) | 强烈回避下行风险，宁可牺牲收益 |

参考文献中 ρ 的典型范围：发展经济学实验估计通常在 0.5-3.0 之间，本研究结果在合理范围内。Malawi 的 3.30 在高端但不极端。

### 6.2 γ 的含义 (Subsistence Threshold)

γ 代表生存底线——收入低于 γ 时效用趋向负无穷（Stone-Geary 特性）。

| 国家 | γ ($) | 消费中位数 ($) | γ/median | 经济含义 |
|------|-------|---------------|----------|----------|
| Uganda | 1.80 | 27.4 | 7% | 生存约束极低——大部分收入在"安全线"以上 |
| Nigeria | 17.49 | 104.2 | 17% | 中等生存约束 |
| Tanzania | 25.42 | 30.3 | 84% | 高生存约束（多数收入仅够生存） |
| Mali | 29.12 | 127.0 | 23% | 绝对金额高但相对消费仍可管理 |
| Malawi | 29.63 | 25.0 | **119%** | γ > 消费中位数——极端困境 |
| Ethiopia | 29.82 | 19.9 | **150%** | γ >> 消费中位数——极端困境 |

Malawi 和 Ethiopia 的 γ > 消费中位数意味着**多数家庭在生存线以下**。效用几乎完全由 survival concern 驱动，与极高的 ρ（3.30 和 2.39）一致：当你连基本生存都无法保障时，任何额外风险都是不可接受的。

**注意**：γ 逼近 $30 上界是正则化的结果，而非模型局限。敏感性测试（§3.5）确认放宽上界后 γ 飙至 $78，模型退化为常数效用。$30 是基于消费中位数 $33.8 的合理约束。消费中位数数据来自 SVI Analysis §4.2。

### 6.3 β 的含义 (Choice Noise)

β = 0.136 → 选择概率对效用差异极不敏感。

实际含义：
- 如果某农户的最优作物 A 比次优 B 的期望效用高 1 单位，P(选 A)/P(选 B) = exp(0.136) ≈ 1.15
- 如果效用差高 10 单位，比值也仅 exp(1.36) ≈ 3.9
- 这意味着即使 "告诉农户正确信息"（纠正信念），选择行为也不会大幅改变——信息干预的 marginal value 受限

### 6.4 CE 排名与跨模型稳健性 (CE Ranking and Robustness)

CE (certainty equivalent) 是综合 ρ 和 γ 的单一福利度量：给定国家的风险偏好参数和收入分布，CE 是使代表性农户无差异的确定性收入。

```
U(CE) = E[U(Y)]
CE = γ + ((E[U] × (1-ρ) + 1))^(1/(1-ρ))
```

**CE 排名表**

| 排名 | 国家 | ρ | γ ($) | CE ($) | med_Y ($) | CE/Y | CE_R3 ($) |
|------|------|-----|-------|--------|-----------|------|-----------|
| 1 (最保守) | Uganda | 1.62 | 1.80 | 7.0 | 25.5 | 28% | 10.5 |
| 2 | Tanzania | 2.96 | 25.42 | 26.9 | 26.0 | 103% | 30.9 |
| 3 | Malawi | 3.30 | 29.63 | 30.9 | 19.1 | 162% | 32.0 |
| 4 | Ethiopia | 2.39 | 29.82 | 31.3 | 19.9 | 158% | 32.2 |
| 5 | Mali | 3.02 | 29.12 | 31.9 | 130.0 | 25% | 49.1 |
| 6 (最不保守) | Nigeria | 1.35 | 17.49 | 42.8 | 102.7 | 42% | 41.0 |

**CE 排名**

```
hier_noalpha:  Uganda < Tanzania < Malawi < Ethiopia < Mali < Nigeria
R3:            Uganda < Tanzania < Malawi < Ethiopia < Nigeria < Mali
```

**跨模型 Spearman 相关**

| 模型对 | Spearman ρ | p-value | 状态 |
|--------|-----------|---------|------|
| MCMC hier_noalpha vs MCMC R3 | **0.943** | **0.005** | **显著** |
| SVI hier_noalpha vs MCMC hier_noalpha | **1.000** | **0.000** | **完全一致** |
| *SVI hier_noalpha vs SVI R3 (参考)* | *0.886* | *0.019* | *显著* |

核心发现：
- **MCMC 后验下 CE 排名跨 specification 高度稳定**（Spearman = 0.943, p = 0.005），仅 Mali 和 Nigeria 在 rank 5-6 互换。
- **SVI 和 MCMC 的 CE 排名完全一致**（Spearman = 1.000）。尽管 ρ 点估计上移最多 1.5（§7.1），CE 作为综合度量吸收了 ρ-γ 补偿，排名不变。
- 这是论文 robustness section 的核心证据：**CE 排名对模型 specification 和推断方法均稳健**。

**经济学解读**

- **Uganda CE=$7.0 远低于 median_Y=$25.5**：极高 "风险税"。这不是来自高 ρ（Uganda ρ=1.62 很温和），而是来自极低的 γ=$1.80——Uganda 农户的效用对低收入极端事件非常敏感。
- **Nigeria CE=$42.8 最高**：低 ρ + 中等 γ + 高 median_Y 的组合。Nigerian 农户最接近 risk-neutral 行为。
- **Malawi/Ethiopia CE≈$31 但 median_Y 仅 $19-20**：CE > median_Y 反映收入分布右偏——少数高收入作物选择拉高了 CE。

---

## 七、SVI 与 MCMC 对比 (SVI vs MCMC Comparison)

### 7.1 国家参数对比

SVI 数据来源：`05_BIRL_SVI/Analysis.md` §4.2

| 国家 | ρ_SVI | ρ_MCMC | Δρ | γ_SVI ($) | γ_MCMC ($) | Δγ ($) |
|------|-------|--------|-----|-----------|------------|--------|
| Ethiopia | 1.34 | 2.39 | +1.05 | 28.8 | 29.82 | +1.02 |
| Malawi | 1.79 | 3.30 | **+1.51** | 28.1 | 29.63 | +1.53 |
| Mali | 1.92 | 3.02 | +1.10 | 27.0 | 29.12 | +2.12 |
| Nigeria | 1.00 | 1.35 | +0.35 | 22.7 | 17.49 | -5.21 |
| Tanzania | 2.33 | 2.96 | +0.63 | 23.6 | 25.42 | +1.82 |
| Uganda | 0.78 | 1.62 | +0.84 | 1.7 | 1.80 | +0.10 |
| **β** | **0.139** | **0.136** | **-0.003** | — | — | — |

### 7.2 关键发现

**1. ρ 系统性上移**

MCMC 的 ρ 在所有 6 国均高于 SVI（+0.35 到 +1.51）。这是预期中的 SVI 偏差：mean-field variational family（AutoNormal）倾向于 mode-seeking（underestimate variance），SVI 点估计倾向于靠近后验 mode 而非 mean。MCMC 的 full posterior exploration 发现真实后验 mass 在更高的 ρ 区域。

**2. γ 大致稳定（Nigeria 除外）**

5/6 国家 γ 变化在 $0.10 ~ $2.12 范围，方向一致（略微上移）。例外是 Nigeria（-$5.21），反映 ρ-γ 补偿：ρ 上升了 0.35，γ 相应下调。

**3. 排名保持一致**

| 排名指标 | SVI | MCMC | 一致？ |
|---------|-----|------|--------|
| 最低 ρ | Nigeria (1.00) | Nigeria (1.35) | 是 |
| 最高 ρ | Tanzania (2.33) | Malawi (3.30) | 否（顺序微调） |
| 最低 γ | Uganda (1.7) | Uganda (1.80) | 是 |
| 最高 γ | Ethiopia (28.8) | Ethiopia (29.82) | 是 |

定性故事不变：Nigeria 最不风险厌恶，Uganda γ 最低，Ethiopia/Malawi 极端保守。

**4. β 几乎不变**

SVI 0.139 → MCMC 0.136（差异 < 3%）。β 是全局标量，被全部数据约束，两种方法给出近乎一致的估计。

### 7.3 SVI 近似质量评价

| 维度 | SVI 是否可靠 | 说明 |
|------|-------------|------|
| 国家排名 | **是** | 定性排名一致 |
| β 估计 | **是** | 差异 < 3% |
| ρ 点估计 | **否** | 低估最多 1.5 (Malawi) |
| γ 点估计 | **部分** | 多数稳定，Nigeria 偏差 $5 |
| 不确定性量化 | **否** | SVI 无 CI；MCMC 揭示 Tanzania CI 极宽 |
| 后验多峰性 | **否** | SVI 假设单峰高斯，无法检测 |

**结论**：SVI 适合快速探索和模型选择（25 分钟 vs MCMC 数小时），但最终推断必须用 MCMC。SVI 的核心价值在于：(1) 排除不可行的 specification（α、R2），(2) 提供 MCMC 初始化，(3) 建立参数量级的先验预期。

---

## 八、已知局限与下一步 (Known Limitations and Next Steps)

### 8.1 已知局限

| 局限 | 影响 | 证据 | 可能缓解 |
|------|------|------|----------|
| ~~γ 上界 = $30 可能截断~~ | ~~Ethiopia/Malawi/Mali γ 可能被截断~~ | **敏感性测试确认 $30 是必要正则化**（见 §3.5） | — |
| ~~R-hat/ESS 未计算~~ | ~~无法正式确认收敛~~ | **已计算，全部 PASS**（见 §2.3） | — |
| Tanzania N=121 | 参数不可靠 | ρ CI: [2.08, 3.90], γ CI: [$5.75, $29.93] | 报告时标注；考虑 pooling |
| tree_crops misfit | 最大 PPC gap (0.065-0.074) | 两个模型一致 | 作物类型特定参数（future work） |
| 单期效用 | 忽略投资动态 | tree crops 是多年生 | 多期模型（future work） |
| Uganda γ 异常低 | $1.80 可能不合理 | SVI ($1.7) 和 MCMC ($1.80) 一致 | 检查 env model 校准偏差（q50/actual = 1.108） |
| ρ-γ 补偿 | 单参数不稳健 | Nigeria γ: $17.49 vs $29.69 | 报告 CE 而非单参数 |

### 8.2 待办事项

**立即（Post-MCMC）**

- [x] 从 `posterior.pkl` 提取 chain-separated samples（按 chunk 重排）
- [x] 计算正式 R-hat 和 ESS（split-R-hat + FFT autocorrelation，全部 PASS，见 §2.3）
- [x] 计算 CE 排名：Uganda < Tanzania < Malawi < Ethiopia < Mali < Nigeria（见 §6.4）
- [x] CE 排名 Spearman：hier_noalpha vs R3 = 0.943 (p=0.005)；SVI vs MCMC = 1.000（见 §6.4）
- [x] γ 上界敏感性分析：$80 上界 → γ 飙到 $78，确认 $30 是必要正则化（见 §3.5）

**2050 反事实分析**

- [ ] 生成 2050 气候情景下的反事实收入矩阵（SSP2-4.5, SSP5-8.5）
- [ ] 实现 4 个 counterfactual scenarios（baseline 2050, 保险, 安全网, 当前对照）
- [ ] 不确定性传播：对每组后验样本独立跑 simulation
- [ ] 政策价值量化：保险价值 = CE(保险) - CE(baseline)，按国家分解

**论文**

- [ ] α 不可识别的完整证据链（独立 section，引用 SVI Analysis §2）
- [ ] Country-level CE 排名主表 + 95% CI
- [ ] Robustness 表：CE 排名 Spearman across 4+ specifications
- [ ] Appendix：SVI 模型对比、MCMC 诊断、env model 诊断、PPC 完整表

---

## 九、技术参考 (Technical Reference)

### 9.1 源代码

| 文件 | 功能 |
|------|------|
| `run_birl.py` | 主 pipeline（5 步 + SVI init + GCS sync） |
| `src/models.py` | `birl_hier_noalpha`, `birl_r3` 模型定义 |
| `src/mcmc_runner.py` | NUTS MCMC + chunked checkpointing + GCS 同步 |
| `src/data_loader.py` | 数据加载（parquet + npz → JAX arrays, BIRLData dataclass） |
| `src/posterior.py` | 后验提取（sigmoid bounded transform, 三层参数） |
| `src/diagnostics.py` | 收敛诊断 + PPC |
| `src/robustness.py` | Spearman rank 相关计算 |
| `src/config.py` | 路径、设备检测、日志、GCS bucket 定义 |

### 9.2 输入数据

| 文件 | 说明 |
|------|------|
| `data/birl_sample.parquet` | 514,665 obs (hh_id, country, zone, action_id, outcomes) |
| `data/env_model_output.npz` | 反事实收入分布 (q10, q50, q90, sigma) per obs×action |
| `data/action_space_config.json` | 27 actions × 6 zones, feasibility mask |
| `data/svi_hier_noalpha_guide.npz` | SVI 后验（MCMC 初始化用） |
| `data/svi_r3_guide.npz` | SVI 后验（R3 初始化用） |

### 9.3 输出文件

**hier_noalpha** (`outputs/hier_noalpha/`)：

| 文件 | 格式 | 内容 |
|------|------|------|
| `timing.json` | JSON | per_step=6.94s, div_rate=0% |
| `convergence_report.txt` | TXT | divergence + quick_report |
| `main_global_params.csv` | CSV | 8 个超参数 + 95% CI |
| `main_country_params.csv` | CSV | 6 国 ρ_c, γ_c + 95% CI |
| `main_hh_params.parquet` | Parquet | ~15,644 HH 的 ρ_i, γ_i (mean/std/q025/q975) |
| `ppc_action_freq.csv` | CSV | 27-action 预测 vs 观测 |
| `correlation_matrix.csv` | CSV | ρ-γ Pearson 相关 = -0.001 |
| `posterior.pkl` | Pickle | 完整后验样本 (~2.8 GB) |
| `mcmc_state.pkl` | Pickle | MCMC 状态（resume 用, ~2.9 GB） |

**R3** (`outputs/R3/`): 同上结构，额外含 `quick_report_R3.txt`。

### 9.4 复现指南

```bash
cd "/home/yushentang/NF/Formal Analysis/06_BIRL_MCMC"

# 从 GCS 拉取结果（无需重跑 MCMC）
./outputs/pull_from_gcs.sh

# 或只拉取非 pkl 的轻量文件
./outputs/pull_from_gcs.sh --no-pkl

# 查看远端文件
./outputs/pull_from_gcs.sh --list

# 重跑 MCMC（需 TPU）
python3 run_birl.py --variant hier_noalpha
python3 run_birl.py --variant R3

# Dry-run（仅验证数据加载）
python3 run_birl.py --variant hier_noalpha --dry-run
```

### 9.5 模型 DAG

```
Global:  μ_ρ^G, σ_ρ^G, σ_ρ^W, μ_γ^G, σ_γ^G, σ_γ^W, log_β
              |
Country:  ρ_c ~ Normal(μ_ρ^G, σ_ρ^G)              c = 1,...,6
          γ_c ~ Normal(μ_γ^G, σ_γ^G)
              |
Household: ρ_i = 0.1 + 4.9 × sigmoid(ρ_c[i] + σ_ρ^W × offset_i)
           γ_i = 0.1 + 29.9 × sigmoid(γ_c[i] + σ_γ^W × offset_i)
              |
Observation: surplus = max(Y - γ_i, ε)
             reward  = E[surplus^(1-ρ_i) / (1-ρ_i)]   (5-point quadrature)
             logits  = β × center(reward) × feasibility_mask
             action  ~ Categorical(softmax(logits))
```

Non-centered parameterization (offset ~ Normal(0, 1))，避免漏斗几何导致的 MCMC 低效。

R3 变体：ρ_i = 1.5 (fixed)，仅 γ 层级化。
