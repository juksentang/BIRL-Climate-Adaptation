# BIRL SVI 分析完整报告

> 日期：2026-03-19
> 阶段：SVI 探索完成，MCMC 待执行

---

## 一、SVI 探索总览

### 1.1 跑过的所有模型

| 变体 | 结构 | α | ρ | γ | β | ELBO (best) | PPC | 耗时 |
|------|------|---|---|---|---|-------------|-----|------|
| main | Stone-Geary, individual params | individual (30K) | 自由 | 自由, ≤30 | learned | -666,556 | 0.065 | 3.4min |
| noalpha | Stone-Geary, pooled | 无 | 自由 | 自由, ≤30 | learned | -676,591 | 0.070 | 2.9min |
| country_alpha | Stone-Geary, country α | country (6) | 自由 | 自由 | learned | -666,238 | 0.064 | 3.2min |
| hier (with α) | Stone-Geary, hier ρ/γ | country (6) | country-level | country-level | learned | -665,129 | 0.065 | 3.1min |
| **hier_noalpha** | **Stone-Geary, hier ρ/γ** | **无** | **country-level** | **country-level** | **learned** | **-677,586** | **0.071** | **2.7min** |
| R2 | E[Y]-λP(Y<γ), hier | 无 | 无 (λ替代) | country-level | learned | -1,848,045 | 0.186 | 2.6min |
| R3 | Stone-Geary, fixed ρ=1.5 | 无 | 固定 1.5 | country-level | learned | -682,686 | 0.074 | 2.3min |
| Nigeria-only | Stone-Geary | single α | 自由 | 自由 | learned | — | — | ~3min |

总计 SVI 探索耗时：约 25 分钟。

### 1.2 关键技术修复时间线

| 问题 | 症状 | 修复 | 发现方式 |
|------|------|------|----------|
| TracerBoolConversionError | JAX tracing 失败 | 消除 Python bool 分支 | SVI 第一次运行 |
| NaN 爆炸 | ELBO → -inf | 数值安全的 log/exp | SVI 训练过程 |
| 参数飞到无穷 | ρ→∞, γ→∞ | bounded transforms | SVI 参数监控 |
| β vs ρ 识别性 | β 和 ρ 互相补偿 | β 改为 learnable | SVI 参数对比 |
| z-score 抹掉尺度 | α/γ 只能改排名，梯度爆炸 | 改为 center-only | MCMC 步长 1e-9 |
| center-only + β=5 仍过尖锐 | MCMC 步长 1e-9 | 全局 RMS 标准化 + β learnable | MCMC timing test |
| γ 撞上界 ($200) | 效用变成常数的捷径 | 上界改为 $30（基于消费中位数 $33.8） | SVI 参数监控 |

---

## 二、核心发现：α 不可识别

### 2.1 证据链

**证据 1：Individual α 后验铺平**

main 模型的 individual α 后验从 -1.5 到 +1.5 近乎均匀分布。30K+ 参数各自不受数据约束。

**证据 2：Country α 撞 bound**

6 个国家中 5 个撞 ±1.5 边界（放宽到 ±3.0 后仍然撞）。唯一收敛的 Nigeria α=+0.122。

**证据 3：α-γ 补偿**

Nigeria-only 模型中 α 符号翻转：6 国模型 α=+0.12（悲观），Nigeria-only α=-0.20（乐观）。原因是 γ 从 0.7 变为 22.1，α 随之翻转。α 的值完全取决于 γ 的假设。

**证据 4：ELBO 差异的真实来源**

| 模型 | ELBO | α 参数量 |
|------|------|----------|
| main (individual α) | -666,556 | 30K+ |
| country_alpha (6 个 α) | -666,238 | 6 |
| noalpha | -676,591 | 0 |

country_alpha 用 6 个参数打平 individual α 的 30K 参数，说明 individual α 的 99.98% 在拟合噪音。而 6 个 country α 中 5 个撞 bound，说明即使 country-level 的 α 也不是在学真实信念偏差，而是在走效用趋同的捷径。

**证据 5：α 存在时 γ 失去经济含义**

| 模型 | γ 中位数 | 经济含义 |
|------|----------|----------|
| 有 α (hier) | 0.2 - 3.2 | 生存底线 ≈ $0-3，等于没有 |
| 无 α (hier_noalpha) | 1.7 - 28.8 | 生存底线占消费 5-85%，合理 |

α 在场时吸收了 γ 的解释力，导致两个参数都不可解释。

### 2.2 不可识别的根本原因

在离散作物选择的 revealed preference 数据中，信念偏差（α）和偏好参数（ρ, γ）之间存在结构性补偿。"农户不选最赚钱的作物"可以被解释为：

- 他不知道那个作物最赚钱（α 故事）
- 他知道但太怕风险（ρ 故事）
- 他知道但需要优先保证生存（γ 故事）

三种解释对观测到的选择产生近乎相同的 likelihood，数据无法区分。

### 2.3 什么数据才能识别 α

- 面板数据：同一农户跨年改变选择，可以识别信念更新
- 主观预期调查：直接测量信念，不需要从选择中反推
- 信息干预 RCT：外生改变信念，观察选择变化
- 更大的 action 间收入差异：β=0.14 说明当前选择噪音极大，信号太弱

---

## 三、航向调转：从信念偏差到风险偏好

### 3.1 原计划叙事

> "Sub-Saharan Africa 农户的作物选择偏离最优，核心原因是信念偏差（α）。纠正信念可以带来显著的福利改善。2050 年气候变化下信息干预的价值更大。"

### 3.2 新叙事

> "Sub-Saharan Africa 农户的作物选择主要由风险厌恶（ρ）和生存约束（γ）驱动。我们尝试从 revealed preference 中识别信念偏差，发现在这类数据下不可识别——这本身是方法论贡献。Country-level 的综合风险态度（CE 排名）在多种模型 specification 下高度稳定。2050 年气候变化下，作物保险和社会安全网（降低有效 ρ 和 γ）的政策价值可以精确量化。"

### 3.3 新叙事的学术贡献

1. **方法论贡献**：首次在 Sub-Saharan Africa 作物选择场景下应用 Bayesian IRL，并诚实报告信念偏差不可识别的条件——为后续研究提供方法论指引
2. **实证贡献**：Country-level 风险偏好结构的估计，以及 CE 排名的跨模型稳健性（Spearman=0.886, p=0.019）
3. **政策贡献**：2050 气候情景下风险管理工具（保险、安全网）的福利价值量化
4. **Negative result 的价值**：信息干预（纠正信念）不能通过作物选择数据来评估，需要直接的信念测量——这改变了政策评估的方法论建议

---

## 四、最终模型：hier_noalpha

### 4.1 模型结构

```
效用函数: U_i(Y) = (max(Y - γ_c, ε))^(1-ρ_c) - 1) / (1-ρ_c)

层级结构:
  群体层:  μ_ρ, σ_ρ, μ_γ, σ_γ (超参数)
  国家层:  ρ_c ~ Normal(μ_ρ, σ_ρ)     c = 1,...,6
           γ_c ~ Normal(μ_γ, σ_γ)     c = 1,...,6
  个体层:  ρ_i ~ Normal(ρ_c, σ_within_ρ)
           γ_i ~ Normal(γ_c, σ_within_γ)

选择模型: P(action_j | obs_i) = softmax(β × U_i(Y_j))
  β ~ LogNormal(prior)，learnable

信念偏差: 无（α = 0）
```

### 4.2 SVI 参数估计

**Country-level 参数**

| 国家 | ρ (风险厌恶) | γ (生存底线, $) | CE ($) | median_Y ($) | CE/Y ratio | n_hh |
|------|-------------|----------------|--------|-------------|------------|------|
| Uganda | 0.78 | 1.7 | 19.5 | 27.4 | 71% | 2,112 |
| Tanzania | 2.33 | 23.6 | 25.5 | 30.3 | 84% | 121 |
| Malawi | 1.79 | 28.1 | 30.0 | 25.0 | 120%* | 2,611 |
| Ethiopia | 1.34 | 28.8 | 31.3 | 19.9 | 157%* | 4,111 |
| Mali | 1.92 | 27.0 | 37.8 | 127.0 | 30% | 1,822 |
| Nigeria | 1.00 | 22.7 | 65.6 | 104.2 | 63% | 4,867 |

*Malawi/Ethiopia 的 CE > median_Y 是因为收入分布右偏，高分位数贡献大。

β = 0.139：选择包含大量噪音，符合 household survey 数据的特征。

**已知问题**

- Uganda γ=1.7 异常低，可能受 env model 校准偏差影响（q50/actual = 1.108，高估 11%）
- Tanzania 只有 121 户，参数估计向群体均值收缩
- ρ 和 γ 单独不可精确识别（ρ-γ 补偿），但 CE 排名稳定

### 4.3 Robustness 证据

**CE 排名跨模型相关性**

| 模型对 | Spearman ρ | p-value |
|--------|-----------|---------|
| hier_noalpha vs R3 (固定 ρ=1.5) | 0.886 | 0.019 |
| hier_noalpha vs hier+alpha | 0.771 | 0.072 |
| hier_noalpha vs country_alpha | 0.714 | 0.111 |
| R3 vs hier+alpha | 0.771 | 0.072 |
| hier+alpha vs country_alpha | 0.943 | 0.005 |

核心结论：CE 排名在不同 specification 下稳定。Uganda 始终最保守（rank 1），Mali/Nigeria 始终最不保守（rank 5-6）。

**R2 (替代效用结构) 的结果**

PPC = 0.186（远差于主模型的 0.071），确认 Stone-Geary CRRA 是更适合的效用结构。R2 不适合作为 robustness check，但其失败支持了主模型的效用函数选择。

---

## 五、MCMC 执行计划

### 5.1 配置

```yaml
模型: hier_noalpha
采样器: NUTS
warmup: 2000
sampling: 3000
chains: 4
hardware: TPU v4-8 (on-demand)
checkpoint: 每 500 步保存到 GCS
预计耗时: 8-12 小时
```

### 5.2 SVI → MCMC 的初始化

用 SVI 的后验均值作为 MCMC 的初始化点，加速 warmup 收敛：

```python
svi_params = load_svi_checkpoint("hier_noalpha")
init_values = {k: v.mean() for k, v in svi_params.items()}
mcmc = MCMC(NUTS(model), num_warmup=2000, num_samples=3000,
            num_chains=4, chain_method="vectorized")
mcmc.run(rng, init_params=init_values, ...)
```

### 5.3 收敛诊断标准

| 指标 | 要求 | 说明 |
|------|------|------|
| R-hat | < 1.01 | 所有参数 |
| ESS (bulk) | > 400 | 群体层超参数是瓶颈 |
| ESS (tail) | > 200 | 确保尾部估计可靠 |
| Divergence rate | < 1% | sampling 阶段 |
| trace plot | 链间混合良好 | 目视检查 ρ_c, γ_c |

### 5.4 同时跑 R3 作为 Robustness

如果 TPU 资源允许，同时跑 R3 hier_noalpha (固定 ρ=1.5) 的 MCMC。两个模型的 CE 排名 Spearman 相关在 MCMC 后验下是否仍然显著，是论文 robustness section 的核心证据。

---

## 六、2050 分析框架

### 6.1 输入

- hier_noalpha 的 MCMC 后验：每个国家的 ρ_c, γ_c 的后验样本
- 2050 气候情景下的反事实收入矩阵（SSP2-4.5 / SSP5-8.5）
- 当前气候条件下的反事实收入矩阵（baseline）

### 6.2 四个 Counterfactual 情景

| 情景 | 气候 | 风险管理 | 政策含义 |
|------|------|----------|----------|
| Baseline 2050 | 2050 | 无干预 (当前 ρ_c, γ_c) | 气候变化下农户自然适应 |
| 保险干预 2050 | 2050 | ρ_effective = ρ_c × 0.5 | 作物保险降低有效风险厌恶 |
| 安全网干预 2050 | 2050 | γ_effective = γ_c × 0.5 | 社会安全网降低生存底线 |
| 当前对照 | 当前 | 无干预 | 当前条件下的现状 |

### 6.3 关键输出

```
每个情景 × 每个国家:
  - 作物选择分布: P(action | scenario, country)
  - 期望收入: E[Y | optimal choice]
  - CE: certainty equivalent
  - 福利变化: ΔCE relative to baseline

政策价值:
  保险价值 = CE(保险干预) - CE(Baseline 2050)
  安全网价值 = CE(安全网干预) - CE(Baseline 2050)
  气候损失 = CE(当前对照) - CE(Baseline 2050)
```

### 6.4 不确定性传播

对每组 MCMC 后验样本独立跑一次 simulation，得到所有输出量的后验分布。报告中位数和 95% credible interval。

### 6.5 预期政策发现

- Uganda（CE 最低，最保守）对保险和安全网的响应可能最大
- Nigeria（ρ≈1, CE 最高）可能已经接近最优选择，干预的边际价值较小
- 保险 vs 安全网的相对价值在不同国家可能不同——高 ρ 国家受益于保险，高 γ 国家受益于安全网

---

## 七、论文结构建议

### Abstract 核心信息

使用 Bayesian IRL 从 Sub-Saharan Africa 6 国 22 万农户的作物选择中估计风险偏好结构。发现信念偏差在这类数据下不可识别，作物选择主要由风险厌恶和生存约束驱动。Country-level 综合风险态度排名跨模型 specification 高度稳定。2050 气候情景下量化了作物保险和社会安全网的福利价值。

### 关键章节

1. **Model**: BIRL + Stone-Geary CRRA，hierarchical country-level ρ 和 γ
2. **Identification**: α 不可识别的完整证据链（Section 专门讨论）
3. **Results**: country-level CE 排名及其经济学解释
4. **Robustness**: CE 排名跨 4 个 specification 的 Spearman 相关
5. **Policy simulation**: 2050 counterfactual，保险 vs 安全网的价值
6. **Discussion**: α 不可识别的方法论含义——什么数据能识别信念偏差

### Appendix

- SVI 全部 8 个模型的参数表和 ELBO 对比
- MCMC 收敛诊断（R-hat, ESS, trace plot）
- Env model 诊断报告（5 项诊断的完整结果）
- ρ-γ 补偿的可视化（scatter plot with country labels）

---

## 八、已排除方向存档

| 方向 | 尝试 | 结果 | 排除原因 |
|------|------|------|----------|
| Individual α | main SVI | 后验铺平 [-1.5, 1.5] | 不可识别 |
| Country α | country_alpha SVI | 5/6 撞 bound | α-γ 补偿 |
| Country α (bound ±3) | Nigeria-only SVI | α 符号随 γ 翻转 | 结构性补偿 |
| Per-obs z-score | P5 修复 v1 | 步长 1e-9，梯度爆炸 | 破坏 obs 间 scale |
| Per-hectare env model | 10% 采样测试 | action importance 3.8% vs 4.2% | 面积不是根因 |
| Crop-specific env model | maize/sorghum 对比 | chose_best 61.7% vs 62.9% | 无改善 |
| R2 效用结构 | hier_noalpha SVI | PPC=0.186 | 严重 misspecified |
| R1 复合效用 | MCMC timing test | 100% diverge | max(0,...) 不可导 |
| R5-R8 | 设计评审 | — | ρ-γ 同时自由，注定不收敛 |

---

## 九、待办事项清单

### 立即执行

- [ ] 确认 TRC on-demand quota 状态（已发邮件）
- [ ] hier_noalpha MCMC：拿到 on-demand v4-8 后立即提交
- [ ] R3 hier_noalpha MCMC：同时提交作为 robustness

### MCMC 完成后

- [ ] 收敛诊断：R-hat, ESS, trace plot, divergence rate
- [ ] 后验分析：country-level ρ_c, γ_c 的完整后验分布
- [ ] CE 计算：基于 MCMC 后验的 CE 排名和 credible interval
- [ ] CE 排名 robustness：主模型 vs R3 的 Spearman 相关（在后验下）

### 2050 分析

- [ ] 生成 2050 气候情景下的反事实收入矩阵
- [ ] 实现 4 个 counterfactual 情景的 simulation 代码
- [ ] 不确定性传播：对每组后验样本独立跑 simulation
- [ ] 政策价值量化：按国家分解保险/安全网的 CE 改善

### 论文写作

- [ ] α 不可识别的证据链（独立 section）
- [ ] Country-level CE 排名表 + robustness 表
- [ ] 2050 policy simulation 的主要结果
- [ ] Appendix：SVI 模型对比、MCMC 诊断、env model 诊断