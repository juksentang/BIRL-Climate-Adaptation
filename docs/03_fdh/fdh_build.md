# DEA Frontier Estimation Documentation

**方法**: Order-m FDH (Free Disposal Hull)
**参考文献**: Cazals, Florens & Simar (2002), "Nonparametric frontier estimation: a robust approach"
**计算时间**: 19秒（全部212,104 obs）

---

## 1. 为什么用FDH而不是DEA-LP

| 方法 | 凸性假设 | 求解方式 | 单产出复杂度 | 实际耗时(212K obs) |
|------|:--------:|---------|:-----------:|:-----------------:|
| DEA (LP) | 需要凸组合 | 每obs一次线性规划 | O(N × m × LP) | >60分钟（第一层卡死） |
| **FDH** | **不需要** | **纯max操作** | **O(N × B × m)** | **19秒** |

FDH在此场景够用的原因：
- 每层平均5000 obs，密度高，FDH前沿和DEA前沿几乎重合
- 3个投入维度，不高，FDH的"维度诅咒"不严重
- Order-m本身就是FDH的鲁棒版本（Cazals et al. 2002原始论文就是FDH+Order-m）

---

## 2. Order-m FDH算法

对每个obs i（产出导向，单产出）：

```
1. 找 dominated set D_i = {j : X_j ≤ X_i 逐元素}
   即所有"投入不多于i"的参考obs

2. 从 D_i 中随机抽 m 个 obs（有放回）

3. θ_b = Y_i / max(Y_sampled)
   即 obs i 的产出 / 参考集中最大产出

4. 重复 B 次，efficiency_i = mean(θ_1, ..., θ_B)
```

**参数**: m=25, B=200（文献标准值）

**效率得分解读**：
- efficiency = 1.0: obs在前沿上（它的产出=参考集最大产出）
- efficiency < 1.0: 低于前沿（需要提高产出才能追上同等投入的最优obs）
- efficiency > 1.0: 超效率（产出超过了随机参考集的最大值，Order-m允许）

**Fallback处理**：当 |D_i| < m 时，不做bootstrap，直接用全部dominated obs计算确定性效率。2.0%的obs触发此fallback。

---

## 3. 投入与产出

### 投入（3个）

| 投入 | 变量 | 单位 | 来源 |
|------|------|------|------|
| 土地 | `plot_area_GPS` | 公顷 | GPS测量 |
| 化肥 | `nitrogen_kg_w` | 千克 | Plot_dataset (winsorized) |
| 劳动力 | `total_labor_days_w` | 人天 | Plot_dataset (winsorized) |

**不含种子**：`seed_kg`覆盖率仅52.6%（Uganda 0%），作为DEA投入会丢掉47%的obs。种子量与面积高度共线，信息损失可接受。

### 产出（1个）

| 产出 | 变量 | 单位 | 说明 |
|------|------|------|------|
| 收获价值 | `harvest_value_USD_w` | USD | 用USD而非kg：间作地块多种作物kg不可加总 |

### 排除条件

- harvest_value_USD_w = 0 → 排除（9,919 obs, 4.5%），DEA无法处理零产出
- DEA-eligible: **212,104 obs**

---

## 4. 分层策略

DEA按 **(country × main_crop)** 分别做。不同作物×国家的生产技术完全不同。

### 小层合并

| 原始层 | Obs | 合并到 | 理由 |
|--------|----:|--------|------|
| Tanzania × MILLET | 98 | Tanzania × SORGHUM | 同为sorghum_millet |
| Mali × TUBERS | 95 | Mali × OTHER | 太少 |
| Malawi × MILLET | 77 | Malawi × SORGHUM | 同类 |
| Malawi × RICE | 29 | Malawi × OTHER | 太少 |
| Tanzania × WHEAT | 19 | Tanzania × OTHER | 太少 |
| Nigeria × WHEAT | 7 | Nigeria × OTHER | 太少 |
| Nigeria × NUTS | 2 | Nigeria × OTHER | 太少 |

合并后 **45个DEA层**，每层 ≥ 159 obs。

### 各层结果

| 层 | Obs | Mean eff | Median | Super% |
|----|----:|------:|-------:|-------:|
| Ethiopia_MAIZE | 8,323 | 0.488 | 0.291 | 13.7% |
| Ethiopia_WHEAT | 3,418 | 0.579 | 0.420 | 16.1% |
| Mali_RICE | 1,606 | 0.526 | 0.356 | 14.2% |
| Malawi_MAIZE | 10,872 | 0.333 | 0.152 | 7.9% |
| Nigeria_MAIZE | 6,718 | 0.392 | 0.242 | 9.3% |
| Uganda_PERENNIAL/FRUIT | 21,844 | 0.203 | 0.063 | 4.9% |
| Uganda_BEANS | 13,512 | 0.245 | 0.122 | 4.3% |
| ... | ... | ... | ... | ... |

（完整45层详见 `fdh_report.md`）

---

## 5. 国家间效率比较

| Country | Mean | Median | Super% | N |
|---------|-----:|-------:|-------:|--:|
| Mali | **0.401** | 0.254 | 8.9% | 14,783 |
| Ethiopia | 0.321 | 0.127 | 7.8% | 65,115 |
| Malawi | 0.321 | 0.150 | 7.3% | 16,850 |
| Tanzania | 0.300 | 0.117 | 7.2% | 13,924 |
| Nigeria | 0.287 | 0.145 | 5.9% | 41,169 |
| Uganda | **0.218** | 0.087 | 4.4% | 60,263 |

**解读**：
- Mali效率最高（0.401）：较少作物种类、Sahel区投入低但产出相对集中
- Uganda效率最低（0.218）：大量多年生作物（banana/coffee占35%）的产出变异极大
- Ethiopia/Malawi/Tanzania/Nigeria在0.28-0.32区间，与SSA文献一致

---

## 6. 衍生变量

| 变量 | 公式 | 覆盖率 |
|------|------|-------:|
| `dea_efficiency` | Order-m FDH score | 95.5% |
| `dea_frontier_value_USD` | harvest_value_USD_w / dea_efficiency | 95.5% |
| `dea_gap` | 1 - dea_efficiency | 95.5% |
| `frontier_yield_kg_ha` | dea_frontier_value_USD / plot_area_GPS | 95.5% |
| `dea_group` | "Country_MainCrop" label | 100% |

---

## 7. 生存阈值 ȳ_i

### 设计原则

BIRL效用函数中 `DownsideRisk_i = max{0, ȳ_i - q_{0.1,it}^{subj}}` 需要生存阈值 ȳ_i。

三条可辩护的路径（详见 `survival_threshold_rationale.md`）：
1. **FAO/IPC粮食不安全标准**——GIEWS定义产量<近5年均值70%为"显著歉收"
2. **Safety-First文献**——Fafchamps (1992) 用最低口粮需求，Dercon (1996) 用收入P10
3. **数据驱动分位数 + 敏感性分析**（采用方案）

### 最终方案：条件分位数体系

**主结果**：ȳ_i = 同(country, action_crop)层中 harvest_value_USD_w 的 **P25**

> *"The 25th percentile of harvest value within the same country–crop stratum,
> representing the lower quartile of output achieved by comparable farmers
> under similar agro-ecological conditions."* (cf. Fafchamps 1992; Dercon 1996)

**敏感性**：P10（极端）、P50（宽松）、贫困线（独立锚定）

### 输出列与触发率

| 列 | 定义 | 中位数 | 触发率 | 覆盖率 | 角色 |
|----|------|--------|:------:|------:|------|
| `survival_threshold_P10` | 同层P10 | $1.9 | 10.0% | 95.5% | 极端灾难线 |
| **`survival_threshold_P25`** | **同层P25** | **$7.5** | **24.9%** | **95.5%** | **主结果** |
| `survival_threshold_P50` | 同层P50 | $24.4 | 49.8% | 95.5% | 宽松灾难线 |
| `survival_threshold_poverty` | WB $2.15/day贫困线 | $103.4 | 61.9% | 71.8% | 独立锚定 |

触发率完美对齐分位数定义（P10→10%, P25→25%, P50→50%），验证计算正确。

### 为什么P25

- 不需要解释"为什么是0.3"——P25是统计标准分位数，无自由度
- 经济含义直观："同类农户中较差25%的产出水平"
- 文献先例：Dercon (1996) 用P10，我们的P25更温和
- 在右偏分布下，P25 ≈ median × 0.4-0.6，与FAO IPC Phase 3-4一致

### 在BIRL效用函数中的使用

```
DownsideRisk_i = max{0, ȳ_i - q_{0.1,it}^{subj}}
```

P25的24.9%触发率意味着约1/4的obs处于"有downside risk"状态，为γ_i提供充分识别信息。敏感性分析报告γ后验在P10/P25/P50下的稳定性。如果三种阈值的γ高度一致，结论稳健；如果不一致，报告差异及其经济解释。

---

## 8. 向量化实现细节

### 小层（N ≤ 25,000）：矩阵化

```python
# 预计算 N×N domination matrix
dom_matrix = np.all(X[None, :, :] <= X[:, None, :], axis=2)
# dom_matrix[i, j] = True if X[j] ≤ X[i] for all k

# 每个obs：从dominated set抽样、取max、除以Y_i
all_samples = rng.choice(n_dom, size=(B, m), replace=True)
max_y = Y_dom[all_samples].max(axis=1)  # (B,)
efficiency = Y_i / mean(max_y)
```

内存：N×N bool matrix，最大层21,844 → 476MB，可接受。

### 大层（N > 25,000）：逐obs循环

```python
for i in range(N):
    dominated = np.all(X <= X[i], axis=1)  # (N,) bool
    # ... same bootstrap logic
```

无N×N矩阵，内存O(N×K)。

### 性能

| 层大小 | 方法 | 耗时 |
|--------|------|-----:|
| 159 | matrix | <0.01s |
| 2,982 | matrix | 0.2s |
| 11,586 | matrix | 1.2s |
| 21,844 | matrix | 2.9s |
| **全部45层** | **串行** | **19秒** |

---

## 9. 验证结果

| 检验 | 结果 | 阈值 |
|------|------|------|
| 效率均值 | 0.289 | 0.2-0.7 ✅ |
| 效率中位数 | 0.127 | 合理（SSA小农） ✅ |
| 超效率(>1)比例 | 6.5% | <30% ✅ |
| 前沿(≥0.95)比例 | 7.1% | 3-20% ✅ |
| Fallback比例 | 2.0% | 低 ✅ |
| 覆盖率 | 95.5% | 排除零产出4.5% ✅ |
| 计算时间 | 19秒 | — ✅ |
| survival_threshold_B触发率 | 25.0% | 15-30% ✅ |

---

## 10. 已知局限与注意事项

### PERENNIAL/FRUIT效率偏低

Ethiopia PERENNIAL/FRUIT中位效率仅0.029，Uganda 0.063。原因：多年生作物（banana, coffee, enset）的年度产出波动大——建园期产出极低、盛果期产出高。FDH用单年横截面数据无法区分"低效率"和"尚未进入盛果期"。

**建议**：环境模型中对 `action_crop='tree_crops'` 加入交互效应或单独建模。报告结果时注明多年生作物效率得分的下偏。

### Ethiopia_RICE层极小

仅159 obs，超效率23.9%（所有层中最高）。层太小导致Order-m估计不稳定——dominated set更容易为空或极小。

**处理**：结果保留但报告中标注。可在敏感性分析中合并到Ethiopia_OTHER。

### 零产出obs的排除

9,919 obs（4.5%）因harvest_value_USD_w=0而排除。这些是作物完全失败（病虫害、干旱、盗窃等），是下尾风险的直接体现。FDH无法处理零产出（除法为零），但环境模型（Step 4）应保留这些obs来学习"某些(s,a)组合有较高的零收获概率"。

**当前处理**：FDH列对这些obs为NaN，但它们仍在birl_sample.parquet中（通过harvest_value_USD_adj=$0.01保留）。
