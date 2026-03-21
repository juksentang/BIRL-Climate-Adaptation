# DEA Frontier Estimation Report (Order-m FDH)

输入: 222,023 obs

## 1. 数据准备

- 零产出排除: 9,919 obs (4.5%)
- 合并: Tanzania×MILLET (98) → Tanzania×SORGHUM
- 合并: Mali×TUBERS / ROOT CROPS (95) → Mali×OTHER
- 合并: Malawi×MILLET (77) → Malawi×SORGHUM
- 合并: Malawi×RICE (29) → Malawi×OTHER
- 合并: Tanzania×WHEAT (19) → Tanzania×OTHER
- 合并: Nigeria×WHEAT (7) → Nigeria×OTHER
- 合并: Nigeria×NUTS (2) → Nigeria×OTHER

## 2. Order-m FDH (m=25, B=200)

| Layer | Obs | Time(s) | Mean eff | Median | Super% | Fallback |
|-------|----:|--------:|------:|-------:|-------:|---------:|
| Ethiopia_BARLEY | 2,982 | 0.2 | 0.429 | 0.290 | 10.2% | 73 |
| Ethiopia_BEANS AND OTHER LEGUMES | 5,745 | 0.4 | 0.370 | 0.200 | 8.1% | 69 |
| Ethiopia_MAIZE | 8,323 | 0.7 | 0.488 | 0.291 | 13.7% | 79 |
| Ethiopia_MILLET | 1,181 | 0.1 | 0.515 | 0.409 | 13.5% | 81 |
| Ethiopia_OTHER | 11,586 | 1.1 | 0.312 | 0.153 | 5.8% | 72 |
| Ethiopia_PERENNIAL/FRUIT | 21,250 | 3.4 | 0.178 | 0.029 | 4.5% | 98 |
| Ethiopia_RICE | 159 | 0.0 | 0.652 | 0.556 | 23.9% | 68 |
| Ethiopia_SORGHUM | 6,135 | 0.5 | 0.408 | 0.249 | 9.7% | 79 |
| Ethiopia_TUBERS / ROOT CROPS | 4,336 | 0.3 | 0.192 | 0.024 | 5.4% | 80 |
| Ethiopia_WHEAT | 3,418 | 0.2 | 0.579 | 0.420 | 16.1% | 80 |
| Malawi_BEANS AND OTHER LEGUMES | 2,897 | 0.2 | 0.271 | 0.140 | 5.1% | 112 |
| Malawi_MAIZE | 10,872 | 0.9 | 0.333 | 0.152 | 7.9% | 87 |
| Malawi_OTHER | 1,154 | 0.1 | 0.420 | 0.213 | 10.9% | 94 |
| Malawi_PERENNIAL/FRUIT | 1,376 | 0.1 | 0.217 | 0.112 | 3.5% | 2 |
| Malawi_SORGHUM | 225 | 0.0 | 0.394 | 0.280 | 10.2% | 57 |
| Malawi_TUBERS / ROOT CROPS | 326 | 0.0 | 0.378 | 0.261 | 10.1% | 56 |
| Mali_BEANS AND OTHER LEGUMES | 734 | 0.0 | 0.456 | 0.201 | 13.2% | 83 |
| Mali_MAIZE | 2,278 | 0.1 | 0.461 | 0.330 | 10.7% | 89 |
| Mali_MILLET | 3,937 | 0.3 | 0.336 | 0.211 | 6.2% | 94 |
| Mali_NUTS | 2,727 | 0.2 | 0.351 | 0.223 | 6.5% | 99 |
| Mali_OTHER | 962 | 0.1 | 0.440 | 0.281 | 11.2% | 68 |
| Mali_RICE | 1,606 | 0.1 | 0.526 | 0.356 | 14.2% | 97 |
| Mali_SORGHUM | 2,539 | 0.1 | 0.392 | 0.263 | 8.3% | 101 |
| Nigeria_BEANS AND OTHER LEGUMES | 5,528 | 0.4 | 0.258 | 0.147 | 4.0% | 140 |
| Nigeria_MAIZE | 6,718 | 0.5 | 0.392 | 0.242 | 9.3% | 161 |
| Nigeria_MILLET | 5,401 | 0.3 | 0.333 | 0.210 | 5.9% | 177 |
| Nigeria_OTHER | 2,161 | 0.1 | 0.183 | 0.073 | 3.4% | 108 |
| Nigeria_PERENNIAL/FRUIT | 1,895 | 0.1 | 0.269 | 0.086 | 6.6% | 118 |
| Nigeria_RICE | 1,856 | 0.1 | 0.423 | 0.292 | 8.7% | 182 |
| Nigeria_SORGHUM | 5,113 | 0.3 | 0.333 | 0.212 | 6.1% | 212 |
| Nigeria_TUBERS / ROOT CROPS | 12,497 | 1.1 | 0.205 | 0.068 | 4.8% | 116 |
| Tanzania_BEANS AND OTHER LEGUMES | 1,274 | 0.1 | 0.368 | 0.202 | 8.8% | 94 |
| Tanzania_MAIZE | 4,053 | 0.2 | 0.392 | 0.213 | 9.5% | 67 |
| Tanzania_NUTS | 740 | 0.0 | 0.220 | 0.080 | 5.0% | 56 |
| Tanzania_OTHER | 5,269 | 0.3 | 0.201 | 0.051 | 4.4% | 50 |
| Tanzania_RICE | 781 | 0.0 | 0.421 | 0.219 | 13.8% | 76 |
| Tanzania_SORGHUM | 351 | 0.0 | 0.430 | 0.281 | 11.1% | 40 |
| Tanzania_TUBERS / ROOT CROPS | 1,456 | 0.1 | 0.284 | 0.132 | 6.6% | 83 |
| Uganda_BEANS AND OTHER LEGUMES | 13,512 | 1.3 | 0.245 | 0.122 | 4.3% | 87 |
| Uganda_MAIZE | 7,421 | 0.5 | 0.228 | 0.096 | 4.6% | 77 |
| Uganda_MILLET | 1,355 | 0.1 | 0.215 | 0.091 | 4.8% | 82 |
| Uganda_OTHER | 3,074 | 0.2 | 0.170 | 0.062 | 3.2% | 100 |
| Uganda_PERENNIAL/FRUIT | 21,844 | 3.2 | 0.203 | 0.063 | 4.9% | 109 |
| Uganda_RICE | 538 | 0.0 | 0.210 | 0.065 | 7.6% | 77 |
| Uganda_SORGHUM | 1,436 | 0.1 | 0.231 | 0.106 | 4.2% | 83 |
| Uganda_TUBERS / ROOT CROPS | 11,083 | 1.0 | 0.219 | 0.108 | 3.8% | 106 |

**总计: 212,104 obs, 20秒, fallback: 4,219 (2.0%)**

## 3. 衍生变量

- `dea_efficiency`: 212,104 (95.5%)
- `dea_frontier_value_USD`: 212,104 (95.5%)
- `dea_gap`: 212,104 (95.5%)
- `frontier_yield_kg_ha`: 212,104 (95.5%)

## 4. 生存阈值

- `survival_threshold_P10`: 212,078 obs, median=$1.9, trigger=10.0%
- `survival_threshold_P25`: 212,078 obs, median=$7.5, trigger=24.9%
- `survival_threshold_P50`: 212,078 obs, median=$24.4, trigger=49.8%
- `survival_threshold_poverty`: 159,457 obs, median=$103.4, trigger=61.9%

## 5. 验证

- 效率均值: 0.289
- 效率中位数: 0.127
- 超效率(>1)比例: 6.5%
- 前沿(>=0.95)比例: 7.1%

### 国家间效率

| Country | Mean | Median | Super% | N |
|---------|-----:|-------:|-------:|--:|
| Ethiopia | 0.321 | 0.127 | 7.8% | 65,115 |
| Malawi | 0.321 | 0.150 | 7.3% | 16,850 |
| Mali | 0.401 | 0.254 | 8.9% | 14,783 |
| Nigeria | 0.287 | 0.145 | 5.9% | 41,169 |
| Tanzania | 0.300 | 0.117 | 7.2% | 13,924 |
| Uganda | 0.218 | 0.087 | 4.4% | 60,263 |

## 输出

- `birl_sample.parquet`: 222,023 × 245 (46.3 MB)
- DEA效率覆盖: 212,104 obs (95.5%)
- 总计算时间: 20秒