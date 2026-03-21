# Action Space Construction Report

输入: 222,023 obs, 15,644 HH

## 1. 作物类别映射 (11 → 9)

| action_crop | 来源 | Obs | % |
|-------------|------|----:|--:|
| maize | — | 41,385 | 18.6% |
| tree_crops | — | 52,633 | 23.7% |
| tubers | — | 31,079 | 14.0% |
| legumes | — | 30,933 | 13.9% |
| sorghum_millet | — | 29,064 | 13.1% |
| teff | — | 5,522 | 2.5% |
| other | — | 19,636 | 8.8% |
| rice | — | 5,192 | 2.3% |
| wheat_barley | — | 6,579 | 3.0% |

## 2. 投入强度分档

### 高施肥作物 (PCA-based 三档)

| Country | Crop | N | P33 fert | P67 fert | P33 labor | P67 labor |
|---------|------|--:|------:|------:|-------:|-------:|
| Ethiopia | teff | 5,522 | 3.8 | 29.6 | 123.3 | 266.6 |
| Ethiopia | wheat_barley | 6,553 | 0.0 | 33.2 | 121.4 | 288.0 |
| Ethiopia | maize | 8,947 | 0.0 | 21.9 | 181.1 | 416.3 |
| Malawi | maize | 11,267 | 16.0 | 59.5 | 119.5 | 260.7 |
| Mali | maize | 2,319 | 0.0 | 55.3 | 72.6 | 234.8 |
| Nigeria | maize | 6,932 | 0.0 | 47.0 | 316.7 | 1404.0 |
| Tanzania | maize | 4,151 | 0.0 | 0.0 | 67.8 | 165.2 |
| Uganda | maize | 7,769 | 0.0 | 0.0 | 79.4 | 192.5 |
| Ethiopia | sorghum_millet | 7,987 | 0.0 | 0.0 | 129.9 | 317.5 |
| Malawi | sorghum_millet | 268 | 0.0 | 0.0 | 82.0 | 197.1 |
| Mali | sorghum_millet | 6,607 | 0.0 | 0.0 | 55.6 | 178.2 |
| Nigeria | sorghum_millet | 10,785 | 0.0 | 26.1 | 228.2 | 999.7 |
| Tanzania | sorghum_millet | 365 | 0.0 | 0.0 | 65.8 | 125.8 |
| Uganda | sorghum_millet | 3,052 | 0.0 | 0.0 | 69.8 | 188.1 |
| Ethiopia | rice | 164 | 0.0 | 39.9 | 122.2 | 248.6 |
| Mali | rice | 1,720 | 0.0 | 45.3 | 106.6 | 361.9 |
| Nigeria | rice | 1,941 | 0.0 | 55.4 | 307.9 | 1418.2 |
| Tanzania | rice | 792 | 0.0 | 0.0 | 102.5 | 243.9 |
| Uganda | rice | 546 | 0.0 | 0.0 | 87.4 | 206.1 |

### 低施肥作物 (二元施肥 × 劳动力中位数)

| Country | Crop | N | % fert>0 | Labor P50 |
|---------|------|--:|--------:|----------:|
| Ethiopia | other | 6,662 | 10.0% | 368.6 |
| Malawi | other | 1,172 | 69.7% | 227.4 |
| Mali | other | 898 | 47.8% | 148.1 |
| Nigeria | other | 2,259 | 43.1% | 698.1 |
| Tanzania | other | 5,398 | 7.1% | 116.4 |
| Uganda | other | 3,247 | 11.1% | 101.2 |
| Ethiopia | legumes | 6,164 | 19.0% | 184.6 |
| Malawi | legumes | 2,994 | 23.3% | 173.9 |
| Mali | legumes | 769 | 11.2% | 127.4 |
| Nigeria | legumes | 5,663 | 40.8% | 443.5 |
| Tanzania | legumes | 1,321 | 7.6% | 125.6 |
| Uganda | legumes | 14,022 | 2.1% | 134.7 |
| Ethiopia | tubers | 4,551 | 8.7% | 549.5 |
| Malawi | tubers | 336 | 14.3% | 208.5 |
| Mali | tubers | 95 | 32.6% | 255.6 |
| Nigeria | tubers | 13,114 | 21.4% | 1318.1 |
| Tanzania | tubers | 1,486 | 5.7% | 127.1 |
| Uganda | tubers | 11,497 | 1.8% | 126.9 |
| Ethiopia | tree_crops | 23,268 | 13.1% | 392.2 |
| Malawi | tree_crops | 1,376 | 0.0% | 0.0 |
| Mali | tree_crops | 2,803 | 6.3% | 189.5 |
| Nigeria | tree_crops | 1,996 | 8.9% | 482.2 |
| Tanzania | tree_crops | 757 | 6.5% | 59.1 |
| Uganda | tree_crops | 22,433 | 2.1% | 123.1 |

## 3. 动作编码

| action_id | Label | Obs | % |
|:---------:|-------|----:|--:|
| 0 | maize_low | 13,805 | 6.2% |
| 1 | maize_medium | 13,789 | 6.2% |
| 2 | maize_high | 13,791 | 6.2% |
| 3 | tree_crops_low | 25,130 | 11.3% |
| 4 | tree_crops_medium | 25,454 | 11.5% |
| 5 | tree_crops_high | 2,049 | 0.9% |
| 6 | tubers_low | 13,829 | 6.2% |
| 7 | tubers_medium | 15,395 | 6.9% |
| 8 | tubers_high | 1,855 | 0.8% |
| 9 | legumes_low | 13,261 | 6.0% |
| 10 | legumes_medium | 15,216 | 6.9% |
| 11 | legumes_high | 2,456 | 1.1% |
| 12 | sorghum_millet_low | 9,691 | 4.4% |
| 13 | sorghum_millet_medium | 9,688 | 4.4% |
| 14 | sorghum_millet_high | 9,685 | 4.4% |
| 15 | teff_low | 1,841 | 0.8% |
| 16 | teff_medium | 1,840 | 0.8% |
| 17 | teff_high | 1,841 | 0.8% |
| 18 | other_low | 8,082 | 3.6% |
| 19 | other_medium | 9,688 | 4.4% |
| 20 | other_high | 1,866 | 0.8% |
| 21 | rice_low | 1,723 | 0.8% |
| 22 | rice_medium | 1,753 | 0.8% |
| 23 | rice_high | 1,716 | 0.8% |
| 24 | wheat_barley_low | 2,184 | 1.0% |
| 25 | wheat_barley_medium | 2,211 | 1.0% |
| 26 | wheat_barley_high | 2,184 | 1.0% |

**理论动作: 27，活跃动作(≥50 obs): 27**

## 4. AEZ可行性掩码

AEZ缺失填补: 12,174 obs → 用country×admin_1众数

| Zone | Feasible actions | Total obs |
|:----:|:----------------:|----------:|
| 312 | 21 | 43,908 |
| 313 | 21 | 38,004 |
| 314 | 21 | 32,926 |
| 322 | 24 | 20,902 |
| 323 | 27 | 48,234 |
| 324 | 27 | 38,049 |

## 5. 验证

- 最小动作频率: 1716 (要求≥50): ✅ PASS
- 最少zone可行动作: 21 (要求≥3): ✅ PASS
- 最高频动作占比: 11.5% (要求<30%): ✅ PASS
- 全部obs有action_id: ✅ PASS

### 投入强度均衡性

| Crop | Low % | Med % | High % |
|------|------:|------:|-------:|
| maize | 33.4% | 33.3% | 33.3% |
| tree_crops | 47.7% | 48.4% | 3.9% |
| tubers | 44.5% | 49.5% | 6.0% |
| legumes | 42.9% | 49.2% | 7.9% |
| sorghum_millet | 33.3% | 33.3% | 33.3% |
| teff | 33.3% | 33.3% | 33.3% |
| other | 41.2% | 49.3% | 9.5% |
| rice | 33.2% | 33.8% | 33.1% |
| wheat_barley | 33.2% | 33.6% | 33.2% |

## 输出

- `birl_sample.parquet`: 222,023 × 236 cols
- 动作数: 27 (理论27, 活跃27)
- AEZ zones: 6
- `action_space_config.json`: saved