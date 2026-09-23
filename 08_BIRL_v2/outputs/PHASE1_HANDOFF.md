# BIRL v2 · Phase 1 交接文档

日期：2026-09-20（集群时间 00:49 核对）
包路径：`/home/yushentang/NF/Formal Analysis/08_BIRL_v2/`（本地）↔ `/scratch/jsentang/birl_v2/08_BIRL_v2/`（Rorqual）
一句话状态：代码已完成并推上 Rorqual，修正后的作业链（21444911–21444916）正在运行；本地只做了静态检查（用户决定），所有真实数据计算在集群上完成。

> **2026-09-20 02:40 更新（本节覆盖下文第 3 节的作业号与状态）**
>
> - **性能 bug 已修**：`src/models.py::per_obs_params` 原来用 `rho_c[obs_country_idx]` 把 6 个国家参数 gather 到 22 万观测，反向传播变成 22 万值 scatter-add 进 6 槽（GPU 原子争用），一次梯度 2.5 秒而前向只要 1.6 毫秒。改成 one-hot 矩阵乘（`Precision.HIGHEST`，避免 TF32 抹掉参数精度、保持输入 dtype 以兼容 float64 测试路径）后梯度 8 毫秒。诊断脚本 `slurm/diag_grad.py` + `diag_grad.sbatch`（1g 切片，几分钟）以后任何改动都先跑它。
> - **门作业结果（修复后）**：`00_smoke` 21449584 退出 0：pytest 96/96、checkpoint 往返通过、SVI 回收 31/36 容差（集 A 17/18，只有 Nigeria 的 s 0.15→0.04 没找回；集 B 14/18，Uganda 的 s 0.45→0.001 同时 ρ 1.5→2.23、β 8→6.3，即 s-ρ 替代，Malawi ρ 4.5→4.23 靠近上界；集 B 的 SVI 估计点对数似然比真值差 6000 nat，说明 3000 步未收敛，正式判定看 NUTS）。`00b_timing` 21449001：4 链整卡 warmup 3.84 s/draw（含早期深树，平均 183 leapfrog），稳态 0.58 s/draw、平均 30 leapfrog（树深约 5）、每 leapfrog 21 ms，`projected_full_run_h = 1.13`，显存峰值 2.5 GB。
> - **正式作业已释放（02:38）**：`01_recover_nuts` 21449585、`02_main` 21449586、`03_gfix` 21449587、`04_smax08` 21449588。预计 02/03/04 各 1 到 1.5 小时，01 约 3 到 5 小时（2g 切片、两套真值、各 2 链 500+500）。
> - 取回结果：`bash slurm/sync_down.sh`；看状态：`bash slurm/status.sh`。

---

## 1. v2 相对 06（`birl_hier_noalpha`）改了什么、为什么

| 06 的缺陷 | v2 的改法 | 为什么 | 代码位置 |
|---|---|---|---|
| 生存底线 γ 用绝对美元约束在 [0.1, 30]，6 国里 4 国贴到上界 30（Ethiopia 29.8、Malawi 29.6、Mali 29.1、Tanzania 25.4；见 `06_BIRL_MCMC/outputs/hier_noalpha/main_country_params.csv`）| `gamma_c = s_c · m_c`，`s_c ∈ (0, S_MAX=0.6)`；`m_c` = 该国"被选行动的 q50"的中位数（Ethiopia 19.86、Malawi 19.12、Mali 129.96、Nigeria 102.74、Tanzania 26.03、Uganda 25.46 USD），由 loader 从数据算出的常数 | 各国收入尺度差 6 倍，绝对美元上界对 Mali/Nigeria 毫无意义；用收入份额 s 让底线在每个国家都可识别 | `src/models.py::country_params_from_latents`，`src/data_loader.py::compute_m_c` |
| Reward = 去均值的期望效用 EU：效用尺度随 ρ 变化，一个全局 β 无法服务所有 ρ，ρ 实际上变成每户的"温度"，与选择噪声混淆 | Reward = 确定性等价 CE（USD）/ `m_c`；CE 直接按 5 个节点的加权幂平均在 log 空间算（`log_power_mean`），**不**先算 EU 再反演 | CE 对任何 ρ 都在收入尺度上，β_c 才是真正的选择噪声参数，ρ 才能被 CE 的曲率而不是尺度识别 | `src/models.py::log_power_mean / ce_from_nodes / compute_logits` |
| 3.1 万个户级潜变量不可识别；NUTS 6.9 s/step（`06/outputs/hier_noalpha/timing.json`）| Phase 1 只保留国家级参数：6 × {ρ, s, β} + 6 个超参；默认 centred（国家参数由数据主导），`--noncentered` / `--flat-priors` 可选 | 先把可识别的 6 国结构做对，户级异质性留到 Phase 2 | `src/models.py::_country_params / v2_country / v2_country_gfix` |
| 硬底线 `max(Y − γ, ε)` 有折点，梯度不连续 | 平滑底线 `surplus = ε + ½(d + √(d² + k²))`，`d = Y − γ − ε`，`k = ε = 0.02·m_c`：C¹、≥ ε、远离折点时等于 `max(Y − γ, ε)`；d<0 用无抵消的等价形式 | NUTS 需要连续梯度；平滑项自身偏差 k²/(4d) 可解析给出（T1 的 2e-3 容差来源）| `src/models.py::smooth_surplus` |
| 幂平均在 ρ≈1 处用 max-shift logsumexp，float32 抵消 ∝ 1/p，只能在 |p|=0.02 切到 Taylor，切换处 ~1e-4 相对跳变 | 中心化直接分支 `m + log1p(Σ w_k expm1(p c_k))/p`（对所有 p≠0 精确，float32 到 |p|=1e-3 仍 ≤ 3e-7 相对误差）；`P_TAYLOR = 1e-3`；Taylor 分支带三阶累积量项 | 似然在 ρ_c 上连续到 ~1e-3 nat，NUTS 轨迹跨过 ρ=1 时没有势能台阶（fix report 的 HIGH 项）| `src/models.py::log_power_mean`，`src/config.py::P_TAYLOR` |
| 输出路径写死在笔记本、无 checkpoint 一致性检查 | 路径相对包目录（`BIRL_DATA_DIR / BIRL_OUT_DIR` 可覆盖）；warmup 状态 + 每 250 draw 的 checkpoint，pkl 为权威（npz 超前则截断）；`--resume` 显式；JAX 持久编译缓存 | 24 h 墙钟内可能跑不完，必须能续跑；同一命令重复提交即可 | `src/config.py`，`src/mcmc_runner.py::run_mcmc_chunked` |
| 诊断只有 Gelman–Rubin 和一个 ρ–γ 相关 | rank-normalised split r̂、bulk/tail ESS（Vehtari 2021）；逐国 corr(ρ,s)、corr(logβ,ρ)、corr(logβ,s)；P(s_c>0.55)；底线占比、p(chosen)<1e-3 占比；Spearman CE(0.3) vs CE(4.5)；float32 分辨率守卫；按 crop×intensity 的 PPC | 这些正是 spec critique 提出的每个 caveat 对应的判据（见第 5 节）| `src/diagnostics.py` |

其余保持与 06 一致：5 节点求积权重 (.1,.2,.4,.2,.1)、`LOG_CLIP=20`、经验可行集掩码（country×zone）、reward 在可行行动上去均值、`obs_action ~ Categorical(β·reward_c)`。`LOG_BETA_HI` 从 5 提到 6。

---

## 2. 验证：本地做了什么、每个 Rorqual 作业验证什么

### 2.1 本地（用户决定：只做静态检查，不加载 222K 行数据、不跑 pytest / timing / recovery / MCMC）

| 检查 | 结果 |
|---|---|
| `python3 -m py_compile` 全部 11 个 .py（`run_v2.py`、`simulate_recover.py`、`src/*.py`、`tests/*.py`）| OK |
| `bash -n` 全部 `slurm/*.sh`、`*.sbatch` | OK |
| 20 行 × 27 行动的合成数据 import 检查（CPU、缓存关闭）| logits 有限、不可行 = −1e10、逐行去均值 4e-7、模拟行动全部可行、loglik 有限 |
| `log_power_mean` vs float64 参考（reviewer 的极端单元 q=0.01/6/1e4、s=0.3、m=20）| 17 个 p 值（含 ±1.01e-3 / ±0.99e-3 切换两侧、0、±4）相对误差 ≤ 2.9e-7 |
| T1–T5、T7 的 toy 版本手工调用（`_t2_check`、`_t2_loglik_check`（x64 路径）、`_t3_check`×7 ρ、`_t4_check`×5 ρ 含 lb=7、`_t5_check`）| 全部通过 |
| 四种参数化（centred / non-centred / flat / gfix）的 numpyro trace | 站点名一致 |
| `choose_chain_method` 决策（cpu/4dev→sequential、显式 parallel→parallel、gpu/1dev→vectorized、gpu/4dev→parallel）| 符合预期 |
| `write_all_diagnostics` 在假的 2×8 后验上 | 所有文件写出、JSON 可序列化、npz 往返 OK |
| `run_v2.py --help` / `simulate_recover.py --help` | 可解析 |

**本地没有验证的**：真实数据上的 T2/T3/T6、timing、recovery、MCMC 收敛——全部交给下面的作业。

### 2.2 每个 Rorqual 作业验证什么

| 作业 | 资源 / 墙钟 | 做什么 | 验证的问题 | 产出 |
|---|---|---|---|---|
| `00_smoke` | MIG `h100_1g.10gb`，4 cpu，32G，3 h | ① `pytest -q tests/`（toy + realdata：T1–T7 在**全部** 222K×27 单元上）；② checkpoint/resume 往返（`run_v2.py --chains 2 --warmup 5 --samples 4 --chunk 2 --tag ckpt`：`--stop-after-chunk 1` → `--resume` → 再 `--resume`，grep 日志确认 `Resumed from checkpoint: 2/4`、`Chunk 2:`、`[SKIP] MCMC: loaded existing`、pkl 已清理）；③ `simulate_recover.py --mode svi`（AutoMVN 点估计冒烟）| 数学正确性（切换连续性、float64 参考、梯度有限、m_c）；24 h 作业赖以生存的续跑路径；模型在真值下能否被 SVI 大致找回；ptxas 12.6 警告数必须为 0 | `outputs/v2_country_ckpt/`、`outputs/recovery/svi/report.{md,json}`，退出码 = 01–04 的 afterok 门 |
| `00b_timing` | 1 × 完整 H100，8 cpu，64G，2 h | `run_v2.py --variant v2_country --timing --chains 4`：50 warmup（collect_warmup）+ 20 采样（含编译）+ 20 采样（同状态、编译已缓存）| 主作业**同布局**（4 链 vectorized、同 GPU）下的 `s_per_leapfrog_steady`、`projected_full_run_h`、`peak_device_gb`——决定 02–04 能否在 24 h 内完成 | `outputs/v2_country_timing/timing.json`、`run.log` 里的 `TIMING …` 行 |
| `01_recover_nuts` | MIG `h100_2g.20gb`，12 h | `simulate_recover.py --mode nuts --sets A B --nuts-chains 2 --nuts-warmup 500 --nuts-samples 500 --resume`（flat priors）| **PASS 判决**：真值集 A（内点）和 B（近边界）在真实 q 数组上模拟行动后能否被找回 | `outputs/recovery/nuts/report.{md,json}`、`nuts_posterior_{A,B}.npz` |
| `02_main` | 1 × H100，24 h | `run_v2.py --variant v2_country --chains 4 --warmup 1000 --samples 1000 --chunk 250 --resume` | 主结果：s_c 是否离开边界、r̂/ESS/散度、ρ 排序 | `outputs/v2_country/{posterior.npz, summary.csv, convergence.txt, model_diagnostics.json, ppc_*.csv}` |
| `03_gfix` | 同上 | `--variant v2_country_gfix --s-fixed 0.3` | 识别对照：固定 s=0.3 后 ρ、β 是否稳定（s–ρ 替代性检验）| `outputs/v2_country_gfix/` |
| `04_smax08` | 同上 | `--variant v2_country --s-max 0.8` | 0.6 上界是否 binding | `outputs/v2_country_smax08/` |

所有 sbatch 共有：`--account=def-zhiming_gpu`、`module load StdEnv/2023 python/3.11 cuda/12.9 cudnn/9.13.1.26 arrow/25.0.0`、venv 激活、`export BIRL_HOST_DEVICES=1`、开跑前 `assert jax.devices()[0].platform == 'gpu'`。软件栈：jax = jaxlib = jax_cuda12_plugin = jax_cuda12_pjrt = 0.10.2，numpyro 0.21.0，numpy 2.4.2，pandas 2.3.3，pyarrow 来自 arrow 模块。

---

## 3. Rorqual 当前状态

### 3.1 作业链（第二次提交，2026-09-20 00:45:15）

```
00_smoke         21444911   RUNNING  rg12803  (h100_1g.10gb, 3 h)
00b_timing       21444912   RUNNING  rg31702  (h100 full,    2 h)
01_recover_nuts  21444913   PD  HELD  afterok:21444911:21444912   (2g.20gb, 12 h)
02_main          21444914   PD  HELD  afterok:21444911:21444912   (h100, 24 h)
03_gfix          21444915   PD  HELD  afterok:21444911:21444912   (h100, 24 h)
04_smax08        21444916   PD  HELD  afterok:21444911:21444912   (h100, 24 h)
```

依赖语义：01–04 同时满足两个条件才会开始——(a) 00 与 00b 都以 0 退出（afterok，`kill_invalid_depend` 开着：任一失败则 01–04 自动取消）；(b) **人工 `scontrol release`**（submit_chain.sh 默认 `sbatch -H`，目的是在读完 TIMING 行之前不让三个 24 h 的 H100 作业自动开跑）。

第一次提交的链 21443250–21443254 已 `scancel`（原因：smoke 里 1 链 50+20 的 timing 在 1g 切片上 70 min 没跑完 70 个 draw；01 只有 3 h、02–04 只有 12 h 没有余量；cuda/12.6 的 ptxas 12.6.77 有 clamping 误编译警告，而似然的梯度路径用了 `jnp.clip`）。旧的部分输出移到 `outputs/v2_country_timing_cancelled_21443250/`，旧的 `.jax_cache`（在 ptxas 12.6 下编译）已删。

### 3.2 smoke / timing 进度（截至 00:49:20 集群时间）

- `00_smoke` 21444911：步骤 ① **pytest 96/96 passed（99 s）**；步骤 ② 进行中：warmup 91 s 完成、`warmup_state.pkl` 已写、`Chunk 1: 2 draws in 36.2s`、`[STOP] stop_after_chunk=1` 已触发，正在跑第二次 `--resume`。（Chunk 1 显示 div=2/4 是 5 步 warmup 没有 adaptation 的正常现象，不是问题。）步骤 ③ SVI 未开始。本日志中 `ptxas with version 12.6` 警告计数 = **0**（旧 smoke 21443250 日志里是 1，证实模块切换生效）。
- `00b_timing` 21444912：数据加载后 RSS 2.90 GB，`chain_method=vectorized (1 gpu device(s) < 4 chains, est. 0.72 GB per chain)`，正在编译 NUTS 内核（XLA 对 `reduce s32[222023]` 常量折叠 > 1 s 的警告是已知现象，编译慢但无害）。**TIMING 行尚未出来。**
- Push report 的监控（30 min，会续到 ~45 min）已布好；01–04 未释放。

### 3.3 命令（都在笔记本 `08_BIRL_v2/` 目录下执行，脚本自动 ssh 到 rorqual；ssh 连接窗口已开）

```bash
# 看队列 + sacct + 已出的 TIMING / verdict / 收敛一行 + 各日志尾部（N 行）
bash slurm/status.sh          # 默认 12 行
bash slurm/status.sh 40

# 直接在 rorqual 上
ssh rorqual 'squeue -u $USER -o "%.10i %.16j %.4t %.10M %.11l %E %R"'
ssh rorqual 'sacct -u $USER -X -j 21444911,21444912,21444913,21444914,21444915,21444916 --format=JobID,JobName%16,State,Elapsed,Timelimit,ExitCode'
ssh rorqual 'cd /scratch/jsentang/birl_v2/08_BIRL_v2 && tail -n 30 outputs/slurm_logs/00_smoke-21444911.out'
ssh rorqual 'cd /scratch/jsentang/birl_v2/08_BIRL_v2 && grep -h "^TIMING" outputs/v2_country_timing/run.log; cat outputs/v2_country_timing/timing.json'

# 00 与 00b 都完成、读过 TIMING 行之后，释放 01–04（二选一）
bash slurm/submit_chain.sh --release
ssh rorqual 'scontrol release 21444913 21444914 21444915 21444916'

# 取回结果（跳过 .jax_cache 与进行中的 checkpoint；-n 为 dry run）
bash slurm/sync_down.sh
# 结果落在本地 08_BIRL_v2/outputs/{v2_country,v2_country_gfix,v2_country_smax08,recovery/{svi,nuts},v2_country_timing,slurm_logs}/

# 某个 24 h 作业 TIMEOUT 后续跑（同一脚本、同一 --gres；--resume 已写在命令里）
ssh rorqual 'cd /scratch/jsentang/birl_v2/08_BIRL_v2 && sbatch slurm/02_main.sbatch'
# 或
bash slurm/submit_chain.sh --skip-smoke --only 02_main

# 注意：作业在队列里时 sync_up.sh 会拒绝同步（rsync --delete 会改掉待运行脚本 import 的代码）；确需覆盖用 --force。
# 本文档在 outputs/ 下，sync_up.sh 本来就不会上传它。
```

释放前要看的数字：`timing.json` 的 `projected_full_run_h`（4 链 vectorized、1000+1000、同一块 H100 = 主作业自己的布局；warmup 项用的是前 50 个 draw 的平均树深，是上界）。经验规则：× ~1.3 后 < 24 h 直接释放；在 20–30 h 之间也可以释放，靠 `warmup_state.pkl` + 250-draw checkpoint 在第二次 24 h 提交里跑完；明显 > 30 h 则先考虑降 `--samples`、拆链或换 `max_tree_depth`（需改代码，不建议在不看 `mean_leapfrog_per_draw` 之前动）。

---

## 4. Phase 1 go/no-go 门槛与读法

### 4.1 门槛

| # | 门槛 | 判据 | 从哪读 |
|---|---|---|---|
| G0 | `00_smoke` 退出 0，`00b_timing` 给出有限的 TIMING 行 | sacct State=COMPLETED, ExitCode 0:0；日志末尾 `SMOKE exit=0`；ptxas-12.6 警告计数 0 | `outputs/slurm_logs/00_smoke-21444911.out`、`00b_timing-21444912.out` |
| G1 | 模拟找回 **PASS** | 两个真值集 A、B 的每个国家：\|Δρ\| ≤ 0.25、\|Δs\| ≤ 0.05、\|Δlogβ\| ≤ 0.2（后验中位数 vs 真值），**且** 36 个真值中 ≥ 80% 落在各自 89% HPDI 内 | `outputs/recovery/nuts/report.md` 首行 `**Verdict: PASS**`；`report.json["verdict"]` |
| G2 | 底线被识别而非贴边 | `v2_country` 中 6 个 `s_c` 后验中位数全部在 **[0.05, 0.55]**，且 `P(s_c > 0.55)` 小（convergence.txt 里 > 0.1 标 WARN）| `summary.csv` 的 `param == s_c` 行；`model_diagnostics.json["p_s_gt_055"]` |
| G3 | 采样收敛 | 所有站点 r̂ < 1.01（rank-normalised）；散度率 < 1%（convergence.txt 用 0.01 判 PASS/FAIL，目标是 0）；bulk / tail ESS > 400（WARN 级）| `summary.csv` 的 `r_hat / ess_bulk / ess_tail` 列；`model_diagnostics.json["convergence"]`、`["divergences"]` |
| G4 | ρ 的国家排序稳定 | `v2_country`、`v2_country_gfix`、06 三者的 ρ_c 排序一致（06 的均值：Malawi 3.30 > Mali 3.02 > Tanzania 2.96 > Ethiopia 2.39 > Uganda 1.62 > Nigeria 1.35，见 `06_BIRL_MCMC/outputs/hier_noalpha/main_country_params.csv`；但注意 06 的 ρ 混了温度效应，只比排序不比数值）；`v2_country_smax08` 复现 `v2_country`（0.6 上界不 binding）| 三个 `summary.csv` 的 `rho_c` 行 |

四个门都过 → Phase 1 结束，进入 Phase 2；任一不过 → 第 6 节。

### 4.2 `summary.csv` 怎么读

一行一个 (param, country)：`param, country, median, mean, sd, hpdi_lo, hpdi_hi, r_hat, ess_bulk, ess_tail, n_chains, n_draws`。超参（`mu_rho, sigma_rho, mu_s, sigma_s, mu_lb, sigma_lb`）的 country 为空；国家站点有 `rho_c, s_c, s_lat_c, gamma_c, beta_c, rho_lat_c, lb_c` 各 6 行（gfix 变体没有 `s_lat_c` 的采样，但 `s_c`、`gamma_c` 仍作为 deterministic 出现，为常数，r̂/ESS 为 NaN 属正常）。

- G2 看 `s_c` 六行的 `median` 与 `hpdi_hi`；`gamma_c`（USD）= s_c·m_c 用来和 06 的 γ 对照。
- G3 看 `r_hat` 列的最大值与 `ess_*` 的最小值；`lb_c` 的 r̂ 若略高而 `beta_c` 正常，通常是 clip 边界附近的量。
- G4 按 `rho_c` 的 `median` 排序；HPDI 重叠的相邻国家不算"排序变化"。
- `convergence.txt` 把上述都排好并标了 PASS/FAIL/WARN，先看它，再回 csv 查数。

### 4.3 `model_diagnostics.json` 怎么读

```
run            variant, n_chains, n_draws, m_c, platform/device, chain_method, mcmc_minutes, peak_device_gb
divergences    count / total / rate                       → G3
convergence    r_hat_max, ess_bulk_min, ess_tail_min      → G3
posterior_median  rho_c / s_c / beta_c 每国               → G2、G4
params         summary.csv 的字典形式
correlations.per_country[country]  rho_s, logbeta_rho, logbeta_s   → 第 5 节 caveat 1、2
correlations.hyper                  mu_rho_mu_s, mu_rho_mu_lb
p_s_gt_055[country]                                       → G2
fit_at_median  loglik, mean_logp_chosen, floor_share_chosen, p_chosen_lt_1e-3（总体 + per_country），
               max_reward, max_abs_logit(+cell), float32_logit_resolution_nat, q90_max_usd   → 第 5 节 caveat 3
spearman_ce_rho[country], overall                         → 第 5 节 caveat 1
ppc            n_draws, max_abs_diff, mean_abs_diff, per_country_max_abs_diff
```

### 4.4 recovery report 怎么读

`outputs/recovery/nuts/report.md`：头部给日期、平台、先验（flat）、NUTS 配置、容差；`**Verdict: PASS|FAIL|INCOMPLETE** - k/36 tolerances met (need all), n/36 = xx% truths inside 89% HPDI (need >= 80%)`。随后每个真值集一节：真值、模拟行动与观测行动的一致率、log-lik(truth) vs log-lik(median)（估计值的 log-lik 应 ≥ 真值的，差值为负且大说明没找到众数），再是一张表 `param | country | truth | median | sd | z | 89% HPDI | in HPDI | abs err | tol | ok`。`gamma_c` 行只报告不判。`INCOMPLETE` = 只跑了一个集（12 h 超时后重新提交同一脚本，`--resume` 会加载已完成的 `nuts_posterior_<set>.npz` 并合并判决）。`outputs/recovery/svi/report.md` 是冒烟版（verdict 固定为 SMOKE），只看点估计大致对不对，不作为门槛。

---

## 5. spec critique 中仍未关闭的 caveat 及对应诊断

| caveat | 含义 | 已做的缓解 | 回答它的诊断（都在 `model_diagnostics.json` / `convergence.txt`）| 判断线 |
|---|---|---|---|---|
| **β–ρ 脊**（β 与 ρ 沿一条脊线互补）| 若 ρ 只是把 CE 整体缩放（像温度），则 β 与 ρ 不可分辨，后验沿脊线拉长 | CE 在收入尺度上、reward 去均值，理论上打断了尺度耦合；但曲率本身随 ρ 变化仍可能残留相关 | `correlations.per_country[c].logbeta_rho`（\|·\| > 0.8 标 WARN）；`spearman_ce_rho[c]`：ρ=0.3 与 ρ=4.5 下可行行动 CE 的 Spearman 相关（> 0.95 标 WARN，意味着 ρ 几乎不改变行动排序、只改尺度）；`summary.csv` 里 `rho_c` 与 `beta_c` 的 ESS 是否同时偏低 | 相关 \|·\| ≤ 0.8 且 Spearman ≤ 0.95 视为可接受；否则 ρ 的数值不可信，只能用排序，并看 03_gfix 中 ρ 是否稳定 |
| **s–ρ 替代**（底线份额 s 与风险厌恶 ρ 互相替代）| 提高 s 压低低收入节点的 surplus，与提高 ρ 对下尾的惩罚效果相似 | s 有上界 0.6 与 P(s>0.55) 监控；03_gfix 固定 s=0.3 作为对照 | `correlations.per_country[c].rho_s`（WARN 线 0.8）；`correlations.hyper.mu_rho_mu_s`；`p_s_gt_055[c]`；03_gfix 与 02_main 的 `rho_c` 中位数之差（若固定 s 后 ρ 移动超过其 HPDI 宽度，说明替代性强）；04_smax08 的 `s_c` 是否越过 0.6 | corr(ρ,s) ≤ 0.8、gfix 的 ρ 排序不变、smax08 的 s 不越 0.6 |
| **reward 的重右尾**（CE/m_c 的极端值）| 少数 q90 极大的单元给出巨大 reward，β·reward 的 logit 在 float32 下分辨率不足，且这些单元主导似然 | loader 对 q > 1e5 USD 报警；`LOG_CLIP=20` 保留（未降到 11.5，避免悄悄改值）；诊断里记录极值单元 | `fit_at_median.max_reward`、`max_abs_logit`（> 1e4 标 WARN）、`max_abs_logit_cell`（obs/action/country）、`float32_logit_resolution_nat`（= max\|logit\|·2⁻²³，应 ≪ 0.1 nat）、`q90_max_usd` 每国；`fit_at_median.p_chosen_lt_1e-3`（被选行动概率 < 1e-3 的占比，总体 + 每国）；`floor_share_chosen`（被选行动 5 个节点全在底线内的占比）；PPC 的 `per_country_max_abs_diff` | max\|logit\| ≤ 1e4、p<1e-3 占比接近 06 水平且各国无异常、PPC 各国最大偏差 ≲ 0.05；若 WARN，记录该 cell，再决定是否降 LOG_CLIP 或 winsorise q90（Phase 2 事项，不在 Phase 1 改）|

另外两个已在实现中处理、但仍值得在结果里核对的点：Taylor 切换连续性（T2 在全部 222K×27 单元上通过即关闭；若 r̂ 在 `rho_c` 接近 1 的国家异常，回看 `run.log` 的散度位置）；`lb_c` 的硬 clip 在 [−4, 6]（`beta_c` 的 `hpdi_hi` 若贴 403 = e⁶，说明 β 上界 binding，需提高 `LOG_BETA_HI`）。

---

## 6. 门槛失败后怎么办

| 失败的门 | 首先做 | 然后 |
|---|---|---|
| G0：smoke 失败（pytest / ckpt 往返 / SVI 任一）| 看 `00_smoke-<id>.out` 中 `!!` 行；01–04 会被自动取消 | 修代码 → 等队列清空 → `bash slurm/sync_up.sh` → `bash slurm/submit_chain.sh` 重新提交整条链 |
| G0：00b 的 `projected_full_run_h` 过大 | 看 `phases.warmup.mean_leapfrog_per_draw` 与 `phases.sampling_steady.mean_leapfrog_per_draw`：若接近 1023（= 2¹⁰−1）说明树深打满，问题在后验几何 | 释放并靠 `--resume` 两次 24 h 完成；或先释放 03_gfix（参数少、几何简单）看它的深度；若仍打满，考虑 `--noncentered` 或降 `--samples` |
| G1：recovery FAIL | 看 `report.md` 哪些行 FAIL：若集 B 的近边界 s（0.05/0.58）失败而 A 全过，是 sigmoid 边界的先验/几何问题，不是模型错；若 β 系统偏高/低，回看 caveat 3 的 `max_abs_logit` | 只在 B 的边界行失败 → 记录并继续（Phase 1 结论按 A 判，B 作为已知限制）；A 也失败 → 停，检查 `compute_logits` 与模拟的一致性（同一函数，先排除数据/mask 差异），再考虑 `--hier-priors` 重跑 recovery |
| G2：某国 s_c 贴 0.55 以上或 P(s>0.55) > 0.1 | 看 04_smax08 中该国的 `s_c`：若跑到 0.6–0.8 之间稳定下来，是 0.6 上界 binding，用 smax08 的结果；若继续贴 0.8，是 s 不可识别（与 ρ 替代）| 用 **03_gfix 作为识别对照**：固定 s=0.3 后 ρ、β 的排序与 HPDI 若与 02_main 一致，Phase 1 的 ρ 结论仍成立，s 只报告区间；否则 Phase 1 不能给出 s 的点估计，转 Phase 2 |
| G2：某国 s_c < 0.05 | 底线对该国不起作用（Mali/Nigeria 收入高时可能如此）| 可接受，记录；gfix 的 s=0.3 对该国会是错误约束，比较时剔除 |
| G3：r̂ > 1.01 / 散度 > 0 | `run.log` 里散度出现在哪个 chunk、哪条链；`correlations` 看是否是 ρ–s 或 β–ρ 脊 | 先 `--noncentered`（同一脚本加 flag，自动 tag `_nc`）重跑主变体；若散度集中在 s 边界，改 `--s-max 0.8`；仍不行则提高 `target_accept`（需改 `mcmc_runner.TARGET_ACCEPT`）|
| G3：ESS < 400 但 r̂ 正常 | 链只是短 | 再提交一次 `--samples 2000 --resume`（checkpoint 的 `n_samples_target` 不同会被判 mismatch，需先删 `mcmc_state.pkl`，或直接改 `--tag` 新目录跑）|
| G4：ρ 排序在 v2_country 与 gfix 之间变化 | 说明 s–ρ 替代性强 | 报告两套排序并以 gfix 为主（s 固定后 ρ 的含义单一）；把它列为 Phase 2 必须解决的问题 |
| G4：v2 与 06 的排序不同 | 06 的 ρ 混了温度效应，不同是**预期内**的 | 不以此判 Phase 1 失败；在 `spearman_ce_rho` 高（> 0.95）的国家里，06 与 v2 的差就是温度混淆的证据，写进结果说明 |

**Phase 2 的方向（任一门失败或全部通过后）**：welfare 的重写——从 Phase 1 的国家级 (ρ, s, β) 出发，把 07_2050_Counter_Fact 里的福利指标改为基于 CE 而非 EU（CE 在美元尺度上可跨国、跨 ρ 直接加总），再决定是否恢复户级异质性（非中心化、部分池化，且只对 β 或只对 s 放开一个维度，避免回到 06 的 3.1 万个不可识别潜变量）。这一步不改 `08_BIRL_v2/` 的模型代码，新开 `09_` 目录。

---

## 附：文件与路径速查

```
本地
  08_BIRL_v2/README.md                         模型、测试表、门槛、集群栈
  08_BIRL_v2/slurm/README.md                   作业含义、预期时长、输出释义、版本选择依据
  08_BIRL_v2/slurm/env.sh                      用户/账户/路径/模块/pin（改 REMOTE_ROOT 时同步改 6 个 sbatch 的 --output）
  08_BIRL_v2/src/{config,models,data_loader,mcmc_runner,diagnostics}.py
  08_BIRL_v2/tests/{conftest,test_models}.py    T1–T7 + guards（-m toy 本地可跑，用户决定不跑）
  08_BIRL_v2/outputs/PHASE1_HANDOFF.md         本文
  06_BIRL_MCMC/outputs/hier_noalpha/{main_country_params.csv, timing.json}   G4 对照
Rorqual
  /scratch/jsentang/birl_v2/08_BIRL_v2/outputs/slurm_logs/<name>-<jobid>.out
  /scratch/jsentang/birl_v2/08_BIRL_v2/outputs/slurm_logs/jobids.txt        两次提交的 job id
  /scratch/jsentang/birl_v2/08_BIRL_v2/outputs/v2_country_timing/timing.json
  /scratch/jsentang/birl_v2/08_BIRL_v2/outputs/recovery/{svi,nuts}/report.md
  /scratch/jsentang/birl_v2/08_BIRL_v2/outputs/{v2_country,v2_country_gfix,v2_country_smax08}/
  /scratch/jsentang/birl_v2/06_BIRL_MCMC/data/   三个 06 数据文件（103 MB）
  /scratch/jsentang/birl_v2/venv                 setup_venv.sh 建好的环境
```
