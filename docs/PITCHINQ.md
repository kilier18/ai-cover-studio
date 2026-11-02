# PitchIn-Q · 歌唱人声诊断（接入契约）

PitchIn-Q 是一个**按需启动、用完即走**的诊断器：给一对音频（分离后的干声 + RVC 输出），
它告诉你哪里出了问题（带时间区间）、为什么、以及 RVC 参数该往哪调。

它不是流水线的一环：**只有 DSH 主动下发诊断指令时才跑**，跑完立刻卸载模型、释放显存。

---

## 1. 它做什么 / 不做什么

| 做 | 不做 |
| --- | --- |
| 歌唱人声诊断、定位音频缺陷（标注时间区间） | 音频分离 |
| 对比源干声与 RVC 输出 | F0 提取（内部只为对齐用，不对外提供） |
| 给出参数方案：RVC 侧（index_rate / rms_mix_rate / protect / f0 算法 / pitch）+ 导出侧（eq_air / eq_presence） | 音频渲染、混音后期（只给建议，不动手） |
| 音色亮度诊断（频谱质心 vs 原唱，偏亮/偏暗两个方向） | 音高转换、去混响 |
| 输出结构化 JSON | 主动触发自己（没有定时任务、没有自动挂载） |

---

## 2. 调度规则（硬约束）

1. **不常驻显存**：默认 `--backend dsp` 只做信号分析，**完全不碰 GPU**。
2. **和 RVC 互斥加载（可配置）**：默认 `--lock-mode auto` —— 先问显卡还剩多少显存，
   够 `模型体积 × 2.2 + 900 MB` 余量就**不加锁、与 RVC 并发跑**；不够才去排队拿 `tools/gpulock.py` 的锁。
   `--lock-mode wait` 永远排队，`--lock-mode none` 完全不加锁（显存紧张时别用，会 OOM）。
3. **异步调度**：拿不到锁就排队（`--wait-gpu 秒数`），等不到返回 `3`，绝不杀别人的进程。
4. **诊断完自动卸载**：模型后端在 `finally` 里 `del model` + `gc.collect()` + `torch.cuda.empty_cache()`，
   并打印 `released_mb` 供核对。
5. **只有 DSH 主动调用才启动**：命令行或 `POST /api/diagnose`；转换流程不会自动调用它。

---

## 3. 显存锁（`tools/gpulock.py`）

RVC 和 PitchIn-Q 都要吃这张 6 GB 卡，所以用**文件锁**做跨进程互斥：

- 锁文件 `work/.gpu.lock`（`O_CREAT|O_EXCL`），持有者信息 `work/.gpu.owner.json`（pid / owner / since）。
- 持有者进程已死 → 自动作废；超过 30 分钟且进程已不在 → 视为陈旧锁清掉。
- **同一进程树可重入**：父进程拿到锁后设 `GPULOCK_HELD=1`，子进程（`rvc_convert.py`）看到就直接放行。
- 只排队，不杀进程。

三方都接了这把锁：

| 入口 | 行为 |
| --- | --- |
| `tools/webui/app.py` 的 `run_job()` | 作业开始前 `hold(f"webui:{jid}")`，失败则作业标错不启动 |
| `tools/rvc_convert.py` 的 `main()` | 自己拿锁（`GPULOCK_WAIT` 秒排队，默认 0 → 忙则退出 `3`） |
| `tools/pitchinq.py --backend model` | `hold("pitchinq", timeout=--wait-gpu)`，拿不到退出 `3` |

手工查看谁占着显存：

```bash
python tools/gpulock.py
# {"busy": true, "owner": "webui:3f2a…", "pid": 12345, "age": 12.4}
```

---

## 4. 命令行

```bash
python tools/pitchinq.py \
  --source   work/separated/歌/stems/vocals.wav \
  --converted work/converted/vocals__m6__….wav \
  --params   work/_diag/last_params.json \
  --out      work/_diag/report.json
```

| 参数 | 说明 |
| --- | --- |
| `--source` | 源干声（分离后、转换前） |
| `--converted` | RVC 输出人声 |
| `--params` | 这次转换用的参数 JSON（可选；用于算 from→to） |
| `--out` | 报告路径（默认 `work/_diag/report.json`） |
| `--backend dsp\|model` | `dsp`＝纯信号分析（默认，不占显存）；`model`＝加载神经网络 |
| `--model-path` | `model` 后端的权重（也可用环境变量 `PITCHINQ_MODEL`） |
| `--f0 yin\|pyin` | 内部音高跟踪器（`pyin` 更准更慢） |
| `--wait-gpu 秒` | 显存忙时最多排队多久；0 表示不排队 |
| `--max-seconds N` | 只分析前 N 秒（调参时快速试） |
| `--json` | 把完整报告打到 stdout |

退出码：`0` 正常；`2` 参数/权重问题；`3` 显存忙（没等到锁）。

---

## 5. HTTP 接口（网页端，给 DSH 用）

```
POST /api/diagnose
{
  "source":    "work/separated/歌/stems/vocals.wav",
  "converted": "work/converted/vocals__m6__b-0.5.wav",
  "params":    {"index_rate": 0.5, "protect": 0.4, "rms_mix_rate": 0.3,
                "f0_method": "crepe", "pitch": 0},
  "wait_gpu":  300
}
→ {"job": "ee383647c1f7", "report": "…/work/_diag/ee383647c1f7.json"}

GET /api/diagnose/<job>
→ {"status": "done", "report": { …完整报告… }, "log": [...]}
```

作业跑在后台线程里，先把结果写成 `work/_diag/<job>.json`，再返回；
`/api/diagnose/<job>` 同时给状态、日志尾巴和解析好的报告。

---

## 6. 报告格式（`schema: pitchinq.report/1`）

```jsonc
{
  "schema": "pitchinq.report/1",
  "generated_at": "2026-11-02T11:50:00+0800",
  "engine": {"name": "PitchIn-Q", "backend": "dsp", "f0_method": "yin",
             "elapsed_s": 6.04, "wall_s": 9.85},
  "audio":  {"src_seconds": 20.0, "conv_seconds": 19.98,
             "analyzed_seconds": 19.98, "sr": 44100},
  "summary": {
    "verdict": "adjust",          // ok | adjust
    "score": 85,                  // 100 满分
    "mean_cents": 19.1,           // 整体音高偏移（正=输出偏高）
    "defect_count": 2,
    "top": ["重复字 @ 5.43–5.58s", "…"]
  },
  "segments": [
    {"type": "doubled_syllable", "start": 5.43, "end": 5.58,
     "severity": "mid",                       // low | mid | high
     "detail": "输出多出一处起音（重复字，常见于和声/双轨没剥干净）",
     "values": {"onset_s": 5.43}}
  ],
  "param_plan": [
    {"param": "index_rate", "from": 0.5, "to": 0.4,
     "why": "多出来的起音多半是和声/双轨残留，降检索可减轻",
     "confidence": 0.5}
  ],
  "inputs_params": {"index_rate": 0.5}
}
```

### 缺陷类型

| type | 含义 | 判据 |
| --- | --- | --- |
| `offset` | 整体时间偏移 | 能量包络互相关 > 60 ms |
| `pitch_off` | 局部音准偏移 | 逐帧 cents 偏差 > 60 音分，持续 > 120 ms |
| `pitch_jump` | 八度/半八度跳变 | 偏差落在 1050–1350 音分 |
| `doubled_syllable` | 重复字 / 和声残留 | 输出多出起音，且离源起音 > 300 ms、源处有声 |
| `dropped_syllable` | 漏字/吞字 | 源有起音而输出没有，且该处能量低 6 dB 以上 |
| `level_jump` | 音量跳变 | 250 ms 窗 RMS 差 > 6 dB（两边都有声才算） |
| `harsh` | 齿音/金属音 | 4–12k 相对 100–4k 的能量比升高 > 3 dB |
| `over_smooth` | 过度平滑 | 逐音符颤音深度保留 < 62% 的音符占 ≥ 40% |
| `wobble` | 抖动/不稳 | 音高换向率 > 0.25 且标准差 > 90 音分 |

### 参数方案（`param_plan`）

| 触发缺陷 | 建议 |
| --- | --- |
| 整体偏移 > 35 音分 | `pitch` ±1 半音 |
| `pitch_off` / `wobble` | `protect` +0.04，`index_rate` −0.05 |
| `pitch_jump` | `f0_method → rmvpe`，`protect` +0.03 |
| `doubled_syllable` | `index_rate` −0.08，并提示去分离阶段剥和声 |
| `dropped_syllable` | `index_rate` +0.08，`protect` −0.03 |
| `harsh` | `filter_radius` +2，`index_rate` −0.05 |
| `over_smooth` | `protect` −0.05，`index_rate` +0.05 |
| `level_jump` | `rms_mix_rate` ±0.1 |

所有数值都夹在合法区间内（`RANGES`：index_rate 0–1 / protect 0–0.5 /
rms_mix_rate 0–1 / filter_radius 0–7 / index_k 0–64），并给出 `confidence`。

---

## 7. model 后端（可选）

权重尚未随仓库分发。给一个 `nn.Module`，输入 `(2, T)`（源/输出两轨），
输出每个时间步的瑕疵分数 `(2, T)` 或 `(T,)`，分数 > 0.5 的区间会被并进 `segments`
（`type: model_flag`）。加载与卸载都在 `run_model_backend()` 里，异常也会走 `finally` 释放。

```bash
python tools/pitchinq.py --source a.wav --converted b.wav \
  --backend model --model-path work/models/pitchinq-q.pt --wait-gpu 600
```

---

## 8. 自检

`work/_diag/selftest.py` 会给每种缺陷合成一对音频并断言检测结果（7/7 通过）：

```bash
python work/_diag/selftest.py
[PASS] 音准偏低 80c         期望 pitch_off        实得 ['pitch_off']
[PASS] 八度跳变             期望 pitch_jump       实得 ['pitch_jump', 'pitch_off']
[PASS] 多出来的字            期望 doubled_syllable 实得 ['doubled_syllable']
[PASS] 漏字                期望 dropped_syllable 实得 ['dropped_syllable']
[PASS] 音量跳 8dB           期望 level_jump       实得 ['level_jump']
[PASS] 齿音变亮             期望 harsh            实得 ['harsh']
[PASS] 过度平滑             期望 over_smooth      实得 ['over_smooth']
7/7 通过
```

真实用例（20 秒原唱干声 → 爱音模型转换）得到 `ok / 94 分 / 2 处重复字`，
耗时约 10 秒（dsp 后端，零显存占用）。

---

## 9. 音色亮度（频谱质心）诊断

用户最常问的一类问题：**「新人声听起来比原唱亮 / 闷」**。这不需要听，能直接量：

- 指标：能量加权**频谱质心**（Hz），只统计高于 -50 dBFS 的帧，避免静音段把结果拉到 0。
- 对比：`centroid_delta_st = 12 * log2(centroid_out / centroid_src)`，单位是**半音**（听感上比 Hz 更好理解）。
- 阈值：`|Δ| ≥ 1.2` 音分记 `mid`，`|Δ| ≥ 2.5` 记 `high`；方向记成 `timbre_bright` / `timbre_dark`。
- 实测参考：原唱干声 2530 Hz、某模型输出 1754 Hz → **-6.2 个半音（明显偏暗）**，这类模型通常高频被抹平。

### 偏亮（比原唱大）怎么办

| 层级 | 动作 | 说明 |
| --- | --- | --- |
| 导出 | `eq_air -2 ~ -4 dB` | 8 kHz 以上高架，直接收掉亮边、齿音、沙沙声 |
| 导出 | `eq_presence -1 ~ -2 dB` | 3 kHz 峰值回调一半，避免只削高频导致音色断层 |
| RVC | `index_rate -0.05` | 降检索，减少「模型味」（模型往往比真人亮） |
| RVC | `filter_radius +1` | 加大滤波半径，压住金属音/齿音 |
| 排查 | 看有无 `doubled_syllable` / `harsh` | 和声残留、齿音本身就会被算成「亮」 |

### 偏暗（比原唱小）怎么办

| 层级 | 动作 | 说明 |
| --- | --- | --- |
| 导出 | `eq_air +2 ~ +3 dB` | 补空气感，最直接 |
| 导出 | `eq_presence +1 dB` | 让人声重新靠前 |
| RVC | `index_rate +0.06` | 提高检索，从音色库补回高频与细节 |
| RVC | `protect +0.03` | 减少过度平滑导致的发闷 |
| 排查 | 看 `over_smooth` / `rms_mix_rate` | 包络混合比太低也会显得闷 |

### 报告里长什么样

```json
"summary": {
  "centroid_src_hz": 2530.0, "centroid_conv_hz": 1754.0,
  "centroid_delta_st": -6.17, "timbre": "dark"
},
"param_plan": [
  {"param": "eq_air", "stage": "mix", "from": 0.0, "to": 2.5,
   "why": "比原唱暗，导出时把 8 kHz 以上加 +2.5 dB，补回空气感", "confidence": 0.7},
  {"param": "eq_presence", "stage": "mix", "from": 0.0, "to": 1.2, "...": "..."},
  {"param": "index_rate", "stage": "rvc", "from": 0.5, "to": 0.56, "...": "..."}
]
```

`param_plan[].stage` 是新增字段：`rvc` 表示改 RVC 页的参数，`mix` 表示改第 3 步导出的 EQ。
UI 上 `eq_air` ↔ `#eqAir`、`eq_presence` ↔ `#eqPres`，直接照着填即可。

---

## 10. 导出侧参数（第 3 步 / `tools/mix.py`）

| 参数 | CLI | 范围 | 作用 |
| --- | --- | --- | --- |
| 亮度 | `--eq-air` | ±6 dB | 8 kHz 以上高架 |
| 靠前度 | `--eq-presence` | ±6 dB | 2.2–4.2 kHz 带通钟形 |
| 厚度 | `--eq-body` | ±6 dB | 180 Hz 低架 |
| 混响 | `--reverb` / `--reverb-decay` / `--reverb-predelay` / `--reverb-damp` | 0–80% / 0.3–6 s / 0–100 ms / 2–16 kHz | 只加在转换后的人声上 |
| 母带匹配 | `--matcher`（HTTP 侧：`matcher.enabled` + `matcher.reference`） | — | Matchering 对齐参考曲的响度与频响 |

## 11. 亮度问题的标定结论（2026-11-02 实测）

同一段 20s 干声（频谱质心 **2530 Hz**），固定其它参数、只改一个 RVC 参数，量输出质心：

| 参数 | 取值 | 质心 | Δ相对原唱 |
|---|---|---|---|
| index_rate | 0.20 / 0.50 / 0.80 | 1751 / 1755 / 1730 Hz | −6.37 / −6.33 / −6.58 半音 |
| protect | 0.20 / 0.45 | 1759 / 1759 Hz | −6.30 / −6.29 半音 |
| filter_radius | 0 / 7 | 1760 / 1758 Hz | −6.29 / −6.30 半音 |
| rms_mix_rate | 0.0 / 1.0 | 1763 / 1749 Hz | −6.26 / −6.39 半音 |

- **RVC 侧这四个参数对质心的影响都 < 0.2 半音**，不能用来修亮度（早期建议「偏亮降 index_rate」已删除）。
- 亮度差的主因是**音色模型本身**（本例该模型固定偏暗 6.3 半音），根本解是换质心更接近原唱的模型。
- 导出侧的 EQ 才是有效杠杆：`eq_air +2.5 dB` 把 1754 Hz 拉到 2373 Hz（≈ +4.4 半音），再过 Matchering 保持 2351 Hz。
- 因此 `param_plan` 里亮度相关条目固定为：`model`（stage=rvc，换模型，置信 0.8）+ `eq_air`（stage=mix，置信 0.75）+ `eq_presence`（stage=mix，取 air 的一半，0.6）。

## 12. 第 3 步的自动填参

- 转换 `onDone` 后 400 ms 自动跑一次诊断（`max_seconds=90`），结果写进第 3 步滑杆：`eq_air` → `#eqAir`、`eq_presence` → `#eqPres`、`eq_body` → `#eqBody`（步进 0.1 dB）。
- 第 1 步分离 `onDone` 时按 `clean` 选项预填混响量：`none` 6% / `backing` 8% / `dereverb` 12% / `both` 14%；用户手动动过混响滑杆就不再覆盖（`S.revTouched`）。
- 第 3 步按钮：`诊断这一步`（手动重跑）、`按建议重填`（重放上一次报告）、右侧提示行显示「质心 2530 → 1729 Hz（−6.6 半音 · 偏暗）→ 已填 …」。
- 诊断不阻塞转换：`/api/diagnose` 用独立 job，`lock_mode=auto` 在显存够用时不吃 gpulock，可与 RVC 并发。

## 13. 亮度是不是硬伤？—— 通用解法：音色对齐（timbre match）

结论：**不是硬伤**。模型音色确实改不动（第 11 节），但可以对输出做一次「按原唱的频谱校正」。

做法（`tools/timbre.py`，已接进 `mix.py` 与第 3 步）：

1. 取转换后干声与**源唱干声**的 STFT 功率谱，按频点求增益 `sqrt(P_ref / P_conv)`，裁到 ±12 dB 后做 9 点频域平滑；
2. 再叠一个全局倾斜（dB/octave），在 `-3 … +3` 网格里闭环测质心，选最接近原唱的（容差 0.4 半音）；
3. 幅度乘增益、保留相位，istft 还原——**不改音高、不改节奏、不改咬字**。

实测（同一段 20s 干声，源质心 2530 Hz，四个不同模型）：

| 模型 | 原始偏差 | 音色对齐后 | LSD（越小越像） |
|---|---|---|---|
| 6_aiyi | −6.61 半音 | **+0.49** | 6.93 → 2.25 dB |
| 1_kiki | −6.03 半音 | **−0.48** | 5.20 → 1.71 dB |
| 3_guanguan | +0.73 半音 | **−0.28** | 4.14 → 2.19 dB |
| 7_deng | +0.28 半音 | **−0.13** | 3.46 → 1.71 dB |

- 另两种备选：**掺原干声**（10/20/30/50% → −6.09/−5.05/−3.76/−1.39 半音，LSD 6.50/5.72/4.95/3.41），能保模型味但会带进原唱音色，且要手动调比例；**LTAS 一次性**（不做倾斜闭环）在本来就接近的模型上会过冲 +1~+3 半音，所以必须闭环。
- 产品化：第 3 步勾选框 `音色对齐（校正到原唱音色）`，默认开；参考用第 1 步分离出的干声，没有干声自动跳过；开启时诊断的 EQ 自动填 0（不叠加）。
- 命令行：`python tools/timbre.py --vocal rvc.wav --reference dry.wav --output out.wav`，或给 `mix.py` 加 `--timbre-match dry.wav`。
