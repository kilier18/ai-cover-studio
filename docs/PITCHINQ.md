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
| 给出 RVC 可调参数方案（index_rate / rms_mix_rate / protect / f0 算法 / pitch） | 音频渲染、混音后期 |
| 输出结构化 JSON | 主动触发自己（没有定时任务、没有自动挂载） |

---

## 2. 调度规则（硬约束）

1. **不常驻显存**：默认 `--backend dsp` 只做信号分析，**完全不碰 GPU**。
2. **和 RVC 互斥加载**：需要走模型后端时，必须持有 `tools/gpulock.py` 的跨进程显存锁。
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