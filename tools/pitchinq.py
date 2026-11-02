"""PitchIn-Q：歌唱人声诊断（按需加载、诊断完立刻释放显存）。

职责（只做这些）
    1. 听一遍"源干声"和"RVC 输出"，定位音频缺陷并给出**时间区间**；
    2. 对比两轨，判断缺陷是分离/调参/内核哪一环造成的；
    3. 给出 RVC 可调参数方案（index_rate / rms_mix_rate / protect / f0 算法 / pitch）；
    4. 输出结构化 JSON。

不做：音频分离、F0 提取产品化、音频渲染、混音后期。
不主动跑：只有被命令行/DSH/网页端调用时才启动，跑完卸载模型、退出。

用法
    python tools/pitchinq.py --source 干声.wav --converted RVC输出.wav \
        --params work/_last_params.json --out work/_diag/report.json

    # 显存被 RVC 占着时最多等 10 分钟（异步调度），等不到返回 3
    python tools/pitchinq.py ... --wait-gpu 600
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpulock                                            # noqa: E402

try:                                                      # Windows 控制台默认 GBK
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                         # noqa: BLE001
    pass

ROOT = Path(__file__).resolve().parent.parent
SR = 44100
HOP = 256
FRAME = 2048
SCHEMA = "pitchinq.report/1"

DEFECT_LABEL = {
    "pitch_off": "音准偏移",
    "pitch_jump": "八度/半八度跳音",
    "doubled_syllable": "多出来的字（重复字/和声残留）",
    "dropped_syllable": "漏字/吞字",
    "level_jump": "音量跳变",
    "harsh": "金属音/齿音过亮",
    "over_smooth": "过度平滑（气声、颤音丢失）",
    "wobble": "抖动/不稳（音高来回摆）",
    "offset": "整体时间偏移",
    "timbre_bright": "音色偏亮（质心高于原唱）",
    "timbre_dark": "音色偏暗（质心低于原唱）",
}
SEVERITY_W = {"low": 1, "mid": 3, "high": 6}


# --------------------------------------------------------------------------- 音频
def load_mono(path: str | Path, sr: int = SR) -> np.ndarray:
    import soundfile as sf
    from scipy.signal import resample_poly

    data, orig = sf.read(str(path), always_2d=True, dtype="float32")
    x = data.mean(axis=1)
    if orig != sr:
        from math import gcd
        g = gcd(int(orig), int(sr))
        x = resample_poly(x, sr // g, int(orig) // g).astype(np.float32)
    return np.nan_to_num(x)


def rms_env(x: np.ndarray, sr: int = SR) -> np.ndarray:
    import librosa

    return librosa.feature.rms(y=x, frame_length=4 * HOP, hop_length=HOP)[0]


def onsets(x: np.ndarray, sr: int = SR) -> np.ndarray:
    """显著起音点（帧号）。先用谱通量归一化，再做峰值挑选，避免颤音被当成起音。"""
    import librosa

    env = librosa.onset.onset_strength(y=x, sr=sr, hop_length=HOP)
    if env.size == 0:
        return np.zeros(0, dtype=int)
    env = env / (float(env.max()) + 1e-9)
    idx = librosa.util.peak_pick(env, pre_max=4, post_max=4, pre_avg=6, post_avg=6,
                                 delta=0.10, wait=4)
    idx = np.asarray(idx, dtype=int)
    if idx.size > 1:                                       # 80ms 内的合并
        keep = [int(idx[0])]
        for i in idx[1:]:
            if (i - keep[-1]) * HOP / sr >= 0.08:
                keep.append(int(i))
        idx = np.asarray(keep, dtype=int)
    return idx


def f0_track(x: np.ndarray, sr: int = SR, method: str = "yin"):
    import librosa

    if method == "pyin":
        f0, voiced, _ = librosa.pyin(x, fmin=65, fmax=1100, sr=sr,
                                     frame_length=FRAME, hop_length=HOP, fill_na=np.nan)
        return np.asarray(f0, dtype=float), np.asarray(voiced, dtype=bool)
    f0 = librosa.yin(x, fmin=65, fmax=1100, sr=sr, frame_length=FRAME, hop_length=HOP)
    f0 = np.asarray(f0, dtype=float)
    env = rms_env(x, sr)
    n = min(len(f0), len(env))
    f0, env = f0[:n], env[:n]
    ref = np.percentile(env[env > 1e-6], 60) if np.any(env > 1e-6) else 1.0
    voiced = env > ref * 0.08
    f0[~voiced] = np.nan
    return f0, voiced


def hf_ratio(x: np.ndarray, sr: int = SR) -> tuple[float, float]:
    """返回 (高频能量相对低频的 dB, 整体能量 dB)。"""
    n = min(len(x), sr * 60)
    if n < 4096:
        return -120.0, -120.0
    spec = np.abs(np.fft.rfft(x[:n] * np.hanning(n))) ** 2
    f = np.fft.rfftfreq(n, 1 / sr)
    low = spec[(f >= 100) & (f < 4000)].sum()
    high = spec[(f >= 4000) & (f < 12000)].sum()
    if low <= 0 and high <= 0:
        return -120.0, -120.0
    hf_db = 10 * np.log10(max(high, 1e-20) / max(low, 1e-20))
    return float(hf_db), float(10 * np.log10((low + high) / n))


def centroid(x: np.ndarray, sr: int = SR, max_sec: float = 60.0) -> float:
    """能量加权频谱质心（Hz）——「音色亮不亮」的客观指标。

    只在有声音的帧上算（低于 -50 dBFS 的帧跳过），避免静音段把质心拉到 0。
    人声通常在 1500–3500 Hz；比原唱高 10% ≈ 亮 1.65 个半音。
    """
    n = min(len(x), int(sr * max_sec))
    win_n, hop = 2048, 512
    if n < win_n * 2:
        return 0.0
    w = np.hanning(win_n)
    f = np.fft.rfftfreq(win_n, 1.0 / sr)
    vals, weights = [], []
    for i in range(0, n - win_n, hop):
        seg = x[i:i + win_n]
        rms = float(np.sqrt((seg ** 2).mean()))
        if rms < 0.00316:                     # -50 dBFS
            continue
        spec = np.abs(np.fft.rfft(seg * w))
        tot = float(spec.sum())
        if tot <= 0:
            continue
        vals.append(float((spec * f).sum() / tot))
        weights.append(rms)
    if not vals:
        return 0.0
    return float(np.average(vals, weights=weights))


def rms_db(x: np.ndarray, win: int) -> np.ndarray:
    n = len(x) // win
    if n == 0:
        return np.zeros(1)
    blocks = x[: n * win].reshape(n, win).astype(np.float64)
    r = np.sqrt((blocks ** 2).mean(axis=1))
    return 20 * np.log10(np.maximum(r, 1e-6))


def align_offset(a: np.ndarray, b: np.ndarray) -> float:
    """用能量包络互相关估计 b 相对 a 的整体时间偏移（秒）。"""
    from scipy.signal import correlate

    ea = rms_env(a)[::4]
    eb = rms_env(b)[::4]
    n = min(len(ea), len(eb))
    if n < 20:
        return 0.0
    ea, eb = ea[:n] - ea[:n].mean(), eb[:n] - eb[:n].mean()
    c = correlate(eb, ea, mode="full")
    lag = int(np.argmax(np.abs(c))) - (n - 1)
    return lag * (HOP * 4) / SR


# --------------------------------------------------------------------------- 分析
def runs(mask: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_len:
                out.append((start, i))
            start = None
    if start is not None and len(mask) - start >= min_len:
        out.append((start, len(mask)))
    return out


def analyze(src: np.ndarray, conv: np.ndarray, params: dict, f0_method: str = "yin") -> dict:
    t0 = time.time()
    t = lambda i: round(i * HOP / SR, 2)                       # noqa: E731
    n = min(len(src), len(conv))
    a, b = src[:n], conv[:n]
    frame_times = np.arange(n // HOP) * HOP / SR

    def clip(start_f: int, end_f: int) -> tuple[float, float]:
        return round(start_f * HOP / SR, 2), round(min(end_f * HOP / SR, n / SR), 2)

    segments: list[dict] = []

    # 1) 整体偏移
    off = align_offset(a, b)
    if abs(off) > 0.06:
        segments.append({"type": "offset", "start": 0.0, "end": round(n / SR, 2),
                         "severity": "mid" if abs(off) < 0.3 else "high",
                         "detail": f"RVC 输出比源干声{'晚' if off > 0 else '早'} {abs(off)*1000:.0f} ms",
                         "values": {"offset_s": round(off, 3)}})

    # 2) 音高
    f0a, va = f0_track(a, method=f0_method)
    f0b, vb = f0_track(b, method=f0_method)
    m = min(len(f0a), len(f0b))
    f0a, f0b, va, vb = f0a[:m], f0b[:m], va[:m], vb[:m]
    both = va & vb & np.isfinite(f0a) & np.isfinite(f0b) & (f0a > 0) & (f0b > 0)
    cents = np.full(m, np.nan)
    cents[both] = 1200 * np.log2(f0b[both] / f0a[both])
    # 八度错（crepe/yin 偶发）会把均值拽走：统计只用 ±600 音分内的帧，且用中位数
    fin = np.isfinite(cents)
    core = fin & (np.abs(cents) <= 600)
    mean_cents = float(np.nanmedian(cents[core])) if core.any() else 0.0
    std_cents = float(np.nanstd(cents[core])) if core.any() else 0.0

    big = core & (np.abs(cents) > 60)
    for s, e in runs(big, int(0.12 * SR / HOP)):
        seg = cents[s:e]
        c = float(np.mean(seg))
        sev = "high" if abs(c) > 150 else "mid"
        segments.append({"type": "pitch_off", "start": t(s), "end": t(e), "severity": sev,
                         "detail": f"这一段{'偏高' if c > 0 else '偏低'} {abs(c):.0f} 音分",
                         "values": {"cents": round(c, 1), "duration": round((e - s) * HOP / SR, 2)}})

    octave = np.isfinite(cents) & (np.abs(cents) > 1050) & (np.abs(cents) < 1350)
    for s, e in runs(octave, int(0.06 * SR / HOP)):
        segments.append({"type": "pitch_jump", "start": t(s), "end": t(e), "severity": "high",
                         "detail": "出现整八度跳变（多/漏一个音区）",
                         "values": {"cents": round(float(np.nanmean(cents[s:e])), 0)}})

    # 3) 抖动：音高在短窗里来回摆
    if np.any(np.isfinite(cents)):
        c0 = np.nan_to_num(cents, nan=0.0)
        d = np.diff(np.sign(c0))
        swings = int(np.sum(np.abs(d) > 0) )
        wobble = swings / max(len(c0), 1)
        if wobble > 0.25 and std_cents > 90:
            segments.append({"type": "wobble", "start": 0.0, "end": round(n / SR, 2),
                             "severity": "mid",
                             "detail": f"音高来回摆动（每百帧换向 {wobble*100:.0f} 次，标准差 {std_cents:.0f} 音分）",
                             "values": {"swing_rate": round(wobble, 3), "std_cents": round(std_cents, 1)}})

    # 4) 多字 / 漏字（除了起音对不上，还要求该处能量真的差很多，避免误报）
    oa, ob = onsets(a), onsets(b)
    tol = int(0.07 * SR / HOP)
    margin = int(0.15 * SR / HOP)
    dur_f = m

    def energy_db(sig: np.ndarray, at: int, span: float = 0.20) -> float:
        i0 = max(0, at)
        i1 = min(len(sig), at + int(span * SR))
        if i1 <= i0:
            return -120.0
        seg = sig[i0:i1]
        return float(20 * np.log10(max(np.sqrt(np.mean(seg.astype(np.float64) ** 2)), 1e-6)))

    extra, missing = [], []
    far = int(0.30 * SR / HOP)
    for i in ob:
        near_src = int(np.min(np.abs(oa - i))) if oa.size else 10 ** 9
        # 离源干声起音太近的多半是同一次起音被检测了两次，不算重复字
        if margin <= i <= dur_f - margin and near_src > far:
            if energy_db(a, i * HOP) > -35.0:
                extra.append(i)
    for i in oa:
        if margin <= i <= dur_f - margin and (ob.size == 0 or np.min(np.abs(ob - i)) > tol):
            if energy_db(b, i * HOP) < energy_db(a, i * HOP) - 6.0:
                missing.append(i)
    for i in extra:
        j = min(i + int(0.15 * SR / HOP), m)
        segments.append({"type": "doubled_syllable", "start": t(max(0, i - 1)), "end": t(j),
                         "severity": "mid",
                         "detail": "输出多出一处起音（重复字，常见于和声/双轨没剥干净）",
                         "values": {"onset_s": t(i)}})
    for i in missing:
        j = min(i + int(0.15 * SR / HOP), m)
        segments.append({"type": "dropped_syllable", "start": t(max(0, i - 1)), "end": t(j),
                         "severity": "high",
                         "detail": "输出少了一处起音（吞字/漏字）",
                         "values": {"onset_s": t(i)}})

    # 5) 音量（只比两边都有声音的地方，一边静音属于漏字）
    win = int(0.25 * SR)
    dba, dbb = rms_db(a, win), rms_db(b, win)
    k = min(len(dba), len(dbb))
    if k > 2:
        dba, dbb = dba[:k], dbb[:k]
        diff = dbb - dba
        voiced = (dba > -50) & (dbb > -50)
        jump = voiced & (np.abs(diff) > 6.0)
        for s, e in runs(jump, 2):
            segments.append({"type": "level_jump", "start": round(s * 0.25, 2),
                             "end": round(min(e * 0.25, n / SR), 2),
                             "severity": "mid" if abs(float(diff[s:e].mean())) < 12 else "high",
                             "detail": f"音量差了 {float(diff[s:e].mean()):+.1f} dB",
                             "values": {"db": round(float(diff[s:e].mean()), 1)}})

    # 6) 齿音/金属音（只在两边响度可比时判断；看相对低频的高频能量涨了多少 dB）
    hf_a, db_a = hf_ratio(a)
    hf_b, db_b = hf_ratio(b)
    if abs(db_a - db_b) < 6 and (hf_b - hf_a) > 3.0:
        segments.append({"type": "harsh", "start": 0.0, "end": round(n / SR, 2),
                         "severity": "mid",
                         "detail": f"高频相对低频多了 {hf_b - hf_a:.1f} dB（齿音/金属音变亮）",
                         "values": {"hf_src_db": round(hf_a, 1), "hf_conv_db": round(hf_b, 1)}})

    # 7) 过度平滑（逐音符比颤音深度；整段比会被旋律本身的音高变化污染）
    def note_spans(x):                                   # 用源干声的起音切音符
        o = onsets(x)
        edges = [0] + [int(i) for i in o] + [m]
        spans = []
        for s, e in zip(edges[:-1], edges[1:]):
            if (e - s) * HOP / SR > 0.25:
                spans.append((s + int(0.10 * SR / HOP), e - int(0.10 * SR / HOP)))
        return spans

    def vib_std(f0, v, s, e):
        seg = f0[s:e]
        vv = v[s:e]
        vals = seg[vv & np.isfinite(seg)]
        if vals.size < 8:
            return 0.0
        c = 1200 * np.log2(vals / np.median(vals))
        return float(np.std(c))

    spans = note_spans(a)
    flat_cnt = tot_cnt = 0
    ratios = []
    for s, e in spans:
        if e <= s or e > m:
            continue
        sa, sb = vib_std(f0a, va, s, e), vib_std(f0b, vb, s, e)
        if sa < 25:                                      # 源本来就没颤音，不判
            continue
        tot_cnt += 1
        ratios.append(sb / sa)
        if sb < sa * 0.62:
            flat_cnt += 1
    if tot_cnt >= 2 and flat_cnt / tot_cnt >= 0.4:
        med = float(np.median(ratios))
        segments.append({"type": "over_smooth", "start": 0.0, "end": round(n / SR, 2),
                         "severity": "mid",
                         "detail": f"{flat_cnt}/{tot_cnt} 个音符的颤音被压掉（保留 {med*100:.0f}%），气声和尾音会发平",
                         "values": {"flat_notes": flat_cnt, "notes": tot_cnt,
                                    "keep_ratio": round(med, 2)}})

    # --- 音色亮度：频谱质心 vs 原唱
    cen_a = centroid(a)
    cen_b = centroid(b)
    cen_st = 0.0
    if cen_a > 0 and cen_b > 0:
        cen_st = 12.0 * float(np.log2(cen_b / cen_a))
        dur = round(n / SR, 2)
        if cen_st >= 1.2:
            sev = "high" if cen_st >= 2.5 else "mid"
            segments.append({
                "type": "timbre_bright", "start": 0.0, "end": dur, "severity": sev,
                "detail": f"音色比原唱亮 {cen_st:+.1f} 个半音（质心 {cen_a:.0f} → {cen_b:.0f} Hz）",
                "values": {"centroid_src": round(cen_a, 1), "centroid_conv": round(cen_b, 1),
                           "delta_semitones": round(cen_st, 2)},
            })
        elif cen_st <= -1.2:
            sev = "high" if cen_st <= -2.5 else "mid"
            segments.append({
                "type": "timbre_dark", "start": 0.0, "end": dur, "severity": sev,
                "detail": f"音色比原唱暗 {cen_st:.1f} 个半音（质心 {cen_a:.0f} → {cen_b:.0f} Hz）",
                "values": {"centroid_src": round(cen_a, 1), "centroid_conv": round(cen_b, 1),
                           "delta_semitones": round(cen_st, 2)},
            })

    segments.sort(key=lambda s: (s["start"], s["type"]))

    # 8) 打分：按「每分钟缺陷密度」扣分，长歌短歌同一把尺
    pen = 0.0
    for s in segments:
        pen += SEVERITY_W.get(s["severity"], 2) * (1.0 if s["type"] != "offset" else 0.5)
    minutes = max(len(a) / SR / 60.0, 0.1)
    score = 100 - pen / minutes * 0.33
    score = int(max(0, min(100, round(score))))

    return {
        "schema": SCHEMA,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "engine": {"name": "PitchIn-Q", "backend": "dsp", "f0_method": f0_method,
                   "elapsed_s": round(time.time() - t0, 2)},
        "audio": {"src_seconds": round(len(src) / SR, 2), "conv_seconds": round(len(conv) / SR, 2),
                  "analyzed_seconds": round(n / SR, 2), "sr": SR},
        "summary": {
            "verdict": "ok" if score >= 85 and not any(s["severity"] == "high" for s in segments) else "adjust",
            "score": score,
            "mean_cents": round(mean_cents, 1),
            "defect_count": len(segments),
            "centroid_src_hz": round(cen_a, 1),
            "centroid_conv_hz": round(cen_b, 1),
            "centroid_delta_st": round(cen_st, 2),
            "timbre": ("bright" if cen_st >= 1.2 else "dark" if cen_st <= -1.2 else "neutral"),
            "top": [f"{DEFECT_LABEL.get(s['type'], s['type'])} @ {s['start']}–{s['end']}s"
                    for s in sorted(segments, key=lambda s: -SEVERITY_W.get(s["severity"], 2))[:3]],
        },
        "segments": segments,
        "param_plan": build_plan(segments, params, mean_cents, cen_st),
        "inputs_params": params,
    }


# --------------------------------------------------------------------------- 参数方案
RANGES = {
    "index_rate": (0.0, 1.0, 0.05),
    "rms_mix_rate": (0.0, 1.0, 0.05),
    "protect": (0.0, 0.5, 0.01),
    "filter_radius": (0, 7, 1),
    "index_k": (0, 64, 2),
}
DEFAULTS = {"index_rate": 0.5, "rms_mix_rate": 0.3, "protect": 0.4,
            "filter_radius": 3, "index_k": 28, "f0_method": "rmvpe", "pitch": 0}


def _clamp(name: str, val: float) -> float:
    lo, hi, step = RANGES[name]
    v = round(round(val / step) * step, 3)
    return float(max(lo, min(hi, v)))


def build_plan(segments: list[dict], params: dict, mean_cents: float,
               timbre_st: float = 0.0) -> list[dict]:
    base = dict(DEFAULTS)
    base.update({k: v for k, v in (params or {}).items() if k in DEFAULTS})
    plan: dict[str, dict] = {}

    def add(param: str, delta: float, why: str, conf: float) -> None:
        cur = plan.get(param)
        if cur:
            cur["to"] = _clamp(param, cur["to"] + delta)
            cur["why"] += "；" + why
            cur["confidence"] = round(max(cur["confidence"], conf), 2)
        else:
            frm = float(base.get(param, 0))
            plan[param] = {"param": param, "from": frm, "to": _clamp(param, frm + delta),
                           "why": why, "confidence": conf}

    kinds = {}
    for s in segments:
        kinds.setdefault(s["type"], []).append(s)

    if abs(mean_cents) > 35:
        shift = -int(round(mean_cents / 100))
        if shift != 0:
            frm = int(base.get("pitch", 0))
            plan["pitch"] = {"param": "pitch", "from": frm, "to": max(-12, min(12, frm + shift)),
                             "why": f"整体音高偏 {mean_cents:+.0f} 音分，建议移调 {shift:+d} 半音",
                             "confidence": 0.75}
    if "pitch_off" in kinds or "wobble" in kinds:
        add("protect", 0.04, "局部音准/颤音不稳，提高保护让原唱音高更多参与", 0.6)
        add("index_rate", -0.05, "同时略降检索强度，减少音色库把音高拽走", 0.5)
    if "pitch_jump" in kinds:
        plan["f0_method"] = {"param": "f0_method", "from": base["f0_method"], "to": "rmvpe",
                             "why": "出现八度跳变，rmvpe 对八度错误最稳", "confidence": 0.7}
        add("protect", 0.03, "抑制跳音", 0.5)
    if "doubled_syllable" in kinds:
        add("index_rate", -0.08, "多出来的起音多半是和声/双轨残留，降检索可减轻", 0.5)
        plan.setdefault("source_note", {"param": "separate", "stage": "separate",
                                        "from": "当前分离",
                                        "to": "先剥和声/双轨再转换",
                                        "why": "重复字的根因通常在分离阶段，不在 RVC 参数",
                                        "confidence": 0.8})
    if "dropped_syllable" in kinds:
        add("index_rate", 0.08, "漏字通常是过度检索抹掉了细节", 0.55)
        add("protect", -0.03, "降低保护，让辅音更容易保留", 0.45)
    if "harsh" in kinds:
        add("filter_radius", 2, "齿音/金属音偏亮，加大滤波半径", 0.6)
        add("index_rate", -0.05, "降低检索降低金属感", 0.45)
    if "over_smooth" in kinds:
        add("protect", -0.05, "气声和颤音被压平，降低保护", 0.6)
        add("index_rate", 0.05, "提高检索补回细节", 0.45)
    if "timbre_bright" in kinds or "timbre_dark" in kinds:
        bright = "timbre_bright" in kinds
        sign = -1.0 if bright else 1.0
        strong = any(x["severity"] == "high"
                     for x in kinds.get("timbre_bright", []) + kinds.get("timbre_dark", []))
        mag = 2.5 if strong else 1.5
        plan["eq_air"] = {"param": "eq_air", "stage": "mix", "from": 0.0, "to": round(sign * mag, 1),
                          "why": (f"备选（不开音色对齐时）：把 8 kHz 以上压 {sign * mag:+.1f} dB，收掉亮边和齿音"
                                  if bright else
                                  f"备选（不开音色对齐时）：把 8 kHz 以上加 {mag:+.1f} dB，补回空气感"),
                          "confidence": 0.6}
        plan["eq_presence"] = {"param": "eq_presence", "stage": "mix", "from": 0.0,
                               "to": round(sign * (mag / 2.0), 1),
                               "why": "3 kHz 靠前度跟着亮度回调一半，避免只改高频、音色出现断层",
                               "confidence": 0.55}
        # 2026-11-02 标定：index_rate / protect / filter_radius / rms_mix_rate 对质心的
        # 影响都 < 0.2 半音（实测 1751/1755/1730 Hz、1759 Hz、1758/1760 Hz、1763/1749 Hz），
        # 所以亮度问题不能靠 RVC 参数修，只能换模型或走导出 EQ。
        plan["timbre_match"] = {"param": "timbre_match", "stage": "mix", "from": "关",
                                "to": "开「音色对齐（校正到原唱音色）」",
                                "why": ("按原唱干声的长时频谱做频带校正，实测 4 个模型 "
                                        "−6.6~+0.7 半音的偏差都能拉回 ±0.5 半音，且不改变咬字/颤音"),
                                "confidence": 0.85}
        plan["model"] = {"param": "model", "stage": "rvc", "from": base.get("model", "当前音色模型"),
                         "to": "换成频谱质心更接近原唱的音色模型（否则用 EQ 补偿）",
                         "why": (f"{abs(timbre_st):.1f} 半音的亮度差主要来自模型本身的音色；"
                                 "实测 index_rate/protect/filter_radius/rms_mix_rate 对质心影响 <0.2 半音"),
                         "confidence": 0.8}
    if "level_jump" in kinds:
        cur = float(base.get("rms_mix_rate", 0.3))
        add("rms_mix_rate", -0.1 if cur >= 0.3 else 0.1,
            "音量包络和源干声差得多，调包络混合比", 0.5)

    out = []
    for k, v in plan.items():
        if k == "source_note":
            continue
        try:
            if v.get("param") not in ("f0_method", "model", "timbre_match") and float(v["from"]) == float(v["to"]):
                continue
        except (TypeError, ValueError):
            pass          # 非数值型（model 这类文本建议）保留
        out.append(v)
    for v in out:
        v.setdefault("stage", "rvc")
    note = plan.get("source_note")
    if note:
        out.append(note)
    out.sort(key=lambda p: -p["confidence"])
    return out


# --------------------------------------------------------------------------- 模型后端
def run_model_backend(src: np.ndarray, conv: np.ndarray, model_path: str | Path,
                      params: dict, f0_method: str) -> dict:
    """可选的神经网络后端：显存按需加载，跑完立刻卸载。

    没给权重时不要走这里（默认 dsp 后端，完全不占显存）。
    """
    import torch

    gpulock.busy()
    if not torch.cuda.is_available():
        raise RuntimeError("没有可用的 CUDA")
    free_before = torch.cuda.mem_get_info()[0] / 1048576
    t_load = time.time()
    model = torch.load(str(model_path), map_location="cuda", weights_only=False)
    if hasattr(model, "eval"):
        model.eval()
    load_ms = int((time.time() - t_load) * 1000)
    try:
        with torch.inference_mode():
            n = min(len(src), len(conv))
            x = torch.from_numpy(np.stack([src[:n], conv[:n]])).float().cuda()
            y = model(x)
            scores = y.detach().float().cpu().numpy().reshape(2, -1)
        report = analyze(src, conv, params, f0_method)
        # 把模型给的帧级分数并进报告（>=0.5 视作有瑕疵的区间）
        bad = scores[1] > 0.5
        step = n / max(len(bad), 1) / SR
        for s, e in runs(bad, 3):
            report["segments"].append({
                "type": "model_flag", "start": round(s * step, 2),
                "end": round(min(e * step, n / SR), 2), "severity": "mid",
                "detail": "PitchIn-Q 模型判定的瑕疵区间",
                "values": {"score": round(float(scores[1][s:e].mean()), 3)}})
        report["engine"]["backend"] = "model"
        report["engine"]["model"] = Path(model_path).name
        return report
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
        free_after = torch.cuda.mem_get_info()[0] / 1048576
        print(json.dumps({"gpu": {"load_ms": load_ms,
                                  "free_before_mb": round(free_before),
                                  "free_after_mb": round(free_after),
                                  "released_mb": round(free_after - free_before)},
                          }, ensure_ascii=False))


def gpu_free_mb() -> float | None:
    """当前显卡剩余显存（MB）；没有 CUDA 时返回 None。"""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        free, _total = torch.cuda.mem_get_info()
        return float(free) / 1048576.0
    except Exception:  # noqa: BLE001
        return None


def model_need_mb(path: str | Path | None) -> float:
    """粗估模型加载后的显存占用：权重体积 × 2.2（fp16 + 激活/缓冲）。"""
    if not path:
        return 0.0
    p = Path(path)
    if not p.exists():
        return 0.0
    return p.stat().st_size / 1048576.0 * 2.2


# --------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="PitchIn-Q 歌唱人声诊断")
    ap.add_argument("--source", required=True, help="源干声（分离后的人声）")
    ap.add_argument("--converted", required=True, help="RVC 输出的人声")
    ap.add_argument("--params", default="", help="这次转换用的参数 JSON（可省）")
    ap.add_argument("--out", default="", help="报告写到哪里（.json）")
    ap.add_argument("--backend", default="dsp", choices=["dsp", "model"],
                    help="dsp=纯信号分析（不占显存，默认）；model=加载神经网络")
    ap.add_argument("--model-path", default=os.environ.get("PITCHINQ_MODEL", ""),
                    help="model 后端的权重路径（也可用环境变量 PITCHINQ_MODEL）")
    ap.add_argument("--f0", dest="f0_method", default="yin", choices=["yin", "pyin"])
    ap.add_argument("--wait-gpu", type=float, default=0,
                    help="显存被占用时最多等多少秒（0 = 不排队直接退出 3）")
    ap.add_argument("--lock-mode", default="auto", choices=["auto", "wait", "none"],
                    help="auto=显存够就和 RVC 并发跑（不加锁）；wait=必须等锁；none=不加锁直接跑")
    ap.add_argument("--max-seconds", type=float, default=0, help="只分析前 N 秒（调试用）")
    ap.add_argument("--json", action="store_true", help="把完整报告打到 stdout")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    t0 = time.time()
    src = load_mono(args.source)
    conv = load_mono(args.converted)
    if args.max_seconds > 0:
        cut = int(args.max_seconds * SR)
        src, conv = src[:cut], conv[:cut]

    params: dict = {}
    if args.params and Path(args.params).exists():
        try:
            params = json.loads(Path(args.params).read_text(encoding="utf-8"))
        except ValueError:
            params = {}

    report: dict
    if args.backend == "model":
        if not args.model_path or not Path(args.model_path).exists():
            print("model 后端需要 --model-path（或 PITCHINQ_MODEL）；没给权重就用 --backend dsp",
                  file=sys.stderr)
            return 2
        handle = None
        mode = args.lock_mode
        if mode == "auto":
            need = model_need_mb(args.model_path)
            free = gpu_free_mb()
            if free is not None and free >= need + 900:
                mode = "none"
                print(f"显存空闲 {free:.0f} MB ≥ 需要 {need:.0f} MB + 900 MB 余量，与 RVC 并发运行",
                      file=sys.stderr)
            else:
                mode = "wait"
        if mode == "wait":
            try:
                handle = gpulock.hold("pitchinq", timeout=args.wait_gpu)
            except gpulock.GpuBusy as exc:
                print(f"显存忙，跳过诊断：{exc}", file=sys.stderr)
                return 3
        try:
            report = run_model_backend(src, conv, args.model_path, params, args.f0_method)
        finally:
            if handle is not None:
                handle.release()
    else:
        report = analyze(src, conv, params, args.f0_method)

    report["engine"]["wall_s"] = round(time.time() - t0, 2)
    out = Path(args.out) if args.out else (ROOT / "work" / "_diag" / "report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(report, ensure_ascii=False))
    elif not args.quiet:
        s = report["summary"]
        print(f"PitchIn-Q · {s['verdict']} · {s['score']}/100 · {s['defect_count']} 处问题")
        for seg in report["segments"][:6]:
            print(f"  [{seg['severity']:>4}] {seg['start']:>6.2f}–{seg['end']:>6.2f}s "
                  f"{DEFECT_LABEL.get(seg['type'], seg['type'])}：{seg['detail']}")
        if report["param_plan"]:
            print("参数建议：")
            for p in report["param_plan"]:
                print(f"  {p['param']}: {p['from']} -> {p['to']}  ({p['why']})")
        print(f"报告：{out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())