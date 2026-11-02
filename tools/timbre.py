# -*- coding: utf-8 -*-
"""音色对齐（timbre match）：把一条人声的长时平均频谱形状拉到参考人声上。

动机：RVC 输出与源唱的音色亮度（频谱质心）差 3–7 半音是**模型本身**决定的，
改 index_rate / protect / filter_radius / rms_mix_rate 都动不了（实测 <0.2 半音）。
但把长时平均谱按频带对齐到源唱，能把 4 个不同模型的偏差都拉回 ±0.5 半音。

做法：
  1. 功率谱按 1/6 倍频程近似（STFT 频点 + 9 点平滑）计算增益，裁剪到 ±max_db；
  2. 再叠一个「全局倾斜」（dB/octave）小网格搜索，闭环测质心，选最接近的；
  3. STFT 幅度乘增益、保留相位，istft 还原。

CLI:
  python tools/timbre.py --vocal converted.wav --reference dry.wav --output out.wav
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SR = 44100
N_FFT = 2048
HOP = 512


def load_mono(path) -> np.ndarray:
    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    return np.asarray(y, dtype=np.float32)


def _frames(y: np.ndarray, max_sec: float = 60.0):
    """一次切帧出幅度谱（F×T）与逐帧 RMS，替代逐帧 Python 循环。

    口径与旧版 centroid 一致：2048 点 hanning 窗、hop 512、< -50 dBFS 的帧丢弃。
    """
    n = min(len(y), int(SR * max_sec))
    win_n, hop = 2048, 512
    if n < win_n * 2:
        return None, None
    idx = np.arange(0, n - win_n + 1, hop)
    fr = y[idx[:, None] + np.arange(win_n)[None, :]]
    mag = np.abs(np.fft.rfft(fr * np.hanning(win_n), axis=1)).T
    rms = np.sqrt((fr ** 2).mean(axis=1))
    return mag.astype(np.float32), rms


def _centroid_of(mag, rms, sr: int = SR) -> float:
    """由幅度谱直接算能量加权质心（只统计有声帧，按帧 RMS 加权）。"""
    if mag is None or mag.size == 0:
        return 0.0
    energy = mag.sum(axis=0)
    keep = (rms >= 0.00316) & (energy > 0)
    if not keep.any():
        return 0.0
    f = np.fft.rfftfreq(mag.shape[0] * 2 - 2, 1.0 / sr)
    c = (f[:, None] * mag[:, keep]).sum(axis=0) / energy[keep]
    return float(np.average(c, weights=rms[keep]))


def centroid(y: np.ndarray, sr: int = SR, max_sec: float = 60.0) -> float:
    """能量加权频谱质心（Hz）——与 pitchinq.centroid 逐行一致的口径。"""
    mag, rms = _frames(y, max_sec)
    return _centroid_of(mag, rms, sr)


def _st(freq: float, ref: float) -> float:
    if freq <= 0 or ref <= 0:
        return 0.0
    return 12.0 * float(np.log2(freq / ref))


def _apply(y: np.ndarray, gain: np.ndarray) -> np.ndarray:
    y = np.nan_to_num(np.asarray(y, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    Y = librosa.stft(y, n_fft=N_FFT, hop_length=HOP)
    out = librosa.istft(Y * gain[:, None], hop_length=HOP, length=len(y))
    peak = float(np.abs(out).max())
    return np.asarray(out / peak * 0.89 if peak > 0 else out, dtype=np.float32)


def ltas_gain(y: np.ndarray, ref: np.ndarray, max_db: float = 12.0,
              smooth: int = 9) -> np.ndarray:
    """平均谱功率比 → 幅度增益（dB），并做频域平滑。"""
    y = np.nan_to_num(np.asarray(y, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    ref = np.nan_to_num(np.asarray(ref, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    py = (np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP)) ** 2).mean(axis=1) + 1e-12
    pr = (np.abs(librosa.stft(ref, n_fft=N_FFT, hop_length=HOP)) ** 2).mean(axis=1) + 1e-12
    g_db = np.clip(10 * np.log10(pr / py), -max_db, max_db)
    g_db = np.convolve(g_db, np.ones(smooth) / smooth, mode="same")
    return g_db


def match_timbre(y: np.ndarray, ref: np.ndarray, max_db: float = 12.0,
                 tol_st: float = 0.4) -> dict:
    """闭环对齐：返回 {audio, st_before, st_after, tilt_db_oct, max_db}。

    加速版：tilt 搜索在谱域解析式完成（只做 1 次 STFT + 1 次 istft），
    不再对每个候选 tilt 都完整正反变换一遍。
    """
    n = min(len(y), len(ref))
    y, ref = y[:n], ref[:n]
    mag, rms = _frames(y)
    mag_r, rms_r = _frames(ref)
    c_ref = _centroid_of(mag_r, rms_r)
    st_before = _st(_centroid_of(mag, rms), c_ref) if mag is not None else 0.0

    g0 = ltas_gain(y, ref, max_db=max_db)
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    octv = np.log2(np.maximum(freqs, 20) / 1000.0)

    best = {"st_after": 0.0, "tilt_db_oct": 0.0, "gain": 10 ** (g0 / 20.0)}
    if mag is not None:
        energy = mag.sum(axis=0)
        keep = (rms >= 0.00316) & (energy > 0)
        f = np.fft.rfftfreq(mag.shape[0] * 2 - 2, 1.0 / SR)
        mk = mag[:, keep]
        wk = rms[keep]

        def st_for(tilt: float) -> tuple:
            gain = 10 ** ((g0 + tilt * octv) / 20.0)
            w = mk * gain[:, None]
            e = w.sum(axis=0)
            good = e > 0
            c = (f[:, None] * w[:, good]).sum(axis=0) / e[good]
            return _st(float(np.average(c, weights=wk[good])), c_ref), gain

        tried = []
        for tilt in (-3.0, -2.0, -1.5, -1.0, -0.7, -0.4, 0.0, 0.4, 0.7, 1.0, 1.5, 2.0, 3.0):
            st, gain = st_for(tilt)
            tried.append((abs(st), st, tilt, gain))
            if abs(st) <= tol_st:
                break
        tried.sort(key=lambda r: r[0])
        _, best_st, best_tilt, best_gain = tried[0]
        best = {"st_after": best_st, "tilt_db_oct": best_tilt, "gain": best_gain}

    best["audio"] = _apply(y, best.pop("gain"))
    best.update({"st_before": st_before, "max_db": max_db})
    return best


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="把线上人声的频谱形状对齐到参考人声")
    p.add_argument("--vocal", required=True, help="要修的人声（RVC 输出）")
    p.add_argument("--reference", required=True, help="参考人声（源唱干声）")
    p.add_argument("--output", required=True)
    p.add_argument("--max-db", type=float, default=12.0, help="单频点最大增益（dB）")
    p.add_argument("--tol", type=float, default=0.4, help="质心差容差（半音）")
    p.add_argument("--json", action="store_true")
    a = p.parse_args(argv)

    y = load_mono(a.vocal)
    ref = load_mono(a.reference)
    r = match_timbre(y, ref, max_db=a.max_db, tol_st=a.tol)
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), r["audio"], SR, subtype="PCM_24")
    info = {"out": str(out), "st_before": round(r["st_before"], 2),
            "st_after": round(r["st_after"], 2), "tilt_db_oct": r["tilt_db_oct"]}
    if a.json:
        print(json.dumps(info, ensure_ascii=False))
    else:
        print(f"timbre-match {info['st_before']:+.2f} -> {info['st_after']:+.2f} 半音 "
              f"(tilt {info['tilt_db_oct']:+.1f} dB/oct) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())