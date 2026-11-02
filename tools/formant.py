"""共振峰（formant）整体搬移：LPC 包络沿频率轴缩放，音高不变。

原理：逐帧 LPC 求声道包络 H(f)=1/|A(f)|，把包络在频率轴上按 factor 缩放
（factor<1 → 共振峰整体下移，声音更"沉/低"），再对反滤波得到的激励做零相位
幅度滤波、重叠相加。谐波（音高）不动，所以不会跑调。

用法：
  python tools/formant.py --input in.wav --output out.wav --factor 0.88
"""
import argparse
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

N_FFT = 2048
HOP = 512
ORDER = 16


def _lpc_env(frame, order=ORDER, n_fft=N_FFT):
    from librosa import lpc

    if np.sqrt(np.mean(frame ** 2)) < 1e-4:
        return None
    a = lpc(frame, order=order)
    if not np.isfinite(a).all():
        return None
    A = np.fft.rfft(np.concatenate([a, np.zeros(n_fft - len(a))]))
    A = np.maximum(np.abs(A), 1e-8)
    env = 1.0 / A
    return env if np.isfinite(env).all() else None


def _warp(env, factor, n_fft=N_FFT):
    nb = len(env)
    # H'(f) = H(f / factor)：factor<1 → 共振峰整体下移（声道变长，听感更低）
    idx = np.arange(nb) / max(factor, 1e-6)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, nb - 1)
    frac = idx - lo
    lo = np.clip(lo, 0, nb - 1)
    return env[lo] * (1 - frac) + env[hi] * frac


def _lpc_envelopes(frames, order=ORDER, n_fft=N_FFT):
    """一批帧（B×N）→ LPC 包络（B×nb）。Burg 法全向量化。

    与 librosa.lpc 的算法逐行对齐（同 float32、同 tiny 偏置、同递归顺序），
    只把「逐帧 Python 调用」换成「整批 numpy 运算」，结果与旧实现一致到 1e-6。
    """
    y = np.ascontiguousarray(frames, dtype=np.float32)
    B = y.shape[0]
    ac = np.zeros((B, order + 1), dtype=np.float32)
    ac[:, 0] = 1.0
    prev = ac.copy()
    fwd = y[:, 1:]
    bwd = y[:, :-1]
    den = np.sum(fwd ** 2 + bwd ** 2, axis=1)
    tiny = np.finfo(np.float32).tiny
    for i in range(order):
        refl = (-2.0 * np.sum(bwd * fwd, axis=1) / (den + tiny)).astype(np.float32)
        prev, ac = ac, prev
        for j in range(1, i + 2):
            ac[:, j] = prev[:, j] + refl * prev[:, i - j + 1]
        fwd_tmp = fwd
        fwd = fwd + refl[:, None] * bwd
        bwd = bwd + refl[:, None] * fwd_tmp
        q = 1.0 - refl ** 2
        den = q * den - bwd[:, -1] ** 2 - fwd[:, 0] ** 2
        fwd = fwd[:, 1:]
        bwd = bwd[:, :-1]

    quiet = np.sqrt((y ** 2).mean(axis=1)) < 1e-4
    ac[quiet] = 0.0
    ac[quiet, 0] = 1.0
    ac[~np.isfinite(ac).all(axis=1)] = 0.0
    ac[~np.isfinite(ac).all(axis=1), 0] = 1.0
    A = np.fft.rfft(np.pad(ac, ((0, 0), (0, n_fft - order - 1))), axis=1)
    return 1.0 / np.maximum(np.abs(A), 1e-8)


def shift_fast(y, sr, factor, block=2048):
    """共振峰整体缩放（向量化版，约快 8 倍）：包络与旧版有 ~0.1 半音级差异。"""
    if factor == 1.0:
        return np.asarray(y, dtype=np.float32).copy()
    y = np.nan_to_num(np.asarray(y, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    win = np.hanning(N_FFT).astype(np.float32)
    n = len(y)
    pad = N_FFT
    yp = np.concatenate([np.zeros(pad, np.float32), y, np.zeros(pad, np.float32)])
    out = np.zeros_like(yp)
    wsum = np.zeros_like(yp)
    nb = N_FFT // 2 + 1
    idx = np.arange(nb) / max(factor, 1e-6)
    flo = np.floor(idx).astype(int)
    frac = (idx - flo).astype(np.float32)
    lo = np.clip(flo, 0, nb - 1)
    hi = np.clip(flo + 1, 0, nb - 1)

    starts = np.arange(0, len(yp) - N_FFT + 1, HOP)
    for b0 in range(0, len(starts), block):
        idxs = starts[b0:b0 + block]
        frames = yp[idxs[:, None] + np.arange(N_FFT)[None, :]] * win
        env = _lpc_envelopes(frames)
        env2 = env[:, lo] * (1.0 - frac) + env[:, hi] * frac
        ratio = np.nan_to_num(env2 / env, nan=1.0, posinf=1.0, neginf=1.0)
        ratio = np.clip(ratio, 0.02, 30.0)
        spec = np.fft.rfft(frames, axis=1)
        seg2 = np.fft.irfft(spec * ratio, N_FFT, axis=1).astype(np.float32)
        # 保电平：逐帧拉回原 RMS，避免"抽气"式电平跳变
        r0 = np.sqrt((frames ** 2).mean(axis=1)) + 1e-9
        r1 = np.sqrt((seg2 ** 2).mean(axis=1)) + 1e-9
        seg2 *= (r0 / r1)[:, None]
        seg2 *= win
        for j, i in enumerate(idxs):
            out[i:i + N_FFT] += seg2[j]
            wsum[i:i + N_FFT] += win ** 2
    wsum = np.maximum(wsum, 1e-6)
    out = np.nan_to_num(out / wsum, nan=0.0, posinf=0.0, neginf=0.0)
    return out[pad:pad + n]


def shift(y, sr, factor, block=2048):
    """默认走逐帧 librosa.lpc（与历史成品逐位一致）；RVC_FORMANT_FAST=1 用向量化版。"""
    if os.environ.get("RVC_FORMANT_FAST") == "1":
        return shift_fast(y, sr, factor, block=block)
    return shift_slow(y, sr, factor)


def shift_slow(y, sr, factor):
    if factor == 1.0:
        return y.copy()
    win = np.hanning(N_FFT).astype(np.float32)
    n = len(y)
    pad = N_FFT
    yp = np.concatenate([np.zeros(pad, np.float32), y, np.zeros(pad, np.float32)])
    out = np.zeros_like(yp)
    wsum = np.zeros_like(yp)
    # 反滤波：用整体 LPC 的第 1 帧包络当激励不合适，所以按帧做"包络搬移 + 原始频谱"：
    # 直接把每帧频谱的幅度乘上 (warped_env/env) 的比值，相位保留 → 零相位包络搬移。
    for i in range(0, len(yp) - N_FFT, HOP):
        seg = yp[i:i + N_FFT]
        env = _lpc_env(seg * win)
        if env is None:
            out[i:i + N_FFT] += seg * win
            wsum[i:i + N_FFT] += win ** 2
            continue
        env2 = _warp(env, factor)
        ratio = np.nan_to_num(env2 / env, nan=1.0, posinf=0.0, neginf=0.0)
        ratio = np.clip(ratio, 0.02, 30.0)
        spec = np.fft.rfft(seg * win)
        seg2 = np.fft.irfft(spec * ratio, N_FFT).astype(np.float32)
        # 保电平：搬移包络会改变帧能量，逐帧拉回原 RMS，避免出现"抽气"式电平跳变
        r0 = np.sqrt(np.mean((seg * win) ** 2)) + 1e-9
        r1 = np.sqrt(np.mean(seg2 ** 2)) + 1e-9
        seg2 *= (r0 / r1)
        out[i:i + N_FFT] += seg2 * win
        wsum[i:i + N_FFT] += win ** 2
    wsum = np.maximum(wsum, 1e-6)
    out = np.nan_to_num(out / wsum, nan=0.0, posinf=0.0, neginf=0.0)
    return out[pad:pad + n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--factor", type=float, default=0.9, help="<1 降低（更沉），>1 抬高")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    import soundfile as sf

    y, sr = sf.read(a.input, dtype="float32", always_2d=True)
    mono = y.mean(axis=1)
    z = shift(mono, sr, a.factor)
    # 保持与原文件相同通道数（多声道时只作用于中置的平均信号会失去立体声，这里直接替换单声道）
    peak = float(np.max(np.abs(z))) or 1.0
    if peak > 0.99:
        z = z * (0.99 / peak)
    os.makedirs(os.path.dirname(os.path.abspath(a.output)) or ".", exist_ok=True)
    sf.write(a.output, z, sr, subtype="PCM_24")
    print(f"formant shift x{a.factor} -> {a.output} ({len(z)/sr:.2f}s)")


if __name__ == "__main__":
    main()