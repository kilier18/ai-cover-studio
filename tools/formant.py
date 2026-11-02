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
    A = np.fft.rfft(np.concatenate([a, np.zeros(n_fft - len(a))]))
    A = np.maximum(np.abs(A), 1e-8)
    return 1.0 / A


def _warp(env, factor, n_fft=N_FFT):
    nb = len(env)
    # H'(f) = H(f / factor)：factor<1 → 共振峰整体下移（声道变长，听感更低）
    idx = np.arange(nb) / max(factor, 1e-6)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, nb - 1)
    frac = idx - lo
    lo = np.clip(lo, 0, nb - 1)
    return env[lo] * (1 - frac) + env[hi] * frac


def shift(y, sr, factor):
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