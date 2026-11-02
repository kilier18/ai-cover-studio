# -*- coding: utf-8 -*-
"""PitchIn-Q 自检：为每种缺陷各造一对音频，检查检测器能否报出对应类型。

跑：D:\\AI\\RVCover\\py311\\python.exe tools/selftest_pitchinq.py
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pitchinq as P                                        # noqa: E402

SR = P.SR
OUT = Path(tempfile.mkdtemp(prefix="pitchinq_selftest_"))
NOTES = [(0.0, 0.8, 220.0), (0.8, 1.6, 293.66), (1.6, 2.4, 246.94), (2.4, 3.2, 329.63),
         (3.2, 4.0, 261.63), (4.0, 4.8, 349.23), (4.8, 5.6, 293.66), (5.6, 6.4, 220.0)]


def synth(notes, detune=0.0, extra=(), drop=(), hf=0.0, gain=None, vib=0.004):
    n = int(6.4 * SR)
    x = np.zeros(n)
    t = np.arange(n) / SR
    for i, (s, e, f) in enumerate(notes):
        if i in drop:
            continue
        a, b = int(s * SR), int(e * SR)
        tt = np.arange(b - a) / SR
        f2 = f * 2 ** (detune / 1200)
        v = 1 + vib * np.sin(2 * np.pi * 5 * tt)
        ph = 2 * np.pi * f2 * np.cumsum(v) / SR
        env = np.minimum(1, np.minimum(tt / 0.02, (len(tt) / SR - tt) / 0.05))
        x[a:b] += (np.sin(ph) + 0.35 * np.sin(2 * ph) + 0.15 * np.sin(3 * ph)) * env
        if i in extra:
            c = int((s + 0.22) * SR)
            d = c + int(0.12 * SR)
            x[c:d] += 0.9 * np.sin(2 * np.pi * f2 * np.arange(d - c) / SR) * np.hanning(d - c)
    if hf > 0:                       # 6k 齿音成分（真实录音里 HF 不是量化噪声）
        rng = np.random.default_rng(7)
        band = rng.standard_normal(len(x))
        from scipy.signal import butter, sosfilt
        sos = butter(4, [5000, 9000], btype="bandpass", fs=SR, output="sos")
        x = x + hf * 0.06 * sosfilt(sos, band)
    if gain is not None:
        x = x * gain(t)
    return (0.25 * x / np.max(np.abs(x))).astype(np.float32)


def run(name, src, conv, expect, forbid=()):
    sp, cp = OUT / "_src.wav", OUT / "_conv.wav"
    sf.write(sp, src, SR, subtype="FLOAT")
    sf.write(cp, conv, SR, subtype="FLOAT")
    rep = P.analyze(P.load_mono(sp), P.load_mono(cp), {})
    types = {s["type"] for s in rep["segments"]}
    ok = expect in types and not (set(forbid) & types)
    print(f"[{'PASS' if ok else 'FAIL'}] {name:<16} 期望 {expect:<16} 实得 {sorted(types)}")
    if not ok:
        for s in rep["segments"]:
            print(f"        - {s['type']:<16} {s['start']:6.2f}-{s['end']:6.2f} {s['detail']}")
    return ok


def main() -> int:
    base = synth(NOTES)
    # 重复字：把一个音切成两段，第二段是新的起音（和声/双轨残留的典型形态）
    split = list(NOTES)
    s0, e0, f0 = split[3]
    mid = (s0 + e0) / 2 - 0.05
    split[3] = (s0, mid, f0)
    split.insert(4, (mid + 0.06, e0, f0))
    cases = [
        ("音准偏低 80c", base, synth(NOTES, detune=-80), "pitch_off"),
        ("八度跳变", base, synth(NOTES, detune=1200), "pitch_jump"),
        ("多出来的字", base, synth(split), "doubled_syllable"),
        ("漏字", base, synth(NOTES, drop={5}), "dropped_syllable"),
        ("音量跳 8dB", base, synth(NOTES, gain=lambda t: np.where(t > 3.2, 2.5, 1.0)), "level_jump"),
        ("齿音变亮", base, synth(NOTES, hf=0.9), "harsh"),
        ("过度平滑", synth(NOTES, vib=0.030), synth(NOTES, vib=0.0), "over_smooth"),
    ]
    results = [run(*c) for c in cases]
    print(f"\n{sum(results)}/{len(results)} 通过")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())