#!/usr/bin/env python
"""Stage 3: mix the RVC-converted vocal back onto the instrumental.

Both inputs are resampled to a common rate, the vocal is placed as mono->stereo,
levels are applied in dB, a soft limiter prevents clipping, and the result is
written as 24-bit WAV (or FLAC/MP3 via ffmpeg when --format says so).

Usage:
  python tools/mix.py --vocals conv.wav --instrumental inst.wav --output final.wav
         [--vocal-gain 0] [--instrumental-gain 0] [--sample-rate 44100]
         [--no-limiter]
"""
from __future__ import annotations

import argparse
import sys

# Windows 控制台默认 GBK，路径里出现「・」这类字符会让 print 直接崩（exit 1）。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

from pathlib import Path


def _load(path: Path):
    import soundfile as sf

    data, sr = sf.read(str(path), always_2d=True, dtype="float32")
    return data, sr


def _resample(data, sr_in: int, sr_out: int):
    if sr_in == sr_out:
        return data
    import librosa

    # librosa handles multichannel when given (channels, samples)
    out = librosa.resample(data.T, orig_sr=sr_in, target_sr=sr_out, res_type="soxr_hq")
    return out.T


def _to_stereo(data):
    import numpy as np

    if data.shape[1] == 1:
        return np.repeat(data, 2, axis=1)
    if data.shape[1] > 2:
        return data[:, :2]
    return data


def _make_ir(sr: int, decay: float, predelay_ms: float, damp_hz: float):
    """Synthetic stereo reverb impulse: decaying noise, damped, energy-normalised."""
    import numpy as np
    from scipy.signal import lfilter

    n = max(int(sr * max(decay, 0.1)), 256)
    t = np.arange(n) / sr
    env = np.exp(-6.9078 * t / max(decay, 0.1))          # -60 dB at `decay`
    rng = np.random.default_rng(20261101)
    ir = rng.standard_normal((n, 2)) * env[:, None]
    if damp_hz and damp_hz < sr / 2:
        a = float(np.exp(-2 * np.pi * damp_hz / sr))
        for c in range(2):
            ir[:, c] = lfilter([1 - a], [1, -a], ir[:, c])
    ir /= np.sqrt((ir ** 2).sum(axis=0, keepdims=True)) + 1e-12
    pre = int(sr * predelay_ms / 1000.0)
    if pre > 0:
        ir = np.vstack([np.zeros((pre, 2)), ir])
    return ir


def apply_reverb(voc, sr: int, percent: float, decay: float,
                 predelay_ms: float = 20.0, damp_hz: float = 6000.0):
    """Send-style reverb: keep the dry signal, add a convolution tail.

    percent=0 returns the input untouched; 20 is a clearly audible but musical
    amount for a cover vocal that was de-reverbed before RVC.
    """
    import numpy as np
    from scipy.signal import fftconvolve

    if percent <= 0:
        return voc
    ir = _make_ir(sr, decay, predelay_ms, damp_hz)
    wet = np.empty_like(voc)
    for c in range(voc.shape[1]):
        # 必须用因果卷积：mode="same" 会把整条湿信号往前挪 (len(ir)-1)//2 个采样，
        # 1.8 秒混响就等于提前约 0.9 秒冒出来 —— 听感正是“混响抢在人声前面”。
        full = fftconvolve(voc[:, c], ir[:, c], mode="full")
        wet[:, c] = full[: voc.shape[0]]
    g = float(percent) / 100.0
    return voc + wet * g


def apply_eq(voc, sr: int, air_db: float = 0.0, presence_db: float = 0.0,
             body_db: float = 0.0):
    """三段架子/峰值 EQ，专治「比原唱亮太多 / 暗太多」。

    air_db     >0 更亮更空气感（8 kHz 以上高架），<0 收掉齿音与沙沙的亮边
    presence_db>0 人声更靠前（3 kHz 附近峰值），<0 退到伴奏里
    body_db    >0 更厚更暖（180 Hz 以下低架），<0 去掉浑浊的低频堆积
    """
    import numpy as np
    from scipy.signal import butter, filtfilt

    if not (air_db or presence_db or body_db):
        return voc

    def _shelf(x, fc, db, kind):
        g = 10 ** (db / 20.0)
        nyq = sr / 2.0
        fc = min(max(fc, 20.0), nyq * 0.95)
        b, a = butter(2, fc / nyq, btype=kind)
        band = filtfilt(b, a, x, axis=0, padlen=min(len(x) - 1, 4096))
        if kind == "low":
            # 低架：x + (g-1)*LP(x)
            return x + (g - 1.0) * band
        # 高架：x + (g-1)*(x - LP(x))
        return x + (g - 1.0) * (x - band)

    out = voc
    if body_db:
        out = _shelf(out, 180.0, body_db, "low")
    if presence_db:
        # 峰值：只抬 2~4 kHz，用带通做钟形
        nyq = sr / 2.0
        b, a = butter(2, [2200.0 / nyq, 4200.0 / nyq], btype="band")
        band = filtfilt(b, a, out, axis=0, padlen=min(len(out) - 1, 4096))
        out = out + (10 ** (presence_db / 20.0) - 1.0) * band
    if air_db:
        out = _shelf(out, 8000.0, air_db, "high")
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="mix converted vocal + instrumental")
    p.add_argument("--vocals", required=True)
    p.add_argument("--instrumental", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--vocal-gain", type=float, default=0.0, help="dB")
    p.add_argument("--instrumental-gain", type=float, default=0.0, help="dB")
    p.add_argument("--sample-rate", type=int, default=44100)
    p.add_argument("--no-limiter", action="store_true")
    p.add_argument("--format", default="wav", choices=["wav", "flac", "mp3", "m4a"])
    p.add_argument("--bitrate", default="320k")
    p.add_argument("--reverb", type=float, default=0.0,
                   help="混响量 %%（0=关；只加在转换后的人声上，用来把干声粘回伴奏的空间）")
    p.add_argument("--reverb-decay", type=float, default=1.8, help="混响时长（秒）")
    p.add_argument("--reverb-predelay", type=float, default=20.0, help="预延迟（毫秒）")
    p.add_argument("--reverb-damp", type=float, default=6000.0, help="混响高频衰减（Hz）")
    p.add_argument("--eq-air", type=float, default=0.0,
                   help="亮度：>0 更亮（8kHz 以上高架），<0 收齿音/亮边，单位 dB（±6 已很明显）")
    p.add_argument("--eq-presence", type=float, default=0.0,
                   help="靠前度：>0 人声更贴脸（3kHz 峰值），<0 退后，单位 dB")
    p.add_argument("--eq-body", type=float, default=0.0,
                   help="厚度：>0 更暖更厚（180Hz 低架），<0 去浑浊，单位 dB")
    args = p.parse_args(argv)

    import numpy as np
    import soundfile as sf

    sr = args.sample_rate
    voc, vsr = _load(Path(args.vocals))
    inst, isr = _load(Path(args.instrumental))
    voc = _to_stereo(_resample(voc, vsr, sr))
    inst = _to_stereo(_resample(inst, isr, sr))

    n = max(len(voc), len(inst))
    voc = np.pad(voc, ((0, n - len(voc)), (0, 0)))
    inst = np.pad(inst, ((0, n - len(inst)), (0, 0)))

    voc = voc * (10 ** (args.vocal_gain / 20.0))
    inst = inst * (10 ** (args.instrumental_gain / 20.0))
    if args.eq_air or args.eq_presence or args.eq_body:
        voc = apply_eq(voc, sr, args.eq_air, args.eq_presence, args.eq_body)
        print(f"eq: air {args.eq_air:+.1f} / presence {args.eq_presence:+.1f} / body {args.eq_body:+.1f} dB")
    if args.reverb and args.reverb > 0:
        voc = apply_reverb(voc, sr, args.reverb, args.reverb_decay,
                           args.reverb_predelay, args.reverb_damp)
        print(f"reverb: {args.reverb:.0f}% / {args.reverb_decay:.1f}s on the converted vocal")
    mix = voc + inst

    if not args.no_limiter:
        peak = float(np.abs(mix).max())
        if peak > 0.98:
            # gentle tanh knee: keep transients, avoid hard clipping
            mix = np.tanh(mix / 0.98) * 0.98

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.format == "wav":
        sf.write(str(out), mix, sr, subtype="PCM_24")
    else:
        tmp = out.with_suffix(".wav")
        sf.write(str(tmp), mix, sr, subtype="PCM_24")
        import shutil
        import subprocess

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        try:
            import settings as _cfg
            _ff = _cfg.ffmpeg()
        except Exception:  # noqa: BLE001
            _ff = None
        ffmpeg = str(_ff) if _ff else (shutil.which("ffmpeg") or "ffmpeg")
        subprocess.run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                        "-i", str(tmp), "-b:a", args.bitrate, str(out)], check=True)
        tmp.unlink()

    print(f"mixed -> {out} ({n / sr:.2f}s, peak={float(np.abs(mix).max()):.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())