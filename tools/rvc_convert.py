#!/usr/bin/env python
"""Stage 2: RVC voice conversion, isolated from the vendor RVC install.

Runs on RVC runtime's *own* embedded Python (3.9 + torch 2.0.0+cu118) but against a
private copy of `RVC_Core/{infer,configs,assets/hubert,assets/rmvpe}` living in
<REPO>\\rvc_core.  The vendor RVC installation is only ever read
(models / index files are opened read-only); nothing is written back to it.

Usage (must be started with Runtime\\python.exe):
  python rvc_convert.py --input vocals.wav --output converted.wav
         --model-dir "F:/.../User_Data/models/10" --model model-x.pth
         [--index "F:/.../10/model-x.index"] [--pitch 0] [--f0-method rmvpe]
         [--index-rate 0.5] [--protect 0.33] [--filter-radius 3]
         [--rms-mix-rate 0.25] [--resample-sr 0] [--device cuda:0] [--fp32]
"""
from __future__ import annotations

import argparse
import os
import sys

# Windows 控制台默认 GBK，路径里出现「・」这类字符会让 print 直接崩（exit 1）。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
try:
    import settings as _cfg
except Exception:  # noqa: BLE001
    _cfg = None

CORE = Path(str((_cfg.get("rvc_core") if _cfg else "") or (HERE.parent / "rvc_core")))

# ffmpeg is needed by infer.lib.audio.load_audio; make sure it is reachable.
_FFMPEG_CANDIDATES = [Path(os.environ.get("FFMPEG_DIR", ""))]
if _cfg is not None and _cfg.ffmpeg():
    _FFMPEG_CANDIDATES.insert(0, _cfg.ffmpeg().parent)


def ensure_ffmpeg() -> None:
    import shutil

    if shutil.which("ffmpeg"):
        return
    for cand in _FFMPEG_CANDIDATES:
        if cand and (cand / "ffmpeg.exe").exists():
            os.environ["PATH"] = str(cand) + os.pathsep + os.environ.get("PATH", "")
            return
    raise RuntimeError("ffmpeg.exe not found on PATH")


def _main(argv=None) -> int:
    p = argparse.ArgumentParser(description="RVC inference (isolated copy of RVC runtime core)")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model-dir", required=True, help="directory holding the .pth")
    p.add_argument("--model", required=True, help=".pth filename inside --model-dir")
    p.add_argument("--index", default="", help="explicit .index path (optional)")
    p.add_argument("--pitch", type=int, default=0)
    p.add_argument("--f0-method", default="rmvpe")
    p.add_argument("--index-rate", type=float, default=0.5)
    p.add_argument("--protect", type=float, default=0.33)
    p.add_argument("--filter-radius", type=int, default=3)
    p.add_argument("--rms-mix-rate", type=float, default=0.25)
    p.add_argument("--resample-sr", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--fp32", action="store_true", help="disable half precision")
    p.add_argument("--no-match-rms", action="store_true",
                   help="do not scale the output back to the input loudness")
    p.add_argument("--peak", type=float, default=0.99, help="hard peak ceiling after matching")
    p.add_argument("--pre-gain", type=float, default=0.0,
                   help="dB boost applied to the input before conversion (quiet "
                        "phrases convert better when fed louder); the output is "
                        "matched back to the original input loudness")
    p.add_argument("--auto-level", type=float, default=0.0,
                   help="lift quiet phrases before conversion: compress the level "
                        "envelope with at most this many dB of extra gain (0 = off)")
    p.add_argument("--post-gain", type=float, default=0.0,
                   help="extra dB applied after the loudness match")
    p.add_argument("--max-segment", type=float, default=0.0,
                   help="longest stretch of audio handed to the model in one pass, "
                        "in seconds (0 = kernel default ~65s).  Bigger = fewer forced "
                        "cut points (fewer seams on long phrases) but more VRAM/time.")
    p.add_argument("--index-k", type=int, default=0,
                   help="how many neighbours the timbre index contributes per frame "
                        "(0 = kernel default 8).  Bigger = smoother/more averaged "
                        "timbre, smaller = sharper but less stable.")
    args = p.parse_args(argv)

    if not (CORE / "infer").is_dir():
        print(f"missing RVC core copy: {CORE}", file=sys.stderr)
        return 2
    ensure_ffmpeg()
    # Resolve paths *before* chdir(CORE), otherwise relative paths point at rvc_core.
    input_path = str(Path(args.input).resolve())
    output_path = str(Path(args.output).resolve())
    model_dir = str(Path(args.model_dir).resolve())
    index_path = str(Path(args.index).resolve()) if args.index else ""
    if index_path and not index_path.isascii():
        # faiss (C++) cannot open non-ASCII paths on Windows -> use an ASCII copy.
        import hashlib
        import shutil as _shutil

        cache = HERE.parent / "work" / "_indices"
        cache.mkdir(parents=True, exist_ok=True)
        tag = hashlib.sha1(index_path.encode("utf-8")).hexdigest()[:8]
        safe = cache / f"{Path(index_path).stem}_{tag}.index"
        if not safe.exists():
            _shutil.copyfile(index_path, safe)
        index_path = str(safe)
        print(f"index cached to ascii path: {safe}")
    os.chdir(CORE)
    sys.path.insert(0, str(CORE))

    # Roots read by infer/modules/vc/{modules,utils}.py and rmvpe.
    os.environ["weight_root"] = model_dir
    os.environ["index_root"] = str(Path(index_path).parent) if index_path else model_dir
    os.environ["rmvpe_root"] = str(CORE / "assets" / "rmvpe")

    # Config() has its own argparse; hide our argv from it.
    ours = sys.argv
    sys.argv = [ours[0]]

    import numpy as np
    import soundfile as sf
    import torch

    from configs.config import Config
    from infer.modules.vc.modules import VC

    # Reference loudness comes from the untouched input, so a --pre-gain boost
    # still lands the converted stem back at normal level.
    ref_rms = None
    try:
        import soundfile as _sf
        _src, _ = _sf.read(input_path, always_2d=True, dtype="float32")
        _src = _src.mean(axis=1)
        ref_rms = float(np.sqrt(np.mean(_src ** 2)))
    except Exception as exc:  # noqa: BLE001
        print(f"pre-gain: could not read reference loudness ({exc})")

    if args.auto_level:
        try:
            import soundfile as _sf
            raw, sr_in = _sf.read(input_path, always_2d=True, dtype="float32")
            mono = raw.mean(axis=1)
            win, hop = 2048, 512
            n = len(mono)
            frames = max(1, 1 + (n - win) // hop)
            rms = np.array([np.sqrt(np.mean(mono[i*hop:i*hop+win] ** 2) + 1e-12)
                            for i in range(frames)])
            target = float(np.percentile(rms[rms > 1e-4], 75)) if (rms > 1e-4).any() else 0.0
            if target > 1e-5:
                max_g = 10.0 ** (args.auto_level / 20.0)
                g = np.clip(target / np.maximum(rms, 1e-6), 1.0, max_g)
                # smooth the gain curve so it does not pump
                k = np.ones(9) / 9.0
                g = np.convolve(g, k, mode="same")
                g = np.clip(g, 1.0, max_g)
                idx = (np.arange(n) // hop).clip(0, frames - 1)
                mono = mono * g[idx]
                peak = float(np.abs(mono).max()) or 1.0
                if peak > 0.98:
                    mono = mono * (0.98 / peak)
                out = mono if raw.shape[1] == 1 else np.repeat(mono[:, None], raw.shape[1], axis=1)
                tmpdir = Path(__file__).resolve().parent.parent / "work" / "_gain"
                tmpdir.mkdir(parents=True, exist_ok=True)
                tmp = tmpdir / f"{Path(input_path).stem}__lvl{args.auto_level:.0f}.wav"
                _sf.write(str(tmp), out, sr_in, subtype="PCM_24")
                input_path = str(tmp)
                boosted = float((g > 1.02).mean() * 100)
                print(f"auto-level: target {20*np.log10(target):.1f} dB, "
                      f"boosted {boosted:.0f}% of frames, max +{args.auto_level:.0f} dB")
        except Exception as exc:  # noqa: BLE001
            print(f"auto-level skipped: {exc}")

    if args.pre_gain:
        try:
            import tempfile
            import soundfile as _sf
            raw, sr_in = _sf.read(input_path, always_2d=True, dtype="float32")
            g = 10.0 ** (args.pre_gain / 20.0)
            peak_raw = float(np.abs(raw).max()) or 1.0
            if peak_raw * g > 0.98:
                g = 0.98 / peak_raw
            boosted = raw * g
            tmpdir = Path(__file__).resolve().parent.parent / "work" / "_gain"
            tmpdir.mkdir(parents=True, exist_ok=True)
            tmp = tmpdir / f"{Path(input_path).stem}__g{args.pre_gain:+.0f}.wav"
            _sf.write(str(tmp), boosted, sr_in, subtype="PCM_24")
            input_path = str(tmp)
            print(f"pre-gain: {20 * np.log10(g):+.1f} dB -> {tmp.name}")
        except Exception as exc:  # noqa: BLE001
            print(f"pre-gain skipped: {exc}")

    config = Config()
    config.device = args.device if torch.cuda.is_available() else "cpu"
    config.is_half = not args.fp32 and config.device.startswith("cuda")

    # x_max = 免查询时长阈值（超过就要强制切段），x_center = 切点间距，
    # x_query = 在切点前后多久的窗口里找最安静的落点。
    if args.max_segment and args.max_segment > 0:
        n = float(args.max_segment)
        center, query = n * 0.85, n * 0.15
        if n <= 65:  # 不超过内核默认值，避免比原来切得更碎
            center, query = min(60.0, center), min(10.0, query)
        config.x_max = n
        # pipeline 用 range()/切片，必须是整数
        config.x_center = int(round(center))
        config.x_query = int(round(query))
        print(f"max-segment: {n:.0f}s per pass "
              f"(cut every ~{center:.0f}s, search window +/-{query:.0f}s)")

    if args.index_k and args.index_k > 0:
        from infer.modules.vc import pipeline as _pipe

        _pipe.INDEX_K = int(args.index_k)
        print(f"index-k: {args.index_k} neighbours per frame")

    vc = VC(config)
    vc.get_vc(args.model)
    info, (out_sr, audio) = vc.vc_single(
        0,
        input_path,
        args.pitch,
        None,
        args.f0_method,
        index_path,
        None,
        args.index_rate,
        args.filter_radius,
        args.resample_sr,
        args.rms_mix_rate,
        args.protect,
    )
    if audio is None:
        print(f"conversion failed: {info}", file=sys.stderr)
        return 1
    audio = np.asarray(audio)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    # the vendor kernel's fork of the RVC pipeline hands back int16-scale samples
    # (pipeline.py: `(audio_opt * max_int16).astype(np.int16)`), so normalize.
    raw_peak = float(np.abs(audio).max()) if audio.size else 0.0
    if raw_peak > 1.5:
        audio /= 32768.0
        print(f"normalized int16-scale output (raw peak {raw_peak:.0f})")

    # Match the loudness of the converted stem to the source stem so mixing with
    # the instrumental stays sane; never push the peak above --peak.
    if not args.no_match_rms and audio.size:
        try:
            if ref_rms is not None:
                src_rms = ref_rms
            else:
                src, _ = sf.read(input_path, always_2d=True, dtype="float32")
                src = src.mean(axis=1)
                src_rms = float(np.sqrt(np.mean(src ** 2)))
            out_rms = float(np.sqrt(np.mean(audio ** 2)))
            if src_rms > 1e-6 and out_rms > 1e-6:
                gain = src_rms / out_rms
                peak = float(np.abs(audio).max())
                ceiling = args.peak / peak if peak > 0 else 1.0
                applied = min(gain, ceiling)
                audio *= applied
                print(f"rms match: source {20*np.log10(src_rms):.1f} dB, "
                      f"converted {20*np.log10(out_rms):.1f} dB, gain {20*np.log10(applied):+.1f} dB")
        except Exception as exc:  # noqa: BLE001
            print(f"rms match skipped: {exc}")

    if args.post_gain:
        audio = audio * (10.0 ** (args.post_gain / 20.0))
        if np.abs(audio).max() > args.peak:
            audio = audio * (args.peak / np.abs(audio).max())
        print(f"post-gain: {args.post_gain:+.1f} dB")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio, int(out_sr))
    print(f"OK {info} -> {output_path} (sr={out_sr}, samples={len(audio)})")
    return 0


def main(argv=None) -> int:
    """拿显存锁再跑：RVC 与 PitchIn-Q 诊断不能同时占这张卡。

    网页端作业会自己先拿锁，并通过环境变量 GPULOCK_HELD=1 传下来，
    这时这里直接放行（同一进程树重入）。命令行单独跑时，
    `GPULOCK_WAIT=秒` 可以排队等（默认 0 = 显存忙就退出 3）。
    """
    try:
        import gpulock
    except Exception:                                     # noqa: BLE001
        gpulock = None
    if gpulock is None or os.environ.get(gpulock.ENV_FLAG) == "1":
        return _main(argv)
    wait = float(os.environ.get("GPULOCK_WAIT", "0") or 0)
    try:
        handle = gpulock.hold("rvc", timeout=wait)
    except gpulock.GpuBusy as exc:
        print(f"GPU busy: {exc}", file=sys.stderr)
        return 3
    try:
        return _main(argv)
    finally:
        handle.release()


if __name__ == "__main__":
    raise SystemExit(main())