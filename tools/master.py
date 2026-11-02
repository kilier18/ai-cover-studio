"""Matchering 母带匹配：把成品混音的响度/频响/音色对齐到原曲参考。

用法：
  python master.py --target final.wav --reference 原曲.mp3 --output out.wav [--format mp3]

只做最后一步「加分」：不改动分轨，只把成品的整体平衡拉回原曲的味道。
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

BASE = Path(__file__).resolve().parent.parent
BIN = BASE / "bin"
if str(BIN) not in sys.path:
    pass


def log(msg: str) -> None:
    print(msg, flush=True)


def probe(path: Path) -> dict:
    info = sf.info(str(path))
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    rms = float(np.sqrt(np.mean(data ** 2))) if data.size else 0.0
    return {
        "path": str(path),
        "sample_rate": sr,
        "duration": float(info.duration),
        "channels": data.shape[1],
        "peak_db": round(20 * np.log10(max(peak, 1e-9)), 2),
        "rms_db": round(20 * np.log10(max(rms, 1e-9)), 2),
    }


def ffmpeg_convert(src: Path, dst: Path, fmt: str, bitrate: str = "320k") -> None:
    exe = BIN / "ffmpeg.exe"
    cmd = [str(exe), "-y", "-v", "error", "-i", str(src)]
    if fmt == "mp3":
        cmd += ["-c:a", "libmp3lame", "-b:a", bitrate]
    elif fmt == "m4a":
        cmd += ["-c:a", "aac", "-b:a", bitrate]
    elif fmt == "flac":
        cmd += ["-c:a", "flac"]
    cmd.append(str(dst))
    subprocess.run(cmd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Matchering 母带匹配到参考曲")
    ap.add_argument("--target", required=True, help="待匹配的成品（混音结果）")
    ap.add_argument("--reference", required=True, help="参考曲目（通常就是原唱整曲）")
    ap.add_argument("--output", required=True, help="输出文件；扩展名决定格式（wav/mp3/flac/m4a）")
    ap.add_argument("--format", default="", help="覆盖输出格式；默认按输出扩展名")
    ap.add_argument("--bitrate", default="320k")
    ap.add_argument("--no-limiter", action="store_true", help="关闭 Matchering 内置限幅器")
    ap.add_argument("--max-length", type=float, default=900.0, help="参考/目标最长秒数（默认 900）")
    ap.add_argument("--fft-size", type=int, default=4096)
    ap.add_argument("--json", action="store_true", help="额外打印 JSON 摘要")
    args = ap.parse_args()

    target = Path(args.target).resolve()
    reference = Path(args.reference).resolve()
    output = Path(args.output).resolve()
    for p in (target, reference):
        if not p.exists():
            log(f"找不到文件：{p}")
            return 2

    fmt = (args.format or output.suffix.lstrip(".") or "wav").lower()
    output.parent.mkdir(parents=True, exist_ok=True)
    work = output.with_suffix(".match24.wav")

    try:
        import matchering as mg
        from matchering.defaults import LimiterConfig
    except Exception as exc:  # pragma: no cover
        log(f"matchering 不可用：{exc}")
        return 2

    before = probe(target)
    log(f"目标 {target.name}  {before['duration']:.1f}s  peak {before['peak_db']} dB  rms {before['rms_db']} dB")
    log(f"参考 {reference.name}")

    config = mg.Config(
        internal_sample_rate=44100,
        max_length=args.max_length,
        fft_size=args.fft_size,
        temp_folder=str(BASE / "work" / "_match_tmp"),
        limiter=None if args.no_limiter else LimiterConfig(),
    )

    t0 = time.time()
    matched = True
    try:
        mg.process(
            target=str(target),
            reference=str(reference),
            results=[mg.pcm24(str(work))],
            config=config,
        )
    except Exception as exc:  # noqa: BLE001
        # 匹配失败不能把整个导出搞崩：退化成「直接转码」，成品照样有
        matched = False
        elapsed = time.time() - t0
        log(f"Matchering 失败（{type(exc).__name__}: {exc}），改为直接转码，不做母带匹配")
    elapsed = time.time() - t0

    if not work.exists():
        # 没拿到匹配结果（失败或没产出）：直接转码，别让导出失败
        if matched:
            log("Matchering 没有产出文件，改为直接转码")
        if fmt == "wav":
            shutil.copyfile(str(target), str(output))
        else:
            ffmpeg_convert(target, output, fmt, args.bitrate)
        log(f"mastered -> {output} ({elapsed:.1f}s, 未做母带匹配)")
        if args.json:
            print(json.dumps({"ok": True, "output": str(output), "format": fmt,
                              "matched": False, "elapsed_s": round(elapsed, 2),
                              "before": before}, ensure_ascii=False), flush=True)
        return 0

    if fmt == "wav":
        shutil.move(str(work), str(output))
    else:
        ffmpeg_convert(work, output, fmt, args.bitrate)
        work.unlink(missing_ok=True)

    after = probe(output) if fmt in ("wav", "flac") else None
    msg = f"mastered -> {output} ({elapsed:.1f}s)"
    if after:
        msg += f"  peak {after['peak_db']} dB  rms {after['rms_db']} dB"
    log(msg)

    if args.json:
        print(json.dumps({
            "ok": True,
            "output": str(output),
            "format": fmt,
            "elapsed_s": round(elapsed, 2),
            "before": before,
            "after": after,
        }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())