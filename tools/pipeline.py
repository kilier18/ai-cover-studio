#!/usr/bin/env python
"""End-to-end local AI cover pipeline (stages 1-3).

  song file
     -> [stage 1] BS-Roformer separation           (py311 + torch/CUDA)
     -> [stage 2] RVC voice conversion             (RVC runtime python 3.9)
     -> [stage 3] remix converted vocal + inst.    (py311)
     -> final.wav

Run with the isolated py311 interpreter:
  py311/python.exe tools/pipeline.py "song.mp3" --model-id 10 --pitch 0

All work happens under <REPO>\\work\\<song>.  The user's RVC runtime install
is only read (models/index files); separation models live in our own folder.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

# Windows 控制台默认 GBK，路径里出现「・」这类字符会让 print 直接崩（exit 1）。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import settings as _cfg  # noqa: E402

RVCORE = _cfg.rvc_core()
RVC_PY = _cfg.rvc_python() or Path(str(_cfg.get("rvc_python") or ""))
_roots = _cfg.model_roots(only_existing=True)
RVC_MODELS = Path(_roots[0]["path"]) if len(_roots) == 1 else (
    next((Path(r["path"]) for r in _roots if not r["writable"]), Path(_roots[0]["path"])))
FFMPEG = _cfg.ffmpeg()


def ffmpeg_bin() -> str:
    import shutil

    found = shutil.which("ffmpeg")
    if found:
        return found
    if FFMPEG.exists():
        return str(FFMPEG)
    raise RuntimeError("ffmpeg not found")


def resolve_model(model_id: str | None, model: str | None, index: str | None,
                  model_dir: str | None) -> tuple[Path, Path | None]:
    """Return (pth path, index path or None)."""
    if model_dir:
        d = Path(model_dir)
        if not model:
            raise SystemExit("--model-dir needs --model")
        pth = d / model
        idx = Path(index) if index else next(iter(sorted(d.glob("*.index"))), None)
        return pth, idx

    d = RVC_MODELS / str(model_id) if model_id else None
    if d is None or not d.is_dir():
        raise SystemExit(f"model dir not found: {d}")
    pths = sorted(d.glob("*.pth"))
    if model:
        pths = [p for p in pths if p.stem == Path(model).stem] or pths
    if not pths:
        raise SystemExit(f"no .pth in {d}")
    idx = Path(index) if index else (next(iter(sorted(d.glob("*.index"))), None))
    return pths[0], idx


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="song -> AI cover, end to end")
    ap.add_argument("input")
    ap.add_argument("--model-id", default=None, help="folder under the RVC runtime's models dir, e.g. 10")
    ap.add_argument("--model", default=None, help=".pth filename (default: first in folder)")
    ap.add_argument("--index", default=None, help="explicit .index (default: first in folder)")
    ap.add_argument("--model-dir", default=None, help="use an explicit model folder instead")
    ap.add_argument("--pitch", type=int, default=0, help="semitones, male->female +12, female->male -12")
    ap.add_argument("--index-rate", type=float, default=0.50)
    ap.add_argument("--protect", type=float, default=0.40)
    ap.add_argument("--f0-method", default="crepe")
    ap.add_argument("--rms-mix-rate", type=float, default=0.30)
    ap.add_argument("--filter-radius", type=int, default=7)
    ap.add_argument("--max-segment", type=float, default=90.0,
                    help="单段推理时长上限（秒），0/65 = 内核默认")
    ap.add_argument("--index-k", type=int, default=28,
                    help="音色检索邻居数，0/8 = 内核默认")
    ap.add_argument("--pre-gain", type=float, default=10.0,
                    help="dB push before RVC (default 12, the validated preset)")
    ap.add_argument("--auto-level", type=float, default=0.0,
                    help="boost quiet frames by up to N dB (0 = off, try 6 or 12)")
    ap.add_argument("--dereverb", dest="dereverb", action="store_true", default=True,
                    help="去混响/去回声（默认开，与界面一致）")
    ap.add_argument("--no-dereverb", dest="dereverb", action="store_false",
                    help="关闭去混响（棚录干净人声用）")
    ap.add_argument("--keep-inst-reverb", dest="keep_inst_reverb", action="store_true", default=False,
                    help="把原唱混响层按 1x 回填进伴奏（默认开，需去混响）")
    ap.add_argument("--no-keep-inst-reverb", dest="keep_inst_reverb", action="store_false",
                    help="伴奏不回填混响")
    ap.add_argument("--inst-reverb-gain", type=float, default=1.0,
                    help="回填增益：1.0=原量，2.0=明显")
    ap.add_argument("--strip-backing", dest="strip_backing", action="store_true", default=False,
                    help="remove harmony/double-tracked vocals (default ON)")
    ap.add_argument("--no-strip-backing", dest="strip_backing", action="store_false",
                    help="keep the harmony layer")
    ap.add_argument("--backing-model", default="mel_band_roformer_karaoke_becruily.ckpt")
    ap.add_argument("--vocal-gain", type=float, default=0.0)
    ap.add_argument("--instrumental-gain", type=float, default=0.0)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--format", default="wav", choices=["wav", "flac", "mp3", "m4a"])
    ap.add_argument("--reuse-stems", action="store_true", help="skip stage 1 if stems exist")
    ap.add_argument("--reuse-vocal", action="store_true", help="skip stage 2 if converted vocal exists")
    args = ap.parse_args(argv)

    src = Path(args.input).resolve()
    if not src.exists():
        raise SystemExit(f"input not found: {src}")
    outdir = Path(args.outdir).resolve() if args.outdir else ROOT / "work" / src.stem
    stems = outdir / "stems"
    outdir.mkdir(parents=True, exist_ok=True)

    pth, idx = resolve_model(args.model_id, args.model, args.index, args.model_dir)
    if not pth.exists():
        raise SystemExit(f"model not found: {pth}")

    # ---- stage 1: separation -------------------------------------------------
    voc, inst = stems / "vocals.wav", stems / "instrumental.wav"
    if args.reuse_stems and voc.exists() and inst.exists():
        print(f"[1/3] reuse stems -> {stems}")
    else:
        print(f"[1/3] separating {src.name} ...")
        import separate as sep  # same folder

        # normalise any container/codec to plain 44.1k stereo wav first (pydub needs ffmpeg)
        norm = outdir / "input.wav"
        if not (args.reuse_stems and norm.exists()):
            r = subprocess.run([ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error",
                                "-i", str(src), "-vn", "-ac", "2", "-ar", "44100",
                                "-c:a", "pcm_s24le", str(norm)])
            if r.returncode != 0:
                raise SystemExit("ffmpeg could not decode the input file")
        res = sep.separate(norm, stems, fp16=True, batch=2,
                           strip_backing=args.strip_backing,
                   dereverb=args.dereverb,
                   keep_inst_reverb=args.keep_inst_reverb,
                   inst_reverb_gain=args.inst_reverb_gain,
                           backing_model=args.backing_model)
        voc, inst = res["vocals"], res["instrumental"]
        print(f"      vocals -> {voc}\n      instrumental -> {inst}")

    # ---- stage 2: RVC conversion --------------------------------------------
    conv = outdir / "vocal_converted.wav"
    if args.reuse_vocal and conv.exists():
        print(f"[2/3] reuse converted vocal -> {conv}")
    else:
        print(f"[2/3] RVC conversion ({pth.stem}, pitch {args.pitch:+d}) ...")
        cmd = [str(RVC_PY), str(HERE / "rvc_convert.py"),
               "--input", str(voc), "--output", str(conv),
               "--model-dir", str(pth.parent), "--model", pth.name,
               "--pitch", str(args.pitch),
               "--f0-method", args.f0_method,
               "--index-rate", str(args.index_rate),
               "--protect", str(args.protect),
               "--rms-mix-rate", str(args.rms_mix_rate),
               "--filter-radius", str(args.filter_radius)]
        if args.pre_gain:
            cmd += ["--pre-gain", str(args.pre_gain)]
        if args.auto_level:
            cmd += ["--auto-level", str(args.auto_level)]
        if args.max_segment:
            cmd += ["--max-segment", str(args.max_segment)]
        if args.index_k:
            cmd += ["--index-k", str(args.index_k)]
        if idx:
            cmd += ["--index", str(idx)]
        r = subprocess.run(cmd)
        if r.returncode != 0:
            raise SystemExit(f"RVC conversion failed (exit {r.returncode})")

    # ---- stage 3: mix --------------------------------------------------------
    final = outdir / f"final.{args.format}"
    print("[3/3] mixing ...")
    cmd = [sys.executable, str(HERE / "mix.py"),
           "--vocals", str(conv), "--instrumental", str(inst),
           "--output", str(final), "--format", args.format,
           "--vocal-gain", str(args.vocal_gain),
           "--instrumental-gain", str(args.instrumental_gain)]
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit(f"mix failed (exit {r.returncode})")

    print(f"\nDONE -> {final}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    raise SystemExit(main())