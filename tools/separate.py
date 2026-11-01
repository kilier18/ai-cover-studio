#!/usr/bin/env python
"""Stage 1 of the local AI-cover pipeline: vocal / instrumental separation.

Wraps nomadkaraoke/python-audio-separator (MDXC / BS-Roformer on CUDA).

Usage
  python tools/separate.py <input_audio> [--outdir DIR] [--model CKPT]
                           [--model-dir DIR] [--fp16] [--batch N]
                           [--segment N] [--overlap F] [--chunk SEC]

Output (canonical names, always 44.1 kHz WAV):
  <outdir>/vocals.wav
  <outdir>/instrumental.wav

Env quirks handled here:
  * all HTTPS on this machine is intercepted by a local proxy on 127.0.0.1:443;
    python/certifi does not trust its root -> we inject a CA bundle built from
    the Windows root store (tmp/ca_bundle.pem, regenerated if missing).
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys

# Windows 控制台默认 GBK，路径里出现「・」这类字符会让 print 直接崩（exit 1）。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CA_BUNDLE = ROOT / "tmp" / "ca_bundle.pem"
WIN_ROOTS = ROOT / "tmp" / "win_roots.pem"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import settings as _cfg  # noqa: E402

DEFAULT_MODEL_DIR = Path(str(_cfg.get("sep_model_dir") or (ROOT / "models" / "mdxc")))
_cfg.ensure_ffmpeg_on_path()
DEFAULT_MODEL = "mel_band_roformer_kim_ft2_bleedless_unwa.ckpt"  # 与界面默认一致（验收通过）
DEREVERB_MODEL = "dereverb-echo_mel_band_roformer_sdr_13.4843_v2.ckpt"
FALLBACK_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"


def ensure_ca_bundle() -> None:
    """Make `requests` trust the machine's TLS interception root."""
    if not CA_BUNDLE.exists():
        import certifi

        parts = [Path(certifi.where()).read_text(encoding="utf-8", errors="ignore")]
        if WIN_ROOTS.exists():
            parts.append(WIN_ROOTS.read_text(encoding="utf-8", errors="ignore"))
        CA_BUNDLE.parent.mkdir(parents=True, exist_ok=True)
        CA_BUNDLE.write_text("\n".join(parts), encoding="utf-8")
    os.environ.setdefault("SSL_CERT_FILE", str(CA_BUNDLE))
    os.environ.setdefault("REQUESTS_CA_BUNDLE", str(CA_BUNDLE))


def separate(
    input_path: str | Path,
    outdir: str | Path,
    model: str = DEFAULT_MODEL,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    fp16: bool = False,
    batch: int | None = None,
    segment: int | None = None,
    overlap: float | None = None,
    chunk: float | None = None,
    log_level: int = 20,
    purify: bool = False,
    purify_model: str = "mel_band_roformer_bleed_suppressor_v1.ckpt",
    strip_backing: bool = False,
    backing_model: str = "mel_band_roformer_karaoke_becruily.ckpt",
    dereverb: bool = False,
    dereverb_model: str = DEREVERB_MODEL,
    keep_inst_reverb: bool = False,
    inst_reverb_gain: float = 1.0,
) -> dict[str, Path]:
    """Separate one file; returns {'vocals': Path, 'instrumental': Path}.

    With purify=True the vocal stem gets a second pass through the bleed
    suppressor and the original is kept as vocals_raw.wav.
    """
    ensure_ca_bundle()
    from audio_separator.separator import Separator

    input_path = Path(input_path)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    mdxc = {"segment_size": 256, "override_model_segment_size": False,
            "batch_size": None, "overlap": None, "pitch_shift": 0}
    if segment is not None:
        mdxc["segment_size"] = segment
    if batch is not None:
        mdxc["batch_size"] = batch
    if overlap is not None:
        mdxc["overlap"] = overlap

    sep = Separator(
        log_level=log_level,
        model_file_dir=str(model_dir),
        output_dir=str(outdir),
        output_format="WAV",
        use_native_fp16=fp16,  # mutually exclusive with use_autocast
        chunk_duration=chunk,
        mdxc_params=mdxc,
    )
    sep.load_model(model_filename=model)
    produced = sep.separate(str(input_path))

    result: dict[str, Path] = {}
    for name in produced:
        src = Path(name)
        if not src.is_absolute():
            src = outdir / src.name
        kind = stem_kind(src.name)
        if kind is None:
            continue
        dst = outdir / f"{kind}.wav"
        shutil.move(str(src), str(dst))
        result[kind] = dst
    for kind in ("vocals", "instrumental"):
        if kind not in result:
            raise RuntimeError(f"separator did not produce a {kind} stem (got {produced})")

    if strip_backing:
        raw2 = outdir / "vocals_with_harmony.wav"
        shutil.copy2(result["vocals"], raw2)
        result["vocals_with_harmony"] = raw2
        lead = strip_backing_vocals(raw2, outdir / "_lead", model=backing_model,
                                  model_dir=model_dir, fp16=fp16, log_level=log_level)
        shutil.move(str(lead), str(result["vocals"]))
        match_level(result["vocals"], raw2)     # 主唱轨偏轻时补回原干声响度
        print("backing removed: lead-only vocal stem written to vocals.wav")

    if purify:
        raw = outdir / "vocals_raw.wav"
        shutil.copy2(result["vocals"], raw)
        result["vocals_raw"] = raw
        clean = purify_vocals(raw, outdir / "_purify", model=purify_model,
                              model_dir=model_dir, fp16=fp16, log_level=log_level)
        shutil.move(str(clean), str(result["vocals"]))
        print("purified: bleed suppressor pass applied to vocals")

    if dereverb:
        wet = outdir / "vocals_wet.wav"
        shutil.copy2(result["vocals"], wet)
        result["vocals_wet"] = wet
        dry = dereverb_vocals(wet, outdir / "_dereverb", model=dereverb_model,
                              model_dir=model_dir, fp16=fp16, log_level=log_level)
        import numpy as np
        import soundfile as sf

        def _rms(path) -> float:
            a, _ = sf.read(str(path), always_2d=True)
            return float(np.sqrt((a.astype("float64") ** 2).mean()))

        rw, rd = _rms(wet), _rms(dry)
        drop_db = 20.0 * np.log10((rd + 1e-12) / (rw + 1e-12))
        if drop_db < -10.0:
            # 模型把整首当成了混响（人声极湿/极散时的典型失败），保留原始干声
            print(f"dereverb skipped: dry stem is {drop_db:.1f} dB below the input "
                  f"(model treated almost everything as reverb) -> keeping the original vocal")
            result["dereverb_skipped"] = True
            keep_inst_reverb = False
        else:
            shutil.move(str(dry), str(result["vocals"]))
            match_level(result["vocals"], wet)  # 去混响会掉一点响度，拉回来
            print(f"dereverb: dry vocal stem written to vocals.wav ({drop_db:.1f} dB removed)")
        rev_layer = outdir / "_dereverb" / "vocals_reverb.wav"
        if keep_inst_reverb and rev_layer.exists():
            import numpy as np
            import soundfile as sf
            dry_inst = outdir / "instrumental_dry.wav"
            if not dry_inst.exists():
                shutil.copy2(result["instrumental"], dry_inst)
            inst, ir = sf.read(str(result["instrumental"]), always_2d=True)
            rev, _ = sf.read(str(rev_layer), always_2d=True)
            if rev.shape[1] != inst.shape[1]:
                rev = np.repeat(rev.mean(axis=1, keepdims=True), inst.shape[1], axis=1)
            n = min(len(inst), len(rev))
            gain = float(inst_reverb_gain)
            merged = inst[:n].astype("float32") + rev[:n].astype("float32") * gain
            peak = float(np.abs(merged).max())
            if peak > 0.99:
                merged *= 0.99 / peak
            sf.write(str(result["instrumental"]), merged, ir, subtype="PCM_24")
            result["instrumental_dry"] = dry_inst
            print(f"instrumental reverb kept: vocal ambience mixed back at {gain:.2f}x "
                  f"-> instrumental.wav (clean version: instrumental_dry.wav)")
    return result


def stem_kind(filename: str) -> str | None:
    """Map a produced stem filename to 'vocals' / 'instrumental'.

    BS-RoFormer files look like `x_(Vocals)_model.wav` while the MelBand models
    emit `x_(vocals)_...` / `x_(other)_...`, so "other" must count as the
    instrumental complement.
    """
    import re

    low = filename.lower()
    # audio-separator tags the stem in parentheses: x_(vocals)_model.wav /
    # x_(other)_model.wav. The model name itself may contain "vocals", so the
    # tag must win over a plain substring search.
    m = re.search(r"\(([^)]+)\)", filename)
    tag = m.group(1).lower() if m else ""
    if tag:
        if "vocal" in tag or tag in {"dry", "lead"}:
            return "vocals"
        if "instrument" in tag or tag in {"other", "inst", "karaoke"}:
            return "instrumental"
    if "instrument" in low or "other" in low:
        return "instrumental"
    if "vocal" in low:
        return "vocals"
    return None


PURIFY_MODEL = "mel_band_roformer_bleed_suppressor_v1.ckpt"


def purify_vocals(
    vocals_path: str | Path,
    outdir: str | Path,
    model: str = PURIFY_MODEL,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    fp16: bool = True,
    log_level: int = 20,
) -> Path:
    """Second pass: run the bleed-suppressor on an already separated vocal stem.

    Bleed = instrument / backing-vocal leftovers inside the vocal stem. Returns
    the cleaned vocal file (`<outdir>/vocals.wav`), falling back to the input
    when the model produced nothing usable.
    """
    ensure_ca_bundle()
    from audio_separator.separator import Separator

    vocals_path = Path(vocals_path)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    sep = Separator(
        log_level=log_level,
        model_file_dir=str(model_dir),
        output_dir=str(outdir),
        output_format="WAV",
        use_native_fp16=fp16,
    )
    sep.load_model(model_filename=model)
    produced = sep.separate(str(vocals_path))
    files = [Path(x) if Path(x).is_absolute() else outdir / Path(x).name for x in produced]
    if not files:
        raise RuntimeError("purify pass produced no stems")
    # The bleed suppressor targets the *bleed* stem: its "(Bleed)" output is the
    # stuff to remove and its "(Instrumental)" output is the dry vocal we want.
    keep = next((f for f in files if "(instrumental)" in f.name.lower() or "(vocals)" in f.name.lower()), None)
    if keep is None:
        keep = next((f for f in files if "bleed" not in f.name.lower()), files[0])
    dst = outdir / "vocals.wav"
    if keep != dst:
        shutil.move(str(keep), str(dst))
    for f in files:
        if f != dst and f.exists():
            f.unlink()
    return dst


def match_level(
    path: str | Path,
    ref: str | Path,
    peak_cap: float = 0.98,
    max_gain_db: float = 12.0,
) -> float:
    """把 path 的 RMS 对齐到 ref 的 RMS（峰值封顶，最多 +max_gain_db）。返回实际增益。"""
    import numpy as np
    import soundfile as sf

    a, sr = sf.read(str(path), always_2d=True)
    b, _ = sf.read(str(ref), always_2d=True)
    ra = float(np.sqrt((a.astype("float64") ** 2).mean()))
    rb = float(np.sqrt((b.astype("float64") ** 2).mean()))
    if ra <= 1e-9 or rb <= 1e-9:
        return 0.0
    gain = rb / ra
    gain = min(gain, 10.0 ** (max_gain_db / 20.0))
    peak = float(np.abs(a).max())
    if peak > 1e-9 and peak * gain > peak_cap:
        gain = peak_cap / peak
    if abs(gain - 1.0) < 0.02:
        return 0.0
    sf.write(str(path), a * gain, sr, subtype="PCM_24")
    print(f"level match: {20 * np.log10(ra):.1f} dB -> {20 * np.log10(ra * gain):.1f} dB "
          f"(gain {20 * np.log10(gain):+.1f} dB)")
    return gain


def strip_backing_vocals(
    vocals_path: str | Path,
    outdir: str | Path,
    model: str = "mel_band_roformer_karaoke_becruily.ckpt",
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    fp16: bool = True,
    log_level: int = 20,
) -> Path:
    """Keep only the lead voice: drop harmonies / backing vocals from a vocal stem.

    A karaoke model applied to an isolated vocal stem puts the lead in its
    "(Vocals)" output and the doubled/harmonised layer in "(Instrumental)".
    Returns the lead-only file (falls back to the input when nothing usable).
    """
    ensure_ca_bundle()
    from audio_separator.separator import Separator

    vocals_path = Path(vocals_path)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    sep = Separator(log_level=log_level, model_file_dir=str(model_dir),
                    output_dir=str(outdir), output_format="WAV",
                    use_native_fp16=fp16)
    sep.load_model(model_filename=model)
    produced = sep.separate(str(vocals_path))
    files = [Path(x) if Path(x).is_absolute() else outdir / Path(x).name for x in produced]
    keep = next((f for f in files if "(vocals)" in f.name.lower()), None)
    if keep is None:
        keep = files[0]
    dst = outdir / "vocals_lead.wav"
    if keep != dst:
        shutil.move(str(keep), str(dst))
    for f in files:
        if f != dst and f.exists():
            f.unlink()
    return dst


def dereverb_vocals(
    vocals_path: str | Path,
    outdir: str | Path,
    model: str = DEREVERB_MODEL,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    fp16: bool = True,
    log_level: int = 20,
) -> Path:
    """Strip room reverb / echo from a vocal stem.

    The model emits two stems: `..._(dry)_...` is the direct sound and
    `..._(No dry)_...` is what was removed (use it to judge how much got cut).
    Returns the dry file. The stripped reverb/echo layer is kept next to it as
    `vocals_reverb.wav` (used to put the original ambience back into the
    instrumental for effect-heavy songs).
    """
    ensure_ca_bundle()
    from audio_separator.separator import Separator

    vocals_path = Path(vocals_path)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    sep = Separator(log_level=log_level, model_file_dir=str(model_dir),
                    output_dir=str(outdir), output_format="WAV",
                    use_native_fp16=fp16)
    sep.load_model(model_filename=model)
    produced = sep.separate(str(vocals_path))
    files = [Path(x) if Path(x).is_absolute() else outdir / Path(x).name for x in produced]

    def tag(f: Path) -> str:
        m = re.search(r"\(([^)]+)\)", f.name)
        return m.group(1).lower() if m else ""

    keep = None
    for f in files:
        t = tag(f)
        if "no dry" in t or "wet" in t or "reverb" in t:
            continue
        if "dry" in t or "vocal" in t:
            keep = f
            break
    if keep is None:
        keep = files[0]
    dst = outdir / "vocals_dereverb.wav"
    removed = None
    for f in files:
        t = tag(f)
        if f is not keep and ("no dry" in t or "wet" in t or "reverb" in t):
            removed = outdir / "vocals_reverb.wav"
            shutil.move(str(f), str(removed))
            break
    if keep != dst:
        shutil.move(str(keep), str(dst))
    for f in files:
        if f not in (dst, removed) and f.exists():
            f.unlink()
    return dst


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="vocal/instrumental separation (stage 1)")
    p.add_argument("input")
    p.add_argument("--outdir", default=None, help="default: work/<input stem>/")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--segment", type=int, default=None)
    p.add_argument("--overlap", type=float, default=None)
    p.add_argument("--chunk", type=float, default=None)
    p.add_argument("--dereverb", action="store_true",
                   help="再跑一遍去混响/去回声模型（慢约 0.6 倍，回声重的歌值得）")
    p.add_argument("--dereverb-model", default=DEREVERB_MODEL)
    p.add_argument("--keep-inst-reverb", action="store_true",
                   help="把原唱混响层回填进伴奏（需同时 --dereverb）")
    p.add_argument("--inst-reverb-gain", type=float, default=1.0,
                   help="回填增益：1.0=原量（自然），2.0=明显，0.5=克制")
    p.add_argument("--purify", action="store_true",
                   help="second pass with the bleed suppressor on the vocal stem")
    p.add_argument("--strip-backing", action="store_true",
                   help="drop harmonies/backing vocals, keep the lead voice only")
    p.add_argument("--backing-model",
                   default="mel_band_roformer_karaoke_becruily.ckpt",
                   help="model used by --strip-backing")
    args = p.parse_args(argv)

    inp = Path(args.input)
    if not inp.exists():
        print(f"input not found: {inp}", file=sys.stderr)
        return 2
    outdir = Path(args.outdir) if args.outdir else ROOT / "work" / inp.stem

    res = separate(inp, outdir, model=args.model, model_dir=args.model_dir,
                   fp16=args.fp16, batch=args.batch, segment=args.segment,
                   overlap=args.overlap, chunk=args.chunk, purify=args.purify,
                   strip_backing=args.strip_backing,
                   backing_model=args.backing_model,
                   dereverb=args.dereverb, dereverb_model=args.dereverb_model,
                   keep_inst_reverb=args.keep_inst_reverb,
                   inst_reverb_gain=args.inst_reverb_gain)
    for k, v in res.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())