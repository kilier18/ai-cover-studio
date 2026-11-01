#!/usr/bin/env python
"""AI cover studio - local web UI (3 independent panels: separate / convert / mix).

Run detached (the .bat launcher does this):
    py311\\pythonw.exe tools\\webui\\app.py --port 8765 --open
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from pathlib import Path

from flask import Flask, jsonify, request, send_file

ROOT = Path(__file__).resolve().parent.parent.parent      # 仓库根目录
TOOLS = ROOT / "tools"
WORK = ROOT / "work"
UPLOADS = WORK / "_uploads"
SEP_MODELS = ROOT / "models" / "mdxc"
LOCAL_MODELS = ROOT / "rvc_models"          # 本地、和 rvc_core 并排的"自己的模型"目录
LOCAL_MODELS.mkdir(parents=True, exist_ok=True)

# 所有外部路径集中在 settings.json（由 tools/settings.py 统一读取），换电脑只改那一个文件。
sys.path.insert(0, str(TOOLS))
import settings as cfg                                    # noqa: E402

SETTINGS = cfg.SETTINGS
MODEL_ROOTS = cfg.model_roots(only_existing=True) or [
    {"path": str(LOCAL_MODELS), "label": "本地模型（本地）", "writable": True}]
RVC_PY = cfg.rvc_python() or Path(str(SETTINGS.get("rvc_python") or ""))
FFMPEG = cfg.ffmpeg() or Path("ffmpeg")
cfg.ensure_ffmpeg_on_path()
RVC_MODELS = Path(MODEL_ROOTS[-1]["path"])
DEFAULT_SEGMENT = 100.0   # 单段推理时长默认（65 = 内核原值；实测 90 气口更稳）
DEFAULT_INDEX_K = 28     # 检索邻居数默认（内核原值 8）

app = Flask(__name__, static_folder=str(Path(__file__).resolve().parent / "static"))
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()
GPU = threading.Semaphore(1)          # one heavy job at a time (6 GB card)


# --------------------------------------------------------------------------- helpers
def ffmpeg_bin() -> str:
    return shutil.which("ffmpeg") or str(FFMPEG)


def ffprobe(path: Path) -> dict:
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json",
                              "-show_format", "-show_streams", str(path)],
                             capture_output=True, text=True, encoding="utf-8",
                             errors="replace", timeout=30).stdout
        j = json.loads(out or "{}")
        dur = float(j.get("format", {}).get("duration") or 0)
        st = next((s for s in j.get("streams", []) if s.get("codec_type") == "audio"), {})
        return {"duration": round(dur, 2), "sample_rate": int(st.get("sample_rate") or 0),
                "channels": int(st.get("channels") or 0)}
    except Exception:  # noqa: BLE001
        return {"duration": 0, "sample_rate": 0, "channels": 0}


def list_rvc_models() -> list[dict]:
    """扫描 settings.json 里的所有模型目录；每个子文件夹（含 .pth）算一个模型。"""
    out = []
    for root in MODEL_ROOTS:
        base = Path(root["path"])
        if not base.is_dir():
            continue
        for d in sorted(base.iterdir()):
            if not d.is_dir():
                continue
            pths = sorted(d.glob("*.pth"))
            if not pths:
                continue
            idx = sorted(d.glob("*.index"))
            imgs = [f.name for f in d.iterdir()
                    if f.suffix.lower() in {".jpg", ".png", ".jpeg"}]
            out.append({
                "id": str(d),                       # 唯一：绝对路径
                "name": d.name,                      # 文件夹名（如 1..10）
                "pth": pths[0].name,
                "index": idx[0].name if idx else None,
                "index_mb": round(idx[0].stat().st_size / 1048576, 1) if idx else None,
                "image": imgs[0] if imgs else None,
                "model_dir": str(d),
                "root": root["label"],
                "writable": bool(root.get("writable")),
            })
    return out


SEP_LABELS = {
    "mel_band_roformer_kim_ft2_bleedless_unwa.ckpt": "Kim 无串音版（推荐 · 人声最干净）",
    "vocals_mel_band_roformer.ckpt": "Kim 原版（保留气声细节）",
    "model_bs_roformer_ep_317_sdr_12.9755.ckpt": "BS-Roformer（伴奏最好）",
    "mel_band_roformer_karaoke_becruily.ckpt": "Karaoke 版（伴奏带和声）",
}
SEP_DEFAULT = "mel_band_roformer_kim_ft2_bleedless_unwa.ckpt"
SEP_PURIFY = "mel_band_roformer_bleed_suppressor_v1.ckpt"


def list_sep_models() -> list[dict]:
    out = []
    if SEP_MODELS.is_dir():
        for p in sorted(SEP_MODELS.glob("*.ckpt")):
            if (p.name.lower().startswith(("melbandroformer_kim",))
                    or "bleed_suppressor" in p.name or "dereverb" in p.name):
                continue  # alias / 二次处理专用模型（净化、去混响）不进分离下拉
            out.append({
                "file": p.name,
                "label": SEP_LABELS.get(p.name, p.name),
                "size_mb": round(p.stat().st_size / 1048576, 1),
                "recommended": p.name == SEP_DEFAULT,
            })
    return out


def new_job(title: str, steps_label: list[str], reveal: str | None = None) -> str:
    jid = uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        JOBS[jid] = {"id": jid, "title": title, "status": "queued", "log": [],
                     "progress": 0, "step": 0, "steps": steps_label,
                     "result": None, "error": None, "proc": None,
                     "reveal": reveal,   # 完成后弹窗要显示/打开的保存位置
                     "t0": time.time(), "t_start": None, "cancel": False}
    return jid


def jlog(jid: str, line: str) -> None:
    line = line.rstrip()
    if not line:
        return
    with JOBS_LOCK:
        job = JOBS[jid]
        job["log"].append(line)
        if len(job["log"]) > 500:
            del job["log"][:-500]
        pcts = re.findall(r"(\d{1,3})%\|", line)
        if pcts:
            job["progress"] = max(job["progress"], min(99, int(pcts[-1])))
        fracs = re.findall(r"(\d+)/(\d+) \[", line)
        if fracs:
            done, total = fracs[-1]
            if int(total):
                job["progress"] = max(job["progress"], min(99, int(100 * int(done) / int(total))))


def finish(jid: str, result=None, error=None, canceled: bool = False) -> None:
    with JOBS_LOCK:
        job = JOBS[jid]
        job["status"] = "canceled" if canceled else ("error" if error else "done")
        job["error"] = error
        job["result"] = result
        job["progress"] = 100 if not canceled else job["progress"]
        job["proc"] = None
        job["elapsed"] = round(time.time() - (job["t_start"] or job["t0"]), 1)


def run_job(jid: str, steps: list[dict]) -> None:
    with GPU:
        with JOBS_LOCK:
            if JOBS[jid]["cancel"]:
                finish(jid, canceled=True)
                return
            JOBS[jid]["status"] = "running"
            JOBS[jid]["t_start"] = time.time()
        result: dict = {}
        try:
            for idx, step in enumerate(steps):
                with JOBS_LOCK:
                    JOBS[jid]["step"] = idx
                    JOBS[jid]["progress"] = 0
                cmd = step["cmd"]
                capture = step.get("capture") or {}
                jlog(jid, "$ " + " ".join(str(c) for c in cmd))
                env = dict(os.environ)
                env["PYTHONUNBUFFERED"] = "1"
                env["PYTHONIOENCODING"] = "utf-8"
                p = subprocess.Popen([str(c) for c in cmd], cwd=step.get("cwd") or str(ROOT),
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     env=env, bufsize=0)
                with JOBS_LOCK:
                    JOBS[jid]["proc"] = p
                found: dict = {}
                buf = b""

                def handle(raw: bytes) -> None:
                    text = raw.decode("utf-8", "replace").strip()
                    if not text:
                        return
                    jlog(jid, text)
                    for key, rx in capture.items():
                        m = re.search(rx, text)
                        if m:
                            found[key] = m.group(1)

                assert p.stdout is not None
                while True:
                    chunk = p.stdout.read(4096)
                    if not chunk:
                        break
                    buf += chunk
                    parts = re.split(rb"[\r\n]+", buf)
                    buf = parts.pop()
                    for raw in parts:
                        handle(raw)
                    if len(buf) > 4000:
                        handle(buf)
                        buf = b""
                handle(buf)
                p.wait()
                with JOBS_LOCK:
                    JOBS[jid]["proc"] = None
                    canceled = JOBS[jid]["cancel"]
                if canceled:
                    finish(jid, canceled=True)
                    return
                if p.returncode != 0:
                    finish(jid, error=f"命令失败 (exit {p.returncode})：{Path(str(cmd[1])).name}")
                    return
                result.update(found)
            finish(jid, result=result or None)
        except Exception as exc:  # noqa: BLE001
            finish(jid, error=f"{type(exc).__name__}: {exc}")


def start(jid: str, steps: list[dict]) -> None:
    threading.Thread(target=run_job, args=(jid, steps), daemon=True).start()


def safe_path(p: str) -> Path:
    path = Path(p)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not str(path).startswith(str(ROOT.resolve())):
        raise ValueError("path outside project root")
    return path


def model_by_id(mid: str) -> dict:
    for m in list_rvc_models():
        if m["id"] == str(mid):
            return m
    raise ValueError(f"model {mid} not found")


def rel(p: Path) -> str:
    return str(Path(p).resolve().relative_to(ROOT)).replace("\\", "/")


def rvc_cmd(vocal: Path, out: Path, p: dict, m: dict) -> list[str]:
    cmd = [str(RVC_PY), str(TOOLS / "rvc_convert.py"),
           "--input", str(vocal), "--output", str(out),
           "--model-dir", m["model_dir"], "--model", m["pth"],
           "--pitch", str(int(p.get("pitch", 0))),
           "--f0-method", p.get("f0_method", "crepe"),
           "--index-rate", str(float(p.get("index_rate", 0.50))),
           "--protect", str(float(p.get("protect", 0.40))),
           "--filter-radius", str(int(p.get("filter_radius", 7))),
           "--rms-mix-rate", str(float(p.get("rms_mix_rate", 0.30))),
           "--resample-sr", str(int(p.get("resample_sr", 0)))]
    if float(p.get("pre_gain", 10) or 0):
        cmd += ["--pre-gain", str(float(p.get("pre_gain", 10)))]
    if float(p.get("auto_level") or 0):
        cmd += ["--auto-level", str(float(p["auto_level"]))]
    # 必须显式传给 rvc_convert：它自己的 0 表示"内核原值 65s / k=8"，
    # 不传就会退回 65s，和界面显示的 90s 不一致
    cmd += ["--max-segment", str(float(p.get("max_segment") or DEFAULT_SEGMENT))]
    cmd += ["--index-k", str(int(p.get("index_k") or DEFAULT_INDEX_K))]
    if m.get("index") and p.get("index_mode", "auto") != "none":
        cmd += ["--index", str(Path(m["model_dir"]) / (p.get("index_file") or m["index"]))]
    if not p.get("is_half", True):
        cmd += ["--fp32"]
    return cmd


def tag_of(p: dict) -> str:
    keys = ("model_id", "pitch", "index_rate", "rms_mix_rate", "protect",
            "filter_radius", "f0_method", "resample_sr", "is_half", "index_mode", "pre_gain", "auto_level")
    payload = {k: p.get(k) for k in keys}
    # 只有离开默认值时才参与指纹，老的 4f046a 才不会因为多了两个滑杆而变
    # 两个滑杆改变的是实际音频结果，所以始终进指纹（不同段长/邻居数不能撞同一个文件）
    payload["max_segment"] = float(p.get("max_segment") or DEFAULT_SEGMENT)
    payload["index_k"] = int(p.get("index_k") or DEFAULT_INDEX_K)
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:6]


# --------------------------------------------------------------------------- routes
@app.post("/api/models/open")
def models_open():
    """在资源管理器里打开模型目录。body: {"root": "<label 或 path>"}"""
    want = (request.json or {}).get("root") or ""
    for r in MODEL_ROOTS:
        if want in (r["label"], r["path"]):
            subprocess.Popen(["explorer", r["path"]])
            return jsonify(ok=True, path=r["path"])
    subprocess.Popen(["explorer", str(LOCAL_MODELS)])
    return jsonify(ok=True, path=str(LOCAL_MODELS))


@app.post("/api/models/import")
def models_import():
    """把 .pth（可选 .index / 预览图）导入 本地本地模型目录。

    表单：files=多个文件，name=模型名（可空，默认用 .pth 文件名）
    会建一个 rvc_models/<name>/ 并把文件放进去。
    """
    files = [f for f in request.files.getlist("files") if f and f.filename]
    if not files:
        return jsonify(error="没有收到文件"), 400
    allowed = {".pth", ".index", ".jpg", ".jpeg", ".png"}
    picked = [f for f in files if Path(f.filename).suffix.lower() in allowed]
    if not any(f.filename.lower().endswith(".pth") for f in picked):
        return jsonify(error="至少要有一个 .pth 模型文件"), 400

    pth = next(f for f in picked if f.filename.lower().endswith(".pth"))
    name = (request.form.get("name") or "").strip()
    # 浏览器之外的调用（比如命令行）可能把中文名传成乱码；乱码就退回用文件名
    if not name or "\ufffd" in name:
        name = Path(pth.filename).stem
    name = re.sub(r'[\\/:*?"<>|\x00-\x1f]+', "_", name).strip(" .") or "model"

    dest = LOCAL_MODELS / name
    dest.mkdir(parents=True, exist_ok=True)
    saved = []
    for f in picked:
        target = dest / Path(f.filename).name
        f.save(str(target))
        saved.append(target.name)
    return jsonify(ok=True, dir=str(dest), saved=saved)


@app.get("/")
def index():
    return app.send_static_file("index.html")


@app.get("/api/models")
def api_models():
    return jsonify({"rvc": list_rvc_models(), "separation": list_sep_models(),
                    "rvc_ready": RVC_PY.exists(), "root": str(ROOT),
                    "roots": [{"label": r["label"], "path": r["path"],
                               "writable": bool(r.get("writable"))} for r in MODEL_ROOTS],
                    "local_models": str(LOCAL_MODELS)})


@app.post("/api/upload")
def api_upload():
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify({"error": "没有文件"}), 400
    UPLOADS.mkdir(parents=True, exist_ok=True)
    name = re.sub(r'[\\/:*?"<>|]+', "_", f.filename)
    dst = UPLOADS / name
    if dst.exists():                                    # keep both copies
        dst = UPLOADS / f"{dst.stem}_{int(time.time())}{dst.suffix}"
    f.save(str(dst))
    return jsonify({"path": str(dst), "rel": rel(dst), "name": dst.name,
                    "mb": round(dst.stat().st_size / 1048576, 2), **ffprobe(dst)})


@app.get("/api/info")
def api_info():
    p = safe_path(request.args["p"])
    if not p.exists():
        return jsonify({"error": "文件不存在"}), 404
    return jsonify({"path": str(p), "rel": rel(p), "name": p.name, **ffprobe(p)})


@app.post("/api/separate")
def api_separate():
    data = request.json or {}
    src = safe_path(data["input"])
    song = WORK / "separated" / re.sub(r'[\\/:*?"<>|]+', "_", str(Path(data.get("name") or src.stem)))
    stems = song / "stems"
    if data.get("outdir"):
        stems = safe_path(data["outdir"])
    cmd = [sys.executable, str(TOOLS / "separate.py"), str(src),
           "--outdir", str(stems),
           "--model", data.get("model") or SEP_DEFAULT,
           "--model-dir", str(SEP_MODELS)]
    if data.get("purify", False):
        cmd.append("--purify")
    if data.get("strip_backing"):
        cmd.append("--strip-backing")
    if data.get("dereverb", True):
        cmd.append("--dereverb")
        if data.get("keep_inst_reverb"):
            cmd += ["--keep-inst-reverb",
                    "--inst-reverb-gain", str(float(data.get("inst_reverb_gain") or 1.0))]
    if data.get("fp16", True):
        cmd += ["--fp16"]
    if data.get("batch"):
        cmd += ["--batch", str(int(data["batch"]))]
    if data.get("segment"):
        cmd += ["--segment", str(int(data["segment"]))]
    if data.get("overlap") not in (None, ""):
        cmd += ["--overlap", str(float(data["overlap"]))]

    steps = ["分离人声与伴奏"]
    if data.get("strip_backing"):
        steps.append("去除和声（只留主唱）")
    if data.get("purify", False):
        steps.append("净化人声（去串音）")
    if data.get("dereverb", True):
        steps.append("去混响 / 去回声")
    if data.get("dereverb", True) and data.get("keep_inst_reverb"):
        steps.append("伴奏保留原唱混响")
    jid = new_job(f"分离 · {src.name}", steps, reveal=str(stems))
    start(jid, [{"cmd": cmd,
                 "capture": {"vocals": r"^vocals:\s*(.+)$",
                             "instrumental": r"^instrumental:\s*(.+)$"}}])
    return jsonify({"job": jid, "stems_dir": str(stems)})


@app.post("/api/convert")
def api_convert():
    data = request.json or {}
    m = model_by_id(str(data["model_id"]))
    vocal = safe_path(data["vocal"])
    out_dir = WORK / "converted"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{vocal.stem}__m{m['name']}__{tag_of(data)}.wav"
    jid = new_job(f"转换 · {vocal.name} · {m['name']} 号模型", ["RVC 人声转换"],
                  reveal=str(out))
    start(jid, [{"cmd": rvc_cmd(vocal, out, data, m),
                 "capture": {"out": r"-> (.+?) \(sr=", "sr": r"sr=(\d+)",
                             "gain": r"gain ([-+][\d.]+) dB"}}])
    return jsonify({"job": jid, "out": str(out), "out_rel": rel(out)})


@app.post("/api/preview")
def api_preview():
    data = request.json or {}
    m = model_by_id(str(data["model_id"]))
    vocal = safe_path(data["vocal"])
    inst = safe_path(data["inst"]) if data.get("inst") else None
    prev = WORK / "preview"
    prev.mkdir(parents=True, exist_ok=True)
    start_s = float(data.get("start", 0))
    dur = float(data.get("duration", 15))
    tag = tag_of(data)
    clip = prev / f"clip_{vocal.stem}_{int(start_s)}_{int(dur)}.wav"
    out = prev / f"voice_{vocal.stem}_{int(start_s)}_{int(dur)}_m{m['name']}_{tag}.wav"
    mixed = prev / f"mix_{vocal.stem}_{int(start_s)}_{int(dur)}_m{m['name']}_{tag}.wav"

    def cut(src: Path, dst: Path) -> list[str]:
        return [ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error", "-ss", str(start_s),
                "-t", str(dur), "-i", str(src), "-c:a", "pcm_s24le", str(dst)]

    steps = [{"cmd": cut(vocal, clip)},
             {"cmd": rvc_cmd(clip, out, data, m),
              "capture": {"sr": r"sr=(\d+)", "gain": r"gain ([-+][\d.]+) dB"}}]
    labels = ["截取试听片段", "RVC 转换"]
    if inst and inst.exists():
        inst_clip = prev / f"inst_{vocal.stem}_{int(start_s)}_{int(dur)}.wav"
        steps.append({"cmd": cut(inst, inst_clip)})
        steps.append({"cmd": [sys.executable, str(TOOLS / "mix.py"), "--vocals", str(out),
                              "--instrumental", str(inst_clip), "--output", str(mixed),
                              "--format", "wav", "--vocal-gain", "1.5",
                              "--instrumental-gain", "-1.0"]})
        labels += ["截取伴奏片段", "混音试听"]
    jid = new_job(f"试听 · {vocal.name} · {m['name']} 号模型", labels,
                  reveal=str(mixed if len(steps) > 2 else out))
    start(jid, steps)
    return jsonify({"job": jid, "voice_rel": rel(out), "clip_rel": rel(clip),
                    "mix_rel": rel(mixed) if len(steps) > 2 else None})


@app.post("/api/mix")
def api_mix():
    data = request.json or {}
    vocal = safe_path(data["vocal"])
    inst = safe_path(data["inst"])
    fmt = data.get("format", "wav")
    out_dir = WORK / "final"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = data.get("name") or f"{vocal.stem}+{inst.stem}"
    safe_stem = re.sub(r'[\\/:*?\"<>|]+', "_", stem)
    out = out_dir / f"{safe_stem}.{fmt}"
    cmd = [sys.executable, str(TOOLS / "mix.py"),
           "--vocals", str(vocal), "--instrumental", str(inst), "--output", str(out),
           "--format", fmt,
           "--vocal-gain", str(float(data.get("vocal_gain", 0))),
           "--instrumental-gain", str(float(data.get("instrumental_gain", 0)))]
    if not data.get("limiter", True):
        cmd += ["--no-limiter"]
    rev = float(data.get("reverb") or 0)
    if rev > 0:
        cmd += ["--reverb", str(rev), "--reverb-decay", str(float(data.get("reverb_decay") or 1.8))]
    steps = ["混音与导出"]
    if rev > 0:
        steps.insert(0, f"给人声加混响（{rev:.0f}% / {float(data.get('reverb_decay') or 1.8):.1f}s）")
    jid = new_job(f"合成 · {out.name}", steps, reveal=str(out))
    start(jid, [{"cmd": cmd, "capture": {"mixed": r"mixed -> (.+?) \("}}])
    return jsonify({"job": jid, "out": str(out), "out_rel": rel(out)})


@app.post("/api/job/<jid>/cancel")
def api_cancel(jid: str):
    with JOBS_LOCK:
        job = JOBS.get(jid)
        if not job:
            return jsonify({"error": "unknown job"}), 404
        job["cancel"] = True
        proc = job.get("proc")
    if proc and proc.poll() is None:
        try:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True)
        except Exception:  # noqa: BLE001
            pass
    return jsonify({"ok": True})


@app.get("/api/job/<jid>")
def api_job(jid: str):
    with JOBS_LOCK:
        job = JOBS.get(jid)
        if not job:
            return jsonify({"error": "unknown job"}), 404
        out = {k: v for k, v in job.items() if k not in ("log", "proc")}
        out["log"] = job["log"][-40:]
        out["elapsed"] = round(time.time() - (job["t_start"] or job["t0"]), 1)
        return jsonify(out)


@app.get("/media")
def media():
    p = safe_path(request.args["p"])
    if not p.exists():
        return jsonify({"error": "not found"}), 404
    return send_file(str(p), conditional=True, as_attachment=bool(request.args.get("dl")))


@app.route("/api/open", methods=["GET", "POST"])
def api_open():
    data = request.get_json(silent=True) or {}
    raw = str(data.get("p") or request.args.get("p") or "").strip()
    if not raw:
        return jsonify({"ok": False, "error": "缺少要打开的路径"}), 400
    p = safe_path(raw)
    if p.exists() and p.is_file():
        target = p.parent
        subprocess.Popen(["explorer", "/select,", str(p)])
    else:
        target = p if p.exists() else p.parent
        subprocess.Popen(["explorer", str(target)])
    return jsonify({"ok": True, "dir": str(target), "path": str(p)})


# ---------------------------- 清理临时文件 ----------------------------
CLEAN_TARGETS = {
    "gain":     {"label": "增益临时文件", "paths": [WORK / "_gain"], "default": True,
                 "desc": "音量对齐时生成的中间 wav，单曲可能上百 MB"},
    "preview":  {"label": "试听片段", "paths": [WORK / "preview"], "default": True,
                 "desc": "第②步试听生成的 clip/voice/mix 小片段"},
    "uploads":  {"label": "上传副本", "paths": [WORK / "_uploads"], "default": True,
                 "desc": "拖进来的歌在 work 里的副本（原文件不受影响）"},
    "scratch":  {"label": "调试残留", "paths": [WORK / "_smoke", WORK / "_tune"], "default": True,
                 "desc": "自检/调参留下的临时目录"},
    "docs":     {"label": "日志", "paths": [WORK / "webui.log", WORK / "webui.out"], "default": False,
                 "desc": "运行日志，排查问题时有用"},
    "indices":  {"label": "音色索引缓存", "paths": [WORK / "_indices"], "default": False,
                 "desc": "faiss 打不开中文路径时的 ASCII 缓存，删了下次会自动重建（较大）"},
    "converted":{"label": "转换结果", "paths": [WORK / "converted"], "default": False,
                 "desc": "第②步产物，下一步还要用；已合成到成品后可以删"},
    "separated":{"label": "分离结果", "paths": [WORK / "separated"], "default": False,
                 "desc": "干声+伴奏，第③步还要用；成品满意后再删"},
}


def _tree_stat(paths) -> tuple[int, int]:
    files = size = 0
    for p in paths:
        if p.is_dir():
            for f in p.rglob("*"):
                if f.is_file():
                    files += 1
                    try:
                        size += f.stat().st_size
                    except OSError:
                        pass
        elif p.is_file():
            files += 1
            try:
                size += p.stat().st_size
            except OSError:
                pass
    return files, size


def _cleanup_scan() -> list[dict]:
    out = []
    for key, meta in CLEAN_TARGETS.items():
        files, size = _tree_stat(meta["paths"])
        out.append({
            "key": key, "label": meta["label"], "desc": meta["desc"],
            "default": meta["default"], "files": files, "bytes": size,
            "mb": round(size / 1024 / 1024, 1),
            "paths": [str(x) for x in meta["paths"] if x.exists()],
        })
    return out


@app.get("/api/cleanup")
def api_cleanup_scan():
    items = _cleanup_scan()
    return jsonify({"items": items, "total_mb": round(sum(i["bytes"] for i in items) / 1024 / 1024, 1),
                    "final_dir": str(WORK / "final")})


@app.post("/api/cleanup")
def api_cleanup_run():
    data = request.get_json(silent=True) or {}
    keys = data.get("keys") or [k for k, v in CLEAN_TARGETS.items() if v["default"]]
    kept_final = 0
    freed = 0
    errors: list[str] = []
    removed: list[str] = []
    for key in keys:
        meta = CLEAN_TARGETS.get(key)
        if not meta:
            continue
        for path in meta["paths"]:
            if not path.exists():
                continue
            files, size = _tree_stat([path])
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
                freed += size
                removed.append(f"{meta['label']}（{round(size/1024/1024,1)} MB）")
            except OSError as exc:
                errors.append(f"{path.name}: {exc}")
    if (WORK / "final").is_dir():
        kept_final = sum(1 for f in (WORK / "final").rglob("*") if f.is_file())
    return jsonify({"ok": not errors, "freed_mb": round(freed / 1024 / 1024, 1),
                    "removed": removed, "errors": errors, "kept_final": kept_final})


@app.post("/api/shutdown")
def api_shutdown():
    def bye() -> None:
        time.sleep(0.6)
        os._exit(0)
    threading.Thread(target=bye, daemon=True).start()
    return jsonify({"ok": True})


def _ensure_streams() -> None:
    """pythonw.exe has no console: sys.stdout/stderr are None and any print() dies."""
    WORK.mkdir(parents=True, exist_ok=True)
    log = WORK / "webui.log"
    for name in ("stdout", "stderr"):
        if getattr(sys, name, None) is None:
            setattr(sys, name, open(log, "a", encoding="utf-8", buffering=1))


def main() -> int:
    _ensure_streams()
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--open", action="store_true")
    args = ap.parse_args()
    UPLOADS.mkdir(parents=True, exist_ok=True)
    url = f"http://{args.host}:{args.port}"

    import socket

    with socket.socket() as s:
        s.settimeout(0.4)
        if s.connect_ex((args.host, args.port)) == 0:
            print(f"already running -> {url}")
            if args.open:
                webbrowser.open(url)
            return 0

    if args.open:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    print(f"AI cover studio -> {url}")
    app.run(host=args.host, port=args.port, threaded=True, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
