"""集中的外部路径配置：所有工具都从这里读 settings.json。

换电脑时只需要改 <REPO>\\settings.json 这一个文件（不用改代码）：

{
  "rvc_python": "……/Runtime/python.exe",     # 跑 RVC 的 Python（RVC runtime 自带）
  "ffmpeg": "……/ffmpeg.exe",                # ffmpeg 可执行文件
  "rvc_core": "<REPO>/rvc_core",      # RVC 内核副本（infer/ configs/ assets/）
  "model_roots": [                            # RVC 模型目录，按顺序列出
    {"path": "<REPO>/rvc_models", "label": "本地模型（本地）", "writable": true},
    {"path": "E:/RVC/User_Data/models", "label": "附加只读目录", "writable": false}
  ]
}

没写进 settings.json 的键会退回默认值；文件不存在会自动生成。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SETTINGS_FILE = ROOT / "settings.json"

DEFAULTS = {
    "rvc_python": "",
    "ffmpeg": "",
    "rvc_core": str(ROOT / "rvc_core"),
    "model_roots": [
        {"path": str(ROOT / "rvc_models"), "label": "本地模型（本地）", "writable": True},
    ],
}


def load() -> dict:
    cfg = json.loads(json.dumps(DEFAULTS))  # deep copy
    if SETTINGS_FILE.exists():
        try:
            cfg.update(json.loads(SETTINGS_FILE.read_text(encoding="utf-8")))
        except Exception as exc:  # noqa: BLE001
            print(f"[settings] 读取失败，用默认值：{exc}")
    else:
        try:
            SETTINGS_FILE.write_text(json.dumps(cfg, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
        except OSError:
            pass
    return cfg


SETTINGS = load()


def get(key: str, default=None):
    return SETTINGS.get(key, default)


def rvc_python() -> Path | None:
    p = Path(str(SETTINGS.get("rvc_python") or ""))
    return p if str(p) and p.exists() else None


def ffmpeg() -> Path | None:
    """ffmpeg.exe：先看 settings.json，再看 PATH，最后看几个常见位置。"""
    cand = str(SETTINGS.get("ffmpeg") or "")
    if cand and Path(cand).exists():
        return Path(cand)
    import shutil
    hit = shutil.which("ffmpeg")
    if hit:
        return Path(hit)
    for c in (
        Path(os.environ.get("FFMPEG_DIR", "")) / "ffmpeg.exe",
        Path(r"C:\ffmpeg\bin\ffmpeg.exe"),
        Path(r"C:\tools\ffmpeg\bin\ffmpeg.exe"),
    ):
        try:
            if c.exists():
                return c
        except OSError:
            pass
    return None


def ensure_ffmpeg_on_path() -> str | None:
    """把 ffmpeg 所在目录塞进 PATH（audio-separator / pydub 都靠 PATH 找它）。"""
    exe = ffmpeg()
    if not exe:
        return None
    d = str(exe.parent)
    parts = os.environ.get("PATH", "").split(os.pathsep)
    if d not in parts:
        os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
    return d


def rvc_core() -> Path:
    return Path(str(SETTINGS.get("rvc_core") or (ROOT / "rvc_core")))


def model_roots(only_existing: bool = False) -> list[dict]:
    out = []
    for r in SETTINGS.get("model_roots") or []:
        p = Path(str(r.get("path") or ""))
        if only_existing and not p.is_dir():
            continue
        out.append({"path": str(p), "label": r.get("label") or p.name,
                    "writable": bool(r.get("writable"))})
    return out