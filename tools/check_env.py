"""环境自检：新电脑上先把找不到的东西一次性列出来。

    python tools\\check_env.py

退出码 0 = 全部就绪，1 = 有缺失项。
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import settings as cfg  # noqa: E402

ok_all = True


def ok(msg: str) -> None:
    print(f"  [ok]   {msg}")


def bad(msg: str, hint: str = "") -> None:
    global ok_all
    ok_all = False
    print(f"  [MISS] {msg}")
    if hint:
        print(f"         -> {hint}")


def section(title: str) -> None:
    print(f"\n== {title} ==")


# ---------- 1. Python + GPU ----------
section("Python / GPU")
print(f"  python {sys.version.split()[0]}  ({sys.executable})")
try:
    import torch

    cuda = torch.cuda.is_available()
    print(f"  torch {torch.__version__}  cuda={cuda}")
    if not cuda:
        bad("CUDA 不可用", "装的是 CPU 版 torch：pip install torch==2.4.1+cu121 torchaudio==2.4.1+cu121 "
                           "--index-url https://download.pytorch.org/whl/cu121")
    elif torch.cuda.get_device_properties(0).total_memory < 5.5e9:
        print("  [warn] 显存 < 6 GB，建议把「单段推理时长」降到 65 s")
except Exception as e:  # noqa: BLE001
    bad(f"import torch 失败：{e}", "pip install torch==2.4.1+cu121 --index-url "
                                  "https://download.pytorch.org/whl/cu121")

for mod in ("audio_separator", "flask", "librosa", "soundfile", "numpy"):
    try:
        __import__(mod)
        ok(f"模块 {mod}")
    except Exception as e:  # noqa: BLE001
        bad(f"模块 {mod} 缺失：{e}", "pip install -r requirements.txt")

# ---------- 2. ffmpeg ----------
section("ffmpeg")
ff = cfg.ffmpeg()
if ff and Path(ff).exists():
    try:
        v = subprocess.run([str(ff), "-version"], capture_output=True, text=True,
                           encoding="utf-8", errors="replace").stdout.splitlines()[0]
        ok(f"ffmpeg {v[:60]}")
    except Exception:  # noqa: BLE001
        ok(f"ffmpeg 存在：{ff}")
else:
    bad("找不到 ffmpeg", "见 DEPLOY.md 第 3 步；或在 settings.json 里写 \"ffmpeg\": \".../ffmpeg.exe\"")

# ---------- 3. RVC 运行时 ----------
section("RVC 运行时")
py = cfg.rvc_python()
if py and Path(py).exists():
    ok(f"rvc_python = {py}")
    try:
        out = subprocess.run([str(py), "-c", "import torch;print(torch.__version__, torch.cuda.is_available())"],
                             capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
        print(f"         {out.stdout.strip() or out.stderr.strip()[:120]}")
    except Exception as e:  # noqa: BLE001
        bad(f"rvc_python 跑不起来：{e}")
else:
    bad("settings.json 里 rvc_python 未配置或不存在", "见 DEPLOY.md 第 4 步")

core = Path(str(cfg.get("rvc_core") or ROOT / "rvc_core"))
need = [
    "infer/modules/vc/modules.py",
    "infer/modules/vc/pipeline.py",
    "configs/config.py",
    "assets/hubert/hubert_base.pt",
    "assets/rmvpe/rmvpe.pt",
    "assets/pretrained_v2/f0G40k.pth",
]
if core.exists():
    missing = [n for n in need if not (core / n).exists()]
    if missing:
        bad(f"rvc_core 缺 {len(missing)} 个文件：{', '.join(missing[:3])}…",
            "从 RVC 整合包复制 infer/ configs/ assets/ 到 rvc_core/，见 DEPLOY.md 第 4 步")
    else:
        ok(f"rvc_core 结构完整：{core}")
        pipe = (core / "infer/modules/vc/pipeline.py").read_text(encoding="utf-8", errors="replace")
        ok("INDEX_K 补丁在位" if "INDEX_K" in pipe else "内核是原版（没有 INDEX_K，index_k 参数会被忽略）")
else:
    bad(f"rvc_core 不存在：{core}", "见 DEPLOY.md 第 4 步")

# ---------- 4. 模型 ----------
section("模型")
sep_dir = Path(str(cfg.get("sep_model_dir") or ROOT / "models" / "mdxc"))
sep_need = {
    "mel_band_roformer_kim_ft2_bleedless_unwa.ckpt": "默认人声/伴奏分离",
    "mel_band_roformer_karaoke_becruily.ckpt": "去和声（可选）",
    "dereverb-echo_mel_band_roformer_sdr_13.4843_v2.ckpt": "去混响（可选）",
}
for name, why in sep_need.items():
    p = sep_dir / name
    if p.exists():
        ok(f"{name} ({p.stat().st_size // 1048576} MB) — {why}")
    else:
        print(f"  [warn] 缺 {name} — {why}（首次用到时 audio-separator 会自动下载）")

roots = cfg.model_roots(only_existing=False)
found_models = []
for r in roots:
    d = Path(r["path"])
    if not d.exists():
        print(f"  [warn] 模型目录不存在：{d}（{r['label']}）")
        continue
    for sub in sorted(d.iterdir()):
        if sub.is_dir() and any(sub.glob("*.pth")):
            found_models.append(sub)
if found_models:
    ok(f"RVC 模型 {len(found_models)} 个：" + "、".join(f"{m.name}" for m in found_models[:5])
       + (" …" if len(found_models) > 5 else ""))
else:
    bad("没有任何 RVC 模型（.pth）", "放到 rvc_models/<名字>/ 下，或用界面「⬆ 导入模型」")

# ---------- 5. settings.json ----------
section("配置")
if cfg.SETTINGS_FILE.exists():
    ok(f"settings.json：{cfg.SETTINGS_FILE}")
    print("         " + json.dumps({k: cfg.SETTINGS.get(k) for k in ("rvc_python", "ffmpeg", "rvc_core")},
                                   ensure_ascii=False))
else:
    bad("没有 settings.json", "复制 settings.example.json 为 settings.json 后改路径")

print("\n" + ("全部就绪，双击 start.bat 即可。" if ok_all else "有缺失项，按上面 [MISS] 的提示补齐。"))
sys.exit(0 if ok_all else 1)