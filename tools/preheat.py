#!/usr/bin/env python
"""把模型文件的热度提起来：顺序读一遍大文件，让 Windows 文件缓存 + 杀毒扫描先做掉。

这解决的是「每次跑一步都要先卡几十秒」——第一次读 600MB 的 ckpt / 300MB 的 hubert
要等磁盘和杀毒；读一遍之后，后续进程加载模型基本是内存速度。

用法（两个环境各跑一次）：
    py311\\python.exe tools\\preheat.py
    rvc_runtime\\python.exe tools\\preheat.py --no-import
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
CHUNK = 4 << 20


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f}{unit}" if unit in ("B", "KB") else f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}GB"


def targets() -> list[Path]:
    out: list[Path] = []
    for pat in ("models/mdxc/*.ckpt", "models/mdxc/*.onnx", "models/mdxc/*.pth",
                "rvc_models/*/*.pth", "rvc_models/*/*.index",
                r"rvc_runtime/assets/hubert/*.pt", r"rvc_runtime/assets/rmvpe/*.pt",
                r"rvc_runtime/assets/pretrained_v2/*"):
        out += sorted(ROOT.glob(pat))
    # 去重、按大小从大到小（大的最值得预热）
    uniq = {p: p for p in out if p.is_file()}
    return sorted(uniq.values(), key=lambda p: p.stat().st_size, reverse=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-import", action="store_true", help="只读文件，不 import torch")
    ap.add_argument("--max-seconds", type=float, default=120.0)
    args = ap.parse_args()

    t_all = time.time()
    if not args.no_import:
        t0 = time.time()
        try:
            import torch  # noqa: F401
            import librosa  # noqa: F401
            print(f"import torch+librosa: {time.time() - t0:.1f}s")
        except Exception as exc:  # noqa: BLE001
            print(f"import 失败（不影响预热）：{exc}")

    files = targets()
    total = sum(p.stat().st_size for p in files)
    print(f"要预热 {len(files)} 个文件，共 {human(total)}")
    done = 0
    for p in files:
        if time.time() - t_all > args.max_seconds:
            print("到时间上限，剩下的跳过")
            break
        t0 = time.time()
        n = 0
        try:
            with p.open("rb") as fh:
                while True:
                    b = fh.read(CHUNK)
                    if not b:
                        break
                    n += len(b)
        except Exception as exc:  # noqa: BLE001
            print(f"  跳过 {p.name}: {exc}")
            continue
        done += n
        mb = n / (1 << 20)
        sec = max(time.time() - t0, 1e-6)
        print(f"  {p.name:<58} {mb:7.1f}MB {sec:5.1f}s {mb / sec:6.0f}MB/s")
    print(f"完成 {human(done)} / {human(total)}，用时 {time.time() - t_all:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())