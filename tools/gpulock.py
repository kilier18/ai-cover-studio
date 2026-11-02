"""跨进程 GPU 互斥锁：RVC 与 PitchIn-Q 不能同时占显存。

设计要点
--------
* 锁是**文件**（`work/.gpu.lock`），所以跨进程有效：网页端作业、命令行 RVC、
  PitchIn-Q 诊断三方都看得见彼此。
* 持有者信息写进 `work/.gpu.owner.json`（pid/名字/开始时间），
  持有者进程已死时自动作废（避免 Ctrl+C 之后永久锁死）。
* **同一进程树内可重入**：网页端拿到锁之后会拉起子进程跑 rvc_convert.py，
  子进程通过环境变量 `GPULOCK_HELD=1` 认出"我爹已经占了"，直接放行。
* 只做"排队"，绝不杀别人的进程——被 RVC 占着就等，等不到就报忙。

用法
----
    from gpulock import hold, busy

    with hold("pitchinq", timeout=600):      # 最多等 10 分钟
        load_model()                          # 这里可以用显存

    busy()  ->  None 或 {"owner": "...", "pid": 123, "since": 12.3}
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "work"
LOCK = WORK / ".gpu.lock"
OWNER = WORK / ".gpu.owner.json"
ENV_FLAG = "GPULOCK_HELD"

STALE_AFTER = 30 * 60      # 30 分钟没进展就当陈旧锁（正常一次 RVC 不会这么久）


class GpuBusy(RuntimeError):
    """等不到显存锁。"""


def _pid_alive(pid: int) -> bool:
    if not pid or pid < 0:
        return False
    if os.name == "nt":
        import ctypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.OpenProcess.restype = ctypes.c_void_p
        k32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_bool, ctypes.c_uint32]
        k32.CloseHandle.argtypes = [ctypes.c_void_p]
        h = k32.OpenProcess(0x1000, False, int(pid))   # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        k32.CloseHandle(ctypes.c_void_p(h))
        return True
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def _read_owner() -> dict | None:
    try:
        info = json.loads(OWNER.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return info if isinstance(info, dict) else None


def busy() -> dict | None:
    """谁占着显存；没人占返回 None。陈旧锁会被顺手清掉。"""
    if not LOCK.exists():
        return None
    info = _read_owner() or {}
    pid = int(info.get("pid") or 0)
    age = time.time() - float(info.get("since") or 0)
    if pid and not _pid_alive(pid):
        _clear()
        return None
    if age > STALE_AFTER and not (pid and _pid_alive(pid)):
        _clear()
        return None
    info.setdefault("owner", "unknown")
    info["age"] = round(age, 1)
    return info


def _clear() -> None:
    for p in (LOCK, OWNER):
        try:
            p.unlink()
        except OSError:
            pass


def _try_take(owner: str) -> bool:
    try:
        fd = os.open(str(LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    payload = {"owner": owner, "pid": os.getpid(), "since": time.time(),
               "host": os.environ.get("COMPUTERNAME", "")}
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    OWNER.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return True


class _Handle:
    def __init__(self, owner: str, reentrant: bool):
        self.owner = owner
        self.reentrant = reentrant
        self.took = False

    def release(self) -> None:
        if self.took:
            _clear()
            self.took = False
        if not self.reentrant:
            os.environ.pop(ENV_FLAG, None)


def hold(owner: str, timeout: float = 0, poll: float = 0.5) -> _Handle:
    """拿显存锁。

    timeout=0  → 拿不到立刻抛 GpuBusy（"不排队"模式）
    timeout>0  → 最多等这么多秒（异步调度：PitchIn-Q 等 RVC 跑完）
    """
    WORK.mkdir(parents=True, exist_ok=True)
    if os.environ.get(ENV_FLAG) == "1":
        return _Handle(owner, reentrant=True)          # 父进程已占用

    deadline = time.time() + max(0.0, float(timeout))
    while True:
        if _try_take(owner):
            os.environ[ENV_FLAG] = "1"
            h = _Handle(owner, reentrant=False)
            h.took = True
            return h
        info = busy() or {}
        if time.time() >= deadline:
            raise GpuBusy("显存被 %s (pid %s) 占用" % (info.get("owner", "?"), info.get("pid", "?")))
        time.sleep(poll)


def state() -> dict:
    info = busy()
    return {"busy": bool(info), "owner": (info or {}).get("owner"),
            "pid": (info or {}).get("pid"), "age": (info or {}).get("age", 0)}


if __name__ == "__main__":       # 方便手工查看：python gpulock.py
    print(json.dumps(state(), ensure_ascii=False))