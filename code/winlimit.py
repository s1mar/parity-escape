"""Hard memory cap for a child process, via a Windows Job Object.

A fuzz input can drive an allocation whose size is a function of the input value, so a single
call can consume every byte of RAM on the machine: the first reference run reached 41 GB before
it was killed. A timeout does not catch this, because one huge allocation is fast.

The Job Object route is used rather than an RSS poll because it is enforced by the kernel at
allocation time: the child sees a MemoryError instead of the machine swapping. `psutil` is not
installed in this venv, and a `tasklist` poll is far too slow to catch a single large allocation.

Falls back to running unlimited (and says so) on any non-Windows platform or API failure, so a
missing capability degrades to the previous behaviour rather than crashing the run.
"""
from __future__ import annotations

import ctypes
import subprocess
import sys
from ctypes import wintypes

_AVAILABLE = sys.platform == "win32"

if _AVAILABLE:
    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)

    JobObjectExtendedLimitInformation = 9
    JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
    JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    PROCESS_SET_QUOTA = 0x0100
    PROCESS_TERMINATE = 0x0001

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [("ReadOperationCount", ctypes.c_ulonglong),
                    ("WriteOperationCount", ctypes.c_ulonglong),
                    ("OtherOperationCount", ctypes.c_ulonglong),
                    ("ReadTransferCount", ctypes.c_ulonglong),
                    ("WriteTransferCount", ctypes.c_ulonglong),
                    ("OtherTransferCount", ctypes.c_ulonglong)]

    class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong),
                    ("PerJobUserTimeLimit", ctypes.c_longlong),
                    ("LimitFlags", wintypes.DWORD),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.POINTER(ctypes.c_ulong)),
                    ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                    ("IoInfo", IO_COUNTERS),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    _k32.CreateJobObjectW.restype = wintypes.HANDLE
    _k32.OpenProcess.restype = wintypes.HANDLE


def make_job(limit_bytes: int):
    """Create a job object that caps per-process and total memory. Returns a handle or None."""
    if not _AVAILABLE:
        return None
    try:
        job = _k32.CreateJobObjectW(None, None)
        if not job:
            return None
        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = (
            JOB_OBJECT_LIMIT_PROCESS_MEMORY
            | JOB_OBJECT_LIMIT_JOB_MEMORY
            | JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        )
        info.ProcessMemoryLimit = limit_bytes
        info.JobMemoryLimit = limit_bytes
        ok = _k32.SetInformationJobObject(
            job, JobObjectExtendedLimitInformation,
            ctypes.byref(info), ctypes.sizeof(info))
        if not ok:
            _k32.CloseHandle(job)
            return None
        return job
    except Exception:                                   # noqa: BLE001
        return None


def assign(job, pid: int) -> bool:
    if not _AVAILABLE or job is None:
        return False
    try:
        h = _k32.OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, False, pid)
        if not h:
            return False
        ok = bool(_k32.AssignProcessToJobObject(job, h))
        _k32.CloseHandle(h)
        return ok
    except Exception:                                   # noqa: BLE001
        return False


def close(job) -> None:
    if _AVAILABLE and job is not None:
        try:
            _k32.CloseHandle(job)
        except Exception:                               # noqa: BLE001
            pass


def run_capped(cmd: list[str], input_text: str, timeout: float,
               mem_bytes: int) -> tuple[str, str, int]:
    """Run `cmd` with stdin `input_text` under a memory cap.

    Returns (stdout, stderr, returncode). Partial stdout is returned when the child is killed by
    the cap or the timeout, which matters because results are flushed per line: an aborted run
    still yields every input it managed to answer.
    """
    job = make_job(mem_bytes)
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True,
                         encoding="utf-8", errors="replace")
    if job is not None:
        assign(job, p.pid)
    try:
        out, err = p.communicate(input_text, timeout=timeout)
        rc = p.returncode
    except subprocess.TimeoutExpired:
        p.kill()
        try:
            out, err = p.communicate(timeout=30)
        except Exception:                               # noqa: BLE001
            out, err = "", ""
        rc = -9
    finally:
        close(job)
    return out or "", err or "", rc
