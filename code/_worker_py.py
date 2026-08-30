
import json, os, sys, threading
sys.setrecursionlimit(20000)
MAX_RESULT_CHARS = 1_000_000

# Raw DAEMON threads, not a ThreadPoolExecutor. A timed-out call cannot be killed in Python, so
# the worker has to be able to exit while it is still running. ThreadPoolExecutor threads are
# non-daemon and are joined by an interpreter atexit hook, so the first non-terminating input
# would hang the worker until the outer subprocess timeout fired: 600 seconds per problem
# instead of 2. Every exit below is os._exit, which skips atexit entirely.
def call_with_timeout(fn, args, secs):
    box = {}
    def run():
        try:
            box["v"] = fn(*args)
        except BaseException as ex:
            box["e"] = type(ex).__name__
    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(secs)
    if t.is_alive():
        return "timeout", None
    if "e" in box:
        return "error", box["e"]
    return "ok", box.get("v")
src = sys.stdin.readline()
src = json.loads(src)
ep = json.loads(sys.stdin.readline())
per_ms = json.loads(sys.stdin.readline())
max_to = json.loads(sys.stdin.readline())
ns = {}
exec(src, ns)
fn = ns[ep]
timeouts = 0
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        args = json.loads(line)
    except Exception:
        print(json.dumps({"ok": False, "e": "BadInput"}), flush=True); continue

    kind, payload = call_with_timeout(fn, args, per_ms / 1000.0)
    if kind == "timeout":
        timeouts += 1
        print(json.dumps({"ok": False, "e": "Timeout"}), flush=True)
        if timeouts >= max_to:
            # exit reporting ONLY the timeout, so the driver sees a trailing Timeout line and
            # knows exactly which input to resume after
            sys.stdout.flush()
            os._exit(3)
        continue
    if kind == "error":
        print(json.dumps({"ok": False, "e": payload}), flush=True)
        continue
    try:
        enc = json.dumps({"ok": True, "v": payload})
    except (TypeError, ValueError):
        enc = json.dumps({"ok": False, "e": "Unserializable"})
    except (OverflowError, MemoryError):
        enc = json.dumps({"ok": False, "e": "Oversize"})
    if len(enc) > MAX_RESULT_CHARS:
        enc = json.dumps({"ok": False, "e": "Oversize"})
    print(enc, flush=True)
sys.stdout.flush()
os._exit(0)
