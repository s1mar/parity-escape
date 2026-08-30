
import json, sys, os
os.environ["HF_DATASETS_OFFLINE"] = "1"
sys.setrecursionlimit(10000)
idx = int(sys.argv[1])
from datasets import load_dataset
d = load_dataset("bigcode/humanevalpack", "python", split="test")
rec = d[idx]
src = rec["declaration"] + rec["canonical_solution"]
ep = rec["entry_point"]

ns = {}
exec(src, ns)
real = ns[ep]
calls = []

def proxy(*args, **kw):
    out = real(*args, **kw)
    if not kw:
        try:
            json.dumps([args, out])
            calls.append([list(args), out])
        except (TypeError, ValueError):
            pass
    return out

ns[ep] = proxy
try:
    exec(rec["test"], ns)
    status = "ok"
except BaseException as e:
    status = "test_error:" + type(e).__name__
print("@@RESULT@@" + json.dumps({"status": status, "calls": calls[:400]}))
