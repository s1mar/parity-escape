
import json, sys
src = json.loads(sys.stdin.readline())
ep  = json.loads(sys.stdin.readline())
runs = json.loads(sys.stdin.readline())     # list of {"tag":..., "inputs":[[...], ...]}

ns = {}
code = compile(src, "<src>", "exec")
exec(code, ns)
fn = ns[ep]

lines = {}
cur = set()
def tracer(frame, event, arg):
    if frame.f_code.co_filename == "<src>":
        if event == "line":
            cur.add(frame.f_lineno)
        return tracer
    return None

out = {}
for r in runs:
    acc = set()
    for args in r["inputs"]:
        cur.clear()
        sys.settrace(tracer)
        try:
            fn(*args)
        except BaseException:
            pass
        finally:
            sys.settrace(None)
        acc |= set(cur)
    out[r["tag"]] = sorted(acc)
print("@@COV@@" + json.dumps(out))
