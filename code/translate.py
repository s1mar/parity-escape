"""Stage 1: produce one Java translation per (model, problem), compile it, and record the result.

Batched by model on purpose. Ollama holds one model in VRAM at a time, so interleaving models
forces a multi-gigabyte reload per call; batching pays the load cost once per arm.

Resumable: an existing per-model output file is loaded and only missing problems are generated,
so an interrupted run costs nothing.

Usage:  python translate.py <python-exe> [model_id ...]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench      # noqa: E402
import models     # noqa: E402

PY = sys.argv[1]
WANT = sys.argv[2:] or [m.mid for m in models.MODELS]
OUTDIR = bench.ROOT / "data" / "translations"
WORK = bench.HERE / "_build"


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    problems = json.loads((bench.ROOT / "data" / "reference.json").read_text(encoding="utf-8"))
    print(f"{len(problems)} problems, models: {WANT}")

    for mid in WANT:
        m = models.BY_ID[mid]
        path = OUTDIR / f"{mid}.json"
        done = {}
        if path.exists():
            done = {r["task_id"]: r for r in json.loads(path.read_text(encoding="utf-8"))}
        todo = [p for p in problems if p["task_id"] not in done]
        print(f"\n=== {mid} ({m.name}) : {len(done)} done, {len(todo)} to go ===", flush=True)

        t0 = time.time()
        for i, p in enumerate(todo):
            prompt = models.TRANSLATE.format(source=p["source"], decl=p["java_decl"])
            raw = models.call(m, prompt, tag=f"translate:{mid}:{p['task_id']}")
            java = models.extract_java(raw)

            rec = {"task_id": p["task_id"], "model": mid,
                   "java": java, "raw_chars": len(raw)}
            if not java:
                rec["compile_ok"] = False
                rec["compile_err"] = "no code extracted from response"
            else:
                wd = WORK / mid / p["task_id"].replace("/", "_")
                ok, err = bench.compile_java(java, wd)
                rec["compile_ok"] = ok
                if not ok:
                    rec["compile_err"] = err
            done[p["task_id"]] = rec

            if (i + 1) % 10 == 0 or i + 1 == len(todo):
                path.write_text(json.dumps(list(done.values())), encoding="utf-8")
                nok = sum(1 for r in done.values() if r.get("compile_ok"))
                print(f"  [{i + 1:3d}/{len(todo)}] compiled {nok}/{len(done)} "
                      f"({time.time() - t0:.0f}s)", flush=True)

        path.write_text(json.dumps(list(done.values())), encoding="utf-8")
        nok = sum(1 for r in done.values() if r.get("compile_ok"))
        print(f"=== {mid}: {nok}/{len(done)} compiled ({time.time() - t0:.0f}s) ===", flush=True)


if __name__ == "__main__":
    main()
