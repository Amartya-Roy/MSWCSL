#!/usr/bin/env python3
"""Run MSWCSL and every benchmark comparator on the CauseMe linear-VAR(5) draws.

Each job runs in its own subprocess through the benchmark harness (official adapters, fixed
hyperparameters, hard timeout) and is scored with mswcsl.metrics.score_all against the
label-rule truth of mswcsl_bench.datasets.causeme. Resumable: finished (method, N, draw)
records in OUT/runs.jsonl are skipped; failures are kept and not retried unless --retry.

Usage: run_causeme_baselines.py --methods PCMCI,PCMCI+ --N 5,10 --draws 1-200 [--workers 8]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from mswcsl.metrics import score_all  # noqa: E402
from mswcsl_bench.baselines.registry import TAKES_S, METHODS, method_hp  # noqa: E402
from mswcsl_bench.baselines.runner import run_job  # noqa: E402
from mswcsl_bench.datasets import causeme  # noqa: E402

OUT = ROOT / "results" / "causeme_baselines"


def parse_draws(s: str) -> list:
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", required=True)
    ap.add_argument("--N", default="5,10")
    ap.add_argument("--draws", default="1-200")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--retry", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    out = Path(a.out)
    (out / "graphs").mkdir(parents=True, exist_ok=True)
    runs = out / "runs.jsonl"
    done = set()
    if runs.exists():
        for line in runs.read_text().splitlines():
            r = json.loads(line)
            if r["status"] == "ok" or not a.retry:
                done.add((r["method"], r["N"], r["draw"]))

    methods = [m.strip() for m in a.methods.split(",") if m.strip()]
    for m in methods:
        if m not in METHODS:
            raise SystemExit(f"unknown method {m!r}")
    jobs = [(m, N, k) for k in parse_draws(a.draws) for N in map(int, a.N.split(",")) for m in methods
            if (m, N, k) not in done]
    print(f"{len(jobs)} jobs to run ({len(done)} already recorded)", flush=True)

    lock = threading.Lock()
    workroot = out / "work"

    def one(job):
        m, N, k = job
        draw = causeme.load(causeme.instance_name(N), {}, f"draw{k}", "")
        hp = method_hp(m)
        if METHODS[m] in TAKES_S:
            hp.setdefault("S", draw.S)
        res = run_job(m, draw.X, draw.p, seed=k, device=a.device, timeout_s=a.timeout, hp=hp,
                      threads=1, workroot=str(workroot))
        rec = dict(method=m, N=N, draw=k, status=res["status"], wall_s=round(res["wall_s"], 3), metrics=None)
        if res["status"] == "ok":
            G = np.asarray(res["G"]).astype(np.int8)
            rec["metrics"] = score_all(draw.W, G)
            rec["flags"] = res.get("meta", {}).get("flags", [])
            np.savez_compressed(out / "graphs" / f"{m.replace('+', 'plus')}_N{N}_draw{k}.npz", G=G)
        else:
            rec["error_tail"] = (res.get("stderr_tail") or "")[-2000:]
        if res.get("workdir"):
            shutil.rmtree(res["workdir"], ignore_errors=True)
        return rec

    with ThreadPoolExecutor(max_workers=a.workers) as pool, runs.open("a") as fh:
        futs = [pool.submit(one, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            rec = f.result()
            with lock:
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
            m = rec["metrics"]
            tag = f"F1 both/lag/sum={m['both']['F1']:.3f}/{m['lagged']['F1']:.3f}/{m['summary']['F1']:.3f}" if m else rec["status"]
            print(f"[{i}/{len(jobs)}] {rec['method']} N={rec['N']} draw{rec['draw']} {tag} ({rec['wall_s']:.1f}s)",
                  flush=True)


if __name__ == "__main__":
    main()
