#!/usr/bin/env python3
"""Append CauseMe N=5 / N=10 columns to the three main benchmark tables from results/causeme_baselines.

Every method is scored on the same draws: per N, the draws for which every listed method has a
record (ok or failed). A method's cell averages its completed draws among them; NC if none.
Re-running replaces previously appended CauseMe columns.

Usage: build_causeme_columns.py [--dry-run]
"""
from __future__ import annotations

import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "results" / "causeme_baselines" / "runs.jsonl"
TABLES = ROOT.parent / "tex" / "tables"
NS = (5, 10)
GROUPS = {
    "both": ["PCMCI+", "DYNOTEARS", "MS-CASTLE", "VarLiNGAM", "STIC", "CDANs", "MSWCSL"],
    "lagged": ["PCMCI", "TCDF", "GVAR", "NAVAR", "ACD", "CUTS", "CUTS+", "Lasso-VAR", "MSWCSL"],
    "summary": ["NBCB", "PCTMI", "oCSE", "MSWCSL"],
}
ALL = sorted({m for g in GROUPS.values() for m in g})


def load():
    recs = {}
    for line in RUNS.read_text().splitlines():
        r = json.loads(line)
        key = (r["method"], r["N"], r["draw"])
        if key not in recs or (recs[key]["status"] != "ok" and r["status"] == "ok"):
            recs[key] = r
    return recs


def common_draws(recs, N):
    per = [{d for (m, n, d) in recs if m == meth and n == N} for meth in ALL]
    return sorted(set.intersection(*per))


def mean_hw(v):
    v = np.asarray(v, float)
    if len(v) == 0:
        return math.nan, math.nan
    if len(v) == 1:
        return float(v[0]), math.nan
    return float(v.mean()), float(stats.t.ppf(.975, len(v) - 1) * v.std(ddof=1) / math.sqrt(len(v)))


def cell(mu, hw, bold, under):
    if not math.isfinite(mu):
        return "NC"
    s = f"{mu:.3f}"[1:] if mu < 1 else "1.00"
    s = rf"\mathbf{{{s}}}" if bold else rf"\underline{{{s}}}" if under else s
    return rf"${s}{{\scriptstyle\pm{hw:.3f}}}$".replace("0.", ".") if math.isfinite(hw) else f"${s}$"


def column(recs, target, N, draws):
    out = {}
    for m in GROUPS[target]:
        f1 = [recs[(m, N, d)]["metrics"][target]["F1"] for d in draws if recs[(m, N, d)]["status"] == "ok"]
        out[m] = (*mean_hw(f1), len(f1))
    levels = sorted({round(v[0], 3) for v in out.values() if math.isfinite(v[0])}, reverse=True)[:2]
    return {m: cell(mu, hw, bool(levels) and round(mu, 3) == levels[0],
                    len(levels) > 1 and round(mu, 3) == levels[1]) for m, (mu, hw, n) in out.items()}, out


def patch_table(path, cols, n_draws):
    text = path.read_text()
    lines = text.split("\n")
    out = []
    for line in lines:
        if line.startswith(r"\begin{tabular*}"):
            line = re.sub(r"\{@\{\\extracolsep\{\\fill\}\}l(c+)@\{\}\}",
                          lambda mo: r"{@{\extracolsep{\fill}}l" + "c" * 4 + "cc@{}}", line)
        elif line.startswith("Method &") and "N{=}5" not in line:
            cells = line.rstrip("\\").split(" & ")[:5]
            line = " & ".join(cells + [r"CauseMe $N{=}5$", r"CauseMe $N{=}10$"]) + r"\\"
        else:
            name = re.sub(r"^\\rowcolor\{[^}]*\}", "", line.split(" & ")[0]) if " & " in line else None
            if name in cols[NS[0]]:
                cells = line.rstrip("\\").split(" & ")[:5]
                line = " & ".join(cells + [cols[N][name] for N in NS]) + r"\\"
        out.append(line)
    text = "\n".join(out)
    note = (rf" CauseMe columns: {n_draws[5]} and {n_draws[10]} linear-VAR(5) draws for $N=5$ and $N=10$"
            r" (Appendix~\ref{app:causeme-delay}), every method on the same draws.")
    text = re.sub(r" CauseMe columns:.*?every method on the same draws\.", "", text)
    text = re.sub(r"(over five development draws\.)", r"\1" + note.replace("\\", "\\\\"), text, count=1)
    return text


def main():
    dry = "--dry-run" in sys.argv
    recs = load()
    draws = {N: common_draws(recs, N) for N in NS}
    print({N: len(d) for N, d in draws.items()})
    for target in GROUPS:
        cols, raw = {}, {}
        for N in NS:
            cols[N], raw[N] = column(recs, target, N, draws[N])
        for m in GROUPS[target]:
            print(target, m, " | ".join(f"N={N}: {raw[N][m][0]:.3f}±{raw[N][m][1]:.3f} (n={raw[N][m][2]})" for N in NS))
        path = TABLES / f"benchmark_main_{target}.tex"
        new = patch_table(path, cols, {N: len(draws[N]) for N in NS})
        if not dry:
            path.write_text(new)
    summary = {N: {"draws": draws[N]} for N in NS}
    (ROOT / "results" / "causeme_baselines" / "common_draws.json").write_text(json.dumps(summary) + "\n")


if __name__ == "__main__":
    main()
