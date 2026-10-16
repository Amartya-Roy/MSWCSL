"""Frozen, matched structural evaluation on independent simulations.

No result-dependent tuning or seed replacement. Save observations, structural
coefficients, truth, predictions, warnings, configurations and source hashes.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import warnings
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
REGIMES = ['equal_gaussian', 'unequal_laplace', 'standardized_laplace',
           'contemporaneous', 'persistent_driver']
METHODS = ['MSWCSL', 'Raw only', 'No screen', 'Stage 0 + Lasso',
           'PCMCI+', 'VarLiNGAM']


def generate(regime, seed, T=500):
    from mswcsl.data.svar import generate_svar
    rng = np.random.default_rng(seed + 800000)
    if regime == 'persistent_driver':
        a, b = .98, .5
        # Exact stationary initialization for the driver; discard 100 rows.
        x = np.empty(T + 102)
        x[0] = rng.normal() / np.sqrt(1-a*a)
        for t in range(1, len(x)):
            x[t] = a*x[t-1] + rng.normal()
        y = b*x[:-1] + rng.normal(size=len(x)-1)
        X = np.column_stack([x[1:], y])[-T:]
        B = np.zeros((2, 2, 2)); B[0, 0, 1] = a; B[0, 1, 1] = b
    elif regime == 'contemporaneous':
        B = np.zeros((6, 6, 2))
        perm = rng.permutation(6)
        for k in range(5):
            B[perm[k], perm[k+1], 0] = rng.choice([-1, 1]) * .7
        X = rng.normal(size=(T, 6)) @ np.linalg.inv(np.eye(6)-B[:, :, 0])
    else:
        sim = generate_svar(d=6, T=T, p=2, seed=seed, lag_mode='mixed',
                            density=.25, persistence=.6, edge_strength=.6)
        X, B = sim['X'], sim['B']
        if regime != 'equal_gaussian':
            scales = np.geomspace(1, 3, 6)
            rng.shuffle(scales)
            E = rng.laplace(size=(T+502, 6)) * scales / np.sqrt(2)
            Z = np.zeros_like(E)
            inv = np.linalg.inv(np.eye(6)-B[:, :, 0])
            for t in range(2, len(Z)):
                Z[t] = (Z[t-1] @ B[:, :, 1] + Z[t-2] @ B[:, :, 2] + E[t]) @ inv
            X = Z[-T:]
            if regime == 'standardized_laplace':
                scale = X.std(0)
                X = (X-X.mean(0))/scale
                B = B * scale[:, None, None] / scale[None, :, None]
    truth = (np.abs(B) > 1e-12).astype(np.int8)
    for r in range(truth.shape[2]):
        np.fill_diagonal(truth[:, :, r], 0)
    return X, B, truth


def run_one(regime, seed, out):
    import torch
    from threadpoolctl import threadpool_limits
    from mswcsl import MSWCSL, MSWCSLConfig
    from mswcsl.refit import refit
    from mswcsl.metrics import score_all
    from mswcsl_bench.baselines.adapters import pcmciplus, varlingam
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        X, B, truth = generate(regime, seed)
        p, d = truth.shape[2]-1, len(truth)
        graphs, records = {}, {}
        base = None
        for method in METHODS:
            start = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                try:
                    details = {}
                    if method == 'MSWCSL':
                        base = MSWCSL(MSWCSLConfig(p=p)).fit(X)
                        graph = base.graph
                    elif method == 'Raw only':
                        graph = MSWCSL(MSWCSLConfig(p=p, finest_band='none', S=0)).fit(X).graph
                    elif method == 'Stage 0 + Lasso':
                        graph = MSWCSL(MSWCSLConfig(p=p, finest_band='none', S=0, refit=False)).fit(X).graph
                    elif method == 'No screen':
                        graph, _, _ = refit([X-X.mean(0)], [np.delete(np.arange(d), j) for j in range(d)],
                                            base.pos, p, np.sqrt(2), .7, refit_stat='tstat', rule='bonferroni')
                    else:
                        adapter = pcmciplus if method == 'PCMCI+' else varlingam
                        fitted = adapter.run([X], p, seed=seed)
                        graph, details = fitted['G'], fitted['meta']
                    graph = graph.copy()
                    for r in range(p+1):
                        np.fill_diagonal(graph[:, :, r], 0)
                    graphs[method] = graph
                    records[method] = dict(status='ok', metrics=score_all(truth, graph), details=details)
                except Exception as exc:
                    records[method] = dict(status='error', error=repr(exc))
            records[method]['wall_s'] = time.perf_counter()-start
            records[method]['warnings'] = sorted(set(str(w.message) for w in caught))
        name = f'{regime}_{seed}.npz'
        np.savez_compressed(Path(out)/name, X=X, B=B, truth=truth, **graphs)
        return dict(regime=regime, seed=seed, artifact=name, methods=records)


def main():
    from mswcsl import MSWCSLConfig
    from mswcsl_bench.baselines.adapters import pcmciplus, varlingam
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=ROOT/'results/single_validation')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--seed-base', type=int, default=51000)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    jobs = [(r, args.seed_base+100*k+i) for k, r in enumerate(REGIMES) for i in range(20)]
    files = [Path(__file__), *sorted((ROOT/'src').rglob('*.py'))]
    manifest = dict(jobs=jobs, methods=METHODS, T=500, base_config=MSWCSLConfig().to_dict(),
                    external_specs={'PCMCI+': pcmciplus.SPEC, 'VarLiNGAM': varlingam.SPEC},
                    versions={p:importlib.metadata.version(p) for p in
                              ['numpy', 'scipy', 'scikit-learn', 'torch', 'PyWavelets', 'tigramite', 'lingam']},
                    sources={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                    protocol='Frozen before evaluation; no seed replacement; all failures retained; 20 independent series per regime.')
    (args.out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    with ProcessPoolExecutor(max_workers=args.workers) as pool, (args.out/'runs.jsonl').open('w') as f:
        futures = {pool.submit(run_one, r, s, str(args.out)): (r, s) for r,s in jobs}
        for i, future in enumerate(as_completed(futures), 1):
            row = future.result()
            f.write(json.dumps(row)+'\n'); f.flush()
            print(f'{i}/{len(jobs)} {row["regime"]} {row["seed"]}: '+
                  ', '.join(f'{m}={v["status"]}' for m,v in row['methods'].items()), flush=True)


if __name__ == '__main__':
    main()
