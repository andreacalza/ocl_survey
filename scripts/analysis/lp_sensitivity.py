#!/usr/bin/env python3
"""Linear-probe sensitivity study: how much does deep forgetting depend on the
probe hyper-parameters (regularisation C, training-set size) rather than on the
representation itself?

Motivation: deep forgetting is read off TWO single points of the LP curve
(learned vs final), so any per-point measurement noise propagates into it. This
script isolates the two candidate causes:
  * regularisation  -> sweep --C-list
  * probe train size -> sweep --cap-list  (samples per class; 0 = full set)
and, unlike compute_lp.py, it draws the training subsample DETERMINISTICALLY
(fixed per task, identical at every checkpoint), so the curve reflects the
backbone and not the sampling.

Crucially, the (expensive) feature extraction is done ONCE per (checkpoint,
task) and reused for every (C, cap) combination.

Usage:
    python scripts/analysis/lp_sensitivity.py results_ep3/er_split_cifar100_10_2000/0 \
        --stride 10 --C-list 0.01,0.1,1,10,100 --cap-list 0,200

Writes <run_dir>/lp_sensitivity.csv with columns step,task,C,cap,probe_acc and
prints a deep-forgetting table per (C, cap).
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from omegaconf import OmegaConf  # noqa: E402

import src.factories.benchmark_factory as benchmark_factory  # noqa: E402
import src.toolkit.utils as utils  # noqa: E402
from src.factories.benchmark_factory import DS_CLASSES  # noqa: E402
from src.toolkit.representation_analysis import (extract_features,  # noqa: E402
                                                 list_checkpoints,
                                                 load_mt_checkpoint,
                                                 task_experience_map)


def rebuild_benchmark(run_dir):
    cfg = OmegaConf.load(os.path.join(run_dir, "config.yaml"))
    utils.set_seed(cfg.experiment.seed)
    fa = OmegaConf.to_container(cfg.benchmark.factory_args, resolve=True)
    benchmark = benchmark_factory.create_benchmark(
        **fa, dataset_root=cfg.benchmark.dataset_root
    )
    n_cls = DS_CLASSES[fa["benchmark_name"]] // fa["n_experiences"]
    return benchmark, n_cls


def stratified_subset(y, per_class, seed):
    """Deterministic ~balanced subset: `per_class` indices for each label."""
    rng = np.random.default_rng(seed)
    idx = []
    for c in np.unique(y):
        ci = np.flatnonzero(y == c)
        take = min(per_class, len(ci))
        idx.append(rng.choice(ci, take, replace=False))
    return np.sort(np.concatenate(idx))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--stride", type=int, default=1,
                    help="use every S-th checkpoint (ignored unless --all-checkpoints)")
    ap.add_argument("--all-checkpoints", action="store_true",
                    help="evaluate every (strided) checkpoint instead of only the "
                         "task-boundary ones. NOTE: a blind stride can miss the "
                         "boundary checkpoint and bias 'learned' downwards, so the "
                         "default (boundaries only) is both cheaper AND exact.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--C-list", default="0.01,0.1,1,10,100",
                    help="comma-separated LogisticRegression C values")
    ap.add_argument("--cap-list", default="0,200",
                    help="comma-separated train samples PER CLASS (0 = full set)")
    ap.add_argument("--max-iter", type=int, default=5000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    Cs = [float(c) for c in args.C_list.split(",")]
    caps = [int(c) for c in args.cap_list.split(",")]

    benchmark, n_cls = rebuild_benchmark(args.run_dir)
    train_map = task_experience_map(benchmark.train_stream)
    test_map = task_experience_map(benchmark.test_stream)

    all_ckpts = list_checkpoints(args.run_dir)
    if not all_ckpts:
        print("No checkpoints found (run trained with experiment.save_checkpoints=true?)")
        return

    if args.all_checkpoints:
        checkpoints = all_ckpts[:: args.stride]
    else:
        # Deep forgetting only ever reads the END of each task's session plus the
        # final checkpoint. Pick exactly those: for task t, the last checkpoint at
        # or before step (t+1)*steps_per_task. This avoids the bias a blind stride
        # introduces by measuring 'learned' before the task actually ended.
        cfg = OmegaConf.load(os.path.join(args.run_dir, "config.yaml"))
        n_tasks = cfg.benchmark.factory_args.n_experiences
        # Derive steps-per-task from the TRAINING CONFIG, not from the last
        # checkpoint: the final step is usually NOT checkpointed (the plugin
        # fires inside after_training_iteration), so max_step underestimates the
        # stream length and every boundary would drift.
        n_per_task = len(train_map[min(train_map)].dataset)
        steps_per_task = (n_per_task // cfg.strategy.train_mb_size) * cfg.strategy.train_epochs
        checkpoints, seen = [], set()
        for t in range(n_tasks):
            target = (t + 1) * steps_per_task
            cand = [c for c in all_ckpts if c[0] <= target + 1e-6]
            if cand and cand[-1][0] not in seen:
                checkpoints.append(cand[-1])
                seen.add(cand[-1][0])
        if all_ckpts[-1][0] not in seen:
            checkpoints.append(all_ckpts[-1])
        print(f"boundary mode: {len(checkpoints)} checkpoints at steps "
              f"{[s for s, _ in checkpoints]} (of {len(all_ckpts)} total)")

    out = args.out or os.path.join(args.run_dir, "lp_sensitivity.csv")
    rows = []
    print(f"{len(checkpoints)} checkpoints | C={Cs} | cap/class={caps} | device={args.device}")

    for step, path in checkpoints:
        model, task_ids = load_mt_checkpoint(path, n_cls, train_map, args.device)
        for t in task_ids:
            # --- expensive part, done ONCE and reused for every (C, cap) ---
            Xtr, ytr = extract_features(model, train_map[t].dataset, args.device)
            Xte, yte = extract_features(model, test_map[t].dataset, args.device)
            Xtr, ytr = Xtr.numpy(), ytr.numpy()
            Xte, yte = Xte.numpy(), yte.numpy()

            for cap in caps:
                if cap > 0:
                    # deterministic and identical at every checkpoint -> the only
                    # thing that changes across steps is the backbone
                    sel = stratified_subset(ytr, cap, seed=1000 + t)
                    Xs, ys = Xtr[sel], ytr[sel]
                else:
                    Xs, ys = Xtr, ytr
                scaler = StandardScaler().fit(Xs)
                Xs_s, Xte_s = scaler.transform(Xs), scaler.transform(Xte)
                for C in Cs:
                    clf = LogisticRegression(C=C, max_iter=args.max_iter)
                    clf.fit(Xs_s, ys)
                    rows.append((step, t, C, cap, clf.score(Xte_s, yte)))
        print(f"  step {step:6d} | tasks {task_ids} done")

    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "task", "C", "cap", "probe_acc"])
        w.writerows(rows)
    print(f"\nCSV -> {out}")

    # ---- deep forgetting per (C, cap) ----
    steps = sorted({r[0] for r in rows})
    tasks = sorted({r[1] for r in rows})
    final_step = steps[-1]
    print(f"\n{'C':>8} {'cap':>5} | {'deep forgetting (pp)':>22} | per-task")
    print("-" * 78)
    for cap in caps:
        for C in Cs:
            acc = {(s, t): a for s, t, c, cp, a in rows if c == C and cp == cap}
            if not acc:
                continue
            fg = []
            per_task = []
            for t in tasks:
                t_steps = [s for (s, tt) in acc if tt == t]
                newest = [s for s in t_steps
                          if max(x for (ss, x) in acc if ss == s) == t]
                ls = max(newest) if newest else max(t_steps)
                af = acc.get((final_step, t))
                if af is None:
                    continue
                d = (acc[(ls, t)] - af) * 100
                per_task.append(f"{d:+.1f}")
                if t != tasks[-1]:
                    fg.append(d)
            capstr = "full" if cap == 0 else str(cap)
            print(f"{C:>8g} {capstr:>5} | {np.mean(fg):>+22.2f} | {' '.join(per_task)}")


if __name__ == "__main__":
    main()
