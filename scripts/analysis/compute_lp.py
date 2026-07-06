#!/usr/bin/env python3
"""Linear-probe (LP) analysis of deep forgetting, per the reference paper.

For every periodic checkpoint of a run, and for every task already trained, we
train a logistic-regression probe (scikit-learn ``LogisticRegression``, C=100, one
probe PER TASK as prescribed for multi-head/TIL) on the FROZEN backbone features
of that task's training set, and measure its accuracy on the task's test set:

    A*_{K,j} = probe accuracy on task j using features of the model at step K

This yields, for each task, a probe-accuracy curve over training (LP curve). From
it we report deep forgetting

    F_deep(j) = A*_{learned,j} - A*_{final,j}

where A*_{learned,j} is the probe accuracy right after task j finished training and
A*_{final,j} is the probe accuracy at the last checkpoint.

Usage:
    python scripts/analysis/compute_lp.py results/er_split_cifar100_10_2000/0 \
        [--stride 1] [--device cuda] [--train-per-class 0] [--out lp.csv]
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

from omegaconf import OmegaConf

import src.factories.benchmark_factory as benchmark_factory
import src.toolkit.utils as utils
from src.factories.benchmark_factory import DS_CLASSES
from src.toolkit.representation_analysis import (extract_features,
                                                 list_checkpoints,
                                                 load_mt_checkpoint,
                                                 task_experience_map)


def rebuild_benchmark(run_dir):
    """Rebuild the exact benchmark used by the run (same seed -> same split)."""
    cfg = OmegaConf.load(os.path.join(run_dir, "config.yaml"))
    utils.set_seed(cfg.experiment.seed)
    fa = OmegaConf.to_container(cfg.benchmark.factory_args, resolve=True)
    benchmark = benchmark_factory.create_benchmark(
        **fa, dataset_root=cfg.benchmark.dataset_root
    )
    n_exp = fa["n_experiences"]
    n_cls = DS_CLASSES[fa["benchmark_name"]] // n_exp
    return benchmark, n_cls


def probe_accuracy(model, train_ds, test_ds, device, C, train_cap):
    Xtr, ytr = extract_features(model, train_ds, device,
                                max_samples=train_cap or None)
    Xte, yte = extract_features(model, test_ds, device)
    Xtr, ytr = Xtr.numpy(), ytr.numpy()
    Xte, yte = Xte.numpy(), yte.numpy()
    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=C, max_iter=1000)
    clf.fit(scaler.transform(Xtr), ytr)
    return clf.score(scaler.transform(Xte), yte)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--stride", type=int, default=1, help="use every S-th checkpoint")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--C", type=float, default=100.0, help="LogisticRegression C (paper: 100)")
    ap.add_argument("--train-per-class", type=int, default=0,
                    help="cap train samples used to fit the probe (0 = full set)")
    ap.add_argument("--out", default=None, help="CSV output path (default <run_dir>/lp_curve.csv)")
    args = ap.parse_args()

    benchmark, n_cls = rebuild_benchmark(args.run_dir)
    train_map = task_experience_map(benchmark.train_stream)
    test_map = task_experience_map(benchmark.test_stream)

    checkpoints = list_checkpoints(args.run_dir)[:: args.stride]
    if not checkpoints:
        print("No checkpoints found (was the run launched with experiment.save_checkpoints=true?)")
        return
    train_cap = args.train_per_class * n_cls if args.train_per_class else 0

    out = args.out or os.path.join(args.run_dir, "lp_curve.csv")
    rows = []  # (step, task, probe_acc)
    print(f"{len(checkpoints)} checkpoints, device={args.device}")
    for step, path in checkpoints:
        model, task_ids = load_mt_checkpoint(path, n_cls, train_map, args.device)
        for t in task_ids:
            acc = probe_accuracy(model, train_map[t].dataset, test_map[t].dataset,
                                 args.device, args.C, train_cap)
            rows.append((step, t, acc))
        seen = [a for (s, tt, a) in rows if s == step]
        print(f"step {step:6d} | tasks {task_ids} | mean probe acc {np.mean(seen):.4f}")

    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "task", "probe_acc"])
        w.writerows(rows)
    print(f"\nLP curve written to {out}")

    # deep forgetting summary: learned (last step where task is the newest) vs final
    steps = sorted({s for s, _, _ in rows})
    tasks = sorted({t for _, t, _ in rows})
    final_step = steps[-1]
    acc = {(s, t): a for s, t, a in rows}
    learned_step = {}
    for t in tasks:
        # last checkpoint at which t is the most-recently-trained task
        t_steps = [s for (s, tt) in acc if tt == t]
        newest = [s for s in t_steps if max(x for (ss, x) in acc if ss == s) == t]
        learned_step[t] = max(newest) if newest else max(t_steps)

    print("\ntask | learned A* | final A* | deep forgetting")
    fg = []
    for t in tasks:
        a_learned = acc[(learned_step[t], t)]
        a_final = acc.get((final_step, t))
        if a_final is None:
            continue
        f = a_learned - a_final
        if t != tasks[-1]:
            fg.append(f)
        print(f"{t:4d} | {a_learned:.4f}     | {a_final:.4f}   | {f:+.4f}")
    if fg:
        print(f"\navg deep forgetting = {np.mean(fg):+.4f}")


if __name__ == "__main__":
    main()
