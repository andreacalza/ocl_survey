#!/usr/bin/env python3
"""Signal-to-Noise Ratio (SNR) analysis of feature separability, per the paper.

For two class feature distributions with means mu and covariances Sigma:

    SNR(c1, c2) = || mu1 - mu2 ||^2 / Tr(Sigma1 + Sigma2)

(higher = more separable; lower-bounds linear separability). For every periodic
checkpoint and every trained task we compute, on the task's TEST features
(population statistics), the average SNR over class pairs, then average over tasks
to obtain the SNR curve over training.

By default pairs are taken WITHIN each task (the relevant separability for a
multi-head / TIL model, where each head discriminates its own classes); use
--pairs all for pairs across all past-task classes as well.

Usage:
    python scripts/analysis/compute_snr.py results/er_split_cifar100_10_2000/0 \
        [--stride 1] [--device cuda] [--pairs within|all] [--out snr.csv]
"""
import argparse
import csv
import itertools
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import src.factories.benchmark_factory as benchmark_factory  # noqa: E402
import src.toolkit.utils as utils  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
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


def class_stats(feats, labels):
    """Per class: mean vector and Tr(Sigma) = sum of per-dim variances."""
    stats = {}
    for c in torch.unique(labels).tolist():
        fc = feats[labels == c]
        if fc.shape[0] < 2:
            continue
        mu = fc.mean(0)
        tr_sigma = fc.var(0, unbiased=True).sum().item()  # Tr(Cov)
        stats[c] = (mu, tr_sigma)
    return stats


def snr(mu1, tr1, mu2, tr2):
    return ((mu1 - mu2) ** 2).sum().item() / (tr1 + tr2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--pairs", choices=["within", "all"], default="within",
                    help="class pairs within each task (default) or across all past classes")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    benchmark, n_cls = rebuild_benchmark(args.run_dir)
    train_map = task_experience_map(benchmark.train_stream)
    test_map = task_experience_map(benchmark.test_stream)

    checkpoints = list_checkpoints(args.run_dir)[:: args.stride]
    if not checkpoints:
        print("No checkpoints found (was the run launched with experiment.save_checkpoints=true?)")
        return

    out = args.out or os.path.join(args.run_dir, "snr_curve.csv")
    rows = []  # (step, task, snr)
    print(f"{len(checkpoints)} checkpoints, device={args.device}, pairs={args.pairs}")
    for step, path in checkpoints:
        model, task_ids = load_mt_checkpoint(path, n_cls, train_map, args.device)
        per_task_stats = {}
        for t in task_ids:
            feats, labels = extract_features(model, test_map[t].dataset, args.device)
            per_task_stats[t] = class_stats(feats, labels)

        if args.pairs == "within":
            for t in task_ids:
                st = per_task_stats[t]
                vals = [snr(*st[a], *st[b]) for a, b in itertools.combinations(st, 2)]
                if vals:
                    rows.append((step, t, float(np.mean(vals))))
        else:
            allc = {c: s for t in task_ids for c, s in per_task_stats[t].items()}
            vals = [snr(*allc[a], *allc[b]) for a, b in itertools.combinations(allc, 2)]
            if vals:
                rows.append((step, -1, float(np.mean(vals))))

        cur = [v for (s, _, v) in rows if s == step]
        print(f"step {step:6d} | tasks {task_ids} | mean SNR {np.mean(cur):.4f}")

    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "task", "snr"])
        w.writerows(rows)
    print(f"\nSNR curve written to {out}")


if __name__ == "__main__":
    main()
