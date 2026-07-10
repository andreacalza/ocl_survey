#!/usr/bin/env python3
"""Aggregate the per-run analysis into the paper's figures.

Reads, for every run under a results root:
  * logs.json      -> shallow forgetting (output accuracy)
  * lp_curve.csv   -> deep forgetting + LP curve over training (compute_lp.py)
  * snr_curve.csv  -> SNR curve over training (compute_snr.py)

and produces (into <results_root>/figures/):
  * lp_curves.png / snr_curves.png     : LP and SNR vs training step, one line per buffer
  * forgetting_vs_buffer.png           : deep vs shallow forgetting vs buffer (replay gap)
  * snr_vs_buffer.png                  : final SNR vs buffer size

Usage:
    python scripts/analysis/plot_results.py results/
"""
import argparse
import csv
import glob
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compute_forgetting as cf  # noqa: E402


def parse_run(run_dir):
    """(strategy, buffer_size) from a run dir like results/er_split_cifar100_10_2000/0."""
    name = os.path.normpath(run_dir).split(os.sep)[-2]  # er_split_cifar100_10_2000
    strategy = name.split("_")[0]
    mem = int(name.split("_")[-1])
    buffer = 0 if strategy == "naive" else mem  # naive = no replay
    return strategy, buffer


def read_curve(csv_path, value_col):
    """{step: mean value over tasks} from a (step, task, <value>) csv."""
    by_step = defaultdict(list)
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            by_step[int(row["step"])].append(float(row[value_col]))
    return {s: sum(v) / len(v) for s, v in sorted(by_step.items())}


def deep_forgetting(lp_csv):
    """learned - final per task, averaged (mirrors compute_lp summary)."""
    rows = []
    with open(lp_csv) as f:
        for r in csv.DictReader(f):
            rows.append((int(r["step"]), int(r["task"]), float(r["probe_acc"])))
    acc = {(s, t): a for s, t, a in rows}
    steps = sorted({s for s, _, _ in rows})
    tasks = sorted({t for _, t, _ in rows})
    if not steps:
        return None
    final_step = steps[-1]
    fg = []
    for t in tasks[:-1]:
        newest = [s for (s, tt) in acc if tt == t
                  and max(x for (ss, x) in acc if ss == s) == t]
        learned = acc[(max(newest), t)] if newest else acc[(steps[0], t)]
        if (final_step, t) in acc:
            fg.append(learned - acc[(final_step, t)])
    return sum(fg) / len(fg) if fg else 0.0


def discover(root):
    runs = {}
    for lp in glob.glob(os.path.join(root, "*", "*", "lp_curve.csv")):
        runs.setdefault(os.path.dirname(lp), {})["lp"] = lp
    for sn in glob.glob(os.path.join(root, "*", "*", "snr_curve.csv")):
        runs.setdefault(os.path.dirname(sn), {})["snr"] = sn
    for lj in glob.glob(os.path.join(root, "*", "*", "logs.json")):
        d = os.path.dirname(lj)
        if d in runs:
            runs[d]["logs"] = lj
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_root")
    args = ap.parse_args()
    outdir = os.path.join(args.results_root, "figures")
    os.makedirs(outdir, exist_ok=True)

    runs = discover(args.results_root)
    if not runs:
        print("No lp_curve.csv / snr_curve.csv found. Run compute_lp.py / compute_snr.py first.")
        return

    per_buffer = {}  # buffer -> dict of metrics
    lp_fig, lp_ax = plt.subplots(figsize=(7, 5))
    snr_fig, snr_ax = plt.subplots(figsize=(7, 5))

    for run_dir, files in sorted(runs.items(), key=lambda kv: parse_run(kv[0])[1]):
        strategy, buffer = parse_run(run_dir)
        label = "naive" if strategy == "naive" else f"ER {buffer}"
        m = per_buffer.setdefault(buffer, {"label": label})

        if "lp" in files:
            curve = read_curve(files["lp"], "probe_acc")
            lp_ax.plot(list(curve), list(curve.values()), marker=".", ms=3, label=label)
            m["deep_forget"] = deep_forgetting(files["lp"])
            m["lp_final"] = list(curve.values())[-1]
        if "snr" in files:
            curve = read_curve(files["snr"], "snr")
            snr_ax.plot(list(curve), list(curve.values()), marker=".", ms=3, label=label)
            m["snr_final"] = list(curve.values())[-1]
        if "logs" in files:
            res = cf.analyse(files["logs"])
            if res:
                m["shallow_forget"] = res["avg_forgetting"]

    lp_ax.set(xlabel="training step", ylabel="linear-probe accuracy (deep)",
              title="LP curve over training")
    lp_ax.legend(); lp_ax.grid(alpha=.3)
    lp_fig.savefig(os.path.join(outdir, "lp_curves.png"), dpi=150, bbox_inches="tight")

    snr_ax.set(xlabel="training step", ylabel="SNR", title="SNR curve over training")
    snr_ax.legend(); snr_ax.grid(alpha=.3)
    snr_fig.savefig(os.path.join(outdir, "snr_curves.png"), dpi=150, bbox_inches="tight")

    # Buffer sizes span 200-50000 (naive=0): log-x reads far better than linear,
    # matching the paper's figures. naive (buffer=0) can't sit on a log axis, so
    # it's drawn as a horizontal reference line instead of a point.
    buffers = sorted(b for b in per_buffer if b > 0)
    naive_m = per_buffer.get(0)

    def plot_vs_buffer(ax, key, marker, label, color=None):
        xs = [b for b in buffers if per_buffer[b].get(key) is not None]
        ys = [per_buffer[b][key] for b in xs]
        line, = ax.plot(xs, ys, marker + "-", label=label, color=color)
        if naive_m is not None and naive_m.get(key) is not None:
            ax.axhline(naive_m[key], ls="--", lw=1, color=line.get_color(), alpha=.6)
            ax.annotate("naive", xy=(xs[0] if xs else 200, naive_m[key]),
                        xytext=(3, 3), textcoords="offset points",
                        fontsize=8, color=line.get_color())

    fig, ax = plt.subplots(figsize=(7, 5))
    plot_vs_buffer(ax, "deep_forget", "o", "deep (LP)")
    plot_vs_buffer(ax, "shallow_forget", "s", "shallow (output)")
    ax.set_xscale("log")
    ax.set(xlabel="buffer size (log scale; dashed = naive, buffer=0)", ylabel="forgetting",
           title="Replay efficiency gap: deep vs shallow forgetting")
    ax.legend(); ax.grid(alpha=.3, which="both")
    fig.savefig(os.path.join(outdir, "forgetting_vs_buffer.png"), dpi=150, bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(7, 5))
    plot_vs_buffer(ax, "snr_final", "o", "SNR")
    ax.set_xscale("log")
    ax.set(xlabel="buffer size (log scale; dashed = naive, buffer=0)", ylabel="final SNR",
           title="Feature separability (SNR) vs buffer size")
    ax.grid(alpha=.3, which="both")
    fig.savefig(os.path.join(outdir, "snr_vs_buffer.png"), dpi=150, bbox_inches="tight")

    print(f"Figures written to {outdir}/")
    print(f"\n{'buffer':>8} {'shallow':>10} {'deep(LP)':>10} {'final SNR':>10}")
    for b in ([0] if naive_m is not None else []) + buffers:
        m = per_buffer[b]
        sh = m.get("shallow_forget"); dp = m.get("deep_forget"); sn = m.get("snr_final")
        print(f"{m['label']:>8} {sh if sh is None else round(sh,4):>10} "
              f"{dp if dp is None else round(dp,4):>10} "
              f"{sn if sn is None else round(sn,4):>10}")


if __name__ == "__main__":
    main()
