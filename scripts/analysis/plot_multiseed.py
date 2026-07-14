#!/usr/bin/env python3
"""Paper-style multi-seed figures for the buffer sweep.

Reads a results folder laid out as <root>/<Name>/<seed>/{logs*.json,
lp_curve*.csv, snr_curve*.csv} where <Name> is "Naive" or "ER-<buffer>"
and <seed> is 0,1,2. Aggregates across seeds (mean ± std) and produces,
in <root>/figures/:

  forgetting_vs_buffer.png   deep vs shallow forgetting (mean ± std, log-x)
  accuracy_gap_vs_buffer.png Top1 vs LP accuracy — the replay efficiency gap
  snr_vs_buffer.png          final SNR vs buffer (mean ± std, log-x)
  lp_curves.png              LP over training, band = ± std across seeds
  snr_curves.png             SNR over training, band = ± std across seeds

Usage: python scripts/analysis/plot_multiseed.py ~/Desktop/Results
"""
import csv
import glob
import json
import os
import re
import sys
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

EXP = re.compile(r"Top1_Acc_Exp/eval_phase/test_stream/Task0*\d+/Exp0*(\d+)$")

BLUE, ORANGE, GRAY = "#1f77b4", "#d95f02", "#666666"
# sequential single-hue ramp for ER buffers ordered by size (light -> dark)
ER_CMAP = LinearSegmentedColormap.from_list("ers", ["#a6cbe3", "#08306b"])


def parse_name(name):
    if name.lower().startswith("naive"):
        return 0
    m = re.search(r"(\d+)$", name)
    return int(m.group(1)) if m else None


def analyse_json(path):
    ckpts = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        accs = {int(m.group(1)): v for k, v in d.items()
                for m in [EXP.match(k)] if m}
        if accs:
            ckpts.append(accs)
    last = {}
    for a in ckpts:
        last[max(a)] = a
    n = max(last) + 1
    R = [[None] * n for _ in range(n)]
    for j, a in last.items():
        for i, v in a.items():
            R[j][i] = v
    final = [R[n - 1][i] for i in range(n)]
    # shallow forgetting, definizione del paper: F = A_jj - A_ij (learned - final),
    # coerente con il deep forgetting calcolato in read_curve().
    fg = [R[i][i] - R[n - 1][i] for i in range(n - 1)]
    return mean(final), mean(fg)


def read_curve(path, col):
    """{step: mean over tasks} plus final-step stats."""
    rows = [(int(r["step"]), int(r["task"]), float(r[col]))
            for r in csv.DictReader(open(path))]
    by_step = {}
    for s, _, v in rows:
        by_step.setdefault(s, []).append(v)
    curve = {s: mean(v) for s, v in sorted(by_step.items())}
    last = max(by_step)
    acc = {(s, t): v for s, t, v in rows}
    tasks = sorted({t for _, t, _ in rows})
    fg = []
    for t in tasks[:-1]:
        tsteps = [s for (s, tt) in acc if tt == t]
        newest = [s for s in tsteps
                  if max(x for (ss, x) in acc if ss == s) == t]
        learned = acc[(max(newest), t)] if newest else acc[(min(tsteps), t)]
        fg.append(learned - acc[(last, t)])
    return curve, curve[last], (mean(fg) if fg else 0.0)


def ms(vals):
    return mean(vals), (stdev(vals) if len(vals) > 1 else 0.0)


def main():
    root = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else ".")
    outdir = os.path.join(root, "figures")
    os.makedirs(outdir, exist_ok=True)

    data = {}  # buffer -> dict of per-seed lists
    for d in sorted(glob.glob(os.path.join(root, "*", "[0-9]"))):
        name = os.path.basename(os.path.dirname(d))
        buf = parse_name(name)
        if buf is None:
            continue
        lj = sorted(glob.glob(os.path.join(d, "logs*.json")))
        lp = sorted(glob.glob(os.path.join(d, "lp_curve*.csv")))
        sn = sorted(glob.glob(os.path.join(d, "snr_curve*.csv")))
        if not (lj and lp and sn):
            print(f"skip {d} (missing files)")
            continue
        top1, shallow = analyse_json(lj[0])
        lp_curve, lp_final, deep = read_curve(lp[0], "probe_acc")
        sn_curve, sn_final, _ = read_curve(sn[0], "snr")
        e = data.setdefault(buf, {"label": "Naive" if buf == 0 else f"ER {buf}",
                                  "top1": [], "shallow": [], "lp": [], "deep": [],
                                  "snr": [], "lp_curves": [], "snr_curves": []})
        e["top1"].append(top1); e["shallow"].append(shallow)
        e["lp"].append(lp_final); e["deep"].append(deep); e["snr"].append(sn_final)
        e["lp_curves"].append(lp_curve); e["snr_curves"].append(sn_curve)

    ers = sorted(b for b in data if b > 0)
    nv = data.get(0)

    # Il Naive ha buffer=0, che non sta su un asse log: lo posizioniamo in un punto
    # fittizio a sinistra del buffer pi\`u piccolo, cos\`i entra nella curva come
    # primo punto (etichetta "0") e la discesa \`e visibile per intero.
    NAIVE_X = ers[0] * 0.45

    def series(key):
        """x, mean, std includendo il Naive come primo punto (a NAIVE_X)."""
        xs = ([NAIVE_X] if nv else []) + ers
        vals = ([nv[key]] if nv else []) + [data[b][key] for b in ers]
        m = [ms(v)[0] for v in vals]
        s = [ms(v)[1] for v in vals]
        return xs, m, s

    def buffer_ticks(ax):
        ax.set_xscale("log")
        ticks = ([NAIVE_X] if nv else []) + ers
        labels = (["0"] if nv else []) + [str(b) for b in ers]
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, rotation=45, fontsize=8)
        ax.minorticks_off()
        # linea verticale sottile che separa "no replay" (Naive) dal regime con replay
        if nv:
            ax.axvline((NAIVE_X * ers[0]) ** 0.5, color="#bbbbbb", lw=.8, ls=":")

    common = dict(alpha=.3, which="major")

    # 1) deep vs shallow forgetting vs buffer
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for key, col, mk, lab in [("deep", BLUE, "o", "deep (linear probe)"),
                              ("shallow", ORANGE, "s", "shallow (output)")]:
        xs, m, s = series(key)
        ax.errorbar(xs, m, yerr=s, marker=mk, ms=6, lw=1.8, capsize=3,
                    color=col, label=lab)
    ax.axhline(0, lw=.8, color="#999999")
    buffer_ticks(ax)
    ax.set(xlabel="buffer size (0 = Naive)", ylabel="forgetting",
           title="Deep vs shallow forgetting across buffer sizes")
    ax.legend(frameon=False); ax.grid(**common)
    fig.savefig(os.path.join(outdir, "forgetting_vs_buffer.png"),
                dpi=200, bbox_inches="tight")

    # 2) Top1 vs LP accuracy — the replay efficiency gap
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for key, col, mk, lab in [("lp", BLUE, "o", "linear-probe acc (features)"),
                              ("top1", ORANGE, "s", "output acc (Top-1)")]:
        xs, m, s = series(key)
        ax.errorbar(xs, m, yerr=s, marker=mk, ms=6, lw=1.8, capsize=3,
                    color=col, label=lab)
    xs, mlp, _ = series("lp"); _, mtop, _ = series("top1")
    ax.fill_between(xs, mtop, mlp, where=[a > b for a, b in zip(mlp, mtop)],
                    color=BLUE, alpha=.08, lw=0)
    buffer_ticks(ax)
    ax.set(xlabel="buffer size (0 = Naive)", ylabel="accuracy",
           title="Replay efficiency gap: features vs classifier")
    ax.legend(frameon=False); ax.grid(**common)
    fig.savefig(os.path.join(outdir, "accuracy_gap_vs_buffer.png"),
                dpi=200, bbox_inches="tight")

    # 3) SNR vs buffer
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    xs, m, s = series("snr")
    ax.errorbar(xs, m, yerr=s, marker="o", ms=6, lw=1.8, capsize=3,
                color=BLUE, label="SNR")
    buffer_ticks(ax)
    ax.set(xlabel="buffer size (0 = Naive)", ylabel="final SNR",
           title="Feature separability (SNR) vs buffer size")
    ax.grid(**common)
    fig.savefig(os.path.join(outdir, "snr_vs_buffer.png"),
                dpi=200, bbox_inches="tight")

    # 4-5) training curves with ± std band across seeds
    for key, ylab, fname in [("lp_curves", "linear-probe accuracy", "lp_curves.png"),
                             ("snr_curves", "SNR", "snr_curves.png")]:
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        if nv:
            steps = sorted(nv[key][0])
            m = [ms([c[s] for c in nv[key]])[0] for s in steps]
            sd = [ms([c[s] for c in nv[key]])[1] for s in steps]
            ax.plot(steps, m, lw=1.8, ls="--", color=GRAY, label="Naive")
            ax.fill_between(steps, [a - b for a, b in zip(m, sd)],
                            [a + b for a, b in zip(m, sd)], color=GRAY, alpha=.15, lw=0)
        for i, b in enumerate(ers):
            col = ER_CMAP(i / max(len(ers) - 1, 1))
            curves = data[b][key]
            steps = sorted(curves[0])
            m = [ms([c[s] for c in curves])[0] for s in steps]
            sd = [ms([c[s] for c in curves])[1] for s in steps]
            ax.plot(steps, m, lw=1.8, color=col, label=f"ER {b}")
            ax.fill_between(steps, [a - b2 for a, b2 in zip(m, sd)],
                            [a + b2 for a, b2 in zip(m, sd)], color=col, alpha=.12, lw=0)
        ax.set(xlabel="training step", ylabel=ylab,
               title=f"{ylab} over training (mean ± std, 3 seeds)")
        ax.legend(frameon=False, fontsize=8, ncols=2)
        ax.grid(alpha=.3)
        fig.savefig(os.path.join(outdir, fname), dpi=200, bbox_inches="tight")

    print(f"Figures written to {outdir}")


if __name__ == "__main__":
    main()
