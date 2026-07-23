#!/usr/bin/env python3
"""Per-task + total forgetting, grouped by train_epochs and strategy, over seeds.

Run it ON THE CLUSTER, pointing it at the root(s) that contain the 1/3/5-epoch
runs. It auto-discovers every run by walking for logs*.json and reads the sibling
config.yaml to recover (train_epochs, strategy name, mem_size, seed) — so it does
NOT depend on how the folders are nested.

    python forgetting_by_epochs.py ~/tesi/ocl_survey/results_1ep \
                                   ~/tesi/ocl_survey/results_3ep \
                                   ~/tesi/ocl_survey/results_5ep
    # or just:  python forgetting_by_epochs.py ~/tesi/ocl_survey

Outputs, next to the current working directory:
  * forgetting_shallow_<E>ep.csv   (one per epoch setting found)
  * forgetting_deep_<E>ep.csv      (only if lp_curve*.csv are present)
and prints the tables to stdout.

Pure standard library (works on the cluster's Python 3.10, no numpy needed).
"""
import csv
import glob
import json
import os
import re
import statistics
import sys

EXP = re.compile(r'Top1_Acc_Exp/eval_phase/test_stream/Task0*\d+/Exp0*(\d+)$')

# strategy display-order helper: Naive first, then ER by ascending buffer.
def strat_key(name):
    if name.lower().startswith("naive") or name.endswith("_0"):
        return (0, 0)
    m = re.search(r'(\d+)$', name)
    return (1, int(m.group(1)) if m else 0)


def read_config(run_dir):
    """Recover (epochs, strategy_label, seed) from config.yaml, with fallbacks."""
    cfg_path = os.path.join(run_dir, "config.yaml")
    epochs = mem = name = seed = None
    if os.path.isfile(cfg_path):
        txt = open(cfg_path).read()
        m = re.search(r'train_epochs:\s*(\d+)', txt);       epochs = int(m.group(1)) if m else None
        m = re.search(r'\bmem_size:\s*(\d+)', txt);          mem = int(m.group(1)) if m else None
        m = re.search(r'\bname:\s*([A-Za-z_]+)', txt);       name = m.group(1) if m else None
        m = re.search(r'\bseed:\s*(\d+)', txt);              seed = int(m.group(1)) if m else None
    # fallbacks from the path: .../<strategyfolder>/<seed>/
    parts = os.path.normpath(run_dir).split(os.sep)
    if seed is None and parts[-1].isdigit():
        seed = int(parts[-1])
    folder = parts[-2] if len(parts) >= 2 else parts[-1]
    if name is None:
        name = "naive" if "naive" in folder.lower() else "er"
    if mem is None:
        m = re.search(r'(\d+)$', folder); mem = int(m.group(1)) if m else 0
    label = "Naive" if name.lower() == "naive" else f"ER-{mem}"
    return epochs, label, seed


def acc_matrix_last(logs_path):
    rows = []
    for line in open(logs_path):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        accs = {}
        for k, v in d.items():
            m = EXP.search(k)
            if m:
                accs[int(m.group(1))] = v
        if accs:
            rows.append(accs)
    if not rows:
        return None
    last = {}
    for a in rows:
        last[max(a)] = a
    n = max(last) + 1
    learned = [last[i][i] for i in range(n)]
    final = [last[n - 1][i] for i in range(n)]
    return [(learned[i] - final[i]) * 100.0 for i in range(n)]   # per-task pp


def deep_from_lp(run_dir):
    f = glob.glob(os.path.join(run_dir, "lp_curve*.csv"))
    if not f:
        return None
    rows = []
    with open(f[0]) as fh:
        for r in csv.DictReader(fh):
            rows.append((int(r["step"]), int(r["task"]), float(r["probe_acc"])))
    if not rows:
        return None
    steps = sorted({s for s, _, _ in rows})
    tasks = sorted({t for _, t, _ in rows})
    acc = {(s, t): a for s, t, a in rows}
    final = steps[-1]
    out = []
    for t in tasks:
        ts = [s for (s, tt) in acc if tt == t]
        newest = [s for s in ts if max(x for (ss, x) in acc if ss == s) == t]
        ls = max(newest) if newest else max(ts)
        af = acc.get((final, t))
        out.append((acc[(ls, t)] - af) * 100.0 if af is not None else 0.0)
    return out


def collect(roots):
    """epochs -> ('shallow'|'deep') -> label -> list of per-task curves (per seed)."""
    data = {}
    seen = set()
    for root in roots:
        for logs in glob.glob(os.path.join(root, "**", "logs*.json"), recursive=True):
            run_dir = os.path.dirname(logs)
            if run_dir in seen:
                continue
            seen.add(run_dir)
            epochs, label, seed = read_config(run_dir)
            ep = epochs if epochs is not None else "?"
            sh = acc_matrix_last(logs)
            if sh is None:
                continue
            data.setdefault(ep, {}).setdefault("shallow", {}).setdefault(label, []).append(sh)
            dp = deep_from_lp(run_dir)
            if dp is not None:
                data[ep].setdefault("deep", {}).setdefault(label, []).append(dp)
    return data


def print_and_save(ep, kind, per_label):
    labels = sorted(per_label, key=strat_key)
    ntask = max(len(c) for cs in per_label.values() for c in cs)
    # pad
    def col(label, t):
        vals = [c[t] for c in per_label[label] if t < len(c)]
        return statistics.mean(vals) if vals else float("nan")

    title = f"{kind.upper()} forgetting — {ep} epoch(s)  [mean over seeds, pp]"
    print("\n" + "=" * len(title)); print(title); print("=" * len(title))
    header = f"{'task':>5} | " + " | ".join(f"{l:>9}" for l in labels)
    print(header); print("-" * len(header))
    for t in range(ntask):
        cells = " | ".join(f"{col(l, t):+9.1f}" for l in labels)
        tag = " (last)" if t == ntask - 1 else ""
        print(f"{t:>5} | {cells}{tag}")
    print("-" * len(header))
    tot = {}
    for l in labels:
        per_seed_tot = [statistics.mean(c[:-1]) for c in per_label[l] if len(c) > 1]
        m = statistics.mean(per_seed_tot)
        s = statistics.pstdev(per_seed_tot) if len(per_seed_tot) > 1 else 0.0
        tot[l] = (m, s)
    print(f"{'TOT':>5} | " + " | ".join(f"{tot[l][0]:+6.1f}±{tot[l][1]:.1f}".rjust(9) for l in labels))
    n_seeds = max(len(cs) for cs in per_label.values())
    print(f"(TOT = mean over tasks 0..{ntask-2}, then mean±std over {n_seeds} seed(s))")

    out = os.path.abspath(f"forgetting_{kind}_{ep}ep.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["strategy", "seed_index"] + [f"task{t}" for t in range(ntask)] + ["avg_task0_last-1"])
        for l in labels:
            for i, c in enumerate(per_label[l]):
                w.writerow([l, i] + [f"{x:.2f}" for x in c] + [f"{statistics.mean(c[:-1]):.2f}"])
        w.writerow([])
        w.writerow(["strategy", "mean_forgetting_pp", "std_over_seeds_pp"])
        for l in labels:
            w.writerow([l, f"{tot[l][0]:.2f}", f"{tot[l][1]:.2f}"])
    print(f"CSV -> {out}")


def main():
    roots = sys.argv[1:] or ["."]
    data = collect(roots)
    if not data:
        print("No logs*.json found under:", roots); return
    for ep in sorted(data, key=lambda x: (x == "?", x)):
        for kind in ("shallow", "deep"):
            if kind in data[ep]:
                print_and_save(ep, kind, data[ep][kind])


if __name__ == "__main__":
    main()
