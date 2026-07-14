#!/usr/bin/env python3
"""Compute final accuracy and forgetting from anytime `logs.json` files.

Each run's `logs.json` (JSON Lines, one dict per training step) contains, at every
anytime-eval checkpoint, the per-experience test accuracy under keys of the form
    Top1_Acc_Exp/eval_phase/test_stream/Task00<i>/Exp00<i>
Only experiences already seen (0..j) are evaluated while training experience j, so
by grouping checkpoints by the highest experience index present we reconstruct the
accuracy matrix R[j][i] = accuracy on experience i right after training experience j.

From R we report, per run:
  * final avg accuracy   = mean_i R[T-1][i]                      (higher = better)
  * average forgetting   = mean_{i<T-1} ( R[i][i] - R[T-1][i] )   (paper Sec. A.1; lower = better)

Usage:
    python scripts/analysis/compute_forgetting.py results/                 # auto-discover runs
    python scripts/analysis/compute_forgetting.py results/er_split_cifar100_10_2000/0 ...
"""
import argparse
import glob
import json
import os
import re

EXP_KEY = re.compile(
    r"Top1_Acc_Exp/eval_phase/test_stream/Task0*\d+/Exp0*(\d+)$"
)


def load_checkpoints(logs_json):
    """Return list of {exp_index: accuracy} dicts, one per eval checkpoint."""
    checkpoints = []
    with open(logs_json) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            accs = {}
            for k, v in d.items():
                m = EXP_KEY.match(k)
                if m:
                    accs[int(m.group(1))] = v
            if accs:
                checkpoints.append(accs)
    return checkpoints


def accuracy_matrix(checkpoints):
    """R[j][i]: acc on exp i at the last checkpoint whose max trained exp is j."""
    last_at_level = {}  # j -> checkpoint dict
    for accs in checkpoints:
        j = max(accs)
        last_at_level[j] = accs
    n = max(last_at_level) + 1
    R = [[None] * n for _ in range(n)]
    for j, accs in last_at_level.items():
        for i, a in accs.items():
            R[j][i] = a
    return R, n


def analyse(logs_json):
    checkpoints = load_checkpoints(logs_json)
    if not checkpoints:
        return None
    R, n = accuracy_matrix(checkpoints)

    final = [R[n - 1][i] for i in range(n)]
    learned = [R[i][i] for i in range(n)]
    # shallow forgetting per the reference paper (Sec. A.1):
    #   F_i = A_ii - A_(T-1),i  (accuracy right after learning task i minus final).
    # Same convention as the deep-forgetting computation, so the two are comparable.
    forgetting = []
    for i in range(n - 1):
        forgetting.append(learned[i] - final[i])

    return {
        "n_experiences": n,
        "n_checkpoints": len(checkpoints),
        "final_acc_per_exp": final,
        "learned_acc_per_exp": learned,
        "avg_final_acc": sum(final) / n,
        "forgetting_per_exp": forgetting,
        "avg_forgetting": sum(forgetting) / len(forgetting) if forgetting else 0.0,
    }


def discover_runs(paths):
    """Expand each path to concrete run dirs containing a logs.json."""
    runs = []
    for p in paths:
        if os.path.isfile(os.path.join(p, "logs.json")):
            runs.append(p)
        else:
            runs.extend(sorted(os.path.dirname(f) for f in glob.glob(
                os.path.join(p, "**", "logs.json"), recursive=True)))
    return runs


def run_label(path):
    # results/er_split_cifar100_10_2000/0 -> er_split_cifar100_10_2000/0
    parts = os.path.normpath(path).split(os.sep)
    for i, part in enumerate(parts):
        if "split_cifar" in part:
            seed = parts[i + 1] if i + 1 < len(parts) else ""
            return f"{part}/{seed}" if seed else part
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="run dirs or a results/ root to scan")
    ap.add_argument("--per-exp", action="store_true",
                    help="also print per-experience final accuracy and forgetting")
    args = ap.parse_args()

    runs = discover_runs(args.paths)
    if not runs:
        print("No logs.json found under the given paths.")
        return

    rows = []
    for run in runs:
        res = analyse(os.path.join(run, "logs.json"))
        label = run_label(run)
        if res is None:
            print(f"[skip] {label}: logs.json empty (job unfinished/killed?)")
            continue
        rows.append((label, res))
        if args.per_exp:
            print(f"\n=== {label} ({res['n_experiences']} exps, "
                  f"{res['n_checkpoints']} checkpoints) ===")
            print("exp | learned | final  | forgetting")
            for i in range(res["n_experiences"]):
                f = res["forgetting_per_exp"][i] if i < len(res["forgetting_per_exp"]) else None
                fs = f"{f:+.4f}" if f is not None else "   -   "
                print(f"{i:3d} | {res['learned_acc_per_exp'][i]:.4f}  | "
                      f"{res['final_acc_per_exp'][i]:.4f} | {fs}")

    print("\n" + "=" * 60)
    print(f"{'run':<32} {'final_acc':>10} {'forgetting':>12}")
    print("-" * 60)
    for label, res in sorted(rows, key=lambda r: r[0]):
        print(f"{label:<32} {res['avg_final_acc']:>10.4f} {res['avg_forgetting']:>12.4f}")


if __name__ == "__main__":
    main()
