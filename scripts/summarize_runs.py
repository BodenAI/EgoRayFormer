#!/usr/bin/env python
"""Summarise training / test runs written by scripts/run_baselines.sh (CSV logs under work_dirs/<run>/lightning_logs/).

For every run: the validation MPJPE curve (best epoch, over-fitting), and the test metrics of every test pass, labelled
with the checkpoint it restored (``work_dirs/logs/<run>_test.log`` -> last, ``<run>_test_best.log`` -> best-val).
These logs name no training attempt: a run with attempt records (work_dirs/<run>/fit_attempt.json,
scripts/ckpt_select.py) is shown with the validation curve of its current attempt's own log directory only and without
test rows - its results are the recorded evaluations of retest_runs.py.

    python scripts/summarize_runs.py v1like_root_rw tsr_root_rw egorayformer_root_rw
    python scripts/summarize_runs.py --glob "*_rw" --dataset unrealego_rw --markdown work_dirs/results_rw.md
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ckpt_select  # noqa: E402

MAIN = ["mpjpe", "pa_mpjpe", "mpjpe_root", "pck", "auc", "mpjve", "group/upper", "group/lower"]
STRATA = ["vis/both", "vis/one", "vis/none", "vis/both_frac", "vis/one_frac", "vis/none_frac", "motion/low", "motion/mid", "motion/high"]
CALIB = ["calib/nll_gauss", "calib/cov68", "calib/cov95", "calib/ause", "calib/spearman", "calib/mean_sigma"]
RATIO = ("pck", "auc", "_frac", "cov68", "cov95", "ause", "spearman")


def version_num(path: str) -> int:
    m = re.search(r"version_(\d+)", path)
    return int(m.group(1)) if m else -1


def read_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r") as f:
        return list(csv.DictReader(f))


def ckpt_of_log(path: str) -> Optional[str]:
    if not os.path.isfile(path):
        return None
    with open(path, "r", errors="ignore") as f:
        m = re.search(r"Restoring states from the checkpoint path at (\S+)", f.read())
    return os.path.basename(m.group(1)) if m else None


def fmt(key: str, v: float) -> str:
    return f"{v:.3f}" if any(r in key for r in RATIO) else f"{v:.2f}"


def summarise(run: str, dataset: Optional[str], logs_dir: str, extra: Optional[List[str]] = None,
              version_dir: Optional[str] = None) -> Dict:
    """``version_dir`` (not None): that log directory only - "" or a directory without metrics: nothing."""
    extra = extra or []
    if version_dir is None:
        versions = sorted(glob.glob(f"work_dirs/{run}/lightning_logs/version_*/metrics.csv"), key=version_num)
    else:
        versions = [m for m in [os.path.join(version_dir, "metrics.csv")] if version_dir and os.path.isfile(m)]
    out: Dict = {"run": run, "val": [], "tests": []}
    test_labels = [ckpt_of_log(os.path.join(logs_dir, f"{run}_test.log")), ckpt_of_log(os.path.join(logs_dir, f"{run}_test_best.log"))]
    test_labels = [t for t in test_labels if t]
    n_test = 0
    for f in versions:
        rows = read_rows(f)
        if not rows:
            continue
        ds = dataset
        if ds is None:  # first dataset tag appearing in a val/test metric
            for k in rows[0]:
                m = re.match(r"(?:val|test)/([^/]+)/mpjpe$", k)
                if m and m.group(1) != "all":
                    ds = m.group(1)
                    break
        pre = f"{ds}/" if ds else ""
        vkey = f"val/{pre}mpjpe"
        val = [(int(float(r["epoch"])), float(r[vkey])) for r in rows if r.get(vkey)]
        if val:
            train = [float(r["train/mpjpe_mm"]) for r in rows if r.get("train/mpjpe_mm")]
            out["val"] = val
            out["train_last"] = train[-1] if train else float("nan")
            continue
        tests = [r for r in rows if r.get(f"test/{pre}mpjpe")]
        if tests:
            r = tests[-1]
            label = test_labels[n_test] if n_test < len(test_labels) else f"test#{n_test}"
            n_test += 1
            vals = {k: float(r[f"test/{pre}{k}"]) for k in MAIN + STRATA + CALIB + extra if r.get(f"test/{pre}{k}")}
            vals["num_frames"] = float(r.get(f"test/{pre}num_frames") or "nan")
            out["tests"].append((label, vals))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--glob", default=None, help="run-name pattern under work_dirs/, e.g. '*_rw'")
    ap.add_argument("--dataset", default=None, help="metric tag (default: first dataset found in the logs)")
    ap.add_argument("--logs_dir", default="work_dirs/logs")
    ap.add_argument("--markdown", default=None, help="also write the summary table to this file")
    ap.add_argument("--best_only", action="store_true",
                    help="table rows only for the best-validation checkpoint of every run (model selection on val only) "
                         "- the last-epoch row is kept when no best-val test exists")
    ap.add_argument("--extra", default="", help="comma separated extra metric keys for the table, e.g. 'dist/near,geo/gate_mean'")
    args = ap.parse_args()
    runs = list(args.runs)
    if args.glob:
        runs += sorted(os.path.basename(p) for p in glob.glob(os.path.join("work_dirs", args.glob)) if os.path.isdir(os.path.join(p, "lightning_logs")))
    extra = [k for k in args.extra.split(",") if k]
    cols = MAIN + extra
    lines = ["| run | ckpt | best val | " + " | ".join(cols) + " | frames |", "|" + "---|" * (len(cols) + 4)]
    for run in runs:
        att = ckpt_select.read_attempt(run)
        if att is None:
            s = summarise(run, args.dataset, args.logs_dir, extra)
        else:   # the current attempt's own log directory, no pipeline test pass
            vdir = ckpt_select.fit_version_dir(run)
            s = summarise(run, args.dataset, args.logs_dir, extra, version_dir=vdir or "")
            s["tests"] = []
            print(f"## {run}: training attempts - current {att['attempt']} "
                  f"({'finished' if ckpt_select.fit_finished(run) else 'not finished'}, "
                  f"{os.path.basename(vdir) if vdir else 'no log directory yet'}); test numbers only from its recorded "
                  "evaluations (retest_runs.py), not from the pipeline logs")
        best_val = ""
        if s["val"]:
            best = min(s["val"], key=lambda x: x[1])
            best_val = f"{best[1]:.1f}@{best[0]}"
            curve = " ".join(f"{e}:{v:.1f}" for e, v in s["val"])
            print(f"## {run}: best val epoch {best[0]} = {best[1]:.2f} mm (last epoch {s['val'][-1][1]:.2f}, last train {s['train_last']:.1f})")
            print(f"   val curve {curve}")
        tests = s["tests"]
        if args.best_only and len(tests) > 1:
            tests = [t for t in tests if t[0] != "last.ckpt"] or tests[-1:]
        for label, vals in tests:
            print(f"   test [{label}] " + ", ".join(f"{k}={fmt(k, vals[k])}" for k in MAIN + STRATA + CALIB + extra if k in vals) + f", frames={vals['num_frames']:.0f}")
            lines.append(f"| {run} | {label} | {best_val} | " + " | ".join(fmt(k, vals[k]) if k in vals else "-" for k in cols) + f" | {vals['num_frames']:.0f} |")
    table = "\n".join(lines)
    print("\n" + table)
    if args.markdown:
        os.makedirs(os.path.dirname(os.path.abspath(args.markdown)), exist_ok=True)
        with open(args.markdown, "w") as f:
            f.write(table + "\n")


if __name__ == "__main__":
    main()
