#!/usr/bin/env python
"""Re-evaluate the best-validation checkpoints of finished runs with each run's saved LightningCLI configuration
(work_dirs/<run>/lightning_logs/version_*/config.yaml):

    python scripts/retest_runs.py --from_config --runs <run> [<run> ...] --mode validate --gpus 0 --out work_dirs/eval \
        --extra "--model.eval_precision fp32 --data.eval_clip_len 0"
    python scripts/retest_runs.py --from_config --glob "egorayformer_device_rw_*" --summarize --out work_dirs/eval

``--epoch N`` evaluates epoch=N.ckpt instead of the best-validation checkpoint, ``--seq_dump`` / ``--diag_dump`` /
``--calib_dump`` write per-sequence error sums, last-layer candidates and per-joint Mahalanobis distances.

Outputs: <out>/<run>/lightning_logs/version_*/metrics.csv, logs in work_dirs/logs/<out name>_<run>_<mode>.log,
<out>/<run>/eval_code_<mode>.txt (code hash and directory of the evaluating code, checkpoint, overrides) and, after a
successful evaluation only, <out>/<run>/eval_done_<mode>.json: the training attempt and checkpoint it evaluated, its
metrics file and the sha1 of the dumps it wrote (readers ignore evaluations of an earlier attempt), and its
identity (scripts/eval_identity.py: effective configuration of the exact arguments, evaluating
code, mode) with the complete parsed configuration in <out>/<run>/eval_config_<mode>.yaml. The evaluation is FAILED if
the configuration it saved itself (lightning_logs/version_*/config.yaml) has another config_id, and refused before it
starts if <out>/<run> holds an evaluation of the same training attempt with another identity (another behaviour or
code in the same directory; ``--replace`` overwrites it). The comparison table (also ``--summarize``) reads the run's
result - the recorded evaluation of the best checkpoint of its finished current attempt with the identity these
arguments give (``--identity legacy``: without the identity check, for directories from before identities existed) -
and with ``--epoch N`` the recorded epoch-N evaluation of that attempt (diagnostics). Its "old best-val" column is the
training pipeline's own test number, which names no attempt: shown only for runs from before attempts existed ("-"
for a run with attempt records).

The best-validation checkpoint comes from scripts/ckpt_select.py (completion record, ModelCheckpoint state or the
validation metric of the fit) - never from a test log, and a run without a determinable best checkpoint is reported, not
evaluated on last.ckpt. ``--mode test`` needs ``--allow_test`` (a frozen configuration; run.py refuses the test split
otherwise).
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ckpt_select  # noqa: E402
import eval_identity  # noqa: E402
from summarize_runs import summarise  # noqa: E402

OUT = "work_dirs/eval"
KEYS = ["mpjpe", "pa_mpjpe", "mpjpe_root", "pck", "pck150", "auc", "mpjve", "group/upper", "group/lower", "dist/far",
        "calib/cov68", "calib/cov95", "calib/cov68_gauss", "calib/mean_sigma", "geo/gate_mean", "geo/mpjpe_geo_info",
        "dist/near", "dist/mid", "motion/high", "motion/high_frac"]
KEY_PREFIXES = ("headmotion/", "session/", "subject/", "ray/", "num_")   # strata (session / subject names vary by dataset)


fit_version_dir = ckpt_select.fit_version_dir


def best_ckpt(run: str, select_key: Optional[str] = None) -> Optional[str]:
    """Best-validation checkpoint of the finished run (scripts/ckpt_select.py); None (with the reason printed) while it
    trains or if unknown. ``--epoch`` evaluations of a running training are possible, but they are not its result."""
    if not ckpt_select.fit_finished(run):
        print(f"[{run}] training not finished (current attempt without completion) - its best checkpoint is not final")
        return None
    try:
        return ckpt_select.best_ckpt(run, select_key=select_key)
    except ckpt_select.CheckpointNotFound as e:
        print(f"[{run}] best-validation checkpoint unknown: {e}")
        return None


def pick_ckpt(run: str, epoch: Optional[int], select_key: Optional[str] = None) -> Optional[str]:
    if epoch is None:
        return best_ckpt(run, select_key)
    d = fit_version_dir(run)
    return ckpt_select._epoch_ckpt(os.path.join(d, "checkpoints"), epoch) if d else None


config_overrides = eval_identity.config_overrides   # backwards-compatible name


def run_head(run: str, from_config: bool) -> List[str]:
    """The arguments an evaluation of ``run`` starts with: its saved config (eval_identity.saved_config_head)."""
    return eval_identity.saved_config_head(run)


def expected_identity(run: str, args) -> Optional[Dict]:
    """The identity an evaluation of ``run`` with these arguments has; None for ``--identity legacy`` and
    ckpt_select.NO_IDENTITY (matches no record) if it cannot be determined."""
    if args.identity == "legacy":
        return None
    try:
        return eval_identity.identity([*run_head(run, args.from_config), *shlex.split(args.extra)], args.mode)[0]
    except eval_identity.IdentityError as e:
        print(f"[{run}] no expected evaluation identity: {e}")
        return ckpt_select.NO_IDENTITY


def eval_code_hash() -> str:
    try:
        return subprocess.run(["bash", "scripts/code_hash.sh"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def eval_one(run: str, gpu: str, args) -> str:
    ck = pick_ckpt(run, args.epoch, args.select_key)
    if ck is None:
        return f"[{run}] no {'epoch=%d' % args.epoch if args.epoch is not None else 'best-validation'} checkpoint found"
    out = os.path.join(args.out, run)
    os.makedirs(out, exist_ok=True)
    logger = f'[{{"class_path":"pytorch_lightning.loggers.CSVLogger","init_args":{{"save_dir":"{out}"}}}}]'
    try:
        head = run_head(run, args.from_config)
    except eval_identity.IdentityError:
        return f"[{run}] no saved config.yaml"
    extra = shlex.split(args.extra) if args.extra else []
    if args.calib_dump:
        extra += ["--model.calib_dump", os.path.join(out, "calib")]
    if args.seq_dump:  # per-sequence error sums / counts for a paired sequence bootstrap
        extra += ["--model.seq_error_dump", os.path.join(out, "seq")]
    if args.diag_dump:  # per-joint last-layer candidates for the error / gate analysis
        extra += ["--model.diag_dump", os.path.join(out, "diag")]
    run_args = [*head, *extra, "--trainer.default_root_dir", out, "--trainer.logger", logger, "--ckpt_path", ck]
    cmd = [sys.executable, "run.py", args.mode, *run_args]
    try:
        ident, parsed = eval_identity.identity(run_args, args.mode)
    except eval_identity.IdentityError as e:
        return f"[{run}] FAILED: no evaluation identity ({e})"
    attempt = ckpt_select.current_attempt(run)
    prev = ckpt_select.read_eval_record(args.out, run, args.mode)
    if (prev and prev.get("attempt") == attempt and prev.get("identity") and not args.replace
            and not ckpt_select.identity_matches(prev["identity"], ident)):
        old = prev["identity"]
        return (f"[{run}] REFUSED: {ckpt_select.eval_done_path(args.out, run, args.mode)} records an evaluation of this "
                f"training attempt with another identity (config {old.get('config_id')}, code {old.get('code')}; this "
                f"one: config {ident['config_id']}, code {ident['code']}) - evaluate into another --out, or --replace it")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
    if args.mode == "test":
        env["EGORAYFORMER_ALLOW_TEST"] = "1"
    with open(os.path.join(out, f"eval_code_{args.mode}.txt"), "w") as f:
        f.write(f"hash {args.eval_hash}\ncode_dir {os.getcwd()}\nckpt {ck}\nattempt {attempt}\nextra {' '.join(extra)}\n"
                f"config_id {ident['config_id']}\n")
    versions = lambda: {os.path.basename(v) for v in glob.glob(os.path.join(out, "lightning_logs", "version_*"))}  # noqa: E731
    before = versions()
    log = f"work_dirs/logs/{os.path.basename(args.out.rstrip('/'))}_{run}_{args.mode}.log"
    with open(log, "w") as f:
        f.write(" ".join(cmd) + "\n")
        f.flush()
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=os.getcwd())
    if rc == 0:   # bind the result to the evaluated training attempt
        new = sorted(versions() - before, key=lambda v: int(v.split("_")[-1]))
        metrics = os.path.join(out, "lightning_logs", new[-1], "metrics.csv") if new else ""
        if not os.path.isfile(metrics):
            return f"[{run}] FAILED: the evaluation wrote no new metrics.csv under {out}/lightning_logs (see {log})"
        ran = os.path.join(os.path.dirname(metrics), "config.yaml")   # what the evaluation itself parsed and saved
        if not os.path.isfile(ran):
            return f"[{run}] FAILED: the evaluation saved no config.yaml next to {metrics} (see {log})"
        with open(ran, "r") as f:
            ran_id = eval_identity.config_id(yaml.safe_load(f) or {})
        if ran_id != ident["config_id"]:
            return (f"[{run}] FAILED: the evaluation ran configuration {ran_id} ({ran}), not the parsed {ident['config_id']} "
                    "- not recorded")
        cfg_file = os.path.join(out, f"eval_config_{args.mode}.yaml")
        with open(cfg_file, "w") as f:
            yaml.safe_dump(parsed, f, sort_keys=False)
        ident = dict(ident, config_file={"path": cfg_file, "sha1": ckpt_select.sha1_file(cfg_file)},
                     run_config={"path": ran, "sha1": ckpt_select.sha1_file(ran)})
        split = "val" if args.mode == "validate" else "test"
        dumps = {"seq": (args.seq_dump, f"seq_{split}.csv"), "diag": (args.diag_dump, f"diag_{split}.npz"),
                 "calib": (args.calib_dump, f"calib_{split}.npz")}
        ckpt_select.record_eval(args.out, run, args.mode, metrics_csv=metrics, ckpt=ck,
                                artifacts={k: os.path.join(out, fn) for k, (on, fn) in dumps.items() if on},
                                eval_hash=args.eval_hash, code_dir=os.getcwd(), overrides=" ".join(extra),
                                select_key=args.select_key, epoch=args.epoch, identity=ident)
    return f"[{run}] {'ok' if rc == 0 else f'FAILED rc={rc} (see {log})'} ({os.path.basename(ck)}, gpu {gpu}, {args.mode})"


def _eval_csvs(out: str, name: str) -> List[str]:
    return sorted(glob.glob(os.path.join(out, name, "lightning_logs", "version_*", "metrics.csv")),
                  key=lambda p: int(re.search(r"version_(\d+)", p).group(1)))


def _split_metrics(files: List[str], dataset: str, prefix: str) -> Optional[Dict[str, float]]:
    for f in reversed(files):  # newest pass that evaluated this split
        rows = [r for r in csv.DictReader(open(f)) if r.get(f"{prefix}/{dataset}/mpjpe")]
        if rows:
            r = rows[-1]
            head = f"{prefix}/{dataset}/"
            keys = set(KEYS) | {c[len(head):] for c in r if c.startswith(head) and c[len(head):].startswith(KEY_PREFIXES)}
            return {k: float(r[f"{head}{k}"]) for k in sorted(keys) if r.get(f"{head}{k}")}
    return None


def read_eval(run: str, dataset: str, out: str, prefix: str = "test", epoch: Optional[int] = None,
              expect: Optional[Dict] = None) -> Optional[Dict[str, float]]:
    """Metrics of the run's evaluation in <out>/<run>. ``epoch`` None: the run's result - the recorded
    evaluation of the best checkpoint of its finished current training attempt (ckpt_select.valid_eval). ``epoch`` N:
    the recorded diagnostic evaluation of that attempt's epoch-N checkpoint (``--epoch``; ckpt_select.epoch_eval).
    ``expect``: the identity of the requested behaviour (scripts/eval_identity.py) - only a record with it counts.
    Without ``expect``, runs and evaluations from before these records existed: the newest pass (results: once the
    training finished). None when no such evaluation exists - an evaluation of another attempt, epoch, checkpoint or
    (with ``expect``) behaviour is never returned."""
    mode = "test" if prefix == "test" else "validate"
    ev = (ckpt_select.valid_eval(out, run, mode, expect=expect) if epoch is None
          else ckpt_select.epoch_eval(out, run, mode, epoch, expect=expect))
    if ev is None:
        return None
    if ev.get("metrics_csv"):
        files = [ev["metrics_csv"]] if os.path.isfile(ev["metrics_csv"]) else []
    else:
        files = _eval_csvs(out, run)
    return _split_metrics(files, dataset, prefix)


read_retest = read_eval  # backwards-compatible name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="unrealego_rw")
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--from_config", action="store_true", help="use each run's saved LightningCLI config (runs from --glob)")
    ap.add_argument("--glob", default=None, help="run-name pattern under work_dirs/ for --from_config")
    ap.add_argument("--runs", nargs="*", default=None, help="explicit run names (with --from_config)")
    ap.add_argument("--seq_dump", action="store_true", help="also write <out>/<run>/seq_{val,test}.csv (per-sequence error sums)")
    ap.add_argument("--diag_dump", action="store_true", help="also write <out>/<run>/diag_{val,test}.npz (last-layer candidates)")
    ap.add_argument("--epoch", type=int, default=None, help="evaluate epoch=N.ckpt instead of the best-validation checkpoint")
    ap.add_argument("--select_key", default=None,
                    help="selection metric of the best checkpoint for multi-dataset runs without val/mpjpe (e.g. val/unrealego_rw/mpjpe)")
    ap.add_argument("--mode", choices=("test", "validate"), default="validate")
    ap.add_argument("--allow_test", action="store_true",
                    help="--mode test: the configuration is frozen (test numbers are a report, not a selection criterion)")
    ap.add_argument("--calib_dump", action="store_true", help="also write <out>/<run>/calib_{val,test}.npz (Mahalanobis distances)")
    ap.add_argument("--extra", default="", help="extra CLI overrides appended to every command")
    ap.add_argument("--identity", choices=("strict", "legacy"), default="strict",
                    help="the comparison table reads only evaluations with the identity of these arguments; legacy: "
                         "without that check (directories from before evaluation identities existed)")
    ap.add_argument("--replace", action="store_true",
                    help="overwrite an evaluation record of the same training attempt with another identity in --out")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--summarize", action="store_true", help="only build the comparison table from finished evaluations")
    ap.add_argument("--markdown", default=None)
    args = ap.parse_args()
    if args.mode == "test" and not args.allow_test and not args.summarize:
        sys.exit("--mode test evaluates the test split: add --allow_test for a frozen configuration "
                 "(development decisions read the validation split)")
    if not args.from_config or not (args.runs or args.glob):
        sys.exit("give --from_config with --runs <run> ... or --glob <pattern>")
    args.eval_hash = eval_code_hash()
    if args.runs:
        runs = [r for r in args.runs if os.path.isdir(os.path.join("work_dirs", r, "lightning_logs"))]
        missing = sorted(set(args.runs) - set(runs))
        if missing:
            print(f"runs without a training directory (skipped): {missing}")
    else:
        runs = sorted(os.path.basename(p) for p in glob.glob(os.path.join("work_dirs", args.glob)) if os.path.isdir(os.path.join(p, "lightning_logs")))
    os.makedirs("work_dirs/logs", exist_ok=True)
    if not args.summarize:
        gpus = [g for g in args.gpus.split(",") if g]
        todo = [r for r in runs if pick_ckpt(r, args.epoch, args.select_key) is not None]
        print(f"evaluating ({args.mode}) {len(todo)} runs on GPUs {gpus}: {todo}")
        with ThreadPoolExecutor(max_workers=len(gpus)) as ex:
            futs = [ex.submit(eval_one, r, gpus[i % len(gpus)], args) for i, r in enumerate(todo)]
            for fu in futs:
                print(fu.result(), flush=True)
    # comparison: the training pipeline's own test number (runs without attempt records) vs this evaluation
    prefix = "val" if args.mode == "validate" else "test"
    cols = ["mpjpe", "group/upper", "group/lower", "dist/far", "mpjve", "calib/cov68", "geo/gate_mean"]
    lines = ["| run | ckpt | old best-val mpjpe | new mpjpe | delta | " + " | ".join(cols[1:]) + " |", "|" + "---|" * (len(cols) + 4)]
    for run in runs:
        if args.identity == "strict" and ckpt_select.read_eval_record(args.out, run, args.mode) is None:
            continue                                  # nothing recorded: no evaluation with any identity
        new = read_eval(run, args.dataset, args.out, prefix, epoch=args.epoch,   # --epoch: diagnostics, not results
                        expect=expected_identity(run, args))
        if new is None:
            continue
        if ckpt_select.read_attempt(run) is None:
            old = dict(summarise(run, args.dataset, "work_dirs/logs", ["dist/far", "geo/gate_mean"])["tests"])
            old_best = next((v for k, v in old.items() if k != "last.ckpt"), old.get("last.ckpt"))
            om = old_best["mpjpe"] if old_best else float("nan")
            old_s, delta_s = f"{om:.2f}", f"{new['mpjpe'] - om:+.2f}"
        else:   # the pipelines' test logs are not bound to a training attempt: no "old" number
            old_s = delta_s = "-"
        ck = os.path.basename(pick_ckpt(run, args.epoch, args.select_key) or "?")
        fmt = lambda k, v: f"{v:.3f}" if any(s in k for s in ("cov", "gate")) else f"{v:.2f}"  # noqa: E731
        lines.append(f"| {run} | {ck} | {old_s} | {new['mpjpe']:.2f} | {delta_s} | "
                     + " | ".join(fmt(k, new[k]) if k in new else "-" for k in cols[1:]) + " |")
    table = "\n".join(lines)
    print("\n" + table)
    if args.markdown:
        head = f"# Evaluation of the best-validation checkpoints ({args.mode}, {args.dataset})\n\n"
        tail = "\n"
        with open(args.markdown, "w") as f:
            f.write(head + table + tail)
        print(f"written {args.markdown}")


if __name__ == "__main__":
    main()
