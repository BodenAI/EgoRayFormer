#!/usr/bin/env python
"""Training identity, completion and the best-validation checkpoint of a run, without any test log.

Training identity. scripts/run_baselines.sh calls ``start`` before every fit: it writes
work_dirs/<run>/fit_attempt.json (a new attempt id and the log directories that already existed), moves an earlier
completion record aside (fit_done.superseded.<attempt>.<time>.json) and appends the attempt to fit_attempts.jsonl.
``record`` after a successful fit writes work_dirs/<run>/fit_done.json for that attempt and its own log directory. A run
whose latest attempt has no matching record is not finished and has no best checkpoint, whatever older attempts left
behind. An evaluation is the run's result only if it evaluated the best checkpoint of the finished current attempt:
scripts/retest_runs.py writes <eval_dir>/<run>/eval_done_<mode>.json (attempt, checkpoint, metrics file, sha1 of the
per-sequence / diagnostic dumps) after a successful evaluation, and ``valid_eval`` rejects a record of another attempt,
of a training that has not finished, or of another checkpoint than its best. ``epoch_eval`` reads a diagnostic
evaluation of one epoch (retest_runs.py --epoch), which is never the run's result. One output directory holds one
evaluation per run and mode. Runs launched before attempts existed (no fit_attempt.json) keep the earlier completion
rules. Evaluation identity: records carry the identity of the evaluated behaviour (scripts/eval_identity.py:
effective configuration, evaluating code, mode); a reader that passes the identity it expects (``expect``) accepts only
a record with that identity - records without one, and the legacy fallback, never match.

The best checkpoint comes from, in this order:
  1. the completion record of the current attempt;
  2. the ModelCheckpoint state stored in last.ckpt when that callback monitored a metric (``best_model_path``);
  3. the selection metric in the attempt's metrics.csv (runs that kept every epoch without a monitor): the first epoch
     with the lowest value -> that epoch's checkpoint.
A best checkpoint that cannot be determined, or whose file is gone, is an error. It is never replaced by last.ckpt.
The selection metric is ``val/mpjpe`` (multi-dataset runs log it from ``model.select_val_dataset`` or the pooled
validation); with several per-dataset metrics and no ``val/mpjpe`` the key must be given (``--select_key``): no
dataset is chosen silently.

    python scripts/ckpt_select.py start <run>                     # before a fit: a new training attempt
    python scripts/ckpt_select.py record <run>                    # after a successful fit
    python scripts/ckpt_select.py finished <run>                  # exit 0 if the current attempt finished
    python scripts/ckpt_select.py best <run> [--select_key KEY]   # print the best-validation checkpoint (exit 1 if unknown)
    python scripts/ckpt_select.py last <run>                      # last.ckpt of the finished current attempt (stage-1 runs)
    python scripts/ckpt_select.py status <run>
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import re
import sys
import time
import uuid
from typing import Dict, List, Optional, Tuple

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
WORK = "work_dirs"
RECORD = "fit_done.json"
ATTEMPT = "fit_attempt.json"
FIT_DONE_MARKERS = ("`Trainer.fit` stopped", "Signaling Trainer to stop")


class CheckpointNotFound(RuntimeError):
    pass


def _version_no(path: str) -> int:
    m = re.search(r"version_(\d+)$", path.rstrip("/"))
    return int(m.group(1)) if m else -1


def _read_json(path: str) -> Optional[Dict]:
    if not os.path.isfile(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def _write_json(path: str, obj: Dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def sha1_file(path: str) -> Optional[str]:
    if not os.path.isfile(path):
        return None
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def version_dirs(run: str, work: str = WORK) -> List[str]:
    return sorted(glob.glob(os.path.join(work, run, "lightning_logs", "version_*")), key=_version_no)


# ------------------------------------------------------------------ training attempts
def read_attempt(run: str, work: str = WORK) -> Optional[Dict]:
    return _read_json(os.path.join(work, run, ATTEMPT))


def current_attempt(run: str, work: str = WORK) -> Optional[str]:
    att = read_attempt(run, work)
    return att["attempt"] if att else None


def read_record(run: str, work: str = WORK) -> Optional[Dict]:
    return _read_json(os.path.join(work, run, RECORD))


def start_attempt(run: str, work: str = WORK) -> Dict:
    """A new training attempt of ``run``: the earlier completion record stops describing the run."""
    d = os.path.join(work, run)
    os.makedirs(d, exist_ok=True)
    now = time.strftime("%Y%m%d-%H%M%S")
    att = {"attempt": f"{now}-{os.getpid()}-{uuid.uuid4().hex[:6]}", "started": time.strftime("%Y-%m-%d %H:%M:%S"),
           "known_versions": [os.path.basename(v) for v in version_dirs(run, work)]}
    old = read_record(run, work)
    if old is not None:
        os.replace(os.path.join(d, RECORD), os.path.join(d, f"fit_done.superseded.{old.get('attempt') or 'legacy'}.{now}.json"))
    _write_json(os.path.join(d, ATTEMPT), att)
    with open(os.path.join(d, "fit_attempts.jsonl"), "a") as f:
        f.write(json.dumps(att) + "\n")
    return att


def attempt_version_dir(run: str, att: Dict, work: str = WORK) -> Optional[str]:
    """Log directory created by attempt ``att``: the newest version that did not exist when it started (the one with
    a last.ckpt if there are several - a validate / test pass of the same run logs into a version of its own)."""
    new = [v for v in version_dirs(run, work) if os.path.basename(v) not in set(att.get("known_versions", []))]
    with_last = [v for v in new if os.path.isfile(os.path.join(v, "checkpoints", "last.ckpt"))]
    return (with_last or new or [None])[-1]


def fit_version_dir(run: str, work: str = WORK) -> Optional[str]:
    """Log directory of the training pass: the current attempt's (runs with attempts); otherwise the newest version
    that holds checkpoints/last.ckpt, else the newest version with a checkpoints dir."""
    att = read_attempt(run, work)
    if att is not None:
        return attempt_version_dir(run, att, work)
    dirs = version_dirs(run, work)[::-1]
    for d in dirs:
        if os.path.isfile(os.path.join(d, "checkpoints", "last.ckpt")):
            return d
    for d in dirs:
        if os.path.isdir(os.path.join(d, "checkpoints")):
            return d
    return None


def fit_log(run: str, work: str = WORK) -> str:
    try:
        with open(os.path.join(work, "logs", f"{run}.log"), "r", errors="ignore") as f:
            return f.read()
    except OSError:
        return ""


def fit_finished(run: str, work: str = WORK) -> bool:
    """The current attempt has its completion record; runs without attempts: a record, or (launched before records
    existed) a finished fit log and a last.ckpt."""
    rec, att = read_record(run, work), read_attempt(run, work)
    if att is not None:
        return rec is not None and rec.get("attempt") == att["attempt"] and bool(rec.get("finished"))
    if rec is not None:
        return bool(rec.get("finished"))
    d = fit_version_dir(run, work)
    if d is None or not os.path.isfile(os.path.join(d, "checkpoints", "last.ckpt")):
        return False
    log = fit_log(run, work)
    return any(m in log for m in FIT_DONE_MARKERS)


# ------------------------------------------------------------------ best checkpoint
def selection_key(columns: List[str], select_key: Optional[str] = None) -> Tuple[Optional[str], str]:
    """-> (key, reason when None)."""
    if select_key:
        return (select_key, "") if select_key in columns else (None, f"no column {select_key}")
    if "val/mpjpe" in columns:
        return "val/mpjpe", ""
    keys = sorted(k for k in columns if k.startswith("val/") and k.endswith("/mpjpe") and "/all/" not in k)
    if len(keys) == 1:
        return keys[0], ""
    if not keys:
        return None, "no validation MPJPE column"
    return None, f"several validation sets ({', '.join(keys)}) and no val/mpjpe: give the selection key (--select_key)"


def _epoch_ckpt(ckdir: str, epoch: int) -> Optional[str]:
    """Checkpoint of ``epoch`` under both naming schemes (``epoch=N.ckpt`` and the bare ``N.ckpt``)."""
    for pat in (f"epoch={epoch}.ckpt", f"epoch={epoch}-*.ckpt", f"{epoch}.ckpt", f"{epoch}-*.ckpt"):
        hits = sorted(glob.glob(os.path.join(ckdir, pat)))
        if hits:
            return hits[0]
    return None


def best_from_metrics(vdir: str, select_key: Optional[str] = None) -> Tuple[str, int, float, str]:
    path = os.path.join(vdir, "metrics.csv")
    if not os.path.isfile(path):
        raise CheckpointNotFound(f"{path} missing")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    key, why = selection_key(list(rows[0]) if rows else [], select_key)
    if key is None:
        raise CheckpointNotFound(f"{path}: {why}")
    best: Optional[Tuple[float, int]] = None
    for r in rows:
        v = r.get(key)
        if v in (None, "") or r.get("epoch") in (None, ""):
            continue
        val, ep = float(v), int(float(r["epoch"]))
        if val == val and (best is None or val < best[0]):   # skip NaN; first minimum wins
            best = (val, ep)
    if best is None:
        raise CheckpointNotFound(f"{path}: no finite {key}")
    ck = _epoch_ckpt(os.path.join(vdir, "checkpoints"), best[1])
    if ck is None:
        raise CheckpointNotFound(f"best epoch {best[1]} ({key} {best[0]:.3f}) has no checkpoint under {vdir}/checkpoints")
    return ck, best[1], best[0], key


def checkpoint_state(last: str) -> Optional[Dict]:
    """The state of the ModelCheckpoint callback that monitored a metric (val/mpjpe preferred), read from last.ckpt."""
    import torch

    sys.path.insert(0, ROOT)   # hyper-parameters may reference egorayformer classes
    try:
        ck = torch.load(last, map_location="cpu", weights_only=False, mmap=True)
    except Exception:  # noqa: BLE001 - old serialisation without mmap support
        ck = torch.load(last, map_location="cpu", weights_only=False)
    states = [s for s in (ck.get("callbacks") or {}).values()
              if isinstance(s, dict) and s.get("monitor") and s.get("best_model_path")]
    if not states:
        return None
    states.sort(key=lambda s: s.get("monitor") != "val/mpjpe")
    s = states[0]
    score = s.get("best_model_score")
    return {"monitor": s["monitor"], "best_model_path": s["best_model_path"],
            "best_model_score": float(score) if score is not None else None}


def _epoch_of(path: str) -> Optional[int]:
    m = re.search(r"(?:epoch=)?(\d+)(?:-[^/]*)?\.ckpt$", os.path.basename(path))
    return int(m.group(1)) if m and os.path.basename(path) != "last.ckpt" else None


def find_best(run: str, work: str = WORK, use_record: bool = True, select_key: Optional[str] = None) -> Dict:
    """-> {"best_ckpt", "best_epoch", "best_score", "monitor", "selection"}; raises CheckpointNotFound."""
    att = read_attempt(run, work)
    rec = read_record(run, work) if use_record else None
    if use_record and att is not None and (rec is None or rec.get("attempt") != att["attempt"]):
        raise CheckpointNotFound(f"{run}: training attempt {att['attempt']} (started {att.get('started')}) has no completion "
                                 "record - running, stopped or failed; earlier attempts do not count")
    if rec is not None and rec.get("best_ckpt") and not select_key:
        if not os.path.isfile(rec["best_ckpt"]):
            raise CheckpointNotFound(f"{run}: the recorded best checkpoint {rec['best_ckpt']} is gone")
        return {k: rec.get(k) for k in ("best_ckpt", "best_epoch", "best_score", "monitor", "selection")}
    vdir = fit_version_dir(run, work)
    if vdir is None:
        raise CheckpointNotFound(f"{run}: no training log directory under {work}/{run}"
                                 + (f" for attempt {att['attempt']}" if att else ""))
    last = os.path.join(vdir, "checkpoints", "last.ckpt")
    if os.path.isfile(last) and not select_key:
        st = checkpoint_state(last)
        if st is not None:
            # the stored path is the training cwd's view; the file lives next to last.ckpt
            ck = os.path.join(vdir, "checkpoints", os.path.basename(st["best_model_path"]))
            if not os.path.isfile(ck):
                raise CheckpointNotFound(f"{run}: best checkpoint {os.path.basename(ck)} ({st['monitor']}) is gone")
            return {"best_ckpt": ck, "best_epoch": _epoch_of(ck), "best_score": st["best_model_score"],
                    "monitor": st["monitor"], "selection": "checkpoint_state"}
    ck, ep, val, key = best_from_metrics(vdir, select_key)
    return {"best_ckpt": ck, "best_epoch": ep, "best_score": val, "monitor": key, "selection": "metrics_csv"}


def best_ckpt(run: str, work: str = WORK, select_key: Optional[str] = None) -> str:
    return find_best(run, work, select_key=select_key)["best_ckpt"]


def last_ckpt(run: str, work: str = WORK) -> str:
    """last.ckpt of the run's finished current training attempt - the checkpoint a stage-1 (heatmap) run hands on; a
    stopped or failed attempt has none."""
    if not fit_finished(run, work):
        raise CheckpointNotFound(f"{run}: the current training attempt has not finished")
    d = fit_version_dir(run, work)
    path = os.path.join(d, "checkpoints", "last.ckpt") if d else ""
    if not os.path.isfile(path):
        raise CheckpointNotFound(f"{run}: no last.ckpt in the training log directory {d}")
    return path


def _code_hash(run: str, work: str = WORK) -> str:
    try:
        with open(os.path.join(work, run, "code_hash.txt"), "r") as f:
            for line in f:
                if line.startswith("hash "):
                    return line.split()[1].strip()
    except OSError:
        pass
    return ""


def write_record(run: str, work: str = WORK) -> Dict:
    """Completion record of the current attempt's fit (of the run's fit for runs without attempts). The best
    checkpoint is stored when it can be determined (heatmap runs have no validation MPJPE: ``best_ckpt`` null with the
    reason)."""
    att = read_attempt(run, work)
    vdir = fit_version_dir(run, work)
    last = os.path.join(vdir, "checkpoints", "last.ckpt") if vdir else None
    rec: Dict = {"run": run, "attempt": att["attempt"] if att else None, "finished": bool(last and os.path.isfile(last)),
                 "version_dir": vdir, "last_ckpt": last if last and os.path.isfile(last) else None,
                 "code_hash": _code_hash(run, work), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        rec.update(find_best(run, work, use_record=False))
    except CheckpointNotFound as e:
        rec.update(best_ckpt=None, best_error=str(e))
    _write_json(os.path.join(work, run, RECORD), rec)
    return rec


# ------------------------------------------------------------------ evaluations bound to an attempt
def eval_done_path(eval_dir: str, run: str, mode: str) -> str:
    return os.path.join(eval_dir, run, f"eval_done_{mode}.json")


def read_eval_record(eval_dir: str, run: str, mode: str) -> Optional[Dict]:
    return _read_json(eval_done_path(eval_dir, run, mode))


IDENTITY_KEYS = ("config_id", "code", "mode")
NO_IDENTITY: Dict = {}   # an expected identity that cannot be determined: no record matches it


def identity_matches(recorded: Optional[Dict], expect: Dict) -> bool:
    """The recorded evaluation identity (scripts/eval_identity.py) is the expected one: the same effective configuration,
    evaluating code and mode. A record without an identity matches nothing."""
    return isinstance(recorded, dict) and all(recorded.get(k) is not None and recorded.get(k) == expect.get(k)
                                              for k in IDENTITY_KEYS)


def record_eval(eval_dir: str, run: str, mode: str, metrics_csv: str, ckpt: str, artifacts: Optional[Dict[str, str]] = None,
                work: str = WORK, **extra) -> Dict:
    """After a successful evaluation: which attempt / checkpoint it evaluated, its metrics file and the sha1 of the
    dumps it wrote (per-sequence errors, diagnostics), so later readers can reject files of another evaluation."""
    ev = {"run": run, "mode": mode, "attempt": current_attempt(run, work), "ckpt": ckpt,
          "ckpt_realpath": os.path.realpath(ckpt), "metrics_csv": metrics_csv,
          "artifacts": {k: {"path": p, "sha1": sha1_file(p)} for k, p in (artifacts or {}).items() if os.path.isfile(p)},
          "time": time.strftime("%Y-%m-%d %H:%M:%S"), **extra}
    _write_json(eval_done_path(eval_dir, run, mode), ev)
    return ev


def evaluates_result(ev: Dict, run: str, work: str = WORK) -> bool:
    """``ev`` evaluated the run's result: its training finished and the evaluated checkpoint is that training's best
    (an evaluation taken during the training, or of another epoch, is not)."""
    if not fit_finished(run, work):
        return False
    try:
        best = find_best(run, work, select_key=ev.get("select_key"))["best_ckpt"]
    except CheckpointNotFound:
        return False
    return os.path.realpath(ev.get("ckpt_realpath") or ev.get("ckpt") or "") == os.path.realpath(best)


def valid_eval(eval_dir: str, run: str, mode: str, work: str = WORK, expect: Optional[Dict] = None) -> Optional[Dict]:
    """The evaluation record of the run's result (same attempt, finished training, its best checkpoint); None otherwise.
    ``expect``: the identity of the requested behaviour - the record must carry it. Without ``expect`` (legacy
    readers), runs and evaluations from before records existed: ``{"legacy": True}`` once the training finished (no
    binding possible)."""
    ev = _read_json(eval_done_path(eval_dir, run, mode))
    att = current_attempt(run, work)
    if ev is None:
        return {"legacy": True} if expect is None and att is None and fit_finished(run, work) else None
    if att is not None and ev.get("attempt") != att:
        return None
    if expect is not None and not identity_matches(ev.get("identity"), expect):
        return None
    return ev if evaluates_result(ev, run, work) else None


def epoch_eval(eval_dir: str, run: str, mode: str, epoch: int, work: str = WORK,
               expect: Optional[Dict] = None) -> Optional[Dict]:
    """The recorded diagnostic evaluation of epoch ``epoch`` of the run's current training attempt (retest_runs.py
    --epoch): same attempt, same epoch, and the evaluated file is that attempt's checkpoint of the epoch; with
    ``expect`` also the identity of the requested behaviour. Without ``expect``, evaluations from before records
    existed: ``{"legacy": True}`` for runs without attempts. Never the run's result."""
    ev = _read_json(eval_done_path(eval_dir, run, mode))
    att = current_attempt(run, work)
    if ev is None:
        return {"legacy": True} if expect is None and att is None else None
    if (att is not None and ev.get("attempt") != att) or ev.get("epoch") != epoch:
        return None
    if expect is not None and not identity_matches(ev.get("identity"), expect):
        return None
    vdir = fit_version_dir(run, work)
    ck = _epoch_ckpt(os.path.join(vdir, "checkpoints"), epoch) if vdir else None
    evaluated = ev.get("ckpt_realpath") or ev.get("ckpt") or ""
    return ev if ck and os.path.realpath(ck) == os.path.realpath(evaluated) else None


def artifact_ok(ev: Optional[Dict], name: str, path: str) -> bool:
    """``path`` is the dump the recorded evaluation wrote (legacy evaluations: any existing file)."""
    if not ev:
        return False
    if ev.get("legacy"):
        return os.path.isfile(path)
    a = (ev.get("artifacts") or {}).get(name)
    return bool(a) and os.path.abspath(a["path"]) == os.path.abspath(path) and sha1_file(path) == a["sha1"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("start", "record", "finished", "best", "last", "status"))
    ap.add_argument("run")
    ap.add_argument("--select_key", default=None, help="best / status: the selection metric column (multi-dataset runs)")
    args = ap.parse_args()
    os.chdir(ROOT)
    if args.command == "start":
        print(start_attempt(args.run)["attempt"])
    elif args.command == "record":
        rec = write_record(args.run)
        print(json.dumps(rec, indent=1))
        if not rec["finished"]:
            sys.exit(1)
    elif args.command == "finished":
        sys.exit(0 if fit_finished(args.run) else 1)
    elif args.command in ("best", "last"):
        try:
            print(best_ckpt(args.run, select_key=args.select_key) if args.command == "best" else last_ckpt(args.run))
        except CheckpointNotFound as e:
            sys.stderr.write(f"{e}\n")
            sys.exit(1)
    else:
        out: Dict = {"run": args.run, "attempt": current_attempt(args.run), "finished": fit_finished(args.run),
                     "version_dir": fit_version_dir(args.run)}
        try:
            out.update(find_best(args.run, select_key=args.select_key))
        except CheckpointNotFound as e:
            out["best_error"] = str(e)
        print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
