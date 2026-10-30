#!/usr/bin/env python
"""Identity of an evaluation.

One checkpoint is evaluated under several inference behaviours (gate overrides, rig perturbations,
diagnostic probes). An evaluation record stands only for the behaviour it ran:
  config_id  sha1 of the effective configuration. The run.py arguments of the real evaluation - the run's saved config,
             the overrides an evaluation adds (config_overrides), then every --config file and CLI override of --extra
             in the order they appear - are parsed by LightningCLI with every default filled in
             (``run.py validate <arguments> --print_config``); the fields in NON_BEHAVIOUR are dropped and the result is
             hashed as JSON with sorted keys. Only the final dictionary is canonicalised: the order of --config files
             and of repeated overrides keeps its meaning, and so does the order of lists.
  code       the egorayformer package (scripts/code_hash.sh) and run.py of the evaluating tree (the current directory)
  mode       validate / test
The training attempt and the evaluated checkpoint (its realpath) are bound by scripts/ckpt_select.py, whose readers
(valid_eval / epoch_eval with ``expect``) compare this identity with the one they expect: a record of another
configuration, code or mode, or a record without an identity, is not an evaluation of the expected behaviour.
scripts/retest_runs.py stores the identity in eval_done_<mode>.json and the complete parsed configuration in
eval_config_<mode>.yaml, and checks that the configuration the evaluation saved itself
(lightning_logs/version_*/config.yaml) has the same config_id.

The configuration is parsed with the validate subcommand also for test evaluations (the two subcommands parse the same
arguments; run.py refuses ``test`` without EGORAYFORMER_ALLOW_TEST and a parse never needs it). A parse takes a few
seconds (imports): parsed configurations are cached under work_dirs/.eval_identity_cache, keyed by the code identity,
the ordered arguments with the sha1 of every --config file, the sha1 of every file under configs/ and the Python /
jsonargparse / pytorch-lightning versions.

    python scripts/eval_identity.py <run> [--mode validate] [--extra "--model.eval_precision fp32 ..."]
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shlex
import subprocess
import sys
import threading
from importlib import metadata
from typing import Dict, List, Optional, Sequence, Tuple

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ckpt_select  # noqa: E402

CACHE = os.path.join("work_dirs", ".eval_identity_cache")
# Fields of the parsed configuration that do not change what an evaluation computes: where its logs, metrics and dumps
# are written, and the checkpoint path (the checkpoint is bound by its realpath: ckpt_select.evaluates_result /
# epoch_eval). Device count, strategy and precision are protocol and stay in the identity.
NON_BEHAVIOUR = ("trainer.logger", "trainer.default_root_dir", "model.seq_error_dump", "model.diag_dump",
                 "model.calib_dump", "ckpt_path")
DISPLAY_CALLBACKS = ("ProgressBar",)   # trainer.callbacks entries whose class_path ends with one of these
EXCLUDED = [*NON_BEHAVIOUR, "trainer.callbacks[*ProgressBar]"]


class IdentityError(RuntimeError):
    pass


def code_identity() -> Dict[str, str]:
    """Code of the evaluating tree (the current directory): the egorayformer package and run.py."""
    try:
        package = subprocess.run(["bash", "scripts/code_hash.sh"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as e:
        raise IdentityError(f"scripts/code_hash.sh failed in {os.getcwd()}: {e}") from e
    run_py = ckpt_select.sha1_file("run.py")
    if not package or not run_py:
        raise IdentityError(f"no package hash or no run.py in {os.getcwd()}")
    return {"package": package, "run_py": run_py[:12]}


def config_files(args: Sequence[str]) -> List[Tuple[int, str]]:
    """(position, path) of every --config file of an argument list, in order."""
    out = []
    for i, a in enumerate(args):
        if a in ("--config", "-c") and i + 1 < len(args):
            out.append((i + 1, args[i + 1]))
        elif a.startswith("--config="):
            out.append((i, a.split("=", 1)[1]))
    return out


def _configs_digest() -> str:
    """sha1 over every file under configs/ (a configuration value may name another file)."""
    h = hashlib.sha1()
    for p in sorted(os.path.join(d, f) for d, _, files in os.walk("configs") for f in files):
        h.update(f"{p}\0{ckpt_select.sha1_file(p) or ''}\0".encode())
    return h.hexdigest()


def _versions() -> Dict[str, str]:
    out = {"python": sys.version.split()[0]}
    for pkg in ("jsonargparse", "pytorch-lightning"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = ""
    return out


def _cache_key(args: Sequence[str], code: Dict[str, str]) -> str:
    files = dict(config_files(args))
    key = {"code": code, "configs": _configs_digest(), "versions": _versions(),
           "args": [{"arg": a, "sha1": ckpt_select.sha1_file(files[i])} if i in files else a for i, a in enumerate(args)]}
    return hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()


def parse(args: Sequence[str]) -> Dict:
    """The configuration LightningCLI builds from ``run.py validate <args>``, defaults filled in - parsed, not run."""
    if "--print_config" in args or any(a.startswith("--print_config=") for a in args):
        raise IdentityError("the arguments already ask for --print_config")
    env = {k: v for k, v in os.environ.items() if k != "EGORAYFORMER_ALLOW_TEST"}
    env["CUDA_VISIBLE_DEVICES"] = ""
    p = subprocess.run([sys.executable, "run.py", "validate", *args, "--print_config"], capture_output=True, text=True,
                       env=env, timeout=900)
    if p.returncode != 0:
        raise IdentityError(f"run.py validate --print_config failed (rc {p.returncode}): {p.stderr.strip()[-1500:]}")
    try:
        cfg = yaml.safe_load(p.stdout)
    except yaml.YAMLError as e:
        raise IdentityError(f"run.py validate --print_config printed no YAML configuration: {e}") from e
    if not isinstance(cfg, dict) or not {"trainer", "model", "data"} <= set(cfg):
        raise IdentityError("run.py validate --print_config printed no complete configuration")
    return cfg


def effective(args: Sequence[str], code: Optional[Dict[str, str]] = None) -> Dict:
    """``parse`` through the cache (work_dirs/.eval_identity_cache)."""
    path = os.path.join(CACHE, f"{_cache_key(args, code or code_identity())}.json")
    if os.path.isfile(path):
        with open(path, "r") as f:
            return json.load(f)
    cfg = parse(args)
    os.makedirs(CACHE, exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}.{threading.get_ident()}"
    with open(tmp, "w") as f:
        json.dump(cfg, f)
    os.replace(tmp, path)
    return cfg


def _drop(cfg: Dict, dotted: str) -> None:
    *head, last = dotted.split(".")
    d = cfg
    for k in head:
        d = d.get(k) if isinstance(d, dict) else None
    if isinstance(d, dict):
        d.pop(last, None)


def _display(cb) -> bool:
    return isinstance(cb, dict) and str(cb.get("class_path", "")).endswith(DISPLAY_CALLBACKS)


def behaviour(cfg: Dict) -> Dict:
    """The parsed configuration without the fields that do not change what the evaluation computes."""
    out = copy.deepcopy(cfg)
    for f in NON_BEHAVIOUR:
        _drop(out, f)
    tr = out.get("trainer")
    if isinstance(tr, dict):
        cbs = tr.get("callbacks")
        if isinstance(cbs, list):
            tr["callbacks"] = [c for c in cbs if not _display(c)]
        elif _display(cbs):
            tr["callbacks"] = None
    return out


def config_id(cfg: Dict) -> str:
    return hashlib.sha1(json.dumps(behaviour(cfg), sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


def identity(args: Sequence[str], mode: str) -> Tuple[Dict, Dict]:
    """-> (identity, parsed configuration) of the evaluation ``run.py <mode> <args>`` in the current directory."""
    code = code_identity()
    cfg = effective(args, code)
    ident = {"config_id": config_id(cfg), "code": code, "mode": mode,
             "sources": [{"path": p, "sha1": ckpt_select.sha1_file(p)} for _, p in config_files(args)],
             "args": list(args), "excluded": list(EXCLUDED)}
    return ident, cfg


def config_overrides(cfg_path: str) -> List[str]:
    """Overrides an evaluation adds to a run's saved config: the evaluated weights come from --ckpt_path, so the
    weights-only initialisation source of a fine-tuning run (model.init_from) is not loaded again."""
    with open(cfg_path, "r") as f:
        saved = yaml.safe_load(f) or {}
    return ["--model.init_from", "null"] if (saved.get("model") or {}).get("init_from") else []


def saved_config_head(run: str) -> List[str]:
    """The arguments an evaluation of ``run`` starts with (retest_runs.py --from_config): the saved config of its
    current training attempt, one device, and config_overrides."""
    d = ckpt_select.fit_version_dir(run)
    cfg = os.path.join(d, "config.yaml") if d else ""
    if not os.path.isfile(cfg):
        raise IdentityError(f"{run}: no saved config.yaml")
    return ["--config", cfg, "--trainer.devices", "1", "--trainer.strategy", "auto", *config_overrides(cfg)]


def expected(run: str, mode: str, extra: str = "") -> Dict:
    """The identity of the evaluation of ``run`` with ``--extra extra`` (retest_runs.py --from_config) under the code
    of the current directory; dump and output arguments do not enter it (NON_BEHAVIOUR)."""
    return identity([*saved_config_head(run), *shlex.split(extra)], mode)[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--mode", choices=("validate", "test"), default="validate")
    ap.add_argument("--extra", default="")
    args = ap.parse_args()
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    try:
        print(json.dumps(expected(args.run, args.mode, args.extra), indent=1))
    except IdentityError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
