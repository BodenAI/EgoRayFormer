#!/usr/bin/env python
"""Pre-check of the effective training configuration.

``PRECHECK=1 bash scripts/run_baselines.sh ...`` builds every fit exactly as a real launch (heatmap initialisation,
INIT_FROM_RUN, EXTRA), lets LightningCLI parse it (``run.py fit ... --print_config``: no training, no attempt) and runs
this script on the printed configuration. It reports what the merged configuration trains:
  - image size, batch, clips scale, fov_circle and the initialisation sources (pending: the source run has not finished);
  - the training and validation domains with their sequences, frames and clip entries, and each training domain's
    expected share of the draws (entry weight x entries, as the sampler draws);
  - optimizer steps per epoch (Lightning's count of the training loader, limit_train_batches and
    train_samples_per_epoch included; with gradient accumulation a batch still draws its clips once), total steps and
    the global steps at which validation runs; a max_steps that truncates the fit is refused;
  - the milestones as parsed and the first step of each learning-rate decay (the training code's own rule:
    egorayformer.training.config.milestone_steps over steps_per_epoch = total steps // max_epochs).

    python scripts/precheck_config.py work_dirs/precheck/<run>.yaml [--json work_dirs/precheck/<run>.json] [--pending "..."]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import OrderedDict
from typing import Dict, List, Optional

import yaml

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)


def num_batches(length: int, limit) -> int:
    """Batches Lightning runs per epoch from a loader of ``length`` batches (trainer _parse_num_batches)."""
    if length == 0 or limit is None:
        return int(length)
    if isinstance(limit, bool):
        raise ValueError(f"limit_train_batches {limit!r}")
    if isinstance(limit, int):
        return min(length, limit)
    return int(length * float(limit))


def schedule(samples: int, world: int, batch_size: int, limit, acc: int, epochs: int, max_steps: int, every: int) -> Dict:
    """Batches, optimizer steps, validation points and clips of one fit. Every training batch draws
    batch_size clips on each device whether or not it ends a gradient accumulation; the optimizer steps after every
    ``acc`` batches and after the last batch of an epoch (Lightning), ceil(batches / acc) times per epoch, and the
    epoch-end validations run at multiples of that. A max_steps that ends the fit before its last epoch is refused: the
    reachable validation points and the domain exposure of a truncated fit are not derived here."""
    loader_batches = math.ceil(samples / world) // batch_size                # per device, drop_last
    batches = num_batches(loader_batches, limit)
    per_epoch = math.ceil(batches / acc)
    total = per_epoch * max(epochs, 1)
    if max_steps != -1 and max_steps < total:
        raise ValueError(f"max_steps {max_steps} ends the fit before its {total} optimizer steps: the precheck does not "
                         "derive the validation points and domain exposure of a truncated schedule")
    return OrderedDict(loader_batches=loader_batches, train_batches=batches, steps_per_epoch=per_epoch, total_steps=total,
                       validation_steps=[per_epoch * e for e in range(1, epochs + 1) if e % every == 0],
                       clips_per_epoch=batches * batch_size * world)


def world_size(trainer: Dict) -> int:
    d, nodes = trainer.get("devices"), int(trainer.get("num_nodes") or 1)
    if isinstance(d, int) and d > 0:
        return d * nodes
    if isinstance(d, (list, tuple)):
        return len(d) * nodes
    if isinstance(d, str) and d.strip().isdigit():
        return int(d) * nodes
    return nodes            # "auto" / -1: decided at launch; pin one device for an exact count


def monitor(trainer: Dict) -> Optional[str]:
    for cb in trainer.get("callbacks") or []:
        if isinstance(cb, dict) and str(cb.get("class_path", "")).endswith("ModelCheckpoint"):
            return (cb.get("init_args") or {}).get("monitor")
    return None


def _sources(ds, sources: List[Dict], weighted: bool) -> List[Dict]:
    """Per source of a dataset: sequences, frames, clip entries and (training) the expected share of the draws."""
    n = [0] * len(sources)
    for si, _, _ in ds.entries:
        n[si] += 1
    w = [float(s.get("weight", 1.0)) for s in sources]
    mass = [w[i] * n[i] if weighted else n[i] for i in range(len(sources))]
    total = sum(mass) or 1.0
    return [OrderedDict(dataset=s["dataset"], split=s["split"], weight=w[i], supervision=s.get("supervision", "full"),
                        sequences=len(ds.indices[i].sequences), frames=ds.indices[i].num_frames, entries=n[i],
                        share=mass[i] / total) for i, s in enumerate(sources)]


def derive(cfg: Dict, pending: str = "") -> Dict:
    from egorayformer.data.datamodule import EgoRayFormerDataModule
    from egorayformer.training.config import milestone_steps

    data, model, trainer = cfg["data"], cfg["model"], cfg["trainer"]
    dm = EgoRayFormerDataModule(**data)
    dm.setup("fit")
    loader = dm.train_dataloader()
    world = world_size(trainer)
    samples = len(loader.sampler)
    acc = int(trainer.get("accumulate_grad_batches") or 1)
    epochs = int(trainer["max_epochs"])
    every, interval = int(trainer.get("check_val_every_n_epoch") or 1), trainer.get("val_check_interval")
    sch = schedule(samples, world, int(data["batch_size"]), trainer.get("limit_train_batches"), acc, epochs,
                   int(trainer.get("max_steps") or -1), every)
    total = sch["total_steps"]
    spe = max(1, total // max(1, epochs))                                   # configure_optimizers
    notes = []
    if interval not in (None, 1.0, 1):
        notes.append(f"val_check_interval {interval}: validation points below are the epoch ends only")
    if world > 1:
        notes.append(f"{world} devices: per-device draws ceil({samples} / {world})")
    if pending:
        notes.append(f"pending: {pending}")
    mm = model["model"]
    train = _sources(dm.train_ds, list(data["train"]), bool(data.get("weighted_sampling")))
    for s in train:
        s["draws_per_epoch"] = s["share"] * sch["clips_per_epoch"]
        s["draws_total"] = s["share"] * sch["clips_per_epoch"] * epochs
    optim = model.get("optim") or {}
    return OrderedDict(
        image_size=OrderedDict(data=data.get("img_size"), model=mm.get("image_size")),
        batch_size=data.get("batch_size"), eval_batch_size=data.get("eval_batch_size"),
        clips_per_epoch_scale=data.get("clips_per_epoch_scale"), clip_len=data.get("clip_len"),
        eval_clip_len=data.get("eval_clip_len"), fov_circle=mm.get("fov_circle"),
        encoder=(mm.get("encoder") or {}).get("type"), stage=model.get("stage"),
        init=OrderedDict(encoder_pretrained=model.get("encoder_pretrained"), init_from=model.get("init_from"),
                         init_from_ema=model.get("init_from_ema")),
        precision=trainer.get("precision"), devices=trainer.get("devices"), strategy=trainer.get("strategy"),
        seed=cfg.get("seed_everything"), eval_precision=model.get("eval_precision"),
        select_val_dataset=model.get("select_val_dataset"), checkpoint_monitor=monitor(trainer),
        weighted_sampling=data.get("weighted_sampling"), train_samples_per_epoch=data.get("train_samples_per_epoch"),
        train=train, val=_sources(dm.val_ds, list(data["val"]), False) if dm.val_ds is not None else [],
        samples_per_epoch=samples, loader_batches=sch["loader_batches"], limit_train_batches=trainer.get("limit_train_batches"),
        train_batches=sch["train_batches"], accumulate_grad_batches=acc,
        steps_per_epoch=spe, max_epochs=epochs, total_steps=total, validation_steps=sch["validation_steps"],
        milestones=optim.get("milestones"), milestone_steps=milestone_steps(optim.get("milestones") or [], spe),
        schedule=optim.get("schedule"), warmup_steps=optim.get("warmup_steps"), notes=notes)


def report(name: str, d: Dict) -> List[str]:
    f = lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) else str(x)   # noqa: E731
    lines = [f"[precheck] {name}",
             f"  image {d['image_size']['data']} (model {d['image_size']['model']}), batch {d['batch_size']}, clips scale "
             f"{d['clips_per_epoch_scale']}, clip {d['clip_len']}, eval clip {d['eval_clip_len']}, fov_circle {d['fov_circle']}, "
             f"encoder {d['encoder']}, stage {d['stage']}, {d['precision']} x {d['devices']} ({d['strategy']}), seed {d['seed']}",
             f"  init: encoder_pretrained {d['init']['encoder_pretrained']}, init_from {d['init']['init_from']}",
             f"  selection: val {d['select_val_dataset'] or '(pooled)'}, ModelCheckpoint monitor {d['checkpoint_monitor']}"]
    for s in d["train"]:
        lines.append(f"  train {s['dataset']}/{s['split']}: {s['sequences']} seq, {f(s['frames'])} frames, {f(s['entries'])} "
                     f"entries, weight {s['weight']:g}, share {100 * s['share']:.1f} %, {f(s['draws_per_epoch'])} draws / epoch, "
                     f"{f(s['draws_total'])} in total")
    for s in d["val"]:
        lines.append(f"  val   {s['dataset']}/{s['split']}: {s['sequences']} seq, {f(s['frames'])} frames, {f(s['entries'])} clips")
    lines.append(f"  weighted_sampling {d['weighted_sampling']}, train_samples_per_epoch {d['train_samples_per_epoch']}: "
                 f"{f(d['samples_per_epoch'])} draws -> {f(d['loader_batches'])} batches, limit_train_batches "
                 f"{d['limit_train_batches']} -> {f(d['train_batches'])} batches, accumulate_grad_batches "
                 f"{d['accumulate_grad_batches']} -> {f(d['steps_per_epoch'])} optimizer steps / epoch x {d['max_epochs']} = "
                 f"{f(d['total_steps'])} steps")
    lines.append(f"  validation after steps {d['validation_steps']} ({len(d['validation_steps'])} points)")
    lines.append(f"  milestones {d['milestones']} -> decay from steps {d['milestone_steps']} ({d['schedule']}, warmup "
                 f"{d['warmup_steps']})")
    lines += [f"  note: {n}" for n in d["notes"]]
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="the configuration printed by run.py fit ... --print_config")
    ap.add_argument("--name", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--pending", default="", help="initialisation sources that are not finished yet (run_baselines.sh)")
    args = ap.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    name = args.name or os.path.splitext(os.path.basename(args.config))[0]
    try:
        d = derive(cfg, args.pending)
    except ValueError as e:
        sys.exit(f"[precheck] {name}: {e}")
    print("\n".join(report(name, d)))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(OrderedDict(run=name, config=args.config, **d), f, indent=1)


if __name__ == "__main__":
    main()
