# -*- coding: utf-8 -*-
"""Verify that a config + CLI overrides reproduce the published training setting.

`train.py` does not read most hyper-parameters from the JSON: it loads the
config and then *unconditionally overwrites* six fields with argparse values
(see train.py:46-52).  The argparse defaults are `slot_num_iteration=3` and
`inter_lambda=0.13`, whereas the published run used 4 and 0.146.  So a
reproducer who edits the JSON alone, or who runs the release config without
flags, silently trains something else.

This tool reproduces that merge exactly and compares the flattened result, key
by key, against the config line that the published checkpoint itself logged.
The reference is the first line of `8_2025-05-14_13-26-12.log`, i.e. the run
behind the shipped weights -- not prose.

Two configs ship for Charades-CG:

    config/charades/main_train_test.json        release default, sigma_neg=4.5
                                                (the 2026 multi-seed setting)
    config/charades/main_train_test.paper.json  sigma_neg=9 (this paper)

Usage:
    python csr/tools/config_parity.py --published     # expect PASS
    python csr/tools/config_parity.py                 # expect FAIL: shows the trap
    python csr/tools/config_parity.py --published --json

Exit code 0 iff every key matches (model_saved_path is ignored: it carries a
timestamp).
"""
import argparse
import ast
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REF_LOG = os.path.join(REPO, "checkpoints", "CharadesSTA", "8_2025-05-14_13-26-12.log")
PAPER_CFG = os.path.join("config", "charades", "main_train_test.paper.json")
RELEASE_CFG = os.path.join("config", "charades", "main_train_test.json")

# argparse defaults, copied from train.py so that a bare run reproduces the
# bare behaviour rather than the published one.
TRAIN_DEFAULTS = dict(seed=8, slot_num_iteration=3, co_cl_loss=1.0,
                      co_qua_loss=1.0, num_props=8, inter_lambda=0.13, vote=False)
# What the published run passed on top of the paper config.
PUBLISHED = dict(seed=8, slot_num_iteration=4, co_cl_loss=0.5,
                 co_qua_loss=1.0, num_props=8, inter_lambda=0.146, vote=False)


def flatten(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flatten(v, prefix + "." + k if prefix else k))
        return out
    return {prefix: d}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None,
                    help="config JSON (default: paper config, or release config with --published off)")
    ap.add_argument("--ref", default=REF_LOG)
    ap.add_argument("--published", action="store_true",
                    help="apply the published CLI recipe instead of train.py's argparse defaults")
    ap.add_argument("--json", action="store_true")
    for k, v in TRAIN_DEFAULTS.items():
        if k == "vote":
            ap.add_argument("--vote", action="store_true", default=v)
        else:
            ap.add_argument("--" + k, type=type(v), default=v)
    args = ap.parse_args()

    cli = dict(seed=args.seed, slot_num_iteration=args.slot_num_iteration,
               co_cl_loss=args.co_cl_loss, co_qua_loss=args.co_qua_loss,
               num_props=args.num_props, inter_lambda=args.inter_lambda,
               vote=args.vote)
    if args.published:
        cli = dict(PUBLISHED)

    cfg_path = args.config or (PAPER_CFG if args.published else RELEASE_CFG)
    if not os.path.isabs(cfg_path):
        cfg_path = os.path.join(REPO, cfg_path)
    if not os.path.exists(cfg_path):
        print("config not found: %s" % cfg_path, file=sys.stderr)
        return 2
    if not os.path.exists(args.ref):
        print("reference log not found: %s\n"
              "the published checkpoint's log must be present to verify parity" % args.ref,
              file=sys.stderr)
        return 2

    # --- the merge performed by train.py:44-52 ---
    eff = json.load(open(cfg_path, encoding="utf-8"))
    eff["train"]["model_saved_path"] = os.path.join(eff["train"]["model_saved_path"], "<stamp>")
    eff["vote"] = cli["vote"]
    eff["model"]["config"]["num_iteration"] = cli["slot_num_iteration"]
    eff["model"]["config"]["num_props"] = cli["num_props"]
    eff["loss"]["co_cl_loss"] = cli["co_cl_loss"]
    eff["loss"]["co_qua_loss"] = cli["co_qua_loss"]
    eff["loss"]["inter_lambda"] = cli["inter_lambda"]
    eff["seed"] = cli["seed"]

    with open(args.ref, encoding="utf-8", errors="replace") as fh:
        ref = ast.literal_eval(fh.readline().split(" - ", 1)[1].strip())

    fa, fb = flatten(ref), flatten(eff)
    keys = [k for k in sorted(set(fa) | set(fb)) if "model_saved_path" not in k]
    diff = [(k, fa.get(k), fb.get(k)) for k in keys if fa.get(k) != fb.get(k)]

    if args.json:
        print(json.dumps({"config": cfg_path, "published": args.published, "keys": len(keys),
                          "diff": [{"key": k, "official": a, "mine": b} for k, a, b in diff]},
                         ensure_ascii=False, indent=2))
    else:
        shown = os.path.relpath(cfg_path, REPO) if cfg_path.startswith(REPO) else cfg_path
        print("config   : %s" % shown)
        print("overrides: %s" % ("published recipe" if args.published else "train.py argparse defaults"))
        print("keys     : %d (model_saved_path ignored)" % len(keys))
        if diff:
            print("DIFF     : %d" % len(diff))
            for k, a, b in diff:
                print("    %-42s official=%-12r mine=%r" % (k, a, b))
            print("RESULT   : FAIL -- this combination does not reproduce the published run")
        else:
            print("DIFF     : 0")
            print("RESULT   : PASS -- identical to the config that produced the shipped weights")

    return 1 if diff else 0


if __name__ == "__main__":
    sys.exit(main())
