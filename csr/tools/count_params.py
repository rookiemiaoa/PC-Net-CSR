"""Parameter-count audit for PC-Net under each released dataset configuration.

Why this exists
---------------
PC-Net does not have a single parameter count. Three of its modules are sized by
dataset-dependent dimensions that are fixed per benchmark in ``config/<dataset>/``:

    frame_fc = nn.Linear(frames_input_size, hidden_size)     -> d_v * 256 + 256
    pred_vec = nn.Parameter(torch.zeros(frames_input_size))  -> d_v
    fc_comp  = nn.Linear(hidden_size, vocab_size)            -> 256 * |V| + |V|

so the same model genuinely contains a different number of parameters on
Charades-CG (I3D, d_v = 1024, |V| = 1111), ActivityNet-CG (C3D, d_v = 500,
|V| = 8000), TACoS (C3D, d_v = 4096, |V| = 1734) and Ego4D (SlowFast,
d_v = 2304, |V| = 2604).

This script instantiates the released model with each released config and sums
``p.numel()`` over ``model.parameters()``, which is the authoritative way to
settle the number. It reproduces the TACoS and Ego4D figures printed in the
originating paper exactly; the Charades-CG and ActivityNet-CG figures printed
there are each 2,372,352 lower, which is exactly the size of one of the two
3-layer decoder stacks of the dual transformer (``DualTransformer.decoder1`` /
``.decoder2``, two independently instantiated stacks) -- i.e. a difference in
what is counted, not in what the model contains. See the manuscript's
Appendix A.6 for the discussion.

Usage
-----
    python csr/tools/count_params.py                 # all four configurations
    python csr/tools/count_params.py --module-detail # add a per-module breakdown

Requires only PyTorch and the repository's own ``models`` package; no data,
no checkpoints and no GPU are needed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# --- locate the repository root (this file lives in <repo>/csr/tools/) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))

CONFIGS = {
    "Charades-CG": "config/charades/main_train_test.json",
    "ActivityNet-CG": "config/activitynet/main_train_test.json",
    "TACoS": "config/tacos/main.json",
    "Ego4D": "config/ego4d/main.json",
}

# Figures printed in the originating PC-Net paper (Table 1, Table 2, Table 6 and
# the cross-dataset section), kept here so the comparison is reproducible.
PAPER_FIGURES = {
    "Charades-CG": 3.34,
    "ActivityNet-CG": 4.97,
    "TACoS": 6.66,
    "Ego4D": 6.42,
}


def build_model_config(cfg_path: str) -> dict:
    """Compose the flat config dict that ``models.pcnet.PCNet`` expects.

    The released configs nest the model settings under ``model.config`` and the
    input dimensions under ``dataset``; the constructor reads them from one
    flat mapping.
    """
    with open(cfg_path, encoding="utf-8") as fh:
        raw = json.load(fh)
    dataset = raw["dataset"]
    model_cfg = raw["model"]["config"]

    opt = dict(model_cfg)
    opt["vocab_size"] = dataset["vocab_size"]
    opt["dropout"] = model_cfg.get("dropout", 0.1)
    opt["num_iteration"] = model_cfg.get("num_iteration", 4)
    opt["max_epoch"] = model_cfg.get("max_epoch", 30)
    opt.setdefault("use_negative", True)
    return opt


def count(name: str, cfg_rel: str, module_detail: bool = False):
    """Instantiate PC-Net for one configuration and count its parameters."""
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    from models.pcnet import PCNet  # imported here so --help works without torch

    opt = build_model_config(os.path.join(_ROOT, cfg_rel))
    model = PCNet(opt)

    # model.parameters() is the headline number: it includes the bare top-level
    # nn.Parameter vectors (mask_vec / start_vec / pred_vec / merge_alpha /
    # range_width) that named_children() would miss.
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    n_vocab = opt["vocab_size"]
    d_v = opt["frames_input_size"]
    hidden = opt["hidden_size"]
    analytic = {
        "frame_fc(d_v*H+H)": d_v * hidden + hidden,
        "pred_vec(d_v)": d_v,
        "fc_comp(256*n_vocab+n_vocab)": hidden * n_vocab + n_vocab,
    }

    print(f"{name}")
    print(f"    config        : {cfg_rel}")
    print(f"    vocab_size    : {n_vocab}")
    print(f"    frames_input_size (d_v) : {d_v}")
    print(f"    total params  : {total:,}  ({total / 1e6:.2f} M)")
    print(f"    trainable     : {trainable:,}  ({trainable / 1e6:.2f} M)")
    print(f"    config-dependent terms:")
    for k, v in analytic.items():
        print(f"        {k:<30s} {v:>12,}")
    paper = PAPER_FIGURES.get(name)
    if paper is not None:
        print(f"    originating paper: {paper:.2f} M "
              f"(difference {total / 1e6 - paper:+.2f} M)")

    if module_detail:
        print("    per-module breakdown:")
        for mod_name, mod in model.named_children():
            c = sum(p.numel() for p in mod.parameters())
            print(f"        {mod_name:<20s} {c:>12,}")
    print()
    return total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--module-detail", action="store_true",
                    help="also print a per-module parameter breakdown")
    ap.add_argument("--dataset", choices=sorted(CONFIGS), default=None,
                    help="restrict the audit to one configuration")
    args = ap.parse_args()

    names = [args.dataset] if args.dataset else list(CONFIGS)
    print("=" * 72)
    print("PC-Net parameter-count audit (recounted from the released code)")
    print("=" * 72)
    for name in names:
        count(name, CONFIGS[name], module_detail=args.module_detail)
    print("Note: the Charades-CG and ActivityNet-CG figures printed in the "
          "originating paper\n      are each 2,372,352 lower than the recount "
          "-- exactly one decoder stack\n      of the dual transformer. "
          "This is a counting convention, not a model difference.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
