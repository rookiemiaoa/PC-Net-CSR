"""Generate Charades-CG configs for CPL by cloning its own config/charades/main.json.

CPL's release checkpoint (checkpoints/charades/model-best.pt) is trained on the
standard Charades-STA train split, so evaluating it on the CG test splits measures
zero-shot transfer.  To get protocol-fair numbers we also expose a cg_main config
that trains on cg_train.json (same protocol as PC-Net / CNM).
"""
import json
import os

# 外部基线工作区：默认与 PC-Net 仓库同级（ws_base/），可用 WS_BASE 覆盖
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = _HERE
while _REPO != os.path.dirname(_REPO) and not os.path.exists(os.path.join(_REPO, 'models', 'pcnet.py')):
    _REPO = os.path.dirname(_REPO)
BASE = os.environ.get('WS_BASE') or os.path.join(os.path.dirname(_REPO), 'ws_base')
BASE = os.path.join(BASE, 'cpl')
CFG = os.path.join(BASE, "config", "charades")
main = json.load(open(os.path.join(CFG, "main.json")))

# training configs on the CG train split (for the fair comparison)
TRAIN = [("cg_main", "cg_train.json")]
for tag, f in TRAIN:
    c = json.loads(json.dumps(main))
    c["dataset"]["train_data"] = "data/charades/" + f
    c["dataset"]["test_data"] = "data/charades/cg_test_trivial.json"
    c["dataset"]["val_data"] = "data/charades/cg_test_trivial.json"
    c["train"]["model_saved_path"] = "checkpoints/cg_main/"
    json.dump(c, open(os.path.join(CFG, tag + ".json"), "w"), indent=2)
    print("wrote", tag)

# eval-only configs, one per CG split
for s in ["test_trivial", "novel_comp", "novel_word"]:
    c = json.loads(json.dumps(main))
    c["dataset"]["test_data"] = "data/charades/cg_" + s + ".json"
    c["dataset"]["val_data"] = "data/charades/cg_" + s + ".json"
    json.dump(c, open(os.path.join(CFG, "eval_" + s + ".json"), "w"), indent=2)
    print("wrote eval_" + s)
print("CPL_PREP_DONE")
