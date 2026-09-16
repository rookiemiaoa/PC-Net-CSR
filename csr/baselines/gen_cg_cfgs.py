# -*- coding: utf-8 -*-
"""gen_cg_cfgs.py —— 为 QMN / PPS / CCR 生成 CG 切分的训练+评测 config。

目标：让这三个谱系内方法在 cg_train 上重训、在三个 CG test 切分上评测，
协议与 CPL / CNM / PC-Net 完全一致（I3D 特征、30 epoch、无切分污染）。

数据契约（与 CPL 一致）：
  feature_path = data/charades/i3d_features.hdf5（软链到共享 I3D）
  vocab_path   = data/charades/glove.pkl
  cg 切分 json 在 data/charades/ 下：cg_train.json / cg_test_trivial.json /
  cg_novel_comp.json / cg_novel_word.json

用法：
  python csr/baselines/gen_cg_cfgs.py                          # 默认用与 PC-Net 同级的 ws_base/
  python csr/baselines/gen_cg_cfgs.py --root /path/to/ws_base
"""
import argparse
import copy
import json
import os
import sys

# ---- 外部基线工作区：默认与 PC-Net 仓库同级（ws_base/），可用 --root 或 WS_BASE 覆盖 ----
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = _HERE
while _REPO != os.path.dirname(_REPO) and not os.path.exists(os.path.join(_REPO, 'models', 'pcnet.py')):
    _REPO = os.path.dirname(_REPO)
sys.path.insert(0, _REPO)

from csr.paths import WS_BASE                 # noqa: E402

SPLITS = {
    'test_trivial': 'cg_test_trivial.json',
    'novel_comp': 'cg_novel_comp.json',
    'novel_word': 'cg_novel_word.json',
}


def load(p):
    with open(p) as f:
        return json.load(f)


def dump(obj, p):
    with open(p, 'w') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print('  wrote', p)


def cpl_style(cfg_path, method_name, model_saved_dir, ckpt_tag):
    """CPL/CNM/QMN/CCR 共用同一套 dataset/train/model/loss 结构。
    method_name 用于 model.name；ckpt_tag 用于 model_saved_path 子目录。"""
    cfg = load(cfg_path)
    d = cfg['dataset']
    d['train_data'] = 'data/charades/cg_train.json'
    d['test_data'] = 'data/charades/cg_test_trivial.json'
    d['val_data'] = 'data/charades/cg_test_trivial.json'
    d['feature_path'] = 'data/charades/i3d_features.hdf5'
    d['vocab_path'] = 'data/charades/glove.pkl'
    cfg['model']['name'] = method_name
    cfg['train']['model_saved_path'] = model_saved_dir
    cfg['_ckpt_tag'] = ckpt_tag  # 仅用于脚本自省，训练代码忽略未知键
    return cfg


def pps_style(cfg_path, model_saved_dir):
    """PPS 的 config 结构不同（dataset.name / train.save_path / model 无 name）。"""
    cfg = load(cfg_path)
    d = cfg['dataset']
    d['feature_path'] = 'data/charades/i3d_features.hdf5'
    d['train_data'] = 'data/charades/cg_train.json'
    d['test_data'] = 'data/charades/cg_test_trivial.json'
    d['vocab_path'] = 'data/charades/glove.pkl'
    cfg['train']['save_path'] = model_saved_dir
    cfg['train']['use_wandb'] = False
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=None,
                    help='外部基线工作区目录；默认取与 PC-Net 仓库同级的 ws_base/')
    args = ap.parse_args()
    root = args.root or WS_BASE

    # ---------- QMN ----------
    qdir = os.path.join(root, 'QMN', 'config', 'charades')
    base = load(os.path.join(qdir, 'main_i3d.json'))
    # 训练 config（train_data=cg_train, 保存到 checkpoints/cg_main/qmn）
    tr = cpl_style(os.path.join(qdir, 'main_i3d.json'), 'QMN',
                   'checkpoints/cg_main/qmn/', 'qmn')
    dump(tr, os.path.join(qdir, 'cg_main.json'))
    # 三个评测 config（test_data 指向对应 cg 切分，val 同）
    for tag, fn in SPLITS.items():
        ev = copy.deepcopy(tr)
        ev['dataset']['test_data'] = 'data/charades/%s' % fn
        ev['dataset']['val_data'] = 'data/charades/%s' % fn
        dump(ev, os.path.join(qdir, 'eval_%s.json' % tag))

    # ---------- PPS ----------
    pdir = os.path.join(root, 'pps', 'config', 'charades')
    tr = pps_style(os.path.join(pdir, 'config.json'), 'checkpoints/cg_main/pps/')
    dump(tr, os.path.join(pdir, 'cg_main.json'))
    for tag, fn in SPLITS.items():
        ev = copy.deepcopy(tr)
        ev['dataset']['test_data'] = 'data/charades/%s' % fn
        dump(ev, os.path.join(pdir, 'eval_%s.json' % tag))

    # ---------- CCR ----------
    cdir = os.path.join(root, 'CCR', 'config', 'charades')
    tr = cpl_style(os.path.join(cdir, 'main.json'), 'CCR',
                   'checkpoints/cg_main/ccr/', 'ccr')
    # 作者原 main.json 里 model.name 误写为 "CPL"（CCR 派生自 CPL 未改名），
    # 但 models/__init__.py 只导出 CCR ⇒ 必须改成 "CCR" 才能 getattr 到类。
    dump(tr, os.path.join(cdir, 'cg_main.json'))
    for tag, fn in SPLITS.items():
        ev = copy.deepcopy(tr)
        ev['dataset']['test_data'] = 'data/charades/%s' % fn
        ev['dataset']['val_data'] = 'data/charades/%s' % fn
        dump(ev, os.path.join(cdir, 'eval_%s.json' % tag))

    print('ALL_CG_CFGS_DONE')


if __name__ == '__main__':
    main()
