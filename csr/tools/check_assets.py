# -*- coding: utf-8 -*-
"""check_assets.py —— 复现前自检：权重、特征、外部仓库、磁盘余量

复现失败的绝大多数原因不是代码，而是资产没到位（权重缺失、hdf5 放错位置、
外部基线仓库不在约定的同级目录）。本脚本把所有前置条件一次性查完，
缺什么就报什么，避免跑到一半才 OSError。

用法
----
    python csr/tools/check_assets.py                 # 权重 + 外部仓库 + 磁盘
    python csr/tools/check_assets.py --weights       # 只查 7 个官方权重
    python csr/tools/check_assets.py --features      # 额外检查 hdf5 可打开且非空
    python csr/tools/check_assets.py --strict        # 缺任何一项即返回 1（CI 用）
    python csr/tools/check_assets.py --json out.json

退出码：0 = 全部就位（或非 strict 下有缺失）；1 = strict 且有缺失。
"""
import argparse
import hashlib
import json
import os
import shutil
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = _HERE
while ROOT != os.path.dirname(ROOT) and not os.path.exists(os.path.join(ROOT, 'models', 'pcnet.py')):
    ROOT = os.path.dirname(ROOT)

WEIGHT_LIST = [
    'checkpoints/ActivityNet/8_2025-05-14_14-07-12/model-best.pt',
    'checkpoints/TACoS/8_2025-05-14_21-05-09/model-best.pt',
    'checkpoints/CharadesSTA/8_2025-05-14_13-26-12/model-best.pt',
    'checkpoints/CharadesSTA/11_2026-09-14_17-09-28/model-best.pt',
    'checkpoints/CharadesSTA/12_2026-09-14_17-09-28/model-best.pt',
    'checkpoints/CharadesSTA/13_2026-09-14_17-09-28/model-best.pt',
    'checkpoints/CharadesSTA/14_2026-09-14_17-39-33/model-best.pt',
]

# 配置里出现的特征文件名 -> 说明（实际路径由 FEAT_PATH / DATA_ROOT 决定）
FEATURE_LIST = [
    ('i3d_features.hdf5', 'Charades-CG（约 3 GB）'),
    ('sub_activitynet_v1-3.c3d.hdf5', 'ActivityNet-CG（约 4.5-5.5 GB）'),
    ('tacos_features_c3d.hdf5', 'TACoS（约 0.4 GB）'),
]

EXTERNAL_DIRS = [
    ('WS_BASE', '../ws_base', 'CPL / QMN / PPS / CCR 的克隆根目录'),
    ('CNM_ROOT', '../CNM-CG', 'CNM（AAAI-2022）克隆根目录'),
]


def md5_of(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_expected_md5():
    p = os.path.join(ROOT, 'checkpoints', 'WEIGHTS.md5')
    exp = {}
    if not os.path.exists(p):
        return exp
    with open(p, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                exp[parts[1].strip().lstrip('*')] = parts[0].strip()
    return exp


def find_feature(name):
    """按 FEAT_PATH -> DATA_ROOT -> 仓库根 -> 仓库根/data 的顺序找。"""
    cands = []
    fp = os.environ.get('FEAT_PATH')
    if fp and os.path.basename(fp) == name:
        cands.append(fp)
    dr = os.environ.get('DATA_ROOT')
    if dr:
        cands.append(os.path.join(dr, name))
    cands += [os.path.join(ROOT, name), os.path.join(ROOT, 'data', name)]
    for c in cands:
        if os.path.exists(c):
            return c
    return None


def main():
    ap = argparse.ArgumentParser(description='复现前资产自检')
    ap.add_argument('--weights', action='store_true', help='只查权重')
    ap.add_argument('--features', action='store_true', help='检查 hdf5 是否可打开')
    ap.add_argument('--strict', action='store_true', help='有缺失即返回 1')
    ap.add_argument('--json', default=None)
    args = ap.parse_args()

    exp = load_expected_md5()
    missing = []
    report = {'root': ROOT, 'weights': [], 'features': [], 'external': [], 'disk': {}}

    print('仓库根: %s' % ROOT)
    print('=' * 78)

    if not args.features:
        print('[1] 官方权重（7 个，未入库；缺失时从发布附件下载）')
        for rel in WEIGHT_LIST:
            p = os.path.join(ROOT, rel)
            item = {'path': rel, 'exists': os.path.exists(p)}
            if item['exists']:
                item['size'] = os.path.getsize(p)
                item['md5'] = md5_of(p)
                want = exp.get(rel)
                item['md5_ok'] = (want is None) or (item['md5'] == want)
                tag = 'OK  ' if item['md5_ok'] else 'FAIL'
                print('   %-4s %-58s %10d B  %s' % (tag, rel, item['size'], item['md5']))
                if not item['md5_ok']:
                    print('        期望 %s' % want)
                    missing.append(rel + ' (md5 不符)')
            else:
                print('   MISS %s' % rel)
                missing.append(rel)
            report['weights'].append(item)

    # ---- 特征
    print('')
    print('[2] 视频特征（本项目不分发，需自备）')
    for name, note in FEATURE_LIST:
        p = find_feature(name)
        item = {'name': name, 'note': note, 'path': p, 'exists': bool(p)}
        if p:
            item['size'] = os.path.getsize(p)
            print('   OK   %-32s %s' % (name, p))
            if args.features:
                try:
                    import h5py
                    with h5py.File(p, 'r') as f:
                        item['n_keys'] = len(f.keys())
                        item['sample_keys'] = list(f.keys())[:3]
                    print('        keys=%d  sample=%s' % (item['n_keys'], item['sample_keys']))
                except Exception as e:                      # noqa: BLE001
                    item['error'] = repr(e)
                    print('        [FAIL] 打不开：%r' % e)
                    missing.append(name + ' (无法打开)')
        else:
            print('   MISS %-32s %s' % (name, note))
            missing.append(name)
        report['features'].append(item)

    # ---- 外部仓库
    print('')
    print('[3] 外部基线仓库（不入仓，需自行克隆）')
    for env, default, note in EXTERNAL_DIRS:
        p = os.environ.get(env) or os.path.join(ROOT, default)
        p = os.path.abspath(p)
        ok = os.path.isdir(p)
        print('   %-4s %-16s %s  (%s)' % ('OK' if ok else 'MISS', env, p, note))
        if not ok:
            missing.append('%s=%s' % (env, p))
        report['external'].append({'env': env, 'path': p, 'exists': ok, 'note': note})

    # ---- 磁盘
    print('')
    print('[4] 磁盘余量（Charades 3.0 GB + TACoS 0.4 + ANET 4.5-5.5 + 权重 0.17）')
    try:
        du = shutil.disk_usage(ROOT)
        free_gb = du.free / 1024 ** 3
        report['disk'] = {'free_bytes': du.free, 'free_gb': round(free_gb, 2)}
        print('    %s 可用 %.2f GB' % (os.path.splitdrive(ROOT)[0] or ROOT, free_gb))
        if free_gb < 6.0:
            print('    [WARN] 不足 6 GB：ANET 阶段请把 Charades 的 hdf5 移出本盘，或挂第二块盘')
    except OSError as e:                                    # noqa: BLE001
        print('    无法读取磁盘信息：%r' % e)

    report['missing'] = missing
    print('')
    print('=' * 78)
    if missing:
        print('MISSING %d 项：' % len(missing))
        for m in missing:
            print('  - %s' % m)
    else:
        print('ALL PRESENT')

    if args.json:
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print('WROTE %s' % args.json)

    return 1 if (args.strict and missing) else 0


if __name__ == '__main__':
    sys.exit(main())
