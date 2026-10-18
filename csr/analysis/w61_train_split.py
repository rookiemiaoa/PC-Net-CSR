# -*- coding: utf-8 -*-
"""w61_train_split.py —— 表 4（训练/测试划分）的驱动与校验

为什么需要这个脚本
  表 4 的 8 个数（4 骨干 × 训练/测试）此前"有产物、没生成器"，读者无法重跑。
  核对原始启动脚本后确认：**指标实现就是 `w34_curve_usability.py`**，w61 只是
  「同一份采集器 + 换一个切分 + 换 K_REP」的编排。本脚本把这层编排固化下来，
  不复制任何指标代码。

口径（写死，改动会使表 4 与表 3 不可比）
  BS=16   WPT=0.1   DDIR=0.08（w34 内部）
  训练列：TEST_DATA = train.json / cg_train.json（N=8281），K_REP=1
  测试列：TEST_DATA = test_trivial.json / cg_test_trivial.json（N=3096），K_REP=3

用法
  python csr/analysis/w61_train_split.py --dry-run          # 只打印 8 条命令
  python csr/analysis/w61_train_split.py --check            # 离线：校验已入库的 8 个 JSON
  python csr/analysis/w61_train_split.py --backbone pcnet --gpu 7
  python csr/analysis/w61_train_split.py --all --gpus 4,5,6,7
  python csr/analysis/w61_train_split.py --all --gpus 4,5,6,7 --manifest

依赖（服务器上）
  WS_BASE 默认 ../ws_base2，可用环境变量覆盖；
  PC-Net 权重默认 checkpoints/CharadesSTA/8_2025-05-14_13-26-12/model-best.pt，
  可用 PCNET_CKPT 覆盖（原始产物用的是 official_8，两者需先用 --check-equivalence 验证）。

退出码：0 = 成功/通过；1 = 有作业失败或校验不符。
"""
import argparse
import concurrent.futures as cf
import glob
import hashlib
import json
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = _HERE
while REPO != os.path.dirname(REPO) and not os.path.exists(os.path.join(REPO, 'models', 'pcnet.py')):
    REPO = os.path.dirname(REPO)

W34 = os.path.join(_HERE, 'w34_curve_usability.py')
OUT_DIR = os.path.join(REPO, 'results', 'table4')
WS_BASE = os.environ.get('WS_BASE') or os.path.normpath(os.path.join(REPO, '..', 'ws_base2'))
PCNET_CKPT = os.environ.get('PCNET_CKPT') or 'checkpoints/CharadesSTA/8_2025-05-14_13-26-12/model-best.pt'

BS = 16
WPT = 0.1
N_PROBE, N_TEST = 8281, 3096

# backbone -> (repo, cfg_probe, cfg_test, ckpt 候选 glob（相对 repo）, probe 切分, test 切分)
BACKBONES = {
    'pcnet': (REPO,
              'config/charades/main_train_test.json',
              'config/charades/main_train_test.json',
              [PCNET_CKPT,
               'checkpoints/CharadesSTA/official_8/model-best.pt'],
              'data/charades/train.json',
              'data/charades/test_trivial.json'),
    'cpl':  ('cpl', 'config/charades/cg_main.json', 'config/charades/eval_test_trivial.json',
             ['checkpoints/cg_main/*/model-best.pt'], 'data/charades/cg_train.json',
             'data/charades/cg_test_trivial.json'),
    'qmn':  ('QMN', 'config/charades/cg_main.json', 'config/charades/eval_test_trivial.json',
             ['checkpoints/cg_main/qmn/*/model-best.pt', 'checkpoints/cg_main/*/model-best.pt'],
             'data/charades/cg_train.json', 'data/charades/cg_test_trivial.json'),
    'ccr':  ('CCR', 'config/charades/cg_main.json', 'config/charades/eval_test_trivial.json',
             ['checkpoints/cg_main/ccr/*/model-best.pt', 'checkpoints/cg_main/*/model-best.pt'],
             'data/charades/cg_train.json', 'data/charades/cg_test_trivial.json'),
}
ORDER = ['pcnet', 'cpl', 'qmn', 'ccr']


def repo_of(key):
    p = BACKBONES[key][0]
    return p if os.path.isabs(p) else os.path.join(WS_BASE, p)


def resolve_ckpt(key):
    """在 repo 内按候选 glob 找权重；多个命中取 mtime 最新的。"""
    root = repo_of(key)
    for pat in BACKBONES[key][3]:
        hits = [p for p in glob.glob(os.path.join(root, pat)) if os.path.exists(p)]
        if hits:
            hits.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return hits[0]
    return None


def jobs(keys, gpu):
    """展开成 8 条（backbone, 列, 命令）三元组。"""
    out = []
    for key in keys:
        root, cfg_p, cfg_t, _, td_p, td_t = BACKBONES[key]
        root = repo_of(key)
        ck = resolve_ckpt(key)
        for col, cfg, td, krep, tag in (
                ('probe', cfg_p, td_p, 1, '%s_TT_probe' % key),
                ('test', cfg_t, td_t, 3, '%s_TT_test' % key)):
            if col == 'test':
                cfg = cfg_t
            out.append((key, col, dict(
                tag=tag, gpu=gpu, repo=root, cfg=cfg, ckpt=ck, test_data=td,
                krep=krep, wpt=WPT, bs=BS, out=os.path.join(OUT_DIR, 'w61_%s.json' % tag))))
    return out


def cmdline(j):
    env = ('TEST_DATA=%s K_REP=%d WPT=%.2f BS=%d OUT=%s CUDA_VISIBLE_DEVICES=%s'
           % (j['test_data'], j['krep'], j['wpt'], j['bs'], j['out'], j['gpu']))
    return 'cd %s && %s python -u %s %s %s %s %s' % (
        j['repo'], env, W34, j['repo'], j['cfg'], j['ckpt'], j['tag'])


def run_one(j, timeout=3600):
    if not j['ckpt']:
        return j, 1, 'ckpt 未找到（repo=%s）' % j['repo']
    env = dict(os.environ)
    env.update(TEST_DATA=j['test_data'], K_REP=str(j['krep']), WPT=str(j['wpt']),
               BS=str(j['bs']), OUT=j['out'], CUDA_VISIBLE_DEVICES=str(j['gpu']))
    t0 = time.time()
    p = subprocess.run([sys.executable, '-u', W34, j['repo'], j['cfg'], j['ckpt'], j['tag']],
                       cwd=j['repo'], env=env, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, timeout=timeout)
    return j, p.returncode, '%.1fs' % (time.time() - t0)


def _md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def cmd_check():
    """离线校验：8 个 JSON 齐全，且 N / K_REP / WPT / split 与协议一致。"""
    fails = []
    for key in ORDER:
        for col, krep, n, splits in (
                ('probe', 1, N_PROBE, ('train.json', 'cg_train.json')),
                ('test', 3, N_TEST, ('test_trivial.json', 'cg_test_trivial.json'))):
            p = os.path.join(OUT_DIR, 'w61_%s_TT_%s.json' % (key, col))
            if not os.path.exists(p):
                fails.append('缺文件 %s' % p)
                continue
            with open(p, encoding='utf-8') as f:
                d = json.load(f)
            if d.get('N') != n:
                fails.append('%s %s: N=%s（应 %d）' % (key, col, d.get('N'), n))
            if d.get('K_REP') != krep:
                fails.append('%s %s: K_REP=%s（应 %d）' % (key, col, d.get('K_REP'), krep))
            if abs(float(d.get('WPT', -1)) - WPT) > 1e-9:
                fails.append('%s %s: WPT=%s（应 %.2f）' % (key, col, d.get('WPT'), WPT))
            if os.path.basename(str(d.get('split'))) not in splits:
                fails.append('%s %s: split=%r 不在 %s' % (key, col, d.get('split'), splits))
            if 'dir_acc_c1' not in d.get('curve', {}):
                fails.append('%s %s: 缺 curve.dir_acc_c1' % (key, col))
    if fails:
        print('CHECK FAIL (%d 项)' % len(fails))
        for s in fails:
            print('  - ' + s)
        return 1
    print('CHECK PASS：8/8 个产物齐全，N=8281/3096、K_REP=1/3、WPT=0.10 全部一致')
    return 0


def cmd_manifest(recs):
    men = dict(generated_at=time.strftime('%Y-%m-%d %H:%M:%S'), w34=W34, bs=BS, wpt=WPT,
               jobs=recs)
    p = os.path.join(OUT_DIR, 'w61_manifest.json')
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(men, f, ensure_ascii=False, indent=1)
    print('WROTE %s' % p)


def main():
    ap = argparse.ArgumentParser(description='表 4（训练/测试划分）驱动与校验')
    ap.add_argument('--dry-run', action='store_true', help='只打印命令')
    ap.add_argument('--check', action='store_true', help='离线校验已入库的 8 个 JSON')
    ap.add_argument('--backbone', choices=ORDER, help='只跑一个骨干（两条）')
    ap.add_argument('--all', action='store_true', help='跑四个骨干')
    ap.add_argument('--gpu', default='0', help='--backbone 时使用的卡号')
    ap.add_argument('--gpus', default='', help='--all 时按骨干并行的卡号，如 4,5,6,7')
    ap.add_argument('--manifest', action='store_true', help='跑完写 w61_manifest.json')
    args = ap.parse_args()

    if args.check:
        return cmd_check()

    if args.dry_run or (not args.backbone and not args.all):
        for key in ORDER:
            for _, _, j in jobs([key], args.gpu):
                print(cmdline(j))
        return 0

    if args.backbone:
        keys = [args.backbone]
        gpu_list = [args.gpu]
    else:
        keys = ORDER
        gpu_list = [g.strip() for g in args.gpus.split(',') if g.strip()] or ['0']
    if len(gpu_list) < len(keys):
        gpu_list = (gpu_list * len(keys))[:len(keys)]

    os.makedirs(OUT_DIR, exist_ok=True)
    recs, fails = [], []
    tasks = []
    for key, gpu in zip(keys, gpu_list):
        for _, _, j in jobs([key], gpu):
            tasks.append(j)

    with cf.ThreadPoolExecutor(max_workers=max(1, len(keys))) as ex:
        futs = [ex.submit(run_one, j) for j in tasks]
        for fu in cf.as_completed(futs):
            j, rc, info = fu.result()
            print('[%s] rc=%d %s' % (j['tag'], rc, info))
            recs.append(dict(tag=j['tag'], gpu=j['gpu'], repo=j['repo'], cfg=j['cfg'],
                             ckpt=j['ckpt'], ckpt_md5=(_md5(j['ckpt']) if j['ckpt'] and os.path.exists(j['ckpt']) else None),
                             test_data=j['test_data'], krep=j['krep'], wpt=j['wpt'], bs=j['bs'],
                             out=j['out'], rc=rc,
                             out_sha256=(hashlib.sha256(open(j['out'], 'rb').read()).hexdigest()
                                         if os.path.exists(j['out']) else None)))
            if rc != 0:
                fails.append(j['tag'])

    if args.manifest:
        cmd_manifest(recs)
    if fails:
        print('FAILED: %s' % ','.join(fails))
        return 1
    return cmd_check()


if __name__ == '__main__':
    sys.exit(main())
