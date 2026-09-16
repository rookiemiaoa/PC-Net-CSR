# -*- coding: utf-8 -*-
"""anet_sens.py —— ActivityNet-CG 上的 CSR 参数敏感性扫描

对 csr/eval/eval_csr.py（历史名 diag40_gen.py）施加不同超参（其余固定），
每次 K_REP=1，解析 ΔALL 与长短分桶，
用于证明：ANET 上不存在能带来正收益的参数点 ⇒ 失效是结构性的而非调参问题。
"""
import os
import re
import subprocess
import sys
import time

# ---- 路径解析：默认自动定位本仓库根目录（含 models/pcnet.py），可用 PCNET_ROOT 覆盖 ----
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = _HERE
while _REPO != os.path.dirname(_REPO) and not os.path.exists(os.path.join(_REPO, 'models', 'pcnet.py')):
    _REPO = os.path.dirname(_REPO)
ROOT = os.environ.get('PCNET_ROOT') or os.environ.get('LM_ROOT') or _REPO
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from csr.paths import OUTPUT_DIR           # noqa: E402

CFG = 'config/activitynet/main_train_test.json'
CKPT = os.environ.get('ANET_CKPT', 'checkpoints/ActivityNet/8_2025-05-14_14-07-12/model-best.pt')
OUT = os.environ.get('ANET_SENS_OUT', os.path.join(OUTPUT_DIR, 'anet_sens'))
os.makedirs(OUT, exist_ok=True)

DEFAULT = dict(SMOOTH='7', BETA='0.20', RHO='0.10', ALPHA='1.00', DMIN='0.01')


def cfg(name, **kw):
    e = dict(DEFAULT)
    e.update({k: str(v) for k, v in kw.items()})
    return name, e


GRID = [
    cfg('s_default'),
    cfg('s_sm3', SMOOTH=3),
    cfg('s_sm15', SMOOTH=15),
    cfg('s_beta010', BETA=0.10),
    cfg('s_beta030', BETA=0.30),
    cfg('s_rho000', RHO=0.00),
    cfg('s_rho025', RHO=0.25),
    cfg('s_alpha050', ALPHA=0.50),
    cfg('s_dmin000', DMIN=0.00),
    cfg('s_dmin005', DMIN=0.05),
]


def parse(txt):
    out = {}
    m = re.search(r'^CSR\s+([-+][\d.]+)\s+([-+][\d.]+)\s+([-+][\d.]+)', txt, re.M)
    if m:
        out['dALL'] = tuple(float(x) for x in m.groups())      # (dR1@.5, dR1@.7, dmIoU)
    m = re.search(r'^CSR-base\s+短\s+\d+\s+([-+][\d.]+)\s+([-+][\d.]+)\s+([-+][\d.]+)', txt, re.M)
    if m:
        out['dshort'] = tuple(float(x) for x in m.groups())
    m = re.search(r'^CSR-base\s+长\s+\d+\s+([-+][\d.]+)\s+([-+][\d.]+)\s+([-+][\d.]+)', txt, re.M)
    if m:
        out['dlong'] = tuple(float(x) for x in m.groups())
    m = re.search(r'^base\s+全\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', txt, re.M)
    if m:
        out['base'] = tuple(float(x) for x in m.groups())
    return out


only = sys.argv[1] if len(sys.argv) > 1 else None
print('%-14s %8s %10s %10s %10s %10s' % (
    'config', 'sec', 'd@.5', 'd@.7', 'dmIoU', 'note'), flush=True)
rows = []
for name, env in GRID:
    if only and only not in name:
        continue
    log = os.path.join(OUT, name + '.log')
    e = dict(os.environ)
    e.update(env)
    e['K_REP'] = '1'
    e['BS'] = '16'
    t0 = time.time()
    with open(log, 'w') as f:
        subprocess.run([sys.executable, '-u',
                        os.path.join('csr', 'eval', 'eval_csr.py'), CFG, CKPT],
                       cwd=ROOT, env=e, stdout=f, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    txt = open(log, errors='replace').read()
    r = parse(txt)
    rows.append((name, dt, r))
    if 'dALL' in r:
        d = r['dALL']
        print('%-14s %8.0f %+10.3f %+10.3f %+10.3f  ok' % (name, dt, d[0], d[1], d[2]), flush=True)
    else:
        print('%-14s %8.0f %10s  PARSE_FAIL' % (name, dt, '-'), flush=True)

print('\n=== summary (K_REP=1, ANET-CG Test-Trivial) ===', flush=True)
print('%-14s %10s %10s %10s | %10s %10s %10s | %10s %10s %10s' % (
    'config', 'dALL@.5', 'dALL@.7', 'dALLmIoU', 'dS@.5', 'dS@.7', 'dSmIoU', 'dL@.5', 'dL@.7', 'dLmIoU'),
    flush=True)
for name, dt, r in rows:
    def g(k):
        return r.get(k, (float('nan'),) * 3)
    a, s, l = g('dALL'), g('dshort'), g('dlong')
    print('%-14s %+10.3f %+10.3f %+10.3f | %+10.3f %+10.3f %+10.3f | %+10.3f %+10.3f %+10.3f' % (
        name, a[0], a[1], a[2], s[0], s[1], s[2], l[0], l[1], l[2]), flush=True)
print('DONE', flush=True)
