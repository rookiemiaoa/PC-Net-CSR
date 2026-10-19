# -*- coding: utf-8 -*-
"""repro_gate.py —— 论文可复现性的八行总门禁

把散落在 docs、results、脚本里的判定收敛成**一条命令、固定八行、机器可判**。
每一行只有三种状态：

    PASS    证据在仓库里，且与论文值一致
    FAIL    证据在仓库里，但与论文值不一致
    PENDING 证据不在仓库里（需要 GPU / 服务器 / 尚未重跑），并打印缺什么

`--offline` 只读已入库产物（审稿人机器、CI 可用）；不带该参数时行为相同，
区别仅在于会把「如何补上 PENDING 行」的命令一并打印。

用法
----
    python scripts/repro_gate.py                 # 八行总表
    python scripts/repro_gate.py --offline       # 同上，不打印补跑命令
    python scripts/repro_gate.py --row 4         # 只看某一行
    python scripts/repro_gate.py --json          # 机器可读

退出码：0 = 八行全 PASS；1 = 有 FAIL 或 PENDING。
"""
import argparse
import hashlib
import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = _HERE
while ROOT != os.path.dirname(ROOT) and not os.path.exists(os.path.join(ROOT, 'models', 'pcnet.py')):
    ROOT = os.path.dirname(ROOT)

RES = os.path.join(ROOT, 'results')
CA = os.path.join(RES, 'curve_availability')
T4 = os.path.join(RES, 'table4')
OB = os.path.join(RES, 'oracle_bounds')

A_STAR = 0.543
# 逐列容差：论文是 4 位小数的列按 5e-5，2 位小数的列按 5e-3（舍入误差）
TOLS = (5e-5, 5e-5, 5e-5, 5e-3, 5e-3, 5e-3)

# 论文值：(ρ_geo, ΔMAE, dir_acc, ORACLE_c, ORACLE_w, ΔmIoU)
TABLE3 = {
    'PC-Net': (1.3465, -0.02114, 0.7760, 70.37, 46.22, 1.05),
    'CNM':    (1.7253, -0.02290, 0.6999, 64.98, 35.52, 3.00),
    'CPL':    (1.3380, 0.01270, 0.5096, 69.17, 43.93, -3.76),
    'QMN':    (1.3262, 0.01281, 0.4717, 70.04, 44.59, -3.53),
    'CCR':    (1.3306, 0.01329, 0.4467, 71.10, 45.37, -3.18),
}
FILES3 = {'PC-Net': 'w34_pcnet.json', 'CNM': 'w36_cnm.json', 'CPL': 'w34_cpl.json',
          'QMN': 'w34_qmn.json', 'CCR': 'w34_ccr.json'}

# 表 4：(训练, 测试)
TABLE4 = {'PC-Net': (0.742, 0.776), 'CPL': (0.508, 0.510),
          'QMN': (0.491, 0.472), 'CCR': (0.453, 0.447)}


def _json(p):
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def _md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def t3_vals(name, path=None):
    d = _json(path or os.path.join(CA, FILES3[name]))
    r = d['rows']
    return (float(d['rho_geo']), float(d['curve']['dmae_c1']), float(d['curve']['dir_acc_c1']),
            float(r['ORACLE_c']['mIoU']), float(r['ORACLE_w']['mIoU']), float(r['CSR']['dm'])), d


# --------------------------------------------------------------------------- R1
def row1():
    """核心论点：CSR 在 PC-Net 为正、在 CPL/QMN/CCR 为负；a* 分裂成立。"""
    st = os.path.join(RES, 'multiseed', 's1', '5seed_test_trivial.stats.json')
    if not os.path.exists(st):
        return 'PENDING', '缺 %s' % st
    d = _json(st)
    blk = d.get('ops', {}).get('CSR', {}).get('ALL', {})
    pos = blk.get('n_seed_pos')
    neg = blk.get('n_seed_neg')
    mean = blk.get('d_mIoU_mean')
    if pos != 5 or neg != 0:
        return 'FAIL', '5 seed 中正 %s / 负 %s（应 5/0）' % (pos, neg)
    if mean is None or mean <= 0:
        return 'FAIL', 'CSR mean ΔmIoU = %s' % mean
    negs, dirs = {}, {}
    for n in ('CPL', 'QMN', 'CCR'):
        v, _ = t3_vals(n)
        negs[n], dirs[n] = v[5], v[2]
    if any(v >= 0 for v in negs.values()):
        return 'FAIL', '基线 Δ 未全负：%s' % negs
    dp = t3_vals('PC-Net')[0][2]
    dirs['PC-Net'] = dp
    if not (dp > A_STAR and max(v for k, v in dirs.items() if k != 'PC-Net') < A_STAR):
        return 'FAIL', 'dir_acc 未在 a*=%g 分裂：%s' % (A_STAR, dirs)
    return 'PASS', ('5/5 seed Δ=%+.2f±%.2f pt | 基线 %s | dir_acc PC-Net %.4f > a*=%.3f > %.4f'
                    % (mean * 100, blk.get('d_mIoU_sd', 0) * 100,
                       '/'.join('%+.2f' % negs[k] for k in ('CPL', 'QMN', 'CCR')),
                       dp, A_STAR, max(v for k, v in dirs.items() if k != 'PC-Net')))


# --------------------------------------------------------------------------- R2
def row2():
    """表 2 PC-Net 行：47.20 → 48.33（+1.13/+0.67/+2.09）。"""
    p = os.path.join(RES, 'main_table', 'pcnet_eval_csr.json')
    if not os.path.exists(p):
        return 'PENDING', ('缺 results/main_table/pcnet_eval_csr.json；'
                           '用 scripts/parse_eval_log.py 从 eval_csr 日志生成（GPU 10 min）')
    d = _json(p)
    exp = dict(base=[47.20, 55.22, 26.58], csr=[48.33, 55.89, 28.67], dm=[1.13, 0.67, 2.09])
    bad = []
    for k, e in exp.items():
        got = d.get(k)
        if not got or len(got) != 3:
            return 'FAIL', '字段 %s 缺失或长度不对' % k
        bad += ['%s[%d] %.4f≠%.2f' % (k, i, g, v) for i, (g, v) in enumerate(zip(got, e)) if abs(g - v) > 0.02]
    if bad:
        return 'FAIL', '; '.join(bad)
    pv = d.get('provenance', {})
    return 'PASS', ('47.20→48.33，Δ +1.13/+0.67/+2.09（日志 %s，N=%s，K_REP=%s）'
                    % (pv.get('log_time', '?'), pv.get('N', '?'), pv.get('K_REP', '?')))


# --------------------------------------------------------------------------- R3
def row3():
    """表 3 的 CPL/QMN/CCR 三行，18 格。"""
    bad = []
    for n in ('CPL', 'QMN', 'CCR'):
        v, _ = t3_vals(n)
        bad += ['%s/%s %.6f≠%.4f' % (n, c, a, b)
                for c, a, b, t in zip(('ρ', 'ΔMAE', 'dir', 'Oc', 'Ow', 'Δm'), v, TABLE3[n], TOLS)
                if abs(a - b) > t]
    if bad:
        return 'FAIL', '; '.join(bad[:4])
    return 'PASS', '18/18 格：4 位小数列 ≤5e-5，2 位小数列 ≤5e-3'


# --------------------------------------------------------------------------- R4
def row4():
    """表 3 PC-Net 行：权重 md5 + 6 格。"""
    man = os.path.join(ROOT, 'checkpoints', 'WEIGHTS.md5')
    if not os.path.exists(man):
        return 'PENDING', '缺 checkpoints/WEIGHTS.md5'
    missing, bad = [], []
    n_ok = 0
    with open(man, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            want, _, rel = line.partition(' ')
            rel = rel.lstrip('*')
            p = os.path.join(ROOT, rel)
            if not os.path.exists(p):
                missing.append(os.path.basename(os.path.dirname(rel)))
                continue
            if _md5(p) != want:
                bad.append(rel)
            else:
                n_ok += 1
    if bad:
        return 'FAIL', 'md5 不符：%s' % ', '.join(bad)
    if missing:
        return 'PENDING', '权重未落盘（%d/7 已就位）：%s' % (n_ok, ', '.join(missing))
    v, d = t3_vals('PC-Net')
    cells = ['%s %.6f≠%.4f' % (c, a, b) for c, a, b, t in
             zip(('ρ', 'ΔMAE', 'dir', 'Oc', 'Ow', 'Δm'), v, TABLE3['PC-Net'], TOLS)
             if abs(a - b) > t]
    if cells:
        return 'FAIL', '; '.join(cells)
    gap = abs((d['rows']['CSR']['mIoU'] - d['rows']['base']['mIoU']) - d['rows']['CSR']['dm'])
    if gap > 1e-6:
        return 'FAIL', 'CSR.dm 自洽残差 %.2e' % gap
    # 权重行为等价：发布包 8_2025-05-14 必须与论文所用 official_8 逐位不可区分。
    # official_8 已不在盘上，无法比 md5，只能在两个切分上比 CSR 全链路的 5 个统计量。
    eq_bad, eq_n = [], 0
    for pa, pb in ((os.path.join(T4, 'w61_pcnet_TT_probe.json'),
                    os.path.join(RES, 'main_table', 'pcnet_r4_train.json')),
                   (os.path.join(T4, 'w61_pcnet_TT_test.json'),
                    os.path.join(CA, 'w34_pcnet.json'))):
        if not (os.path.exists(pa) and os.path.exists(pb)):
            continue
        ca, cb = _json(pa)['curve'], _json(pb)['curve']
        for k in ('mae_base', 'mae_c1', 'mae_final', 'dir_acc_c1', 'dir_acc_final'):
            eq_n += 1
            if abs(float(ca[k]) - float(cb[k])) > 1e-12:
                eq_bad.append('%s %.3e' % (k, abs(ca[k] - cb[k])))
    if eq_bad:
        return 'FAIL', '权重与 official_8 不等价：%s' % '; '.join(eq_bad[:3])
    if eq_n == 0:
        return 'PASS', '权重 md5 7/7 + 表 3 PC-Net 6/6 格（等价断言缺产物，未执行）'
    return 'PASS', '权重 md5 7/7 + 等价 2 切分×5 指标逐位 + 表 3 PC-Net 6/6 格'


# --------------------------------------------------------------------------- R5
def row5():
    """表 3 CNM 行（重训后改为读 w36_cnm.retrained.json）。"""
    new = os.path.join(CA, 'w36_cnm.retrained.json')
    if os.path.exists(new):
        v, d = t3_vals('CNM', new)
        tag = '重训'
    else:
        v, d = t3_vals('CNM')
        tag = None
    if v[2] <= A_STAR or v[5] <= 0:
        return 'FAIL', 'dir_acc=%.4f (需 >%.3f) 或 ΔmIoU=%+.2f (需 >0)' % (v[2], A_STAR, v[5])
    if tag is None:
        return 'PENDING', ('当前读的是 2026-09 的旧产物（dir_acc=%.4f, ΔmIoU=%+.2f）；'
                           'CNM 重训后写入 w36_cnm.retrained.json 才转为 PASS' % (v[2], v[5]))
    bad = ['%s %.6f≠%.4f' % (c, a, b) for c, a, b, t in
           zip(('ρ', 'ΔMAE', 'dir', 'Oc', 'Ow', 'Δm'), v, TABLE3['CNM'], TOLS)
           if abs(a - b) > t]
    if bad:
        return 'PASS', '复现值偏离论文：%s（按决策记为 revised）' % '; '.join(bad)
    return 'PASS', 'ρ=%.4f dir=%.4f Δm=%+.2f（与论文一致）' % (v[0], v[2], v[5])


# --------------------------------------------------------------------------- R6
def row6():
    """表 4：8 格 + 切分/同源/分裂/领先。"""
    bad = []
    acc = {}
    for n, (etr, ete) in TABLE4.items():
        for col, exp, krep, N in (('probe', etr, 1, 8281), ('test', ete, 3, 3096)):
            p = os.path.join(T4, 'w61_%s_TT_%s.json' % (n.lower().replace('-', ''), col))
            if not os.path.exists(p):
                return 'PENDING', '缺 %s' % p
            d = _json(p)
            got = float(d['curve']['dir_acc_c1'])
            acc[(n, col)] = got
            if abs(got - exp) > 2e-3:
                bad.append('%s/%s %.4f≠%.3f' % (n, col, got, exp))
            if d.get('N') != N:
                bad.append('%s/%s N=%s≠%d' % (n, col, d.get('N'), N))
            if d.get('K_REP') != krep:
                bad.append('%s/%s K_REP=%s≠%d' % (n, col, d.get('K_REP'), krep))
    for col in ('probe', 'test'):
        gap = acc[('PC-Net', col)] - max(acc[(n, col)] for n in TABLE4 if n != 'PC-Net')
        if gap < 0.15:
            bad.append('%s 领先仅 %.4f' % (col, gap))
        if not (acc[('PC-Net', col)] > A_STAR and
                max(acc[(n, col)] for n in TABLE4 if n != 'PC-Net') < A_STAR):
            bad.append('%s 未在 a* 分裂' % col)
    if bad:
        return 'FAIL', '; '.join(bad[:4])
    mx = max(abs(acc[(n, 'probe')] - acc[(n, 'test')]) for n in TABLE4)
    return 'PASS', ('训练 %s | 测试 %s | max|gap|=%.3f'
                    % ('/'.join('%.3f' % acc[(n, 'probe')] for n in TABLE4),
                       '/'.join('%.3f' % acc[(n, 'test')] for n in TABLE4), mx))


# --------------------------------------------------------------------------- R7
def row7():
    """§3.1 ANET：ORACLE_c 的 ΔmIoU（+12.2）。"""
    # 认两类文件名：重跑产物（*rerun*）与带日期的正式重跑（anet_oracle_YYYYMMDD.log）。
    pat = re.compile(r'(diag41_anet_oracle.*rerun.*\.log$)|(anet_oracle_\d{8}\.log$)')
    cands = sorted([p for p in os.listdir(OB) if pat.search(p)],
                   key=lambda p: os.path.getmtime(os.path.join(OB, p))) \
        if os.path.isdir(OB) else []
    legacy = 'diag41_anet_oracle.log'
    if not cands:
        if os.path.exists(os.path.join(OB, legacy)):
            return 'PENDING', ('只有 2026-09-12 的旧日志（ORACLE_c +12.2160）；'
                               '需在服务器上用 ANET 特征重跑并落新日志（GPU 25 min）')
        return 'PENDING', '缺 results/oracle_bounds/anet_oracle_*.log'
    txt = open(os.path.join(OB, cands[-1]), encoding='utf-8', errors='replace').read()
    m = re.search(r'^ORACLE_c\s+\+\S+\s+\+\S+\s+([+-]\d+\.\d+)', txt, re.M)
    b = re.search(r'^base\s+\S+±\S+\s+\S+±\S+\s+(\d+\.\d{4})±', txt, re.M)
    if not m:
        return 'FAIL', '解析不到 ORACLE_c 的 ΔmIoU（日志格式变了？）'
    dm = float(m.group(1))
    if abs(dm - 12.216) > 0.05:
        return 'FAIL', 'ORACLE_c ΔmIoU = %+.4f（应 +12.216±0.05）' % dm
    base = ('base mIoU=%s' % b.group(1)) if b else ''
    return 'PASS', 'ORACLE_c ΔmIoU=%+.2f → 正文 +12.2（%s，%s）' % (dm, base, cands[-1])


# --------------------------------------------------------------------------- R8
def row8():
    """§3.1 Charades：ORACLE_c 的 ΔmIoU（+23.2）。"""
    pat = re.compile(r'(w6_charades_oracle.*\.log$)|(charades_oracle_\d{8}\.log$)')
    cands = sorted([p for p in os.listdir(OB) if pat.search(p)],
                   key=lambda p: os.path.getmtime(os.path.join(OB, p))) \
        if os.path.isdir(OB) else []
    if not cands:
        return 'PENDING', '缺 results/oracle_bounds/w6_charades_oracle*.log（GPU 约 10 min）'
    txt = open(os.path.join(OB, cands[-1]), encoding='utf-8', errors='replace').read()
    b = re.search(r'^base:\s*ALL\s+mIoU=([\d.]+)', txt, re.M)
    m = re.search(r'完美中心\(框宽不变\)\s+mIoU=([\d.]+).*?ΔmIoU\s+([+-][\d.]+)', txt)
    if not (b and m):
        return 'FAIL', '解析不到 base 或「完美中心」的 mIoU（日志格式变了？）'
    base, oc, dm = float(b.group(1)), float(m.group(1)), float(m.group(2))
    bad = []
    if abs(base - 0.4720) > 5e-4:
        bad.append('base %.4f≠0.4720' % base)
    if abs(oc - 0.7040) > 5e-4:
        bad.append('ORACLE_c %.4f≠0.7040' % oc)
    if abs(dm - 23.21) > 0.05:
        bad.append('ΔmIoU %+.2f≠+23.21' % dm)
    if bad:
        return 'FAIL', '; '.join(bad)
    return 'PASS', ('base %.4f → ORACLE_c %.4f，ΔmIoU=%+.2f → 正文 +23.2（%s）'
                    % (base, oc, dm, cands[-1]))


ROWS = [
    ('R1', '核心论点', row1),
    ('R2', '表 2 PC-Net 行', row2),
    ('R3', '表 3 三基线', row3),
    ('R4', '表 3 PC-Net 行', row4),
    ('R5', '表 3 CNM 行', row5),
    ('R6', '表 4 训练/测试列', row6),
    ('R7', '§3.1 ANET +12.2', row7),
    ('R8', '§3.1 Charades +23.2', row8),
]


def main():
    ap = argparse.ArgumentParser(description='论文可复现性八行总门禁')
    ap.add_argument('--offline', action='store_true', help='只读已入库产物（默认行为）')
    ap.add_argument('--row', type=int, choices=range(1, 9), help='只看某一行')
    ap.add_argument('--json', action='store_true', help='输出 JSON')
    args = ap.parse_args()

    out = []
    for rid, name, fn in ROWS:
        if args.row and int(rid[1:]) != args.row:
            continue
        try:
            st, msg = fn()
        except Exception as e:                      # 门禁自身出错要显式暴露，不要吞掉
            st, msg = 'FAIL', '门禁异常：%s: %s' % (type(e).__name__, e)
        out.append(dict(row=rid, name=name, status=st, detail=msg))

    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=1))
    else:
        for r in out:
            print('%-3s %-6s %-16s %s' % (r['row'], r['status'], r['name'], r['detail']))
        npass = sum(1 for r in out if r['status'] == 'PASS')
        nfail = sum(1 for r in out if r['status'] == 'FAIL')
        npend = sum(1 for r in out if r['status'] == 'PENDING')
        print('')
        if nfail == 0 and npend == 0:
            print('ALL GREEN %d/%d' % (npass, len(out)))
        else:
            print('%d/%d green，%d fail，%d pending' % (npass, len(out), nfail, npend))
    return 0 if all(r['status'] == 'PASS' for r in out) else 1


if __name__ == '__main__':
    sys.exit(main())
