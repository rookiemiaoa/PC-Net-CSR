# -*- coding: utf-8 -*-
"""compare_cnm_trace.py —— 证明 CNM 重训与 2026-09 那次是**同一条训练轨迹**

为什么需要这个
  CNM 的最优 checkpoint 已经丢失，重训得到的是新权重。若新旧两次训练的 loss 轨迹
  不同，那么即使最终 mIoU 接近，也不能说"复现了论文那一行"——只能说"训了个差不多的
  模型"。本脚本逐 epoch、逐 batch 比对 6 个 loss 分量，把这件事变成可判定的。

判据
  两份日志里 (epoch, batch) 相同的记录，其
      final_loss / nll_loss / neg_nll_loss / ivc_loss / easy_neg_loss / hard_neg_loss
  必须全部相同（日志只打印 4 位小数，故绝对容差 5e-5）。``seconds/batch`` 是硬件
  速度，**不参与**比对。

用法
  python scripts/compare_cnm_trace.py <旧日志> <新日志> [--tol 5e-5]
  python scripts/compare_cnm_trace.py            # 用默认路径（仓库内 evidence/ 与 results/）

退出码：0 = 轨迹一致；1 = 有分歧或无重叠。
"""
import argparse
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BATCH_RE = re.compile(
    r'Epoch\s+(\d+),\s*Batch\s+(\d+),\s*lr\s*=\s*([\d.eE+-]+),\s*'
    r'final_loss\s*=\s*([\d.]+),\s*nll_loss\s*=\s*([\d.]+),\s*'
    r'neg_nll_loss\s*=\s*([\d.]+),\s*ivc_loss\s*=\s*([\d.]+),\s*'
    r'easy_neg_loss\s*=\s*([\d.]+),\s*hard_neg_loss\s*=\s*([\d.]+)')
EPOCH_RE = re.compile(r'\|\s*mIoU\s+([\d.]+)\s*\|.*?IoU@0\.7\s+([\d.]+)')

FIELDS = ('final_loss', 'nll_loss', 'neg_nll_loss', 'ivc_loss',
          'easy_neg_loss', 'hard_neg_loss')


def parse(path):
    """返回 ({(epoch,batch): [6个loss]}, [每 epoch 的 mIoU])"""
    batches, mious = {}, []
    order = []
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            m = BATCH_RE.search(line)
            if m:
                e, b = int(m.group(1)), int(m.group(2))
                vals = [float(m.group(i)) for i in range(4, 10)]
                batches[(e, b)] = vals
                order.append((e, b))
                continue
            m = EPOCH_RE.search(line)
            if m:
                mious.append(float(m.group(1)))
    return batches, mious, order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('old', nargs='?',
                    default=os.path.join(ROOT, 'results', 'cnm_training',
                                         'cnm_train_2026-09.log'),
                    help='2026-09 那次训练的日志（轨迹基线）')
    ap.add_argument('new', nargs='?',
                    default=os.path.join(ROOT, 'results', 'cnm_training',
                                         'cg_main_2026-10-18.log'),
                    help='本次重训的日志')
    ap.add_argument('--tol', type=float, default=5e-5,
                    help='loss 容差（日志 4 位小数，理论精度 5e-5）')
    a = ap.parse_args()

    for p in (a.old, a.new):
        if not os.path.exists(p):
            print('[FAIL] 缺日志：%s' % p)
            return 1

    bo, mo, _ = parse(a.old)
    bn, mn, order = parse(a.new)
    print('旧（2026-09）：%d 条 batch 记录，%d 个 epoch 的 mIoU' % (len(bo), len(mo)))
    print('新（本次）  ：%d 条 batch 记录，%d 个 epoch 的 mIoU' % (len(bn), len(mn)))
    print('=' * 74)

    common = [k for k in order if k in bo and k in bn]
    if not common:
        print('CHECK FAIL：两份日志没有共同的 (epoch, batch) 记录，无法比对')
        return 1

    bad = []
    maxd = 0.0
    for k in common:
        for i, name in enumerate(FIELDS):
            d = abs(bo[k][i] - bn[k][i])
            maxd = max(maxd, d)
            if d > a.tol:
                bad.append('E%d/B%d %s %.6f≠%.6f (Δ%.2e)'
                           % (k[0], k[1], name, bo[k][i], bn[k][i], d))

    print('共同记录 %d 条 × %d 个 loss 分量 = %d 次比对'
          % (len(common), len(FIELDS), len(common) * len(FIELDS)))
    print('最大绝对差 %.3e（容差 %.0e）' % (maxd, a.tol))

    # 逐 epoch 的 mIoU 也对一遍（两份日志长度可能不同，只比共同前缀）
    n_ep = min(len(mo), len(mn))
    if n_ep:
        dmax = max(abs(mo[i] - mn[i]) for i in range(n_ep))
        print('前 %d 个 epoch 的 mIoU 最大绝对差 %.4f' % (n_ep, dmax))
        if dmax > 1e-4:
            bad.append('mIoU 轨迹不同：最大 Δ %.4f' % dmax)

    print('-' * 74)
    if bad:
        print('CHECK FAIL（%d 处分歧）' % len(bad))
        for s in bad[:10]:
            print('  - ' + s)
        return 1
    print('CHECK PASS：重训轨迹与 2026-09 那次逐位一致')
    print('⇒ 丢失的只是权重文件，不是训练配方；同 seed 同配置可重放同一条轨迹。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
