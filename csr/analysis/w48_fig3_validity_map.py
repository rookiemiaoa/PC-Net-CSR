# -*- coding: utf-8 -*-
"""w48_fig3_b.py —— 重做论文图 3(b)：把训练轨迹的 18 个互异工作点并入第二坐标。

为什么重做
  旧版图 3(b) 只有 5 个收敛工作点，并在图上标注「观测空档 0.51–0.70」。
  审稿意见（建议 #4）指出该空档内没有工作点。我们用三个骨干（CNM / CPL / CCR）
  的逐 epoch 检查点在完全相同的口径下补测了训练轨迹（w45, w47），
  18 个互异工作点沿第二坐标完全分开：Δ<0 者 a ≤ 0.510，Δ>0 者 a ≥ 0.569。
  本脚本把轨迹点画为小空心点 + 骨干联线，收敛点保持大实心点。

数据来源（全部服务器实测）
  (a) rho_geo 与 CSR ΔmIoU：w32 / w36 / 表 19 / 表 27
  (b) 收敛点 a：w34_*.json / w36_cnm.json 的 curve.dir_acc_c1
      轨迹点 a：w45_*.json 的 curve.dir_acc_c1（w47 去重后 18 个互异工作点）

运行：
  python csr/analysis/w48_fig3_validity_map.py

依赖：matplotlib（含中文字体，否则中文标签会渲染成空白方框）。
      Ubuntu 上可 `apt install fonts-wqy-zenhei` 或把中文字体放入 ~/.fonts 后
      `rm -rf ~/.cache/matplotlib`；脚本已按「Microsoft YaHei → SimHei → Noto Sans CJK
      → WenQuanYi Zen Hei → DejaVu Sans」顺序回退。
输出：csr/analysis/figures/fig3_validity_map.png（可用 CSR_FIG_DIR 覆盖目录）
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 中文字体回退链：Linux 服务器通常没有 YaHei/SimHei，需给出开源 CJK 字体兜底。
plt.rcParams['font.sans-serif'] = [
    'Microsoft YaHei', 'SimHei',                  # Windows
    'Noto Sans CJK SC', 'Noto Sans CJK JP',       # Debian/Ubuntu fonts-noto-cjk
    'WenQuanYi Zen Hei', 'WenQuanYi Micro Hei',   # fonts-wqy-zenhei / -microhei
    'Source Han Sans SC', 'PingFang SC',          # macOS / 思源黑体
    'DejaVu Sans',                                # 最终兜底（无 CJK 字形）
]
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 200

HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = os.environ.get('CSR_FIG_DIR') or os.path.join(HERE, 'figures')
os.makedirs(OUTD, exist_ok=True)


def _warn_if_no_cjk():
    """若回退链里一个 CJK 字体都没有，明确提示（避免静默出空白方框）。"""
    try:
        from matplotlib import font_manager as fm
        have = {f.name for f in fm.fontManager.ttflist}
    except Exception:
        return
    if not (have & set(plt.rcParams['font.sans-serif'][:-1])):
        print('[warn] 未找到任何中文字体，图中中文标签将显示为方框。')
        print('       请安装 fonts-noto-cjk / fonts-wqy-zenhei，或设置 '
              'CSR_FIG_LATIN=1 使用英文标签。')


_warn_if_no_cjk()

# ---------------- (a) 第一坐标：几何（与旧版一致） ----------------
GEO = [
    ('CCR', 1.331, -3.18),
    ('QMN', 1.326, -3.53),
    ('CPL', 1.338, -3.76),
    ('PC-Net', 1.347, 1.05),
    ('CNM', 1.725, 3.00),
    ('ANET-CG\n/PC-Net', 3.60, -0.17),
    ('TACoS\n/PC-Net', 11.28, 0.10),
]

# ---------------- (b) 第二坐标：信息 ----------------
# 收敛点（大实心）：标签, a, ΔmIoU, rho_geo
CONV = [
    ('CCR', 0.4467, -3.18, 1.331),
    ('QMN', 0.4717, -3.53, 1.326),
    ('CPL', 0.5096, -3.76, 1.338),
    ('PC-Net', 0.7760, 1.05, 1.346),
    ('CNM', 0.6999, 3.00, 1.725),
]

# 训练轨迹点（小空心）：骨干, a, ΔmIoU（来自 w47 去重后的 18 个互异工作点，
# 此处只列轨迹点，收敛点已在上表）
TRAJ = {
    'CNM': [(0.5691, 2.92), (0.7172, 3.14), (0.7216, 3.26),
            (0.7349, 2.91), (0.7436, 2.74), (0.7628, 2.81), (0.7724, 3.45)],
    'CPL': [(0.4601, -0.84), (0.4672, -3.50), (0.4705, -3.70)],
    'CCR': [(0.4400, -3.26), (0.4500, -3.23), (0.5009, -0.56)],
}

A_HI_NEG = 0.5096   # 所有 Δ<0 的点中最大 a（CPL 收敛点）
A_LO_POS = 0.5691   # 所有 Δ>0 的点中最小 a（CNM 第 2 epoch）

C_POS = '#2e7d32'
C_NEG = '#c0392b'
C_FLAT = '#7f8c8d'
C_TRAJ = {'CNM': '#1565c0', 'CPL': '#8e24aa', 'CCR': '#ef6c00'}

fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.3))


def colr(v):
    return C_POS if v > 0.5 else (C_NEG if v < -0.5 else C_FLAT)


# ---------------- (a) 第一坐标：几何 ----------------
ax = axes[0]
ax.axvspan(1.20, 2.00, color='#e8f5e9', alpha=0.85, zorder=0)
ax.axvspan(3.00, 20.0, color='#fdeaea', alpha=0.85, zorder=0)
ax.axhline(0, color='#666', lw=1.0, zorder=1)
off = {'CCR': (16, 9, 'left'), 'QMN': (16, 0, 'left'), 'CPL': (16, -13, 'left'),
       'PC-Net': (0, 13, 'center'), 'CNM': (0, 13, 'center'),
       'ANET-CG\n/PC-Net': (0, -21, 'center'), 'TACoS\n/PC-Net': (0, 11, 'center')}
for nm, rho, dm in GEO:
    ax.plot([rho], [dm], 'o', ms=11, color=colr(dm), zorder=5,
            markeredgecolor='white', markeredgewidth=1.2)
    dx, dy, ha = off[nm]
    ax.annotate(nm, (rho, dm), textcoords='offset points', xytext=(dx, dy),
                ha=ha, va='center', fontsize=8.6, color='#222')
ax.set_xscale('log')
ax.set_xticks([1.33, 1.73, 3.60, 11.28])
ax.set_xticklabels(['1.33', '1.73', '3.60', '11.28'], fontsize=9)
ax.set_xlim(1.20, 20)
ax.set_ylim(-6.4, 5.4)
ax.set_xlabel(r'$\rho_{geo}$ = 基础框宽中位数 / 目标时长中位数  （对数轴）', fontsize=10.5)
ax.set_ylabel(r'CSR 的 $\Delta$mIoU（pt，相对各自基础模型）', fontsize=10.5, labelpad=2)
ax.set_title(r'(a) 第一坐标（几何）：$\rho_{geo}$ 不足以预测符号', fontsize=11.5, pad=22)
ax.text(1.53, 4.55, '宽度相当\n几何上中心轴可用', ha='center', va='center',
        fontsize=8.6, color='#2e7d32')
ax.text(5.0, 0.9, '框覆盖全视频\n两轴皆无效', ha='center', va='center', fontsize=8.6, color='#a33')
ax.annotate('同一几何区间（1.33–1.35）内\n三位骨干显著退化、一位获益',
            xy=(1.40, -3.6), xytext=(2.75, -4.15), fontsize=8.8, color='#a33',
            va='center', ha='left',
            arrowprops=dict(arrowstyle='->', color='#a33', lw=1.1))
ax.grid(alpha=0.22, linestyle=':')

# ---------------- (b) 第二坐标：信息 ----------------
ax = axes[1]
# 符号区间：由 18 个互异工作点夹逼
ax.axvspan(0.40, A_HI_NEG, color='#fdeaea', alpha=0.85, zorder=0)
ax.axvspan(A_LO_POS, 0.85, color='#e8f5e9', alpha=0.85, zorder=0)
ax.axvspan(A_HI_NEG, A_LO_POS, facecolor='#f3f3f3', alpha=0.95, zorder=0,
           hatch='///', edgecolor='#bbbbbb')
ax.axhline(0, color='#666', lw=1.0, zorder=1)
ax.axvline(0.5, color='#888', lw=1.2, ls='--', zorder=1)

# 训练轨迹：同骨干联线 + 小空心点
for bk, pts in TRAJ.items():
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs, ys, '-', lw=1.0, color=C_TRAJ[bk], alpha=0.55, zorder=3)
    ax.plot(xs, ys, 'o', ms=6.5, mfc='white', mec=C_TRAJ[bk], mew=1.5,
            zorder=6, label='%s 训练轨迹' % bk)

# 收敛点：大实心
for nm, acc, dm, rho in CONV:
    ax.plot([acc], [dm], 'o', ms=11, color=colr(dm), zorder=7,
            markeredgecolor='white', markeredgewidth=1.2)
ax.annotate('CNM\n$\\rho_{geo}$=1.73', (0.6999, 3.00), textcoords='offset points',
            xytext=(0, 13), ha='center', fontsize=8.4, color='#222')
ax.annotate('PC-Net\n$\\rho_{geo}$=1.35', (0.7760, 1.05), textcoords='offset points',
            xytext=(0, 13), ha='center', fontsize=8.4, color='#222')
ax.annotate('CPL、QMN、CCR\n$\\rho_{geo}$ = 1.34 / 1.33 / 1.33',
            xy=(0.476, -3.5), xytext=(0.596, -5.0), fontsize=8.6, color='#a33',
            ha='left', va='center',
            arrowprops=dict(arrowstyle='->', color='#a33', lw=1.1))
# 夹逼区间标注
ax.annotate('', xy=(A_HI_NEG, -6.05), xytext=(A_LO_POS, -6.05),
            arrowprops=dict(arrowstyle='<->', color='#555', lw=1.1))
ax.text((A_HI_NEG + A_LO_POS) / 2, -5.72, '实测夹逼\n0.510–0.569', ha='center', va='bottom',
        fontsize=7.8, color='#444')
ax.set_xlim(0.40, 0.85)
ax.set_ylim(-6.4, 5.4)
ax.set_xlabel('曲线方向准确率（确有偏移的子群上，估计方向与真值方向一致的比例）', fontsize=10.5)
ax.set_ylabel(r'CSR 的 $\Delta$mIoU（pt）', fontsize=10.5, labelpad=2)
ax.set_title('(b) 第二坐标（信息）：18 个互异工作点按方向准确率完全分开', fontsize=11.5, pad=22)
ax.text(0.452, -5.5, '曲线不定位目标\n精化有害', ha='center', va='bottom', fontsize=8.6, color='#a33')
ax.text(0.80, -5.5, '曲线可用\n增益为正', ha='center', va='bottom', fontsize=8.6, color='#2e7d32')
ax.text(0.493, 3.9, '随机水平 0.5', fontsize=8.4, color='#555', rotation=90,
        ha='right', va='center')
ax.legend(loc='upper left', fontsize=7.6, framealpha=0.92, ncol=1,
          bbox_to_anchor=(0.005, 0.995))
ax.grid(alpha=0.22, linestyle=':')

fig.tight_layout()
# 论文图 3 —— 文件名与论文/README 引用一致（历史脚本曾误写为 fig2_*）
fig.savefig(os.path.join(OUTD, 'fig3_validity_map.png'), bbox_inches='tight', facecolor='white')
plt.close(fig)

with open(os.path.join(OUTD, 'fig3_stats.json'), 'w') as f:
    json.dump({'geo': GEO, 'conv': CONV, 'traj': TRAJ,
               'a_hi_neg': A_HI_NEG, 'a_lo_pos': A_LO_POS}, f, ensure_ascii=False, indent=1)
print('WROTE %s/fig3_validity_map.png' % OUTD)
print('  轨迹点总数 = %d（+ %d 个收敛点 = %d 个互异工作点）' % (
    sum(len(v) for v in TRAJ.values()), len(CONV),
    sum(len(v) for v in TRAJ.values()) + len(CONV)))
