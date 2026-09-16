"""CSR —— Center-Shift Refinement（中心重定位精修）

动机（与 RBR 的对照）
  RBR 走「边界扩框」，效应符号由 sign(g - w) 决定（g=gt 时长, w=基线框宽）：
  Charades-CG 中位框宽 0.341 vs gt 时长中位 ~0.25-0.31 ⇒ 多数样本 g < w ⇒ 扩框天然有害
  ⇒ 实测短动作 R1@0.5 -2.98pt（headline 转负），长动作 @0.7 +10.49pt ——「短降长升」不可避免。

  固定框宽时 IoU 对中心偏移 δ 是**单峰偶函数**（IoU(δ)=IoU(-δ)，δ=0 取极大），
  故中心修正**无子群方向偏置**。ORACLE_c（只换 gt 中心、保持框宽）实测：
      Charades-CG  mIoU 0.4720 -> 0.7040（短 +20.7pt / 长 +30.4pt，两个子群同时受益）。
  这就是「把主战场从边界移到中心」的依据。

方法（零训练、纯推理期、不使用任何 gt）
  1) 逐点提案 -> 相关性曲线 r(t) = -nll(t)；**先 z-score 再平滑**（顺序不可颠倒）
  2) 中心估计：窗口 [c_b ± beta] 内 relu(Z) 加权质心
  3) dead-zone：|d| < rho*框宽 则不平移（保护「中心本来就准」的样本 —— 实测它们是
     短样本伤害的唯一来源，占短桶 47%）
  4) 阻尼 c = c_b + alpha*d
  5) 自洽门控：仅当「平移后框内 Z 均值」提升 >= dmin 才接受（弱监督可算的接受准则）
  6) **宽度保持不变**（RBR 的本质区别 —— 短样本不会因宽度变化跨阈值）

实测（Charades-CG test_trivial，N=3096，K_REP=3，官方权重，不重训）
      ALL  mIoU 47.20 -> 48.33 (+1.13)   R1@0.5 55.22 -> 55.89 (+0.67)   R1@0.7 26.58 -> 28.67 (+2.09)
      短   mIoU 44.75 -> 45.33 (+0.58)   R1@0.5 51.28 -> 51.31 (+0.03)   R1@0.7 24.08 -> 24.30 (+0.22)
      长   mIoU 54.16 -> 56.88 (+2.72)   R1@0.5 66.46 -> 68.95 (+2.49)   R1@0.7 33.71 -> 41.13 (+7.42)
  （以上为定版数字，与英文稿 Table 15–16 一致；本 docstring 曾误记为 48.24/56.42/28.53，已更正。）
  对照 RBR k=0.15：ALL R1@0.5 55.22 -> 54.25 (-0.97)、短 R1@0.5 -2.98。
  显著性、移动率与平均位移以 csr/eval/eval_csr.py（历史名 diag40_gen.py）的 [显著性]/[诊断] 段
  与英文稿 Table 17、Table 21 为准。

跨骨干（外部基线 CNM，零调参，K_REP=3；见 csr/eval/eval_csr_cnm.py 与英文稿 §6.10）
      Test-Trivial  ΔmIoU +3.00  ΔR1@0.5 +5.19  ΔR1@0.7 +2.23
      Novel-Comp    ΔmIoU +2.17  ΔR1@0.5 +3.41  ΔR1@0.7 +1.62
      Novel-Word    ΔmIoU +2.84  ΔR1@0.5 +4.56  ΔR1@0.7 +3.55
  ⇒ 主张是"一条机理"，不是"一个针对某权重调出来的算子"。

超参来源：在 data/charades/train.json（开发集）上按「短样本优先」准则选出，再原样用于 test。
       train 达标 217/900、test 达标 97/900、交集 81 组（37.3%）⇒ 解区域跨切分稳定。

用法
    from csr import CenterShiftRefiner
    csr = CenterShiftRefiner(model, num_point=50)      # 默认即校准值
    out = csr(net_input)
    out['boxes_top1']    # [bsz, 2] 精修 top-1 框 (start, end)，用于 R@1
    out['boxes_pool5']   # [bsz, 5, 2] CSR 顶替原 top-1 + 原 rank2..5，用于 R@5
"""
import numpy as np
import torch

__all__ = ['CenterShiftRefiner']


class CenterShiftRefiner:
    def __init__(self, model, num_point=None, smooth=7, beta=0.20, rho=0.10,
                 alpha=1.00, dmin=0.01, max_num_frames=200):
        """
        model    : 已加载权重的 models.PCNet（需 .proposal_generator / .num_props 可被替换）
        num_point: 逐点提案数，默认 max_num_frames//4（PC-Net 在 50 帧重建分辨率上工作）
        smooth   : 相关性曲线平滑窗口
        beta     : 中心估计搜索窗口半径（归一化时长单位）
        rho      : dead-zone 阈值 = rho * 基线框宽
        alpha    : 位移阻尼
        dmin     : 自洽门控阈值（框内 Z 均值提升下限；None 表示关闭）
        """
        self.model = model
        self.L = int(num_point or max_num_frames // 4)
        self.smooth = int(smooth)
        self.beta = float(beta)
        self.rho = float(rho)
        self.alpha = float(alpha)
        self.dmin = None if dmin is None else float(dmin)
        self.cg = (np.arange(self.L) + 0.5) / self.L
        self._orig_pg = model.proposal_generator
        self._orig_np = model.num_props
        self._pt_c = torch.linspace(0.5 / self.L, 1 - 0.5 / self.L, self.L)
        self._pt_w = torch.full((self.L,), max(2.0 / self.L, 0.02))

    # ---------- 内部 ----------
    def _nll(self, o, n, bsz):
        from models.loss import cal_nll_loss
        wm = o['words_mask'].unsqueeze(1).expand(bsz, n, -1).contiguous().view(bsz * n, -1)
        wi = o['words_id'].unsqueeze(1).expand(bsz, n, -1).contiguous().view(bsz * n, -1)
        nl, _ = cal_nll_loss(o['words_logit'], wi, wm)
        return nl.view(bsz, n)

    @staticmethod
    def _smooth(z, w):
        if w <= 1:
            return z
        pad = w // 2
        zp = np.concatenate([np.repeat(z[:, :1], pad, axis=1), z,
                             np.repeat(z[:, -1:], pad, axis=1)], axis=1)
        cs = np.concatenate([np.zeros((len(z), 1)), np.cumsum(zp, axis=1)], axis=1)
        return (cs[:, w:] - cs[:, :-w]) / float(w)

    def _inside(self, z, lo, hi):
        cs = np.concatenate([np.zeros((len(z), 1)), np.cumsum(z, axis=1)], axis=1)
        li = np.clip(np.searchsorted(self.cg, lo), 0, self.L - 1)
        hii = np.clip(np.searchsorted(self.cg, hi, side='right') - 1, 0, self.L - 1)
        cnt = np.maximum(hii - li + 1, 1)
        tot = cs[np.arange(len(z)), hii + 1] - cs[np.arange(len(z)), li]
        return tot / cnt

    @torch.no_grad()
    def __call__(self, net_input):
        """对一批输入执行 CSR，返回精修后的框与诊断信息。

        Args
        ----
        net_input : dict
            PC-Net forward 的输入（与 ``model(epoch=0, **net_input)`` 一致）。

        Returns
        -------
        dict
            ``boxes_top1``   [bsz, 2] 精修后 top-1 框 (start, end)，用于 R@1。
            ``boxes_pool5``  [bsz, 5, 2] CSR 框顶替原 top-1 + 原 rank2..5，用于 R@5。
            ``shift``        [bsz] 平移幅度 |c2 - cb|。
            ``moved``        [bsz] bool，是否发生平移。
            ``center_raw`` / ``width_raw``  原提案的 center / width（诊断用）。
            ``relevance``    [bsz, L] 逐点相关性曲线 r(t)（诊断用）。
        """
        m = self.model
        bsz = next(iter(net_input.values())).shape[0]
        P = self._orig_np

        # (1) 原提案 -> top-1 中心/宽度
        m.proposal_generator = self._orig_pg
        m.num_props = P
        o8 = m(epoch=0, **net_input)
        nll8 = self._nll(o8, P, bsz).cpu().numpy()
        C = o8['center'].view(bsz, P).cpu().numpy()
        W = o8['width'].view(bsz, P).cpu().numpy()

        # (2) 逐点提案 -> 相关性曲线，先 z-score 再平滑
        m.proposal_generator = lambda h, b: (self._pt_c.repeat(b).cuda(), self._pt_w.repeat(b).cuda())
        m.num_props = self.L
        op = m(epoch=0, **net_input)
        rel = (-self._nll(op, self.L, bsz)).cpu().numpy()
        m.proposal_generator = self._orig_pg
        m.num_props = P

        Z = (rel - rel.mean(1, keepdims=True)) / (rel.std(1, keepdims=True) + 1e-8)
        Z = self._smooth(Z, self.smooth)

        order = np.argsort(nll8, axis=1)
        j1 = order[:, 0]
        r = np.arange(bsz)
        cb, wb = C[r, j1], W[r, j1]
        bL = np.clip(cb - wb / 2, 0, 1)
        bH = np.clip(cb + wb / 2, 0, 1)

        # (3) 窗口内正部加权质心
        mask = np.abs(self.cg[None, :] - cb[:, None]) <= self.beta
        Pm = np.where(mask, np.maximum(Z, 0.0), 0.0)
        ssum = Pm.sum(1)
        c1 = (Pm @ self.cg) / np.maximum(ssum, 1e-12)
        bad = ssum < 1e-12
        if bad.any():
            c1[bad] = cb[bad]

        # (4) dead-zone
        d = c1 - cb
        d = np.where(np.abs(d) < self.rho * wb, 0.0, d)

        # (5) 阻尼
        c2 = np.clip(cb + self.alpha * d, 0, 1)

        # (6) 自洽门控
        if self.dmin is not None:
            din = self._inside(Z, c2 - wb / 2, c2 + wb / 2) - self._inside(Z, cb - wb / 2, cb + wb / 2)
            c2 = np.where(din >= self.dmin, c2, cb)

        # (7) 保宽输出
        lo = np.clip(c2 - wb / 2, 0, 1)
        hi = np.clip(c2 + wb / 2, 0, 1)
        top1 = np.stack([lo, hi], 1)

        # R@5 框池：CSR 框顶替原 top-1 + 原 rank2..5
        pool = np.zeros((bsz, 5, 2), dtype=np.float64)
        pool[:, 0] = top1
        for t in range(1, 5):
            jj = order[:, t]
            c_, w_ = C[r, jj], W[r, jj]
            pool[:, t, 0] = np.clip(c_ - w_ / 2, 0, 1)
            pool[:, t, 1] = np.clip(c_ + w_ / 2, 0, 1)

        return {'boxes_top1': top1, 'boxes_pool5': pool,
                'shift': np.abs(c2 - cb), 'moved': np.abs(c2 - cb) > 1e-9,
                'center_raw': C, 'width_raw': W, 'relevance': rel}
