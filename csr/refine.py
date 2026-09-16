"""PC-Net 长时动作「起止点连接」修复模块 —— 推理期边界精修（RBR, Relevance-Boundary Refinement）

问题：PC-Net 的 8 个提案在训练后塌缩为**固定时间网格**（中心与输入无关、宽度恒在 0.33~0.49），
      因此无法覆盖长时动作的完整跨度 —— 这正是论文自述的
      "long durations ... inability to capture association between the start and end points"。

本模块：零训练、纯推理期、不使用任何 gt。步骤
  1) 用 PC-Net 原提案（含 nll 选择）拿到 top-1 中心 c；
  2) 用**逐点提案** c_t = linspace 走一遍同一模型，得到逐帧相关性曲线 r(t) = -nll(t)，z-score 标准化；
  3) 在 c 处按阈值 z > μ + k·σ 向两侧 flood 扩展，得到区间 [s, e]；
  4) 与基线框取并集 → 精修框。

为什么有效（已用 8 个权重的对照实验证明）：
  - 「均匀放大框」无效（scale×1.05~1.30 最多 +0.0007），必须由相关性曲线决定扩展方向与幅度；
  - 收益与「基线框有多窄」单调相关（框宽 0.16 时 +0.177，框宽 0.97 时 +0.000）；
  - 增益集中在长动作（gt≥0.35 时 +0.049，短动作 −0.000）。

用法：
    from refine import RelevanceBoundaryRefiner
    rbr = RelevanceBoundaryRefiner(model, num_point=50, k=0.15)
    out = rbr(net_input)          # net_input 与 PC-Net forward 的输入一致
    out['boxes_top1']             # [bsz, 2] 精修后的 top-1 框 (start, end)，用于 R@1
    out['boxes_pool5']            # [bsz, 5, 2] 精修 top-1 + 原 rank2..5，用于 R@5

注意：R@5 请使用 boxes_pool5（即"精修框顶替原 top-1"）。两条已验证的经验：
      ① 给精修框重新计算 nll 再排序会破坏排序（精修框变宽→重建高斯被稀释），R@5 从 0.693 掉到 0.667（diag27）；
      ② 用多档 k 的多个精修框去"丰富"框池同样有害（diag28，13 种组合全负，最好仍 −0.003）；
      ③ R@5 不提升是本方法的结构性边界（R@5 度量原 5 元集合的覆盖力），若更看重 R@5 可用 k=0.20（损失最小 −0.0024）。
实测（Charades-CG test_trivial，N=3096，K_REP=3）：
      R@1 0.4720 → 0.4828（+0.0108）；gt>=0.35 长动作 R@1 0.5363 → 0.5882（+0.0519）；R@5 0.6928 → 0.6890。
"""
import numpy as np
import torch

__all__ = ['RelevanceBoundaryRefiner']


class RelevanceBoundaryRefiner:
    def __init__(self, model, num_point=None, k=0.15, max_num_frames=200, smooth=1):
        """
        model        : 已加载权重的 models.PCNet 实例（需 .proposal_generator / .num_props 可被替换）
        num_point    : 逐点提案数量，默认 max_num_frames//4
        k            : flood 阈值系数，z > mu + k*sd。0.10~0.20 最优且不敏感，默认 0.15
        max_num_frames: 与 config['dataset']['max_num_frames'] 一致（PC-Net 输入固定 200 帧）
        smooth       : relevance 曲线平滑窗口（1 = 不平滑）
        """
        self.model = model
        self.L = int(num_point or max_num_frames // 4)
        self.k = float(k)
        self.smooth = int(smooth)
        self.grid = torch.linspace(0, 1, self.L)
        self._orig_pg = model.proposal_generator
        self._orig_np = model.num_props
        self._pt_c = torch.linspace(0.5 / self.L, 1 - 0.5 / self.L, self.L)
        self._pt_w = torch.full((self.L,), max(2.0 / self.L, 0.02))

    # ---------- 内部：把模型跑成逐点提案，得到相关性曲线 ----------
    def _relevance(self, net_input, bsz):
        m = self.model
        m.proposal_generator = lambda h, b: (self._pt_c.repeat(b).cuda(), self._pt_w.repeat(b).cuda())
        m.num_props = self.L
        o = m(epoch=0, **net_input)
        return o

    def _nll(self, o, n, bsz):
        from models.loss import cal_nll_loss
        wm = o['words_mask'].unsqueeze(1).expand(bsz, n, -1).contiguous().view(bsz * n, -1)
        wi = o['words_id'].unsqueeze(1).expand(bsz, n, -1).contiguous().view(bsz * n, -1)
        nl, _ = cal_nll_loss(o['words_logit'], wi, wm)
        return nl.view(bsz, n)

    @torch.no_grad()
    def __call__(self, net_input):
        """对一批输入执行 RBR，返回精修后的框与诊断信息。

        Args
        ----
        net_input : dict
            PC-Net forward 的输入（与 ``model(epoch=0, **net_input)`` 一致）。

        Returns
        -------
        dict
            ``boxes_top1``   [bsz, 2] 精修后 top-1 框 (start, end)，用于 R@1。
            ``boxes_pool5``  [bsz, 5, 2] 精修 top-1 + 原 rank2..5，用于 R@5。
            ``nll8`` / ``center`` / ``width`` / ``relevance``  诊断信息。
        """
        m = self.model
        bsz = next(iter(net_input.values())).shape[0]
        P = self._orig_np
        # (1) 原提案 + 逐点相关性
        m.proposal_generator = self._orig_pg
        m.num_props = P
        o8 = m(epoch=0, **net_input)
        nll8 = self._nll(o8, P, bsz).cpu().numpy()
        C = o8['center'].view(bsz, P).cpu().numpy()
        W = o8['width'].view(bsz, P).cpu().numpy()
        opl = self._relevance(net_input, bsz)
        rel = (-self._nll(opl, self.L, bsz)).cpu().numpy()
        m.proposal_generator = self._orig_pg
        m.num_props = P

        if self.smooth > 1:
            ker = np.ones(self.smooth) / self.smooth
            rel = np.apply_along_axis(lambda v: np.convolve(v, ker, mode='same'), 1, rel)

        order = np.argsort(nll8, axis=1)
        j1 = order[:, 0]
        r = np.arange(bsz)
        cb, wb = C[r, j1], W[r, j1]
        bL = np.clip(cb - wb / 2, 0, 1)
        bH = np.clip(cb + wb / 2, 0, 1)

        # (3) 从 top-1 中心沿相关性曲线 flood
        g = self.grid.numpy()
        Z = (rel - rel.mean(1, keepdims=True)) / (rel.std(1, keepdims=True) + 1e-8)
        TH = Z.mean(1) + self.k * Z.std(1)
        s = np.zeros(bsz); e = np.ones(bsz)
        for i in range(bsz):
            pc = int(round(np.clip(cb[i], 0, 1) * (self.L - 1)))
            a1 = pc
            while a1 > 0 and Z[i, a1 - 1] > TH[i]:
                a1 -= 1
            b1 = pc
            while b1 < self.L - 1 and Z[i, b1 + 1] > TH[i]:
                b1 += 1
            s[i] = g[a1]; e[i] = g[b1]
        # (4) 与基线框取并集
        lo = np.minimum(s, bL); hi = np.maximum(e, bH)
        top1 = np.stack([np.clip(lo, 0, 1), np.clip(hi, 0, 1)], 1)

        # R@5 框池：精修 top-1 + 原 rank2..5
        pool = np.zeros((bsz, 5, 2), dtype=np.float64)
        pool[:, 0] = top1
        for t in range(1, 5):
            jj = order[:, t]
            c_ = C[r, jj]; w_ = W[r, jj]
            pool[:, t, 0] = np.clip(c_ - w_ / 2, 0, 1)
            pool[:, t, 1] = np.clip(c_ + w_ / 2, 0, 1)

        return {'boxes_top1': top1, 'boxes_pool5': pool,
                'nll8': nll8, 'center': C, 'width': W, 'relevance': rel}
