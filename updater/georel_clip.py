# -*- coding: utf-8 -*-
import math
import sys
from copy import deepcopy

import torch
import torch.nn.functional as F
from ignite.utils import convert_tensor

# =============== 以下代码完全保留你的原始逻辑 ===============
sys.path.append("/root/autodl-tmp/")

# =============== 导入 AnchorBank 模块 ===============
try:
    from anchor_bank_kmeans import AnchorBank
    ANCHOR_BANK_AVAILABLE = True
except Exception:
    ANCHOR_BANK_AVAILABLE = False
    AnchorBank = None
    print("[Updater] Warning: anchor_bank.py not available; AnchorBank features disabled.")


# =============== 核心工具函数 ===============
def grad_global_norm(model) -> float:
    """梯度全局范数（训练稳定性监控）"""
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.pow(2).sum().item()
    return math.sqrt(total) if total > 0 else 0.0


def uniformity_loss(feats, beta=0.05):
    """特征均匀性正则损失（训练用，不等同论文Uniformity指标）"""
    if feats.size(0) < 2 or beta <= 0:
        return feats.new_tensor(0.0)
    f = F.normalize(feats, dim=-1)
    idx = torch.randperm(f.size(0), device=f.device)
    val = torch.exp(-2.0 * (f - f[idx]).pow(2).sum(dim=-1)).mean()
    return beta * val


def _pair_align(zi, zt):
    """配对余弦相似度（训练过程监控用，不等同论文Alignment指标）"""
    return (F.normalize(zi, dim=-1) * F.normalize(zt, dim=-1)).sum(dim=-1).mean().item()


def _mgap(zi, zt):
    """模态间隙（角度版监控用，不等同论文Modality Gap指标）"""
    c1 = F.normalize(zi.mean(dim=0, keepdim=True), dim=-1)
    c2 = F.normalize(zt.mean(dim=0, keepdim=True), dim=-1)
    return (1 - (c1 * c2).sum(dim=-1)).mean().item()


def _uni_rbf_stack(zi, zt):
    """特征均匀性监控（训练过程proxy，不等同论文Uniformity指标）"""
    z = torch.cat([zi, zt], dim=0)
    z = F.normalize(z, dim=-1)
    if z.size(0) <= 1:
        return 0.0
    idx = torch.randperm(z.size(0), device=z.device)[: min(256, z.size(0))]
    d = (1 - (z[idx] @ z[idx].T)).clamp_min_(0)
    return (-2.0 * d).exp().mean().log().item()


# ===================================================
# GeoRel-CLIP updater: AnchorBank + Relational KD
# ===================================================
class GeoRelCLIPUpdater:
    def __init__(
        self,
        *args,
        # 主损失核心参数
        lambda_cont=1.0,

        # AnchorBank 核心参数
        use_anchor_bank=False,
        num_anchors=128,
        anchor_assign_topk=2,
        anchor_noise_std=0.03,
        T_rotate=60,

        # 正则化核心参数
        beta_uni=0.05,
        use_anchor_diversity=True,
        gamma_div=0.03,
        anchor_soft_temp=0.1,

        # Relational KD 参数
        use_rkd=False,
        lambda_rkd=0.0,
        rkd_temperature=1.5,

        **kwargs,
    ):
        # 基础参数初始化
        self.model = kwargs.pop("model")
        self.optimizer = kwargs.pop("optimizer")
        self.device = kwargs.pop("device")
        self.lambda_cont = float(lambda_cont)

        # 正则化参数
        self.beta_uni = float(beta_uni)
        self.use_anchor_bank = bool(use_anchor_bank) and ANCHOR_BANK_AVAILABLE
        self.use_anchor_diversity = bool(use_anchor_diversity)
        self.gamma_div = float(gamma_div)
        self.anchor_soft_temp = float(anchor_soft_temp)

        # Relational KD 参数
        self.use_rkd = bool(use_rkd)
        self.lambda_rkd = float(lambda_rkd)
        self.rkd_temperature = float(rkd_temperature)

        # AnchorBank 核心参数
        self.anchor_noise_std = float(anchor_noise_std)
        self.T_rotate = int(T_rotate)
        self.anchor_assign_topk = int(anchor_assign_topk)
        self.num_anchors = int(num_anchors)
        self.anchor_bank = None

        if self.use_anchor_bank:
            feat_dim = 512
            if hasattr(self.model, "visual") and hasattr(self.model.visual, "output_dim"):
                feat_dim = int(self.model.visual.output_dim)
            elif hasattr(self.model, "text_projection") and hasattr(self.model.text_projection, "out_features"):
                feat_dim = int(self.model.text_projection.out_features)

            self.anchor_bank = AnchorBank(
                dim=feat_dim,
                num_anchors=self.num_anchors,
                device=self.device,
                assign_topk=self.anchor_assign_topk,
            )

        # ====== 构建 teacher（RKD 用）======
        self.teacher = None
        if self.use_rkd:
            self.teacher = deepcopy(self.model).to(self.device)
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad = False
            print("[GeoRel-CLIP] Relational KD enabled: teacher model has been frozen.")

    # ---- 数据加载 ----
    @staticmethod
    def _get_batch(batch, device=None, non_blocking=True):
        x, y = batch
        return (
            convert_tensor(x, device=device, non_blocking=non_blocking),
            convert_tensor(y, device=device, non_blocking=non_blocking),
        )

    # ---- CLIP 核心对比损失（InfoNCE）----
    @staticmethod
    def _clip_loss(zi, zt, logit_scale, device):
        logits_i2t = zi @ zt.T
        logits_t2i = zt @ zi.T
        labels = torch.arange(logits_i2t.size(0), device=device)
        return 0.5 * (
            F.cross_entropy(logit_scale * logits_i2t, labels)
            + F.cross_entropy(logit_scale * logits_t2i, labels)
        )

    # ---- 可微软分配（Anchor-usage 用）----
    def _soft_anchor_assignment(self, feats):
        anchors = F.normalize(self.anchor_bank.anchors, dim=-1)  # [M, D]
        similarity = feats @ anchors.T                            # [B, M]
        soft_assignment = F.softmax(similarity / self.anchor_soft_temp, dim=-1)
        return soft_assignment

    # ---- Relational KD（相似度矩阵 MSE 蒸馏）----
    def _relational_kd(self, zi_s, zt_s, images, texts):
        if (not self.use_rkd) or (self.teacher is None):
            device = zi_s.device
            return torch.tensor(0.0, device=device)

        B = zi_s.size(0)
        if B < 2:
            device = zi_s.device
            return torch.tensor(0.0, device=device)

        with torch.no_grad():
            if hasattr(self.teacher, "encode_image") and hasattr(self.teacher, "encode_text"):
                zi_t = self.teacher.encode_image(images, normalized=True)
                texts_in = texts.squeeze(1) if texts.dim() != 1 else texts
                zt_t = self.teacher.encode_text(texts_in, normalized=True)
            else:
                texts_in = texts.squeeze(1) if texts.dim() != 1 else texts
                out_t = self.teacher(images, texts_in)
                zi_t = F.normalize(out_t["image_features"], dim=-1)
                zt_t = F.normalize(out_t["text_features"], dim=-1)

        tau = self.rkd_temperature if self.rkd_temperature > 0 else 1.0
        S_t = (zi_t @ zt_t.T) / tau
        S_s = (zi_s @ zt_s.T) / tau

        eps = 1e-6
        S_t_norm = (S_t - S_t.mean()) / (S_t.std(unbiased=False) + eps)
        S_s_norm = (S_s - S_s.mean()) / (S_s.std(unbiased=False) + eps)

        return F.mse_loss(S_s_norm, S_t_norm)

    # ==========================================================
    # ✅ 训练结束后：按论文公式在“整个测试集”上计算三项指标（只算一次）
    # ==========================================================
    @torch.no_grad()
    def evaluate_paper_metrics_on_test(
        self,
        test_loader,
        chunk_size: int = 1024,
        uniformity_max_M: int = 20000,
    ):
        self.model.eval()

        img_feats = []
        txt_feats = []

        for batch in test_loader:
            images, texts = self._get_batch(batch, device=self.device)
            texts_in = texts if texts.dim() == 1 else texts.squeeze(1)

            if hasattr(self.model, "encode_image") and hasattr(self.model, "encode_text"):
                zi = self.model.encode_image(images, normalized=True)
                zt = self.model.encode_text(texts_in, normalized=True)
            else:
                out = self.model(images, texts_in)
                zi = F.normalize(out["image_features"], dim=-1)
                zt = F.normalize(out["text_features"], dim=-1)

            img_feats.append(zi.detach().float().cpu())
            txt_feats.append(zt.detach().float().cpu())

        zi_all = torch.cat(img_feats, dim=0)  # [N,D]
        zt_all = torch.cat(txt_feats, dim=0)  # [N,D]
        N = int(zi_all.size(0))

        # ---- (11) Modality Gap: || mean(fV) - mean(fT) ||_2^2
        fV_bar = zi_all.mean(dim=0)
        fT_bar = zt_all.mean(dim=0)
        modality_gap = (fV_bar - fT_bar).pow(2).sum().item()

        # ---- (12) Alignment: mean_i || fV(x_i) - fT(t_i) ||_2^2
        alignment = (zi_all - zt_all).pow(2).sum(dim=-1).mean().item()

        # ---- (13) Uniformity: (1/(2N)) * sum_{f1,f2 in F} exp(-2||f1-f2||^2)
        Z = torch.cat([zi_all, zt_all], dim=0).to(self.device)  # [M,D], M=2N
        Z = F.normalize(Z, dim=-1)
        M = int(Z.size(0))

        if M > int(uniformity_max_M):
            raise RuntimeError(
                f"[Paper Uniformity] M=2N={M} too large for exact O(M^2). "
                f"Increase uniformity_max_M or reduce test size."
            )

        # normalized: ||a-b||^2 = 2 - 2*sim -> exp(-2||a-b||^2) = exp(-4 + 4*sim)
        total_sum = 0.0
        for s in range(0, M, int(chunk_size)):
            zc = Z[s : s + int(chunk_size)]     # [C,D]
            sim = zc @ Z.T                      # [C,M]
            val = torch.exp(-4.0 + 4.0 * sim)   # [C,M]
            total_sum += val.sum().item()

        uniformity = total_sum / float(M)  # 1/(2N)=1/M

        return {
            "paper_modality_gap": float(modality_gap),
            "paper_alignment": float(alignment),
            "paper_uniformity": float(uniformity),
            "N_test": int(N),
        }

    # ---- 核心训练逻辑 ----
    def __call__(self, engine, batch):
        report = {}
        self.model.train()
        step = engine.state.iteration if hasattr(engine.state, "iteration") else 0

        # 1. 数据加载与特征提取（梯度保留）
        images, texts = self._get_batch(batch, device=self.device)
        out = self.model(images, texts if texts.dim() == 1 else texts.squeeze(1))

        feat_v_raw = out["image_features"]
        feat_t_raw = out["text_features"]

        zi = F.normalize(feat_v_raw, dim=-1)
        zt = F.normalize(feat_t_raw, dim=-1)

        # 2. 主损失：CLIP 对比损失
        loss_clip = self._clip_loss(zi, zt, out["logit_scale"], self.device)
        total_loss = self.lambda_cont * loss_clip

        # 3. 可微锚点多样性损失（软分配 + KL 散度）
        loss_div_val = 0.0
        L_div = torch.tensor(0.0, device=zi.device)
        if self.use_anchor_bank and self.use_anchor_diversity:
            _ = self.anchor_bank.nearest(zi, zt, t_anchor=0.30)

            soft_zi = self._soft_anchor_assignment(zi)
            soft_zt = self._soft_anchor_assignment(zt)

            anchor_usage = (soft_zi.sum(dim=0) + soft_zt.sum(dim=0)) / 2
            p = (anchor_usage / anchor_usage.sum()).clamp_min(1e-8)
            q = torch.ones_like(p) / self.num_anchors

            L_div = (p * (p.log() - q.log())).sum().clamp_min(0.0)
            total_loss += self.gamma_div * L_div
            loss_div_val = float(L_div.detach())

        # 4. 特征均匀性正则损失（训练用）
        loss_uni_val = 0.0
        if self.beta_uni > 0:
            all_feats = torch.cat([zi, zt], dim=0)
            loss_uni = uniformity_loss(all_feats, self.beta_uni)
            total_loss += loss_uni
            loss_uni_val = float(loss_uni.detach())

        # 5. Relational KD（相似度矩阵 MSE KD）
        loss_kd_rel_val = 0.0
        if self.use_rkd and self.lambda_rkd > 0:
            L_kd_rel = self._relational_kd(zi, zt, images, texts)
            total_loss += self.lambda_rkd * L_kd_rel
            loss_kd_rel_val = float(L_kd_rel.detach())

        # 6. 锚点旋转（仅更新锚点，无梯度）
        if self.use_anchor_bank and self.T_rotate > 0 and step > 0 and (step % self.T_rotate == 0):
            with torch.no_grad():
                self.anchor_bank.rotate(noise_std=self.anchor_noise_std)

        # 7. 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # 8. 日志指标（训练过程监控）
        sim = zi @ zt.T
        diag = sim.diag().mean().item()
        offdiag = ((sim.sum() - sim.diag().sum()) / (sim.numel() - sim.size(0))).item()

        zi_norm = F.normalize(zi, dim=-1)
        zt_norm = F.normalize(zt, dim=-1)
        feat_gap = (1 - (zi_norm * zt_norm).sum(dim=-1)).mean().detach()

        report.update({
            "loss": float(loss_clip.detach()),
            "total_loss": float(total_loss.detach()),
            "train_loss": float(loss_clip.detach()),
            "loss_uni": loss_uni_val,
            "loss_div": loss_div_val,
            "loss_kd_rel": loss_kd_rel_val,
            "align": _pair_align(zi, zt),
            "modality_gap": float(F.mse_loss(zi.mean(dim=0), zt.mean(dim=0)).detach()),
            "mgap": _mgap(zi, zt),
            "uniRBF": _uni_rbf_stack(zi, zt),
            "feat_gap": float(feat_gap),
            "train_feat_gap": float(feat_gap),
            "sim_diag": diag,
            "sim_offdiag": offdiag,
            "margin": diag - offdiag,
            "grad_norm": grad_global_norm(self.model),
        })

        return report
