# anchor_bank.py
import torch
import math
import torch.nn.functional as F

# 用刚刚生成的 COCO 特征文件
JOINT_FEAT_PATH = "/root/autodl-tmp/anchor_feats_coco.pt"


def _kmeans(z, k, iters=10):
    """
    简单 K-means:
        z: [N, D] 已归一化的特征
        k: 聚类数
    返回:
        centers: [k, D]
    """
    N, D = z.shape
    if k > N:
        k = N

    # 初始化：从样本中随机挑 k 个点
    idx = torch.randperm(N, device=z.device)[:k]
    centers = z[idx].clone()  # [k, D]

    for _ in range(iters):
        # 1. assignment：每个样本找最近中心
        z2 = (z * z).sum(dim=1, keepdim=True)                  # [N, 1]
        c2 = (centers * centers).sum(dim=1, keepdim=True).T    # [1, k]
        dist = z2 + c2 - 2.0 * (z @ centers.T)                 # [N, k]

        assign = dist.argmin(dim=1)                            # [N]

        # 2. update：按 assignment 重新计算中心
        new_centers = torch.zeros_like(centers)
        counts = torch.zeros(k, device=z.device, dtype=torch.long)

        new_centers.index_add_(0, assign, z)
        counts.index_add_(0, assign, torch.ones_like(assign))

        # 避免空簇：对没有样本的中心重新随机选一个样本
        for ci in range(k):
            if counts[ci] > 0:
                new_centers[ci] /= counts[ci].float()
            else:
                ridx = torch.randint(0, N, (1,), device=z.device)
                new_centers[ci] = z[ridx]

        centers = new_centers

    return centers


class AnchorBank(torch.nn.Module):
    def __init__(self, dim, num_anchors=128, device='cuda', assign_topk=1):
        super().__init__()
        self.dim = dim
        self.num_anchors = num_anchors
        self.assign_topk = assign_topk

        # ========= 仅此处：用 (zi+zt)/2 的 joint_feats 做 K-means 初始化锚点 =========
        data = torch.load(JOINT_FEAT_PATH, map_location=device)

        if "joint_feats" not in data:
            raise RuntimeError(
                f"[AnchorBank] feature file '{JOINT_FEAT_PATH}' 必须包含 'joint_feats' 键。"
            )

        joint_feats = data["joint_feats"].to(device)  # [N, D]
        N, D = joint_feats.shape

        if D != dim:
            raise RuntimeError(
                f"[AnchorBank] 维度不匹配: joint_feats dim={D}, 但 AnchorBank dim={dim}."
            )
        if N < 1:
            raise RuntimeError("[AnchorBank] joint_feats 为空。")

        # 先归一化，再聚类会稳定一些
        z = F.normalize(joint_feats, dim=-1)

        k = min(num_anchors, N)
        centers = _kmeans(z, k=k, iters=10)  # [k, D]

        if k < num_anchors:
            # 样本少于 num_anchors，重复填满
            reps = (num_anchors + k - 1) // k
            anchors = centers.repeat(reps, 1)[:num_anchors].clone()
        else:
            anchors = centers.clone()

        anchors = F.normalize(anchors, dim=-1)
        # ========= 初始化结束 =========

        self.register_buffer('anchors', anchors, persistent=True)

    @torch.no_grad()
    def rotate(self, noise_std=0.05):
        # 轻量随机扰动替代高维正交矩阵旋转，稳定且便宜
        noise = torch.randn_like(self.anchors) * noise_std
        self.anchors.add_(noise)
        self.anchors.copy_(F.normalize(self.anchors, dim=-1))

    @torch.no_grad()
    def nearest(self, z_img, z_txt, t_anchor=0.1):
        # z_img, z_txt: (B, D), 已归一化
        # 选能同时靠近图像与文本的锚点：最小化两者到锚点的合计距离
        # 等价于最大化与 (z_img + z_txt)/2 的余弦相似度
        target = F.normalize((z_img + z_txt) * 0.5, dim=-1)  # (B,D)
        sims = target @ self.anchors.T                        # (B,M)
        
        # 软分配温度下限（防止过尖）
        t_anchor = max(t_anchor, 0.2)  # 下限
        
        if self.assign_topk == 1:
            idx = sims.argmax(dim=1)                         # (B,)
            return self.anchors.index_select(0, idx)         # (B,D)
        else:
            topv, topi = sims.topk(self.assign_topk, dim=1)  # (B,K)
            w = torch.softmax(topv / t_anchor, dim=1)        # (B,K) 较温和的权重
            # 保存调试用
            self.w_last = w.detach()
            self.topi = topi.detach()
            selected = self.anchors.index_select(0, topi.reshape(-1))
            selected = selected.view(z_img.size(0), self.assign_topk, -1)  # (B,K,D)
            zref = (w.unsqueeze(-1) * selected).sum(dim=1)   # (B,D)
            return F.normalize(zref, dim=-1)

    def get_anchors(self):
        """返回当前锚点，用于调试或可视化"""
        return self.anchors.clone()

    def get_anchor_stats(self):
        """返回锚点统计信息，用于监控"""
        with torch.no_grad():
            # 计算锚点间的平均距离
            pairwise_dist = torch.pdist(self.anchors, p=2)
            avg_dist = pairwise_dist.mean().item()
            
            # 计算锚点到原点的距离（应该都接近1）
            center_dist = torch.norm(self.anchors, dim=1)
            center_mean = center_dist.mean().item()
            center_std = center_dist.std().item()
            
            return {
                'avg_pairwise_dist': avg_dist,
                'center_dist_mean': center_mean,
                'center_dist_std': center_std,
                'num_anchors': self.num_anchors
            }
