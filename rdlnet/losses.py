"""Training losses and bipartite matching (DETR-style, paper Sec. 3.4, Eq. 8–12)."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .hungarian import linear_sum_assignment
from .model import RDLNetConfig


def dice_loss(pred: Tensor, target: Tensor, eps: float = 1e-6) -> Tensor:
    """Eq. (9), averaged over spatial dims; pred/target [B, Nq, H, W] in [0,1]."""
    p = pred.flatten(2)
    t = target.flatten(2)
    num = 2 * (p * t).sum(-1)
    den = p.sum(-1) + t.sum(-1) + eps
    return (1 - num / den).mean()


def _points_valid_mask_from_padding(tp: Tensor) -> Tensor:
    """
    Build a [Nt, P*2] float mask where padded point pairs are excluded.

    Convention: a point pair (x,y) is considered padding / invalid if either coordinate is negative.
    """
    # tp: [Nt, P*2]
    pts = tp.view(tp.shape[0], -1, 2)
    valid_pt = ((pts[..., 0] >= 0.0) & (pts[..., 1] >= 0.0)).float()  # [Nt, P]
    return valid_pt.unsqueeze(-1).repeat(1, 1, 2).view(tp.shape[0], -1)  # [Nt, P*2]


def _masked_l1_cost(pred: Tensor, tgt: Tensor, mask: Tensor, eps: float = 1e-6) -> Tensor:
    """
    pred: [Nq, D], tgt: [Nt, D], mask: [Nt, D] in {0,1}
    returns: [Nq, Nt] mean L1 over masked dims per target.
    """
    diff = (pred[:, None, :] - tgt[None, :, :]).abs()
    w = mask[None, :, :]
    num = (diff * w).sum(dim=-1)
    den = w.sum(dim=-1).clamp_min(eps)
    return num / den


def build_matcher(cfg: RDLNetConfig, cost_class: float = 2.0, cost_mask: float = 5.0, cost_point: float = 2.0):
    return HungarianMatcher(cfg, cost_class=cost_class, cost_mask=cost_mask, cost_point=cost_point)


class HungarianMatcher(nn.Module):
    """Minimal bipartite matching following Carion et al. (DETR) for triple-branch costs."""

    def __init__(self, cfg: RDLNetConfig, cost_class: float = 2.0, cost_mask: float = 5.0, cost_point: float = 2.0) -> None:
        super().__init__()
        self.cfg = cfg
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_point = cost_point

    @torch.no_grad()
    def forward(
        self,
        pred_logits: Tensor,
        pred_masks: Tensor,
        pred_points: Tensor,
        tgt_labels: List[Tensor],
        tgt_masks: List[Tensor],
        tgt_points: List[Tensor],
    ) -> List[Tuple[Tensor, Tensor]]:
        """
        pred_*: batched predictions [B, Nq, ...].
        tgt_*: list length B of [Ni], [Ni,H,W], [Ni, P*2] on the **same device** as ``pred_logits``.
        Returns list of (src_idx, tgt_idx) per image.
        """
        b, nq, _ = pred_logits.shape
        out: List[Tuple[Tensor, Tensor]] = []
        prob = pred_logits.softmax(-1)

        for i in range(b):
            if tgt_labels[i].numel() == 0:
                empty = torch.tensor([], dtype=torch.long, device=pred_logits.device)
                out.append((empty, empty))
                continue

            tl = tgt_labels[i]
            tm = tgt_masks[i].float()
            tp = tgt_points[i]

            # classification cost [Nq, Nt]
            cost_c = -prob[i][:, tl]
            # mask cost: downsample if needed
            if tm.shape[-2:] != pred_masks.shape[-2:]:
                pm = F.interpolate(pred_masks[i : i + 1], size=tm.shape[-2:], mode="bilinear", align_corners=False)[0]
            else:
                pm = pred_masks[i]
            # torch.cdist CUDA does not support bfloat16; compute mask cost in fp32.
            pm = pm.sigmoid().float()
            tm_f = tm.float()
            cost_m = torch.cdist(pm.flatten(1), tm_f.flatten(1), p=1) / (tm.shape[-1] * tm.shape[-2])

            # point L1 cost [Nq, Nt]
            m = _points_valid_mask_from_padding(tp)
            cost_p = _masked_l1_cost(pred_points[i], tp, m)

            C = self.cost_class * cost_c + self.cost_mask * cost_m + self.cost_point * cost_p
            src, dst = linear_sum_assignment(C)
            out.append((src, dst))
        return out


class RDLNetLoss(nn.Module):
    """Eq. (12): L_total = λ1 L_cls + λ2 L_distance + λ3 L_dice + λ4 L_mask."""

    def __init__(
        self,
        cfg: RDLNetConfig,
        matcher: HungarianMatcher,
        lambda_cls: float = 2.0,
        lambda_dist: float = 2.0,
        lambda_dice: float = 5.0,
        lambda_mask: float = 5.0,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.matcher = matcher
        self.l1 = lambda_cls
        self.l2 = lambda_dist
        self.l3 = lambda_dice
        self.l4 = lambda_mask
        self.empty_weight = torch.ones(cfg.num_classes + 1)
        self.empty_weight[-1] = 0.1

    def forward(
        self,
        pred_logits: Tensor,
        pred_masks: Tensor,
        pred_points: Tensor,
        tgt_labels: List[Tensor],
        tgt_masks: List[Tensor],
        tgt_points: List[Tensor],
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """``tgt_*`` must live on the same device as ``pred_logits`` (e.g. moved in the training loop)."""
        b, nq, _ = pred_logits.shape
        device = pred_logits.device
        self.empty_weight = self.empty_weight.to(device)

        indices = self.matcher(pred_logits, pred_masks, pred_points, tgt_labels, tgt_masks, tgt_points)

        loss_cls = torch.zeros(1, device=device)
        loss_mask = torch.zeros(1, device=device)
        loss_dice = torch.zeros(1, device=device)
        loss_dist = torch.zeros(1, device=device)
        n_inst = 0

        for i, (src_i, tgt_i) in enumerate(indices):
            if tgt_i.numel() == 0:
                # background queries
                tgt_c = torch.full((nq,), self.cfg.num_classes, dtype=torch.long, device=device)
                loss_cls = loss_cls + F.cross_entropy(pred_logits[i], tgt_c, self.empty_weight)
                continue

            n_inst += src_i.numel()
            tgt_c = torch.full((nq,), self.cfg.num_classes, dtype=torch.long, device=device)
            tgt_c[src_i] = tgt_labels[i][tgt_i]
            loss_cls = loss_cls + F.cross_entropy(pred_logits[i], tgt_c, self.empty_weight)

            pm = pred_masks[i][src_i]
            tm = tgt_masks[i][tgt_i]
            if pm.shape[-2:] != tm.shape[-2:]:
                tm = F.interpolate(tm.unsqueeze(1), size=pm.shape[-2:], mode="nearest").squeeze(1)
            pm_sig = pm.sigmoid()
            loss_mask = loss_mask + F.binary_cross_entropy_with_logits(pm, tm)
            loss_dice = loss_dice + dice_loss(pm_sig, tm)

            pp = pred_points[i][src_i]
            tp = tgt_points[i][tgt_i]
            m = _points_valid_mask_from_padding(tp)  # [Nt, P*2] in {0,1}
            diff = (pp - tp).abs() * m

            den = m.sum(dim=-1)  # [Nt]
            valid = den > 0

            if valid.any():
                per_inst = diff.sum(dim=-1) / den.clamp_min(1.0)  # [Nt]
                loss_dist = loss_dist + per_inst[valid].mean()

        n_inst = max(n_inst, 1)
        loss_cls = loss_cls / b
        loss_mask = loss_mask / n_inst
        loss_dice = loss_dice / n_inst
        loss_dist = loss_dist / n_inst

        total = self.l1 * loss_cls + self.l2 * loss_dist + self.l3 * loss_dice + self.l4 * loss_mask
        return total, {
            "loss": total,
            "loss_cls": loss_cls.detach(),
            "loss_dist": loss_dist.detach(),
            "loss_dice": loss_dice.detach(),
            "loss_mask": loss_mask.detach(),
        }
