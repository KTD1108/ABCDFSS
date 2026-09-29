import torch
import torch.nn.functional as F
import math
from utils import segutils


def buildHyperCol(feat_pyram):
    # concatenate along channel dim
    # upsample spatial size to largest feat vol space available
    target_size = feat_pyram[0].shape[-2:]
    upsampled = []
    for layer in feat_pyram:
        # if idx < self.stack_ids[0]: continue
        upsampled.append(F.interpolate(layer, size=target_size, mode='bilinear', align_corners=False))
    return torch.cat(upsampled, dim=1)


# accepts both:
# s_feat_vol: [bsz,k,c,h,w]->[bsz,c,h,w*k]
# s_mask: [bsz,k,h,w]->[bsz,h,w*k]
def paste_supports_together(supports):
    return torch.cat(supports.unbind(dim=1), dim=-1)


# Attention regular:
# 1. Dot product
# 2. Divide by square root of key length (#nchannels)
# 3. Softmax
# 4. Multiply with V (mask)

def buildDenseAffinityMat(qfeat_volume, sfeat_volume, softmax_arg2=True):  # bsz,C,H,W
    qfeat_volume, sfeat_volume = qfeat_volume.permute(0, 2, 3, 1), sfeat_volume.permute(0, 2, 3, 1)
    bsz, Hq, Wq, C = qfeat_volume.shape
    Hs, Ws = sfeat_volume.shape[1], sfeat_volume.shape[2]
    # [px,C][C,px]=[px,px]
    dense_affinity_mat = torch.matmul(qfeat_volume.view(bsz, Hq * Wq, C),
                                      sfeat_volume.view(bsz, Hs * Ws, C).transpose(1, 2))
    if softmax_arg2 is False: return dense_affinity_mat
    dense_affinity_mat_softmax = (dense_affinity_mat / math.sqrt(C)).softmax(
        dim=-1)  # each query pixel's affinities sum up to 1 over support pxls
    return dense_affinity_mat_softmax


# filter with support mask following DAM
def filterDenseAffinityMap(dense_affinity_mat, downsampled_smask):
    # for each query pixel, aggregate all correlations where the support mask ==1
    # [px,px][px,1]=[px,1]
    bsz, HWq, HWs = dense_affinity_mat.shape
    # let mean(V)=1 -> sum(V)=len(V) -> d_mask / mean(d_mask)
    # downsampled_smask_norm = downsampled_smask / downsampled_smask.mean()
    q_coarse = torch.matmul(dense_affinity_mat, downsampled_smask.view(bsz, HWs, 1))
    return q_coarse.view(bsz, HWq)


def upsample(volume, h, w):
    return F.interpolate(volume, size=(h, w), mode='bilinear', align_corners=False)

class SoftmaxWeightedFusion(torch.nn.Module):
    r""" Softmax-weighted layer fusion replacing flat average q_fused """
    def __init__(self, num_layers=13, mode='softmax_margin', temperature=1.0):
        super(SoftmaxWeightedFusion, self).__init__()
        self.num_layers = num_layers
        self.mode = mode
        self.temperature = max(float(temperature), 1e-3)
        self.learnable_delta = torch.nn.Parameter(torch.zeros(num_layers))

    @torch.no_grad()
    def compute_layer_margins(self, s_feat_t, s_mask, l0=3):
        r""" Computes discriminative margin delta_l for each layer l >= l0 on support set """
        margins = []
        for sft in s_feat_t[l0:]:
            # sft: [bsz, k, c, hs, ws]
            sft = sft.detach()
            bsz, k, c, hs, ws = sft.shape
            sft_flat = sft.view(bsz * k, c, hs * ws)

            smask_d = [segutils.downsample_mask(m.float(), hs, ws) for m in s_mask.unbind(1)]
            smask_cat = torch.stack(smask_d, dim=1).view(bsz * k, 1, hs * ws)

            fg_mask = smask_cat.float()
            bg_mask = 1.0 - fg_mask

            fg_count = fg_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
            bg_count = bg_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)

            # Foreground and background prototypes
            fg_proto = (sft_flat * fg_mask).sum(dim=-1, keepdim=True) / fg_count
            bg_proto = (sft_flat * bg_mask).sum(dim=-1, keepdim=True) / bg_count

            fg_proto_norm = F.normalize(fg_proto, p=2, dim=1)
            bg_proto_norm = F.normalize(bg_proto, p=2, dim=1)

            # Margin = 1 - CosineSimilarity(fg_proto, bg_proto)
            cos_sim = (fg_proto_norm * bg_proto_norm).sum(dim=1).mean()
            margin = 1.0 - cos_sim
            margins.append(margin)

        return torch.stack(margins)

    def forward(self, q_pred_coarses_t, s_feat_t=None, s_mask=None, l0=3):
        bsz, L, h0, w0 = q_pred_coarses_t.shape

        if self.mode == 'softmax_margin' and s_feat_t is not None and s_mask is not None:
            delta = self.compute_layer_margins(s_feat_t, s_mask, l0=l0).to(q_pred_coarses_t.device)
            weights = F.softmax(delta / self.temperature, dim=0).detach()
        elif self.mode == 'learnable':
            weights = F.softmax(self.learnable_delta[:L] / self.temperature, dim=0).to(q_pred_coarses_t.device)
        else:  # 'mean' or fallback uniform
            weights = torch.full((L,), 1.0 / L, device=q_pred_coarses_t.device)

        q_fused = (q_pred_coarses_t * weights.view(1, L, 1, 1)).sum(dim=1)
        return q_fused, weights


class DAMatComparison:
    def __init__(self, fusion_mode='softmax_margin', fusion_temp=1.0):
        self.fusion_mode = fusion_mode
        self.fusion_temp = fusion_temp
        self.fusion_module = SoftmaxWeightedFusion(mode=fusion_mode, temperature=fusion_temp)

    def algo_mean(self, q_pred_coarses_t, s_feat_t=None, s_mask=None):
        return q_pred_coarses_t.mean(1)

    def algo_softmax_weighted(self, q_pred_coarses_t, s_feat_t=None, s_mask=None):
        q_fused, _ = self.fusion_module(q_pred_coarses_t, s_feat_t, s_mask)
        return q_fused

    def calc_q_pred_coarses(self, q_feat_t, s_feat_t, s_mask, l0=3):
        q_pred_coarses = []
        h0, w0 = q_feat_t[l0].shape[-2:]
        for (qft, sft) in zip(q_feat_t[l0:], s_feat_t[l0:]):
            qft, sft = qft.detach(), sft.detach()
            bsz, c, hq, wq = qft.shape
            hs, ws = sft.shape[-2:]

            sft_row = torch.cat(sft.unbind(1), -1)  # bsz,k,c,h,w -> bsz,c,h,w*k
            smasks_downsampled = [segutils.downsample_mask(m.float(), hs, ws) for m in s_mask.unbind(1)]
            smask_row = torch.cat(smasks_downsampled, -1)

            damat = buildDenseAffinityMat(qft, sft_row)
            filtered = filterDenseAffinityMap(damat, smask_row)
            q_pred_coarse = upsample(filtered.view(bsz, 1, hq, wq), h0, w0).squeeze(1)
            q_pred_coarses.append(q_pred_coarse)
        return torch.stack(q_pred_coarses, dim=1)

    def forward(self, q_feat_t, s_feat_t, s_mask, upsample=True, debug=False):
        q_pred_coarses_t = self.calc_q_pred_coarses(q_feat_t, s_feat_t, s_mask)

        if debug: display(segutils.pilImageRow(*q_pred_coarses_t.unbind(1), q_pred_coarses_t.mean(1)))

        # select the algorithm
        if self.fusion_mode == 'mean':
            postprocessing_algorithm = self.algo_mean
        else:
            postprocessing_algorithm = self.algo_softmax_weighted

        # do the fusion
        logit_mask = postprocessing_algorithm(q_pred_coarses_t, s_feat_t=s_feat_t, s_mask=s_mask)
        if upsample:  # if query and support have different shape, then you must do upsampling yourself afterwards
            logit_mask = segutils.downsample_mask(logit_mask, *s_mask.shape[-2:])

        return logit_mask