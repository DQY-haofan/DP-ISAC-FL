# 文件名: stga.py
# 作用: R-JORA 子模块 1 - 时空图聚合器。计算信任分数，过滤异常更新。
# 版本: Fixed (Added Norm Clipping & Softened Softmax to prevent Defense Backfire)

import torch
import torch.nn.functional as F
from collections import deque
import copy


class STGAAggregator:
    def __init__(self, config):
        self.conf = config['r_jora']
        self.device = config['device']
        self.alpha = self.conf['stga_alpha']
        self.history_updates = deque(maxlen=5)
        self.diagnostics = {'avg_trust_benign': 0.0, 'avg_trust_malicious': 0.0}

    def _flatten(self, update_dict):
        vec = []
        for k in sorted(update_dict.keys()):
            if update_dict[k].dtype in [torch.float32, torch.float64]:
                vec.append(update_dict[k].view(-1))
        return torch.cat(vec)

    def aggregate(self, updates, client_types=None):
        if not updates: return None

        # 回退检查
        if not self.conf['enabled'] or not self.conf['enable_stga']:
            return self._fedavg(updates)

        # 1. 准备数据
        flat_updates = [self._flatten(u) for u in updates]
        update_matrix = torch.stack(flat_updates).to(self.device)

        # [Critical Fix 1] 范数裁剪 (Norm Clipping)
        # 防御 VGAE 的 scale_factor 攻击
        update_norms = torch.norm(update_matrix, p=2, dim=1)
        median_norm = torch.median(update_norms)
        # 动态阈值：中位数的 1.5 倍
        threshold = median_norm * 1.5
        clip_factor = torch.clamp(threshold / (update_norms + 1e-6), max=1.0)
        # 将裁剪因子应用到矩阵 (广播)
        update_matrix = update_matrix * clip_factor.unsqueeze(1)

        # 2. 空间一致性 (结合余弦和距离)
        spatial_center = torch.median(update_matrix, dim=0).values

        # Cosine 相似度
        s_spat_cos = F.cosine_similarity(update_matrix, spatial_center.unsqueeze(0), dim=1)
        # Euclidean 距离 (转化为相似度: exp(-dist))
        dists = torch.norm(update_matrix - spatial_center, p=2, dim=1)
        sigma = torch.median(dists) + 1e-6
        s_spat_dist = torch.exp(-dists / sigma)

        # 综合空间分数
        s_spat = (s_spat_cos + 1) / 2 * 0.5 + s_spat_dist * 0.5

        # 3. 时间一致性 (Historical Expectation)
        if len(self.history_updates) > 0:
            expected_update = self.history_updates[-1].to(self.device)
            s_temp = F.cosine_similarity(update_matrix, expected_update.unsqueeze(0), dim=1)
        else:
            s_temp = torch.ones(len(updates)).to(self.device)

        # 4. 信任分数
        s_temp_norm = (s_temp + 1) / 2
        trust_scores = self.alpha * s_temp_norm + (1 - self.alpha) * s_spat

        # [Critical Fix 2] 软化 Softmax
        # 原来的 exp(*5) 太激进，改用温和的 softmax
        weights = F.softmax(trust_scores * 2.0, dim=0)

        # 诊断
        if client_types:
            b_w = [w.item() for w, t in zip(weights, client_types) if t == 'benign']
            m_w = [w.item() for w, t in zip(weights, client_types) if t != 'benign']
            self.diagnostics['avg_trust_benign'] = sum(b_w) / len(b_w) if b_w else 0
            self.diagnostics['avg_trust_malicious'] = sum(m_w) / len(m_w) if m_w else 0

        # 5. 聚合
        weighted_update_vec = torch.mv(update_matrix.t(), weights)
        self.history_updates.append(weighted_update_vec.detach().cpu())

        return self._unflatten(weighted_update_vec, updates[0])

    def _fedavg(self, updates):
        avg = {k: torch.zeros_like(v) for k, v in updates[0].items()}
        n = len(updates)
        for u in updates:
            for k, v in u.items(): avg[k] += v
        for k in avg: avg[k] /= n
        return avg

    def _unflatten(self, vec, template):
        res = {}
        idx = 0
        for k in sorted(template.keys()):
            v = template[k]
            if v.dtype in [torch.float32, torch.float64]:
                numel = v.numel()
                res[k] = vec[idx:idx + numel].view(v.shape)
                idx += numel
            else:
                res[k] = copy.deepcopy(v)
        return res