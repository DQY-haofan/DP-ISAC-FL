# 文件名: stga.py
# 作用: R-JORA 子模块 1 - 时空图聚合器。计算信任分数，过滤异常更新。
# 版本: Final

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

        # 2. 空间一致性 (Geometric Median Approx)
        spatial_center = torch.median(update_matrix, dim=0).values
        s_spat = F.cosine_similarity(update_matrix, spatial_center.unsqueeze(0), dim=1)

        # 3. 时间一致性 (Historical Expectation)
        if len(self.history_updates) > 0:
            expected_update = self.history_updates[-1].to(self.device)
            s_temp = F.cosine_similarity(update_matrix, expected_update.unsqueeze(0), dim=1)
        else:
            s_temp = torch.ones(len(updates)).to(self.device)

        # 4. 信任分数
        s_spat_norm = (s_spat + 1) / 2
        s_temp_norm = (s_temp + 1) / 2
        trust_scores = self.alpha * s_temp_norm + (1 - self.alpha) * s_spat_norm

        # 放大差异 (Softmax-like)
        trust_scores = torch.exp(trust_scores * 5)
        weights = trust_scores / trust_scores.sum()

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