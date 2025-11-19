# 文件名: aggregators.py
# 作用: 基线聚合器 (Krum, Median, FedAvg)。
# 版本: Final

import torch


class BaseAggregator:
    def aggregate(self, updates, **kwargs): raise NotImplementedError


class FedAvgAggregator(BaseAggregator):
    def aggregate(self, updates, **kwargs):
        if not updates: return None
        avg = {k: torch.zeros_like(v) for k, v in updates[0].items()}
        n = len(updates)
        for u in updates:
            for k, v in u.items(): avg[k] += v
        for k in avg: avg[k] /= n
        return avg


class MedianAggregator(BaseAggregator):
    def aggregate(self, updates, **kwargs):
        if not updates: return None
        res = {}
        for k in updates[0].keys():
            stacked = torch.stack([u[k] for u in updates])
            res[k] = torch.median(stacked, dim=0).values
        return res


class KrumAggregator(BaseAggregator):
    def __init__(self, f_malicious=2):
        self.f = f_malicious

    def aggregate(self, updates, **kwargs):
        if not updates: return None
        if len(updates) < self.f + 2: return MedianAggregator().aggregate(updates)

        flat_updates = [torch.cat([v.view(-1) for k, v in sorted(u.items()) if v.dtype == torch.float32]) for u in
                        updates]
        stack = torch.stack(flat_updates)
        dists = torch.cdist(stack, stack)

        scores = []
        k = len(updates) - self.f - 2
        for i in range(len(updates)):
            d_sorted, _ = torch.sort(dists[i])
            scores.append(torch.sum(d_sorted[1:1 + k]))  # Exclude self

        best_idx = torch.argmin(torch.tensor(scores)).item()
        return updates[best_idx]