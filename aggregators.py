# 文件名: aggregators.py
# 作用: 基线聚合器 (FedAvg, Median, Multi-Krum)。
# 版本: Final (Upgraded Krum to Multi-Krum for Non-IID robustness)

import torch


class BaseAggregator:
    def aggregate(self, updates, **kwargs): raise NotImplementedError


class FedAvgAggregator(BaseAggregator):
    def aggregate(self, updates, **kwargs):
        if not updates: return None
        # 初始化累加器
        avg = {k: torch.zeros_like(v) for k, v in updates[0].items()}
        n = len(updates)

        for u in updates:
            for k, v in u.items():
                avg[k] += v

        # 平均
        for k in avg:
            avg[k] /= n
        return avg


class MedianAggregator(BaseAggregator):
    def aggregate(self, updates, **kwargs):
        if not updates: return None
        res = {}
        # Coordinate-wise Median: 对每个参数维度取中位数
        for k in updates[0].keys():
            # Stack 所有的 update: [n_clients, param_shape]
            stacked = torch.stack([u[k] for u in updates])
            res[k] = torch.median(stacked, dim=0).values
        return res


class KrumAggregator(BaseAggregator):
    """
    Multi-Krum Aggregator
    相比 Single-Krum，它选择 m 个最好的更新并求平均。
    这在 Non-IID 场景下至关重要，因为它保留了平均（Averaging）带来的偏差消除能力。
    """

    def __init__(self, f_malicious=2):
        self.f = f_malicious

    def aggregate(self, updates, **kwargs):
        if not updates: return None
        n = len(updates)

        # 如果更新数太少，无法执行 Krum (需要 n >= 2f + 3 理论上，这里放宽一点)
        # 回退到 Median，比报错好
        if n < self.f + 2:
            return MedianAggregator().aggregate(updates)

        # 1. 展平更新向量 [n, d]
        flat_updates = [
            torch.cat([v.view(-1) for k, v in sorted(u.items()) if v.dtype == torch.float32])
            for u in updates
        ]
        stack = torch.stack(flat_updates)

        # 2. 计算成对欧氏距离矩阵 [n, n]
        dists = torch.cdist(stack, stack)

        # 3. 计算 Krum Score
        # 对于每个 i，累加距离它最近的 n - f - 2 个邻居的距离
        # k_neighbors 是用来打分的邻居数量
        k_neighbors = n - self.f - 2
        if k_neighbors < 1: k_neighbors = 1

        scores = []
        for i in range(n):
            # 排序距离 (从小到大)
            d_sorted, _ = torch.sort(dists[i])
            # d_sorted[0] 是自己到自己(0)，从索引 1 开始累加前 k 个邻居
            # sum(d_sorted[1:1+k])
            scores.append(torch.sum(d_sorted[1: 1 + k_neighbors]))

        scores = torch.tensor(scores)

        # 4. Multi-Krum 选择逻辑
        # 我们不仅仅选 1 个，而是选 n - f 个最好的（假设 f 个是坏的，剩下都是好的）
        m = max(1, n - self.f)

        # 获取分数最小的 m 个索引 (largest=False)
        top_k_indices = torch.topk(scores, m, largest=False).indices

        # 5. 提取选中的更新
        selected_updates = [updates[i] for i in top_k_indices]

        # 6. 对选中的更新做平均 (FedAvg)
        return FedAvgAggregator().aggregate(selected_updates)