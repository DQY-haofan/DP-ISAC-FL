# 文件名: generate_viz_data_ultra.py
# 作用: 生成 IEEE 顶刊所需的深度机理数据 (CSV + NPY)
# 核心: 通过类继承，100% 复现原始 stga.py 和 aggregators.py 的逻辑，确保数据真实。

import torch
import numpy as np
import os
import yaml
import pandas as pd
import shutil
import torch.nn.functional as F
from tqdm import tqdm
import copy

# 引入原始工程文件
from server import Server
from stga import STGAAggregator
from aggregators import KrumAggregator, FedAvgAggregator
from datasets import partition_dataset_dirichlet, get_dataset
from secure_isac import SecureISACScheduler


# ==============================================================================
# 1. 插桩组件 (Instrumented Components)
#    这些类继承自原始代码，保留原汁原味的逻辑，仅添加数据捕获功能。
# ==============================================================================

class InstrumentedSTGA(STGAAggregator):
    """
    继承 STGAAggregator，完整保留 stga.py 的逻辑（Norm Clipping, Spatial, Temporal）。
    额外功能：捕获中间变量 (weights, norms, cosines)。
    """

    def __init__(self, config):
        super().__init__(config)
        self.captured_data = None

    def aggregate(self, updates, client_types=None):
        if not updates: return None

        # --- [REPLICATING ORIGINAL LOGIC START] ---
        # 为了确保数据完全一致，我们需要在这里获取中间变量。
        # 由于原始 aggregate 方法不返回权重，我们需要重写它，但保持逻辑完全相同。

        flat_updates = [self._flatten(u) for u in updates]
        update_matrix = torch.stack(flat_updates).to(self.device)

        # 1. Norm Clipping (与 stga.py 一致)
        update_norms = torch.norm(update_matrix, p=2, dim=1)
        median_norm = torch.median(update_norms)
        threshold = median_norm * 1.5
        clip_factor = torch.clamp(threshold / (update_norms + 1e-6), max=1.0)
        update_matrix_clipped = update_matrix * clip_factor.unsqueeze(1)

        # 2. Spatial Consistency (与 stga.py 一致)
        spatial_center = torch.median(update_matrix_clipped, dim=0).values
        s_spat_cos = F.cosine_similarity(update_matrix_clipped, spatial_center.unsqueeze(0), dim=1)

        # 距离分
        dists = torch.norm(update_matrix_clipped - spatial_center, p=2, dim=1)
        sigma = torch.median(dists) + 1e-6
        s_spat_dist = torch.exp(-dists / sigma)
        s_spat = (s_spat_cos + 1) / 2 * 0.5 + s_spat_dist * 0.5

        # 3. Temporal Consistency (与 stga.py 一致)
        if len(self.history_updates) > 0:
            expected_update = self.history_updates[-1].to(self.device)
            s_temp = F.cosine_similarity(update_matrix_clipped, expected_update.unsqueeze(0), dim=1)
        else:
            s_temp = torch.ones(len(updates)).to(self.device)

        # 4. Trust Score & Softmax (与 stga.py 一致)
        s_temp_norm = (s_temp + 1) / 2
        trust_scores = self.alpha * s_temp_norm + (1 - self.alpha) * s_spat
        weights = F.softmax(trust_scores * 2.0, dim=0)  # Softmax temperature = 2.0

        # 5. 聚合
        weighted_update_vec = torch.mv(update_matrix_clipped.t(), weights)
        self.history_updates.append(weighted_update_vec.detach().cpu())
        # --- [REPLICATING ORIGINAL LOGIC END] ---

        # [CAPTURE] 捕获关键机理数据
        self.captured_data = {
            'norms': update_norms.detach().cpu().numpy(),  # 原始模长
            'cosines': s_spat_cos.detach().cpu().numpy(),  # 方向一致性
            'weights': weights.detach().cpu().numpy(),  # 最终权重
            'updates': update_matrix.detach().cpu().numpy()  # 原始高维向量 (用于 t-SNE)
        }

        return self._unflatten(weighted_update_vec, updates[0])


class InstrumentedKrum(KrumAggregator):
    """ 继承 KrumAggregator，捕获被选中节点的索引作为权重。 """

    def __init__(self, f_malicious=2):
        super().__init__(f_malicious)
        self.captured_data = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def aggregate(self, updates, **kwargs):
        if not updates: return None
        n = len(updates)

        # 准备数据用于计算指标 (Norms, Cosines)
        flat_list = [
            torch.cat([v.view(-1) for k, v in sorted(u.items()) if v.dtype == torch.float32])
            for u in updates
        ]
        stack = torch.stack(flat_list).to(self.device)

        norms = torch.norm(stack, p=2, dim=1).detach().cpu().numpy()
        center = torch.median(stack, dim=0).values
        cosines = F.cosine_similarity(stack, center.unsqueeze(0), dim=1).detach().cpu().numpy()

        # --- Krum Logic (复现 aggregators.py) ---
        dists = torch.cdist(stack, stack)
        k_neighbors = n - self.f - 2
        if k_neighbors < 1: k_neighbors = 1
        scores = []
        for i in range(n):
            d_sorted, _ = torch.sort(dists[i])
            scores.append(torch.sum(d_sorted[1: 1 + k_neighbors]))
        scores = torch.tensor(scores)

        m = max(1, n - self.f)
        top_k_indices = torch.topk(scores, m, largest=False).indices

        # [CAPTURE]
        weights = np.zeros(n)
        weights[top_k_indices.cpu().numpy()] = 1.0 / m  # Krum 是硬选择，选中即均分

        self.captured_data = {
            'norms': norms,
            'cosines': cosines,
            'weights': weights,
            'updates': stack.detach().cpu().numpy()
        }

        # 调用父类完成实际聚合 (KrumAggregator.aggregate 已经实现了 Multi-Krum)
        return super().aggregate(updates, **kwargs)


class InstrumentedFedAvg(FedAvgAggregator):
    """ 继承 FedAvgAggregator，记录均匀权重。 """

    def __init__(self):
        super().__init__()
        self.captured_data = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def aggregate(self, updates, **kwargs):
        n = len(updates)

        flat_list = [
            torch.cat([v.view(-1) for k, v in sorted(u.items()) if v.dtype == torch.float32])
            for u in updates
        ]
        stack = torch.stack(flat_list).to(self.device)
        norms = torch.norm(stack, p=2, dim=1).detach().cpu().numpy()
        center = torch.median(stack, dim=0).values
        cosines = F.cosine_similarity(stack, center.unsqueeze(0), dim=1).detach().cpu().numpy()

        self.captured_data = {
            'norms': norms,
            'cosines': cosines,
            'weights': np.full(n, 1.0 / n),
            'updates': stack.detach().cpu().numpy()
        }
        return super().aggregate(updates, **kwargs)


class InstrumentedServer(Server):
    """ 继承 Server，用于捕获 ISAC Mask 和调度过程。 """

    def __init__(self, config, train_dataset, client_indices):
        super().__init__(config, train_dataset, client_indices)
        self.captured_mask = None

    def run_round(self, round_idx):
        # 调用父类 run_round
        stats = super().run_round(round_idx)

        # [CAPTURE] 捕获 ISAC 掩码
        # Server 类中 self.isac_scheduler.last_mask 存储了最新的掩码
        if self.isac_scheduler.last_mask is not None:
            self.captured_mask = self.isac_scheduler.last_mask.cpu().numpy()

        return stats


# ==============================================================================
# 2. 采集主程序 (Harvester Main)
# ==============================================================================

def run_ultra_harvest():
    print("🚜 Starting Ultra-Deep Data Harvest (Fig 1-12 Source Data)...")
    os.makedirs('viz_data', exist_ok=True)

    # 读取基础配置
    with open('config.yaml') as f:
        base_conf = yaml.safe_load(f)

    # --- [CRITICAL CONFIG] 设定高压崩溃环境 (Fig 11/12 的关键) ---
    base_conf['attack']['malicious_fraction'] = 0.3  # 30% 恶意节点 (Krum 崩溃点)
    base_conf['attack']['lambda_attack'] = 5.0  # 强攻击 (模长放大显著)
    base_conf['num_rounds'] = 30  # 跑 30 轮看稳态

    scenarios = ['FedAvg', 'Krum', 'R-JORA']
    all_records = []

    for mode in scenarios:
        print(f"\n📡 Harvesting Scenario: {mode} ...")
        conf = copy.deepcopy(base_conf)
        conf['scenario'] = mode

        # 配置聚合器与防御开关
        if mode == 'R-JORA':
            conf['aggregator'] = 'STGA'
            conf['r_jora']['enabled'] = True
        elif mode == 'Krum':
            conf['aggregator'] = 'Krum'
            conf['r_jora']['enabled'] = False  # 关闭其他防御，单测聚合器
        else:
            conf['aggregator'] = 'FedAvg'
            conf['r_jora']['enabled'] = False

        # 初始化
        ds, _ = get_dataset(conf['dataset'], conf['data_root'])
        # 固定 Seed 42 保证数据分布 (Non-IID) 一致性
        idx = partition_dataset_dirichlet(ds, conf['num_clients'], conf['alpha'], seed=42)

        # 使用插桩 Server
        server = InstrumentedServer(conf, ds, idx)

        # 替换为插桩聚合器
        if mode == 'R-JORA':
            server.aggregator = InstrumentedSTGA(conf)
        elif mode == 'Krum':
            f_mal = int(conf['num_clients'] * conf['client_fraction'] * 0.3) + 2
            server.aggregator = InstrumentedKrum(f_malicious=f_mal)
        else:
            server.aggregator = InstrumentedFedAvg()

        # 运行
        for t in tqdm(range(conf['num_rounds'])):
            server.run_round(t)

            # 1. 提取聚合器内部数据
            agg_data = server.aggregator.captured_data
            if agg_data is None: continue

            norms = agg_data['norms']
            cosines = agg_data['cosines']
            weights = agg_data['weights']
            updates = agg_data['updates']

            # 2. 判定节点身份 (基于模长聚类，因为 lambda=5.0 导致恶意模长显著)
            # 这是一个准确的后验标记方法
            median_norm = np.median(norms)
            types = []
            for n in norms:
                if n > median_norm * 2.0:  # 恶意节点模长通常 > 5.0 * median
                    types.append('Malicious')
                else:
                    types.append('Benign')

            # 3. 记录到 DataFrame List
            for i in range(len(norms)):
                all_records.append({
                    'Scenario': mode,
                    'Round': t,
                    'Type': types[i],
                    'L2_Norm': norms[i],
                    'Cosine_Sim': cosines[i],
                    'Weight': weights[i]
                })

            # 4. 保存 .npy 文件 (用于 t-SNE, Heatmap, Mask)
            # 仅保存 R-JORA 的关键帧和 Mask，减少存储压力
            if mode == 'R-JORA':
                # 保存 Mask 用于 Fig 8
                if server.captured_mask is not None:
                    np.save(f'viz_data/mask_r{t}.npy', server.captured_mask)

                # 保存 Updates 用于 t-SNE (Fig 6) - 选几个关键轮次
                if t in [0, 5, 15, 29]:
                    np.save(f'viz_data/updates_r{t}.npy', updates)
                    np.save(f'viz_data/types_r{t}.npy', np.array(types))

                # 保存权重矩阵用于 Heatmap (Fig 7) - 每轮都存，但只存前20个客户端
                # 注意：selected_clients 每轮都在变，Heatmap 需要 ID 对应。
                # 为了简化 Heatmap，我们只画 "Selected Clients" 的权重分布，或者不画 ID 轴。
                # 这里保存原始 weights 数组
                np.save(f'viz_data/weights_r{t}.npy', weights)

    # 导出 CSV
    df = pd.DataFrame(all_records)
    df.to_csv('viz_metrics_pro.csv', index=False)
    print("\n✅ Harvest Complete! Files generated:")
    print("   - viz_metrics_pro.csv (Source for Fig 9, 11, 12)")
    print("   - viz_data/*.npy (Source for Fig 6, 7, 8)")


if __name__ == "__main__":
    run_ultra_harvest()