# ============================================================
# 脚本名: generate_viz_data.py
# 作用: 采集用于高级可视化(t-SNE, Heatmap)的高维数据
# 版本: Fixed (Solved KeyError: 'stga_alpha')
# ============================================================
import torch
import numpy as np
import os
import copy
import yaml
import torch.nn.functional as F
from runner import SimulationRunner
from server import Server
from stga import STGAAggregator
from datasets import partition_dataset_dirichlet, get_dataset


# 1. 定义一个“间谍”聚合器，用来把内部权重偷出来
class InstrumentedSTGA(STGAAggregator):
    def __init__(self, config):
        super().__init__(config)
        self.captured_weights = None
        self.captured_updates = None  # Flattened

    def aggregate(self, updates, client_types=None):
        # 复用父类的预处理逻辑
        if not updates: return None

        # 重新实现核心打分逻辑以捕获数据 (保持与原版 stga.py 完全一致)
        flat_updates = [self._flatten(u) for u in updates]
        update_matrix = torch.stack(flat_updates).to(self.device)

        # 保存用于 t-SNE 的原始向量 (只存 CPU 版以省显存)
        self.captured_updates = update_matrix.detach().cpu().numpy()

        # --- STGA 逻辑复现 (为了获取 weights) ---
        # 1. Norm Clipping
        update_norms = torch.norm(update_matrix, p=2, dim=1)
        median_norm = torch.median(update_norms)
        threshold = median_norm * 1.5
        clip_factor = torch.clamp(threshold / (update_norms + 1e-6), max=1.0)
        update_matrix_clipped = update_matrix * clip_factor.unsqueeze(1)

        # 2. Spatial
        spatial_center = torch.median(update_matrix_clipped, dim=0).values
        s_spat_cos = F.cosine_similarity(update_matrix_clipped, spatial_center.unsqueeze(0), dim=1)
        dists = torch.norm(update_matrix_clipped - spatial_center, p=2, dim=1)
        sigma = torch.median(dists) + 1e-6
        s_spat_dist = torch.exp(-dists / sigma)
        s_spat = (s_spat_cos + 1) / 2 * 0.5 + s_spat_dist * 0.5

        # 3. Temporal
        if len(self.history_updates) > 0:
            expected_update = self.history_updates[-1].to(self.device)
            s_temp = F.cosine_similarity(update_matrix_clipped, expected_update.unsqueeze(0), dim=1)
        else:
            s_temp = torch.ones(len(updates)).to(self.device)
        s_temp_norm = (s_temp + 1) / 2

        # 4. Weights
        # [Fix] 确保 self.alpha 存在 (父类已初始化)
        trust_scores = self.alpha * s_temp_norm + (1 - self.alpha) * s_spat
        weights = F.softmax(trust_scores * 2.0, dim=0)

        # 捕获权重！
        self.captured_weights = weights.detach().cpu().numpy()

        # 调用父类完成实际聚合 (确保训练不受影响)
        return super().aggregate(updates, client_types)


# 2. 运行采集流程
def run_harvest():
    print("🚜 Starting Visualization Data Harvest (20 Rounds)...")
    os.makedirs('viz_data', exist_ok=True)

    # 读取配置
    with open('config.yaml') as f:
        conf = yaml.safe_load(f)

    # [Critical Fix] 不要覆盖整个 r_jora 字典，而是更新它
    # 这样可以保留 config.yaml 里的 stga_alpha
    if 'r_jora' not in conf: conf['r_jora'] = {}
    conf['r_jora'].update({
        'enabled': True,
        'enable_stga': True,
        'enable_optimal_dp': True,
        'enable_secure_isac': True
    })

    # 兜底：万一 config.yaml 里真的没有，赋默认值
    if 'stga_alpha' not in conf['r_jora']:
        conf['r_jora']['stga_alpha'] = 0.5

    conf['num_rounds'] = 20  # 只跑20轮
    conf['scenario'] = 'Viz_Harvest'
    conf['aggregator'] = 'STGA'

    # 使用较强的攻击来凸显防御效果
    conf['attack'] = {'malicious_fraction': 0.2, 'lambda_attack': 3.0, 'tau_sim': 0.5, 't_vgae': 1, 'q_eaves': 0.8,
                      'eaves_sigma': 0.005, 'vgae_epochs': 5, 'vgae_lr': 0.01, 'latent_dim': 16}

    if torch.cuda.is_available(): conf['device'] = 'cuda'

    # 初始化环境
    ds, _ = get_dataset(conf['dataset'], conf['data_root'])
    idx = partition_dataset_dirichlet(ds, conf['num_clients'], conf['alpha'], seed=42)

    server = Server(conf, ds, idx)

    # [注入] 替换聚合器为间谍聚合器
    # 注意：必须重新传入完整的 conf
    server.aggregator = InstrumentedSTGA(conf)

    # 记录客户端类型
    client_types = np.array(
        ['Malicious' if i in server.malicious_ids else 'Benign' for i in range(conf['num_clients'])])
    np.save('viz_data/client_types.npy', client_types)

    # 循环
    for t in range(conf['num_rounds']):
        print(f"   Harvesting Round {t}...")
        server.run_round(t)

        # A. 保存权重 (Heatmap)
        if server.aggregator.captured_weights is not None:
            np.save(f'viz_data/weights_r{t}.npy', server.aggregator.captured_weights)

        # B. 保存更新向量 (t-SNE) - 仅保存关键轮次
        if t in [0, 5, 10, 19]:
            np.save(f'viz_data/updates_r{t}.npy', server.aggregator.captured_updates)

        # C. 保存 ISAC 掩码
        if hasattr(server.isac_scheduler, 'last_mask') and server.isac_scheduler.last_mask is not None:
            np.save(f'viz_data/mask_r{t}.npy', server.isac_scheduler.last_mask.cpu().numpy())

    print("✅ Data Harvest Complete. Check 'viz_data/' folder.")


if __name__ == "__main__":
    run_harvest()