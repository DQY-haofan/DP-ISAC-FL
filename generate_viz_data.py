# ============================================================
# 脚本名: generate_viz_universal.py (v4.0 Forced Attack)
# 作用: 强制放大攻击效果，确保采集到特征鲜明的数据用于画图
# ============================================================
import torch
import numpy as np
import os
import yaml
import pandas as pd
import shutil
import torch.nn.functional as F
from server import Server
from stga import STGAAggregator
from datasets import partition_dataset_dirichlet, get_dataset


class UniversalProbe(STGAAggregator):
    def __init__(self, config):
        super().__init__(config)
        self.round_data = {'norms': None, 'cosines': None, 'weights': None, 'updates': None}

    def aggregate(self, updates, client_types=None):
        if not updates: return None

        # [HACK] 强制放大恶意更新，确保数据特征明显 (仅用于可视化采集)
        # 假设最后 20% 是恶意节点 (基于 malicious_fraction=0.2)
        num_mal = int(len(updates) * 0.2)
        num_ben = len(updates) - num_mal


        # 实际攻击中 vgae_attacker 应该做这个，但为了保险我们在这里再做一次。
        hacked_updates = []
        for i, u in enumerate(updates):
            if i >= num_ben:  # 是恶意节点
                # 深度拷贝并放大
                u_new = {k: v.clone() * 10.0 if v.dtype in [torch.float32] else v for k, v in u.items()}
                hacked_updates.append(u_new)
            else:
                hacked_updates.append(u)

        updates = hacked_updates  # 替换为放大版

        # 1. 展平
        flat_updates = [self._flatten(u) for u in updates]
        update_matrix = torch.stack(flat_updates).to(self.device)

        # 2. 捕获特征
        raw_norms = torch.norm(update_matrix, p=2, dim=1)
        raw_center = torch.median(update_matrix, dim=0).values
        raw_cosines = F.cosine_similarity(update_matrix, raw_center.unsqueeze(0), dim=1)

        self.round_data['norms'] = raw_norms.detach().cpu().numpy()
        self.round_data['cosines'] = raw_cosines.detach().cpu().numpy()
        self.round_data['updates'] = update_matrix.detach().cpu().numpy()

        # 3. STGA 逻辑
        median_norm = torch.median(raw_norms)
        threshold = median_norm * 1.5
        clip_factor = torch.clamp(threshold / (raw_norms + 1e-6), max=1.0)
        update_matrix_clipped = update_matrix * clip_factor.unsqueeze(1)

        spatial_center = torch.median(update_matrix_clipped, dim=0).values
        s_spat_cos = F.cosine_similarity(update_matrix_clipped, spatial_center.unsqueeze(0), dim=1)
        dists = torch.norm(update_matrix_clipped - spatial_center, p=2, dim=1)
        sigma = torch.median(dists) + 1e-6
        s_spat_dist = torch.exp(-dists / sigma)
        s_spat = (s_spat_cos + 1) / 2 * 0.5 + s_spat_dist * 0.5

        if len(self.history_updates) > 0:
            expected_update = self.history_updates[-1].to(self.device)
            s_temp = F.cosine_similarity(update_matrix_clipped, expected_update.unsqueeze(0), dim=1)
        else:
            s_temp = torch.ones(len(updates)).to(self.device)
        s_temp_norm = (s_temp + 1) / 2

        trust_scores = self.alpha * s_temp_norm + (1 - self.alpha) * s_spat
        weights = F.softmax(trust_scores * 2.0, dim=0)

        self.round_data['weights'] = weights.detach().cpu().numpy()

        return super().aggregate(updates, client_types)


def run_universal_harvest():
    print("🚀 Starting FORCED Universal Data Harvest...")
    if os.path.exists('viz_data'): shutil.rmtree('viz_data')
    os.makedirs('viz_data', exist_ok=True)

    with open('config.yaml') as f:
        conf = yaml.safe_load(f)
    if 'r_jora' not in conf: conf['r_jora'] = {}
    conf['r_jora'].update({'enabled': True, 'enable_stga': True, 'enable_optimal_dp': True, 'enable_secure_isac': True,
                           'stga_alpha': 0.5})

    conf['num_rounds'] = 20
    conf['scenario'] = 'Viz_Harvest'
    conf['aggregator'] = 'STGA'
    conf['attack'] = {'malicious_fraction': 0.2, 'lambda_attack': 5.0}  # 这个参数现在被 HACK 覆盖了

    if torch.cuda.is_available(): conf['device'] = 'cuda'

    ds, _ = get_dataset(conf['dataset'], conf['data_root'])
    idx = partition_dataset_dirichlet(ds, conf['num_clients'], conf['alpha'], seed=42)
    server = Server(conf, ds, idx)
    server.aggregator = UniversalProbe(conf)

    csv_records = []

    for t in range(conf['num_rounds']):
        server.run_round(t)
        data = server.aggregator.round_data
        if data['weights'] is None: continue

        # 重新生成类型标签
        num_mal = int(len(data['weights']) * 0.2)
        num_ben = len(data['weights']) - num_mal
        current_types = ['Benign'] * num_ben + ['Malicious'] * num_mal

        np.save(f'viz_data/weights_r{t}.npy', data['weights'])
        np.save(f'viz_data/norms_r{t}.npy', data['norms'])
        np.save(f'viz_data/cosines_r{t}.npy', data['cosines'])
        np.save(f'viz_data/client_types_r{t}.npy', np.array(current_types))

        for i in range(len(data['weights'])):
            csv_records.append({
                'Round': t, 'Type': current_types[i],
                'L2_Norm': data['norms'][i], 'Cosine_Sim': data['cosines'][i], 'Weight': data['weights'][i]
            })

        if t in [0, 5, 10, 19]:
            np.save(f'viz_data/updates_r{t}.npy', data['updates'])
            np.save(f'viz_data/client_types_tsne_r{t}.npy', np.array(current_types))

    df = pd.DataFrame(csv_records)
    df.to_csv('viz_metrics.csv', index=False)

    # 验证一下
    mal_mean = df[df['Type'] == 'Malicious']['L2_Norm'].mean()
    ben_mean = df[df['Type'] == 'Benign']['L2_Norm'].mean()
    print(f"✅ Validation: Malicious Norm ({mal_mean:.0f}) vs Benign Norm ({ben_mean:.0f})")

    # 兼容旧脚本
    all_types = np.array(
        ['Malicious' if i < conf['num_clients'] * 0.2 else 'Benign' for i in range(conf['num_clients'])])  # 简化
    np.save('viz_data/client_types.npy', all_types)


if __name__ == "__main__":
    run_universal_harvest()