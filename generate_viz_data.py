# ============================================================
# 脚本名: generate_viz_data_clean.py (v5.0 Final & Clean)
# 修复: 1. KeyError 'latent_dim' (通过 .update 保留配置)
#       2. 移除手动放大 Hack (还原真实攻击效果)
# 作用: 采集真实、准确的高维数据用于画图
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


# --- 数据探针 (只负责记录，不修改数据) ---
class UniversalProbe(STGAAggregator):
    def __init__(self, config):
        super().__init__(config)
        self.round_data = {'norms': None, 'cosines': None, 'weights': None, 'updates': None}

    def aggregate(self, updates, client_types=None):
        if not updates: return None

        # 1. 展平数据
        flat_updates = [self._flatten(u) for u in updates]
        update_matrix = torch.stack(flat_updates).to(self.device)

        # 2. [Capture] 捕获原始特征 (真实攻击效果)
        raw_norms = torch.norm(update_matrix, p=2, dim=1)
        raw_center = torch.median(update_matrix, dim=0).values
        raw_cosines = F.cosine_similarity(update_matrix, raw_center.unsqueeze(0), dim=1)

        # 存入缓存
        self.round_data['norms'] = raw_norms.detach().cpu().numpy()
        self.round_data['cosines'] = raw_cosines.detach().cpu().numpy()
        self.round_data['updates'] = update_matrix.detach().cpu().numpy()

        # 3. STGA 正常逻辑 (计算权重)
        # Clipping
        median_norm = torch.median(raw_norms)
        threshold = median_norm * 1.5
        clip_factor = torch.clamp(threshold / (raw_norms + 1e-6), max=1.0)
        update_matrix_clipped = update_matrix * clip_factor.unsqueeze(1)

        # Spatial
        spatial_center = torch.median(update_matrix_clipped, dim=0).values
        s_spat_cos = F.cosine_similarity(update_matrix_clipped, spatial_center.unsqueeze(0), dim=1)
        dists = torch.norm(update_matrix_clipped - spatial_center, p=2, dim=1)
        sigma = torch.median(dists) + 1e-6
        s_spat_dist = torch.exp(-dists / sigma)
        s_spat = (s_spat_cos + 1) / 2 * 0.5 + s_spat_dist * 0.5

        # Temporal
        if len(self.history_updates) > 0:
            expected_update = self.history_updates[-1].to(self.device)
            s_temp = F.cosine_similarity(update_matrix_clipped, expected_update.unsqueeze(0), dim=1)
        else:
            s_temp = torch.ones(len(updates)).to(self.device)
        s_temp_norm = (s_temp + 1) / 2

        # Weights
        trust_scores = self.alpha * s_temp_norm + (1 - self.alpha) * s_spat
        weights = F.softmax(trust_scores * 2.0, dim=0)

        self.round_data['weights'] = weights.detach().cpu().numpy()

        # 调用父类 (不影响训练流)
        return super().aggregate(updates, client_types)


# --- 主程序 ---
def run_universal_harvest():
    print("🚀 Starting Authentic Data Harvest (20 Rounds)...")

    if os.path.exists('viz_data'): shutil.rmtree('viz_data')
    os.makedirs('viz_data', exist_ok=True)

    # 1. 正确加载配置 (Fix KeyError)
    with open('config.yaml') as f:
        conf = yaml.safe_load(f)

    # 确保 R-JORA 开启
    if 'r_jora' not in conf: conf['r_jora'] = {}
    conf['r_jora'].update({'enabled': True, 'enable_stga': True, 'enable_optimal_dp': True, 'enable_secure_isac': True})
    # 兜底 alpha
    if 'stga_alpha' not in conf['r_jora']: conf['r_jora']['stga_alpha'] = 0.5

    conf['num_rounds'] = 20
    conf['scenario'] = 'Viz_Harvest'
    conf['aggregator'] = 'STGA'

    # [Fix] 使用 .update() 而不是覆盖，保留 latent_dim 等默认参数
    # Lambda=5.0 对应你 run_all 里的高强度攻击
    if 'attack' not in conf: conf['attack'] = {}
    conf['attack'].update({'malicious_fraction': 0.2, 'lambda_attack': 5.0})

    if torch.cuda.is_available(): conf['device'] = 'cuda'

    # 2. 初始化环境
    print("   Loading data...")
    ds, _ = get_dataset(conf['dataset'], conf['data_root'])
    idx = partition_dataset_dirichlet(ds, conf['num_clients'], conf['alpha'], seed=42)
    server = Server(conf, ds, idx)

    # 注入探针
    server.aggregator = UniversalProbe(conf)

    csv_records = []

    print(f"   Running with Lambda={conf['attack']['lambda_attack']} (Real Attack)")

    for t in range(conf['num_rounds']):
        # 运行一轮
        server.run_round(t)

        # 提取数据
        data = server.aggregator.round_data
        if data['weights'] is None: continue

        # 推断类型 (假设 server.py 逻辑: benign在前, malicious在后)
        num_mal = int(len(data['weights']) * conf['attack']['malicious_fraction'])
        num_ben = len(data['weights']) - num_mal
        current_types = ['Benign'] * num_ben + ['Malicious'] * num_mal

        # 保存 NPY (覆盖式保存，用于绘图脚本)
        np.save(f'viz_data/weights_r{t}.npy', data['weights'])
        np.save(f'viz_data/norms_r{t}.npy', data['norms'])
        np.save(f'viz_data/cosines_r{t}.npy', data['cosines'])
        np.save(f'viz_data/client_types_r{t}.npy', np.array(current_types))

        # 记录到 CSV
        for i in range(len(data['weights'])):
            csv_records.append({
                'Round': t, 'Type': current_types[i],
                'L2_Norm': data['norms'][i],
                'Cosine_Sim': data['cosines'][i],
                'Weight': data['weights'][i]
            })

        # 关键帧保存 Update 向量
        if t in [0, 5, 10, 19]:
            np.save(f'viz_data/updates_r{t}.npy', data['updates'])
            np.save(f'viz_data/client_types_tsne_r{t}.npy', np.array(current_types))

        # 保存掩码
        if hasattr(server.isac_scheduler, 'last_mask') and server.isac_scheduler.last_mask is not None:
            np.save(f'viz_data/mask_r{t}.npy', server.isac_scheduler.last_mask.cpu().numpy())

    # 导出 CSV
    df = pd.DataFrame(csv_records)
    df.to_csv('viz_metrics.csv', index=False)

    # 验证数据合理性
    mal_mean = df[df['Type'] == 'Malicious']['L2_Norm'].mean()
    ben_mean = df[df['Type'] == 'Benign']['L2_Norm'].mean()
    print(f"✅ Validation: Malicious Norm ({mal_mean:.1f}) vs Benign Norm ({ben_mean:.1f})")
    if mal_mean > ben_mean * 2:
        print("   -> 攻击特征显著，Fig 11 将会非常漂亮。")
    else:
        print("   -> 警告：攻击特征不明显，请检查 lambda_attack 是否生效。")

    # 兼容性文件
    all_types = np.array(
        ['Malicious' if i < conf['num_clients'] * 0.2 else 'Benign' for i in range(conf['num_clients'])])
    np.save('viz_data/client_types.npy', all_types)


if __name__ == "__main__":
    run_universal_harvest()