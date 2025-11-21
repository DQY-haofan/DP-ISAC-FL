# ============================================================
# 脚本名: generate_viz_universal.py
# 作用: 1. 采集所有可视化数据 (.npy 用于绘图)
#       2. 导出 viz_metrics.csv (用于人工检查和调试)
# 修复: 在 Clipping 之前捕获 Norms，解决 Fig 11 问题
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


# --- 间谍聚合器 (数据探针) ---
class UniversalProbe(STGAAggregator):
    def __init__(self, config):
        super().__init__(config)
        # 临时存储本轮数据
        self.round_data = {
            'norms': None, 'cosines': None, 'weights': None, 'updates': None
        }

    def aggregate(self, updates, client_types=None):
        if not updates: return None

        # 1. 展平数据
        flat_updates = [self._flatten(u) for u in updates]
        update_matrix = torch.stack(flat_updates).to(self.device)

        # [Fix for Fig 11] 在任何防御处理之前，捕获原始特征！
        raw_norms = torch.norm(update_matrix, p=2, dim=1)

        # 计算相对于原始中心的余弦相似度 (看攻击者伪装得像不像)
        raw_center = torch.median(update_matrix, dim=0).values
        raw_cosines = F.cosine_similarity(update_matrix, raw_center.unsqueeze(0), dim=1)

        # 存入缓存
        self.round_data['norms'] = raw_norms.detach().cpu().numpy()
        self.round_data['cosines'] = raw_cosines.detach().cpu().numpy()
        self.round_data['updates'] = update_matrix.detach().cpu().numpy()

        # --- 执行正常的 STGA 逻辑 (以计算权重) ---
        # 1. Clipping
        median_norm = torch.median(raw_norms)
        threshold = median_norm * 1.5
        clip_factor = torch.clamp(threshold / (raw_norms + 1e-6), max=1.0)
        update_matrix_clipped = update_matrix * clip_factor.unsqueeze(1)

        # 2. Spatial Score
        spatial_center = torch.median(update_matrix_clipped, dim=0).values
        s_spat_cos = F.cosine_similarity(update_matrix_clipped, spatial_center.unsqueeze(0), dim=1)
        dists = torch.norm(update_matrix_clipped - spatial_center, p=2, dim=1)
        sigma = torch.median(dists) + 1e-6
        s_spat_dist = torch.exp(-dists / sigma)
        s_spat = (s_spat_cos + 1) / 2 * 0.5 + s_spat_dist * 0.5

        # 3. Temporal Score
        if len(self.history_updates) > 0:
            expected_update = self.history_updates[-1].to(self.device)
            s_temp = F.cosine_similarity(update_matrix_clipped, expected_update.unsqueeze(0), dim=1)
        else:
            s_temp = torch.ones(len(updates)).to(self.device)
        s_temp_norm = (s_temp + 1) / 2

        # 4. Final Weights
        trust_scores = self.alpha * s_temp_norm + (1 - self.alpha) * s_spat
        weights = F.softmax(trust_scores * 2.0, dim=0)

        # 存入缓存
        self.round_data['weights'] = weights.detach().cpu().numpy()

        return super().aggregate(updates, client_types)


# --- 主程序 ---
def run_universal_harvest():
    print("🚀 Starting Universal Data Harvest...")

    # 1. 环境清理与配置
    if os.path.exists('viz_data'): shutil.rmtree('viz_data')
    os.makedirs('viz_data', exist_ok=True)

    with open('config.yaml') as f:
        conf = yaml.safe_load(f)

    # 强制配置: R-JORA + 强攻击 (为了让特征明显)
    if 'r_jora' not in conf: conf['r_jora'] = {}
    conf['r_jora'].update({'enabled': True, 'enable_stga': True, 'enable_optimal_dp': True, 'enable_secure_isac': True})
    if 'stga_alpha' not in conf['r_jora']: conf['r_jora']['stga_alpha'] = 0.5

    conf['num_rounds'] = 20
    conf['scenario'] = 'Viz_Harvest'
    conf['aggregator'] = 'STGA'
    # 使用 Lambda=5.0 确保 Fig 11 中恶意节点 Norm 飞起来
    conf['attack'] = {'malicious_fraction': 0.2, 'lambda_attack': 5.0, 'tau_sim': 0.5, 't_vgae': 1, 'q_eaves': 0.8,
                      'eaves_sigma': 0.005, 'vgae_epochs': 5, 'vgae_lr': 0.01, 'latent_dim': 16}

    if torch.cuda.is_available(): conf['device'] = 'cuda'

    # 2. 初始化
    ds, _ = get_dataset(conf['dataset'], conf['data_root'])
    idx = partition_dataset_dirichlet(ds, conf['num_clients'], conf['alpha'], seed=42)
    server = Server(conf, ds, idx)
    server.aggregator = UniversalProbe(conf)  # 注入探针

    # 全局 CSV 数据容器
    csv_records = []

    # 3. 运行循环
    print(f"   Running {conf['num_rounds']} rounds with Lambda={conf['attack']['lambda_attack']}...")
    for t in range(conf['num_rounds']):
        server.run_round(t)

        # 提取探针数据
        data = server.aggregator.round_data
        if data['weights'] is None: continue

        # 获取本轮选中的客户端 ID (假设 run_round 内部顺序一致)
        # 这里我们要稍微 hack 一下：server.run_round 里的 selected_clients 是局部变量。
        # 但我们知道 updates 的顺序就是 selected_clients 的顺序。
        # 且我们知道 types。

        # 重新推导 Client Type (通过权重推断：权重极低的大概率是恶意，但这不严谨)
        # 严谨做法：我们需要 server 告诉我们这轮选了谁。
        # 简化做法：我们在 CSV 里只记录 'Type' (Malicious/Benign) 而不记录具体 ID，这足够画图了。

        # 这里的 data['weights'] 长度为 K (比如10)。
        # 我们怎么知道哪个是 Malicious？
        # 回到 server.py, malicious_updates 是后加进去的。
        # 通常 server.run_round 里: benign_updates + malicious_updates
        # 所以前 N 个是 Benign，后 M 个是 Malicious。

        num_mal = int(len(data['weights']) * conf['attack']['malicious_fraction'])  # 2
        num_ben = len(data['weights']) - num_mal  # 8

        # 构造类型标签列表
        current_types = ['Benign'] * num_ben + ['Malicious'] * num_mal

        # 保存到 NPY (用于 plot_ieee 脚本)
        np.save(f'viz_data/weights_r{t}.npy', data['weights'])
        np.save(f'viz_data/norms_r{t}.npy', data['norms'])
        np.save(f'viz_data/cosines_r{t}.npy', data['cosines'])
        np.save(f'viz_data/client_types_r{t}.npy', np.array(current_types))  # 每轮存一份类型

        # 保存到 CSV 列表
        for i in range(len(data['weights'])):
            csv_records.append({
                'Round': t,
                'Client_Index_In_Batch': i,
                'Type': current_types[i],
                'L2_Norm': data['norms'][i],
                'Cosine_Sim': data['cosines'][i],
                'Weight': data['weights'][i]
            })

        # 保存 t-SNE 向量 (仅关键帧)
        if t in [0, 5, 10, 19]:
            np.save(f'viz_data/updates_r{t}.npy', data['updates'])
            # 对应的类型也存一下，方便 t-SNE 画图脚本读取
            np.save(f'viz_data/client_types_tsne_r{t}.npy', np.array(current_types))

    # 4. 导出 CSV
    df = pd.DataFrame(csv_records)
    df.to_csv('viz_metrics.csv', index=False)
    print(f"✅ Saved 'viz_metrics.csv' ({len(df)} rows).")
    print("   Columns: Round, Type, L2_Norm, Cosine_Sim, Weight")

    # 顺便生成一个 client_types.npy 给旧绘图脚本兼容
    # 注意：旧脚本可能假设这里存的是所有 100 个客户端的类型。
    # 为了兼容，我们生成一个全量的
    all_types = np.array(['Malicious' if i in server.malicious_ids else 'Benign' for i in range(conf['num_clients'])])
    np.save('viz_data/client_types.npy', all_types)


if __name__ == "__main__":
    run_universal_harvest()