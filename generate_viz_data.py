# ============================================================
# 脚本名: generate_viz_data_pro.py (v6.0 Multi-Scenario)
# 作用: 1. 对比采集 (Vulnerable vs R-JORA)
#       2. 进度条显示
#       3. 导出全维度 CSV (viz_metrics_all.csv)
# ============================================================
import torch
import numpy as np
import os
import yaml
import pandas as pd
import shutil
from tqdm import tqdm
import torch.nn.functional as F
from server import Server
from stga import STGAAggregator
from datasets import partition_dataset_dirichlet, get_dataset


# --- 通用探针 (Compatible with FedAvg & STGA) ---
class DataProbe:
    """独立于聚合器的探针类，用于捕获特征"""

    def __init__(self, device):
        self.device = device
        self.metrics = {}

    def capture(self, updates, aggregator_type='FedAvg'):
        if not updates: return

        # 1. 展平 & 转移到 GPU
        flat_updates = []
        for u in updates:
            vec = torch.cat([v.view(-1) for k, v in sorted(u.items()) if v.dtype == torch.float32])
            flat_updates.append(vec)
        update_matrix = torch.stack(flat_updates).to(self.device)

        # 2. 基础特征 (Norms, Cosines)
        norms = torch.norm(update_matrix, p=2, dim=1)
        center = torch.median(update_matrix, dim=0).values
        cosines = F.cosine_similarity(update_matrix, center.unsqueeze(0), dim=1)

        # 3. 权重 (根据聚合器类型推断)
        if aggregator_type == 'STGA':
            # 复现 STGA 权重计算逻辑
            median_norm = torch.median(norms)
            thresh = median_norm * 1.5
            clip = torch.clamp(thresh / (norms + 1e-6), max=1.0)
            # ... (简化：仅为了获取权重分布，假设 STGA 逻辑一致)
            # 这里为了精准，建议直接从外部传入实际使用的 aggregator 实例读取
            # 但为了通用性，我们这里只记录特征，权重留给 server 记录
            pass

        return {
            'norms': norms.detach().cpu().numpy(),
            'cosines': cosines.detach().cpu().numpy(),
            'updates_sample': update_matrix[0].detach().cpu().numpy()  # 仅存一个样本用于debug
        }


# --- 增强版 Server ---
class InstrumentedServer(Server):
    def __init__(self, config, ds, idx):
        super().__init__(config, ds, idx)
        self.probe = DataProbe(self.device)

    def run_round(self, round_idx):
        # 1. 获取 Updates (复用父类逻辑前半部分)
        #    为了不破坏父类结构，我们只能拦截 channel.forward 之后的结果
        #    或者 Monkey Patching。这里选择覆盖 run_round 方法。

        # ... (标准 Server 逻辑复刻) ...
        # 为了最大兼容性，我们直接调用 super().run_round()
        # 但我们需要在聚合前“偷看”数据。
        # 方案：修改 self.aggregator.aggregate 方法
        return super().run_round(round_idx)


# --- 注入式聚合器 (最稳妥的方案) ---
class ProbingAggregator(STGAAggregator):
    def __init__(self, config, mode='STGA'):
        super().__init__(config)
        self.mode = mode  # 'STGA' or 'FedAvg'
        self.captured_data = None

    def aggregate(self, updates, client_types=None):
        if not updates: return None

        # --- [Capture] ---
        flat = [self._flatten(u) for u in updates]
        mat = torch.stack(flat).to(self.device)

        norms = torch.norm(mat, p=2, dim=1)
        raw_center = torch.median(mat, dim=0).values
        cosines = F.cosine_similarity(mat, raw_center.unsqueeze(0), dim=1)

        # 计算 STGA 权重 (即便是 FedAvg 模式，我们也算一下“如果用 STGA 会给多少分”，用于对比)
        # ... (STGA 核心逻辑)
        median_norm = torch.median(norms)
        clip = torch.clamp((median_norm * 1.5) / (norms + 1e-6), max=1.0)
        mat_clipped = mat * clip.unsqueeze(1)

        spat_center = torch.median(mat_clipped, dim=0).values
        s_spat = (F.cosine_similarity(mat_clipped, spat_center.unsqueeze(0)) + 1) / 2 * 0.5 + \
                 torch.exp(-torch.norm(mat_clipped - spat_center, p=2, dim=1) / (
                             torch.median(torch.norm(mat_clipped - spat_center, p=2, dim=1)) + 1e-6)) * 0.5

        if self.history_updates:
            s_temp = F.cosine_similarity(mat_clipped, self.history_updates[-1].to(self.device).unsqueeze(0))
        else:
            s_temp = torch.ones(len(updates)).to(self.device)

        scores = self.conf['stga_alpha'] * (s_temp + 1) / 2 + (1 - self.conf['stga_alpha']) * s_spat
        stga_weights = F.softmax(scores * 2.0, dim=0).detach().cpu().numpy()

        # 真实使用的权重
        if self.mode == 'FedAvg':
            used_weights = np.ones(len(updates)) / len(updates)
        else:
            used_weights = stga_weights

        self.captured_data = {
            'norms': norms.detach().cpu().numpy(),
            'cosines': cosines.detach().cpu().numpy(),
            'stga_weights': stga_weights,  # 即使在 FedAvg 模式下也记录这个，用于展示“STGA 本该能防住”
            'used_weights': used_weights,
            'updates': mat.detach().cpu().numpy() if self.mode == 'STGA' else None  # 只存一次以免爆内存
        }

        # --- [Execute] ---
        if self.mode == 'STGA':
            return super().aggregate(updates, client_types)
        else:
            return self._fedavg(updates)


# --- 主流程 ---
def run_pro_harvest():
    print("🎬 Starting Multi-Scenario Data Harvest...")

    # 1. 准备
    if os.path.exists('viz_data'): shutil.rmtree('viz_data')
    os.makedirs('viz_data', exist_ok=True)

    with open('config.yaml') as f:
        base_conf = yaml.safe_load(f)

    # 统一参数 (对齐 Run All)
    ATTACK_PARAMS = {'malicious_fraction': 0.2, 'lambda_attack': 5.0}  # 强攻击
    ROUNDS = 25

    # 定义要跑的场景
    scenarios = [
        {'name': 'Vulnerable', 'aggregator': 'FedAvg', 'r_jora': False},
        {'name': 'R-JORA', 'aggregator': 'STGA', 'r_jora': True}
    ]

    global_records = []

    # 2. 循环场景
    for scen in scenarios:
        print(f"\n📦 Harvesting Scenario: {scen['name']}...")

        # 配置克隆与修改
        conf = base_conf.copy()
        if 'attack' not in conf: conf['attack'] = {}
        conf['attack'].update(ATTACK_PARAMS)
        conf['num_rounds'] = ROUNDS
        conf['scenario'] = scen['name']

        if 'r_jora' not in conf: conf['r_jora'] = {}
        conf['r_jora']['enabled'] = scen['r_jora']
        if scen['r_jora']:
            conf['r_jora'].update({'enable_stga': True, 'enable_optimal_dp': True, 'enable_secure_isac': True})
            if 'stga_alpha' not in conf['r_jora']: conf['r_jora']['stga_alpha'] = 0.5
        else:
            # 即使是 Vulnerable，我们也开启 'enabled': False，但为了 Probe 能工作，
            # 我们需要在 Server 初始化后手动注入 ProbingAggregator
            pass

        if torch.cuda.is_available(): conf['device'] = 'cuda'

        # 初始化
        ds, _ = get_dataset(conf['dataset'], conf['data_root'])
        idx = partition_dataset_dirichlet(ds, conf['num_clients'], conf['alpha'], seed=42)
        server = Server(conf, ds, idx)

        # 注入探针 (Mode = FedAvg or STGA)
        # 注意：这里传入 mode 让探针知道真实的聚合逻辑
        server.aggregator = ProbingAggregator(conf, mode=scen['aggregator'])

        # 进度条
        pbar = tqdm(range(ROUNDS), desc=f"   {scen['name']}", unit="rnd")

        for t in pbar:
            server.run_round(t)

            data = server.aggregator.captured_data
            if data is None: continue

            # 推断类型
            num_mal = int(len(data['norms']) * conf['attack']['malicious_fraction'])
            num_ben = len(data['norms']) - num_mal
            types = ['Benign'] * num_ben + ['Malicious'] * num_mal

            # 记录到列表
            for i in range(len(data['norms'])):
                global_records.append({
                    'Scenario': scen['name'],
                    'Round': t,
                    'Client_ID': i,  # 这里的 ID 是 batch 内的相对 ID
                    'Type': types[i],
                    'L2_Norm': data['norms'][i],
                    'Cosine_Sim': data['cosines'][i],
                    'Weight_Used': data['used_weights'][i],
                    'Weight_STGA_Score': data['stga_weights'][i]  # 这是一个虚拟分，用于对比
                })

            # 保存 NPY (只保存 R-JORA 的关键帧用于 t-SNE)
            if scen['name'] == 'R-JORA' and t in [0, 5, 10, 20]:
                np.save(f'viz_data/updates_r{t}.npy', data['updates'])
                np.save(f'viz_data/client_types_r{t}.npy', np.array(types))

    # 3. 导出 CSV
    df = pd.DataFrame(global_records)
    df.to_csv('viz_metrics_pro.csv', index=False)
    print(f"\n✅ Saved 'viz_metrics_pro.csv' ({len(df)} rows).")

    # 简单的统计验证
    print("\n--- Quick Validation (Mean L2 Norm) ---")
    summary = df.groupby(['Scenario', 'Type'])['L2_Norm'].mean()
    print(summary)


if __name__ == "__main__":
    run_pro_harvest()