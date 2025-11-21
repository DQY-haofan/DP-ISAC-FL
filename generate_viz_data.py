# ============================================================
# 脚本名: generate_viz_data_final.py (v3.0 Ultimate)
# 作用: 一站式采集所有可视化所需的高维数据
# 新增: L2范数 (Norms) + 余弦相似度 (Cosines) -> 完美解释攻防机理
# ============================================================
import torch
import numpy as np
import os
import yaml
import torch.nn.functional as F
from runner import SimulationRunner
from server import Server
from stga import STGAAggregator
from datasets import partition_dataset_dirichlet, get_dataset


# --- 定义间谍聚合器 ---
class InstrumentedSTGA(STGAAggregator):
    def __init__(self, config):
        super().__init__(config)
        # 数据缓存区
        self.captured_weights = None
        self.captured_updates = None
        self.captured_norms = None  # [New] 解释为什么能防住 (幅度异常)
        self.captured_cosines = None  # [New] 解释为什么 Krum 防不住 (方向伪装)

    def aggregate(self, updates, client_types=None):
        if not updates: return None

        # 1. 预处理
        flat_updates = [self._flatten(u) for u in updates]
        update_matrix = torch.stack(flat_updates).to(self.device)

        # [Capture 1] 原始更新向量 (用于 t-SNE)
        self.captured_updates = update_matrix.detach().cpu().numpy()

        # [Capture 2] L2 范数 (用于 Boxplot)
        norms = torch.norm(update_matrix, p=2, dim=1)
        self.captured_norms = norms.detach().cpu().numpy()

        # --- 复现 STGA 逻辑以捕获中间变量 ---

        # 计算空间中心 (用于计算余弦相似度)
        # 注意：为了公平对比，我们计算相对于“未裁剪”中心的相似度，看攻击者伪装得有多像
        raw_center = torch.median(update_matrix, dim=0).values
        cos_sim = F.cosine_similarity(update_matrix, raw_center.unsqueeze(0), dim=1)

        # [Capture 3] 余弦相似度 (用于证明攻击者的方向伪装)
        self.captured_cosines = cos_sim.detach().cpu().numpy()

        # === 正常的 STGA 处理流程 ===
        # Norm Clipping
        median_norm = torch.median(norms)
        threshold = median_norm * 1.5
        clip_factor = torch.clamp(threshold / (norms + 1e-6), max=1.0)
        update_matrix_clipped = update_matrix * clip_factor.unsqueeze(1)

        # Spatial Score
        spatial_center = torch.median(update_matrix_clipped, dim=0).values
        s_spat_cos = F.cosine_similarity(update_matrix_clipped, spatial_center.unsqueeze(0), dim=1)
        dists = torch.norm(update_matrix_clipped - spatial_center, p=2, dim=1)
        sigma = torch.median(dists) + 1e-6
        s_spat_dist = torch.exp(-dists / sigma)
        s_spat = (s_spat_cos + 1) / 2 * 0.5 + s_spat_dist * 0.5

        # Temporal Score
        if len(self.history_updates) > 0:
            expected_update = self.history_updates[-1].to(self.device)
            s_temp = F.cosine_similarity(update_matrix_clipped, expected_update.unsqueeze(0), dim=1)
        else:
            s_temp = torch.ones(len(updates)).to(self.device)
        s_temp_norm = (s_temp + 1) / 2

        # Final Weights
        trust_scores = self.alpha * s_temp_norm + (1 - self.alpha) * s_spat
        weights = F.softmax(trust_scores * 2.0, dim=0)

        # [Capture 4] 最终权重 (用于 Heatmap)
        self.captured_weights = weights.detach().cpu().numpy()

        # 调用父类完成实际聚合
        return super().aggregate(updates, client_types)


# --- 主程序 ---
def run_harvest():
    print("🚜 Starting Final Visualization Data Harvest (25 Rounds)...")
    # 清理旧数据
    import shutil
    if os.path.exists('viz_data'): shutil.rmtree('viz_data')
    os.makedirs('viz_data', exist_ok=True)

    # 1. 加载并修补配置
    with open('config.yaml') as f:
        conf = yaml.safe_load(f)

    # 强制开启 R-JORA
    if 'r_jora' not in conf: conf['r_jora'] = {}
    conf['r_jora'].update({
        'enabled': True, 'enable_stga': True, 'enable_optimal_dp': True, 'enable_secure_isac': True
    })
    # 确保必要参数存在
    if 'stga_alpha' not in conf['r_jora']: conf['r_jora']['stga_alpha'] = 0.5

    # 设置采集参数
    conf['num_rounds'] = 25  # 跑25轮足够展示收敛初期的动态
    conf['scenario'] = 'Viz_Harvest'
    conf['aggregator'] = 'STGA'

    # [关键] 使用能产生显著对比的攻击参数 (Exp 1/2 验证过的参数)
    # Lambda=5.0 能产生巨大的 Norm 差异，非常适合画图
    conf['attack'] = {
        'malicious_fraction': 0.2,
        'lambda_attack': 5.0,
        'tau_sim': 0.5,
        't_vgae': 1,
        'q_eaves': 0.8,
        'eaves_sigma': 0.005,
        'vgae_epochs': 5,
        'vgae_lr': 0.01,
        'latent_dim': 16
    }

    if torch.cuda.is_available(): conf['device'] = 'cuda'

    # 2. 初始化
    print("   Loading data...")
    ds, _ = get_dataset(conf['dataset'], conf['data_root'])
    # 固定种子 42，确保和论文里的 Exp 1/2 一致
    idx = partition_dataset_dirichlet(ds, conf['num_clients'], conf['alpha'], seed=42)

    server = Server(conf, ds, idx)

    # 注入间谍聚合器
    server.aggregator = InstrumentedSTGA(conf)

    # 保存客户端类型标签 (0=Benign, 1=Malicious)
    client_types = np.array(
        ['Malicious' if i in server.malicious_ids else 'Benign' for i in range(conf['num_clients'])])
    np.save('viz_data/client_types.npy', client_types)
    print(f"   Setup complete. Malicious nodes: {len(server.malicious_ids)}")

    # 3. 运行循环
    for t in range(conf['num_rounds']):
        print(f"   Harvesting Round {t + 1}/{conf['num_rounds']}...")
        server.run_round(t)

        # 保存各类数据
        if server.aggregator.captured_weights is not None:
            # 权重
            np.save(f'viz_data/weights_r{t}.npy', server.aggregator.captured_weights)
            # 范数 [新增]
            np.save(f'viz_data/norms_r{t}.npy', server.aggregator.captured_norms)
            # 余弦相似度 [新增]
            np.save(f'viz_data/cosines_r{t}.npy', server.aggregator.captured_cosines)

        # 模型向量 (仅保存关键帧，文件较大)
        if t in [0, 5, 10, 15, 20, 24]:
            np.save(f'viz_data/updates_r{t}.npy', server.aggregator.captured_updates)

        # ISAC 掩码
        if hasattr(server.isac_scheduler, 'last_mask') and server.isac_scheduler.last_mask is not None:
            np.save(f'viz_data/mask_r{t}.npy', server.isac_scheduler.last_mask.cpu().numpy())

    print("✅ Data Harvest Complete. All high-dim data saved in 'viz_data/'.")


if __name__ == "__main__":
    run_harvest()