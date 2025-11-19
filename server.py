# 文件名: server.py
# 作用: 中央服务器 (上帝视角)。负责协调训练、攻击、防御、信道模拟和参数聚合。
# 版本: Final (修复 Mask 索引越界 Bug)

import torch
import numpy as np
import math
from models import build_model
from client import BenignClient, MaliciousClient
from channel import AbstractISACChannel
from vgae_attacker import VGAEAttacker

# 引入防御模块与基线聚合器
from stga import STGAAggregator
from aggregators import FedAvgAggregator, KrumAggregator, MedianAggregator
from optimal_dp import OptimalDPSearcher
from secure_isac import SecureISACScheduler


class Server:
    def __init__(self, config, train_dataset, client_indices):
        self.conf = config
        self.device = config['device']
        # 默认场景为 Ideal，如果 config 里没写
        self.scenario = config.get('scenario', 'Ideal')

        # 1. 初始化全局模型
        self.global_model = build_model(config['model']).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()

        # 2. 初始化各个组件 (工厂模式)
        self._init_components(config)

        # 3. 初始化客户端
        self._init_clients(config, train_dataset, client_indices)

        # 状态追踪: 动态调整的 DP 噪声 sigma
        self.current_dp_sigma = config['dp_sigma_z']

    def _init_components(self, config):
        """根据配置初始化攻击者、信道、聚合器和防御模块"""

        # 计算模型参数总量 (用于 VGAE 输入维度)
        param_count = sum(p.numel() for p in self.global_model.parameters() if p.requires_grad)

        # 基础组件
        self.attacker = VGAEAttacker(config, param_count)
        self.channel = AbstractISACChannel(config)

        # 聚合器选择
        agg_type = config.get('aggregator', 'FedAvg')
        print(f"Server Initialized with Aggregator: {agg_type}")

        if agg_type == 'STGA':
            self.aggregator = STGAAggregator(config)
        elif agg_type == 'Krum':
            # Krum 需要知道 f (假设的恶意节点数)
            mal_frac = config['attack']['malicious_fraction']
            # 这里我们用真实的恶意比例来配置 Krum，给予它“公平”的竞争环境
            f_malicious = int(config['num_clients'] * config['client_fraction'] * mal_frac) + 2
            self.aggregator = KrumAggregator(f_malicious=f_malicious)
        elif agg_type == 'Median':
            self.aggregator = MedianAggregator()
        else:
            self.aggregator = FedAvgAggregator()

        # R-JORA 防御特定组件
        self.dp_searcher = OptimalDPSearcher(config)
        self.isac_scheduler = SecureISACScheduler(config)

    def _init_clients(self, config, train_dataset, client_indices):
        """初始化良性和恶意客户端"""
        self.clients = []
        num_clients = config['num_clients']

        # 如果是 Ideal 模式，强制无恶意客户端
        if self.scenario == 'Ideal':
            malicious_frac = 0.0
        else:
            malicious_frac = config['attack']['malicious_fraction']

        num_malicious = int(num_clients * malicious_frac)

        # 随机选择恶意 ID
        all_ids = np.arange(num_clients)
        np.random.seed(config['seed'])
        np.random.shuffle(all_ids)
        self.malicious_ids = set(all_ids[:num_malicious])

        print(f"Clients: {num_clients - num_malicious} Benign, {num_malicious} Malicious")

        for i in range(num_clients):
            if i in self.malicious_ids:
                # 恶意客户端持有攻击者引用
                client = MaliciousClient(i, self.attacker, config)
            else:
                # 良性客户端
                client = BenignClient(
                    client_id=i,
                    dataset=train_dataset,
                    indices=client_indices[i],
                    # 传入 lambda 延迟实例化模型 (显存优化)
                    model_fn=lambda: build_model(config['model']),
                    config=config
                )
            self.clients.append(client)

    def select_clients(self):
        """随机采样客户端"""
        num_select = max(int(self.conf['num_clients'] * self.conf['client_fraction']), 1)
        selected_indices = np.random.choice(len(self.clients), num_select, replace=False)
        return [self.clients[i] for i in selected_indices]

    def run_round(self, round_idx):
        """
        运行一轮联邦学习 (包含 R-JORA 的三层循环逻辑)
        """
        # 获取 R-JORA 配置 (如果没有则默认为空/禁用)
        r_conf = self.conf.get('r_jora', {'enabled': False})

        # --- 1. 外层循环: Optimal DP (仅当 R-JORA 启用时) ---
        if r_conf['enabled'] and r_conf.get('enable_optimal_dp', False):
            if round_idx % r_conf['t_outer'] == 0:
                new_sigma2 = self.dp_searcher.find_optimal_sigma2()
                if new_sigma2:
                    self.current_dp_sigma = math.sqrt(new_sigma2)
                    # 动态更新所有良性客户端的噪声参数
                    for c in self.clients:
                        if isinstance(c, BenignClient):
                            c.conf['dp_sigma_z'] = self.current_dp_sigma

        # --- 2. 中层循环: Secure ISAC (仅当 R-JORA 启用时) ---
        selected_clients = self.select_clients()
        client_ids = [c.client_id for c in selected_clients]

        mask_t = None
        instability = 0.0

        if r_conf['enabled'] and r_conf.get('enable_secure_isac', False):
            # 生成动态掩码 mask_t (长度 = len(selected_clients))
            mask_t = self.isac_scheduler.update_schedule(client_ids, round_idx)
            instability = self.isac_scheduler.instability_metric

        # --- 3. 内层循环: 训练与攻击 ---

        # 分类客户端
        benigns = []
        malicious = []

        # [FIX] 同时分离 mask，确保 mask 索引与 benign_updates 对齐
        benign_mask = None

        if mask_t is not None:
            # mask_t 是 Tensor, 需要拆分
            # selected_clients 里的顺序是固定的，mask_t 对应这个顺序
            benign_indices = [i for i, c in enumerate(selected_clients) if isinstance(c, BenignClient)]
            benigns = [selected_clients[i] for i in benign_indices]
            malicious = [c for c in selected_clients if isinstance(c, MaliciousClient)]

            # 只取 Benign Clients 对应的 mask
            # 这样 attacker 拿到的 mask 长度就等于 benign_updates 的长度了
            benign_mask = mask_t[benign_indices]
        else:
            benigns = [c for c in selected_clients if isinstance(c, BenignClient)]
            malicious = [c for c in selected_clients if isinstance(c, MaliciousClient)]

        global_params = self.global_model.state_dict()

        # A. 良性训练 (Benign Training)
        benign_updates = []
        for c in benigns:
            benign_updates.append(c.local_train(global_params))

        # B. 攻击者观测 (Attacker Observation)
        # [CRITICAL FIX]: 传递切分后的 benign_mask
        self.attacker.observe_dp_updates(benign_updates, round_idx, external_mask=benign_mask)

        # 训练 VGAE (如果到了 T_vgae 周期)
        self.attacker.train_vgae_if_needed(round_idx)

        # C. 恶意生成 (Malicious Generation)
        malicious_updates = []
        for c in malicious:
            malicious_updates.append(c.local_train(global_params))

        # D. 信道传输 (Channel Transmission)
        # 注意：这里我们需要把 updates 重新拼回去或者直接合并，顺序不影响 FedAvg 但可能影响调试
        all_updates = benign_updates + malicious_updates
        # 经过 ISAC 信道叠加通信噪声
        received_updates = self.channel.forward(all_updates)

        # E. 聚合 (Aggregation)
        # 准备标签用于 STGA 诊断
        # 注意 received_updates 的顺序是先 Benign 后 Malicious (因为上面是 list +)
        client_types = ['benign'] * len(benigns) + ['malicious'] * len(malicious)

        # 调用选定的聚合器
        if self.conf.get('aggregator') == 'STGA':
            avg_update = self.aggregator.aggregate(received_updates, client_types)
            theta_b = self.aggregator.diagnostics['avg_trust_benign']
            theta_m = self.aggregator.diagnostics['avg_trust_malicious']
        else:
            avg_update = self.aggregator.aggregate(received_updates)
            theta_b, theta_m = 0, 0

            # F. 更新全局模型
        if avg_update:
            current_params = self.global_model.state_dict()
            for k in current_params:
                if current_params[k].dtype in [torch.float32, torch.float64]:
                    current_params[k] += avg_update[k]
            self.global_model.load_state_dict(current_params)

        # 返回统计信息给 Runner/Logger
        return {
            'n_benign': len(benigns),
            'stga_trust_benign': theta_b,
            'stga_trust_mal': theta_m,
            'isac_instability': instability,
            'dp_sigma': self.current_dp_sigma
        }

    def evaluate(self, test_loader):
        """在测试集上评估全局模型"""
        self.global_model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                test_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        acc = 100. * correct / len(test_loader.dataset)
        return test_loss, acc