# 文件名: server.py
# 作用: 中央服务器 (修复 Mask 索引越界 Bug)

import torch
import numpy as np
import math
from models import build_model
from client import BenignClient, MaliciousClient
from channel import AbstractISACChannel
from vgae_attacker import VGAEAttacker
from stga import STGAAggregator
from aggregators import FedAvgAggregator, KrumAggregator, MedianAggregator
from optimal_dp import OptimalDPSearcher
from secure_isac import SecureISACScheduler


class Server:
    def __init__(self, config, train_dataset, client_indices):
        self.conf = config
        self.device = config['device']
        self.scenario = config.get('scenario', 'Ideal')
        self.global_model = build_model(config['model']).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self._init_components(config)
        self._init_clients(config, train_dataset, client_indices)
        self.current_dp_sigma = config['dp_sigma_z']

    def _init_components(self, config):
        param_count = sum(p.numel() for p in self.global_model.parameters() if p.requires_grad)
        self.attacker = VGAEAttacker(config, param_count)
        self.channel = AbstractISACChannel(config)

        agg_type = config.get('aggregator', 'FedAvg')
        if agg_type == 'STGA':
            self.aggregator = STGAAggregator(config)
        elif agg_type == 'Krum':
            mal_frac = config['attack']['malicious_fraction']
            f_mal = int(config['num_clients'] * config['client_fraction'] * mal_frac) + 2
            self.aggregator = KrumAggregator(f_malicious=f_mal)
        elif agg_type == 'Median':
            self.aggregator = MedianAggregator()
        else:
            self.aggregator = FedAvgAggregator()

        self.dp_searcher = OptimalDPSearcher(config)
        self.isac_scheduler = SecureISACScheduler(config)

    def _init_clients(self, config, train_dataset, client_indices):
        self.clients = []
        num_clients = config['num_clients']
        malicious_frac = 0.0 if self.scenario == 'Ideal' else config['attack']['malicious_fraction']
        num_malicious = int(num_clients * malicious_frac)

        all_ids = np.arange(num_clients)
        np.random.seed(config['seed'])
        np.random.shuffle(all_ids)
        self.malicious_ids = set(all_ids[:num_malicious])

        for i in range(num_clients):
            if i in self.malicious_ids:
                client = MaliciousClient(i, self.attacker, config)
            else:
                client = BenignClient(i, train_dataset, client_indices[i], lambda: build_model(config['model']), config)
            self.clients.append(client)

    def select_clients(self):
        num_select = max(int(self.conf['num_clients'] * self.conf['client_fraction']), 1)
        selected_indices = np.random.choice(len(self.clients), num_select, replace=False)
        return [self.clients[i] for i in selected_indices]

    def run_round(self, round_idx):
        r_conf = self.conf.get('r_jora', {'enabled': False})

        if r_conf['enabled'] and r_conf.get('enable_optimal_dp', False):
            if round_idx % r_conf['t_outer'] == 0:
                new_sigma2 = self.dp_searcher.find_optimal_sigma2()
                if new_sigma2:
                    self.current_dp_sigma = math.sqrt(new_sigma2)
                    for c in self.clients:
                        if isinstance(c, BenignClient): c.conf['dp_sigma_z'] = self.current_dp_sigma

        selected_clients = self.select_clients()
        client_ids = [c.client_id for c in selected_clients]

        mask_t = None
        instability = 0.0
        if r_conf['enabled'] and r_conf.get('enable_secure_isac', False):
            mask_t = self.isac_scheduler.update_schedule(client_ids, round_idx)
            instability = self.isac_scheduler.instability_metric

        benigns = []
        malicious = []
        benign_mask = None

        if mask_t is not None:
            benign_indices = [i for i, c in enumerate(selected_clients) if isinstance(c, BenignClient)]
            benigns = [selected_clients[i] for i in benign_indices]
            malicious = [c for c in selected_clients if isinstance(c, MaliciousClient)]
            # [CRITICAL FIX] 只切分出属于 Benign 的 mask 传给 attacker
            benign_mask = mask_t[benign_indices]
        else:
            benigns = [c for c in selected_clients if isinstance(c, BenignClient)]
            malicious = [c for c in selected_clients if isinstance(c, MaliciousClient)]

        global_params = self.global_model.state_dict()
        benign_updates = [c.local_train(global_params) for c in benigns]

        # 传递切分后的 mask，防止索引越界
        self.attacker.observe_dp_updates(benign_updates, round_idx, external_mask=benign_mask)
        self.attacker.train_vgae_if_needed(round_idx)

        malicious_updates = [c.local_train(global_params) for c in malicious]

        all_updates = benign_updates + malicious_updates
        received_updates = self.channel.forward(all_updates)

        client_types = ['benign'] * len(benigns) + ['malicious'] * len(malicious)

        if self.conf.get('aggregator') == 'STGA':
            avg_update = self.aggregator.aggregate(received_updates, client_types)
            theta_b = self.aggregator.diagnostics['avg_trust_benign']
            theta_m = self.aggregator.diagnostics['avg_trust_malicious']
        else:
            avg_update = self.aggregator.aggregate(received_updates)
            theta_b, theta_m = 0, 0

        if avg_update:
            current_params = self.global_model.state_dict()
            for k in current_params:
                if current_params[k].dtype in [torch.float32, torch.float64]:
                    current_params[k] += avg_update[k]
            self.global_model.load_state_dict(current_params)

        return {'stga_trust_benign': theta_b, 'stga_trust_mal': theta_m, 'isac_instability': instability,
                'dp_sigma': self.current_dp_sigma}

    def evaluate(self, test_loader):
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
        return test_loss / len(test_loader.dataset), 100. * correct / len(test_loader.dataset)