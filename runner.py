# 文件名: runner.py
# 作用: 实验执行引擎。封装了 Exp1-5 的具体逻辑。
# 版本: Final (Hardware-aware)

import yaml
import torch
import numpy as np
import copy
from torch.utils.data import DataLoader
from datasets import get_dataset, partition_dataset_dirichlet
from server import Server
from logger import ExperimentLogger


class SimulationRunner:
    def __init__(self, base_config_path='config.yaml'):
        with open(base_config_path, 'r') as f: self.base_config = yaml.safe_load(f)
        self.data_ready = False

    def _prepare_data(self):
        if self.data_ready: return
        self.train_ds, self.test_ds = get_dataset(self.base_config['dataset'], self.base_config['data_root'])
        self.test_loader = DataLoader(self.test_ds, batch_size=1000, shuffle=False)
        self.data_ready = True

    def run_single_seed(self, config_override, seed, log_file):
        self._prepare_data()
        config = copy.deepcopy(self.base_config)
        self._recursive_update(config, config_override)

        if torch.cuda.is_available():
            print("GPU detected! Using CUDA.")
            config['device'] = 'cuda'
        else:
            config['device'] = 'cpu'

        config['seed'] = seed
        torch.manual_seed(seed);
        np.random.seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed(seed)

        client_indices = partition_dataset_dirichlet(self.train_ds, config['num_clients'], alpha=config['alpha'],
                                                     seed=seed)
        logger = ExperimentLogger(log_file)
        server = Server(config, self.train_ds, client_indices)

        print(f"-> Running {config['scenario']} (Seed {seed})...")
        acc = 0
        for t in range(config['num_rounds']):
            stats = server.run_round(t)
            loss, acc = server.evaluate(self.test_loader)
            stats.update({'accuracy': acc, 'loss': loss})
            meta = {'scenario': config['scenario'], 'seed': seed, 'beta': config['attack']['malicious_fraction'],
                    'sigma_z': stats['dp_sigma']}
            logger.log_round(t, stats, meta)
            if t % 10 == 0: print(f"   Round {t}: Acc={acc:.2f}%")
        return acc

    def _recursive_update(self, d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = self._recursive_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    def run_exp1_vulnerability(self, n_seeds=3):
        scenarios = {
            'Ideal': {'scenario': 'Ideal', 'aggregator': 'FedAvg', 'attack': {'malicious_fraction': 0.0}},
            'Vulnerable': {'scenario': 'Vulnerable', 'aggregator': 'FedAvg', 'attack': {'malicious_fraction': 0.2}}
        }
        for name, conf in scenarios.items():
            for s in range(n_seeds): self.run_single_seed(conf, 42 + s, f'logs/exp1/{name}_seed{s}.csv')

    def run_exp2_efficacy(self, n_seeds=3):
        scenarios = {
            'Ideal': {'scenario': 'Ideal', 'aggregator': 'FedAvg', 'attack': {'malicious_fraction': 0.0}},
            'Vulnerable': {'scenario': 'Vulnerable', 'aggregator': 'FedAvg', 'attack': {'malicious_fraction': 0.2}},
            'R-JORA': {'scenario': 'R-JORA', 'aggregator': 'STGA', 'attack': {'malicious_fraction': 0.2},
                       'r_jora': {'enabled': True, 'enable_stga': True, 'enable_optimal_dp': True,
                                  'enable_secure_isac': True}}
        }
        for name, conf in scenarios.items():
            for s in range(n_seeds): self.run_single_seed(conf, 42 + s, f'logs/exp2/{name}_seed{s}.csv')

    def run_exp3_baselines(self, n_seeds=3):
        betas = [0.1, 0.2, 0.3]
        modes = ['Vulnerable', 'Krum', 'Median', 'HighDP', 'R-JORA']
        for beta in betas:
            for mode in modes:
                conf = {'attack': {'malicious_fraction': beta}, 'scenario': f'{mode}_beta{beta}'}
                if mode == 'Vulnerable':
                    conf['aggregator'] = 'FedAvg'
                elif mode == 'Krum':
                    conf['aggregator'] = 'Krum'
                elif mode == 'Median':
                    conf['aggregator'] = 'Median'
                elif mode == 'HighDP':
                    conf['aggregator'] = 'FedAvg'
                    conf['dp_sigma_z'] = 0.1
                elif mode == 'R-JORA':
                    conf['aggregator'] = 'STGA'
                    conf['r_jora'] = {'enabled': True, 'enable_stga': True, 'enable_optimal_dp': True,
                                      'enable_secure_isac': True}
                for s in range(n_seeds): self.run_single_seed(conf, 42 + s, f'logs/exp3/{mode}_beta{beta}_seed{s}.csv')

    def run_exp4_pru_tradeoff(self):
        sigmas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
        modes = ['Vulnerable', 'R-JORA']
        for sigma in sigmas:
            for mode in modes:
                conf = {'scenario': f'{mode}_sigma{sigma}', 'dp_sigma_z': sigma}
                if mode == 'Vulnerable':
                    conf['aggregator'] = 'FedAvg'
                else:
                    conf['aggregator'] = 'STGA'
                    conf['r_jora'] = {'enabled': True, 'enable_stga': True, 'enable_optimal_dp': False,
                                      'enable_secure_isac': True}
                self.run_single_seed(conf, 42, f'logs/exp4/{mode}_sigma{sigma}.csv')

    def run_exp5_ablation(self, n_seeds=3):
        base = {'enabled': True, 'enable_stga': True, 'enable_optimal_dp': True, 'enable_secure_isac': True}
        configs = {'Full': base, 'No-STGA': {**base, 'enable_stga': False},
                   'No-OptDP': {**base, 'enable_optimal_dp': False}, 'No-ISAC': {**base, 'enable_secure_isac': False}}
        for name, rc in configs.items():
            conf = {'scenario': name, 'r_jora': rc, 'aggregator': 'STGA' if rc['enable_stga'] else 'FedAvg'}
            for s in range(n_seeds): self.run_single_seed(conf, 42 + s, f'logs/exp5/{name}_seed{s}.csv')