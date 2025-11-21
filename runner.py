# 文件名: runner.py
# 作用: 实验执行引擎。修复了 HighDP/Vulnerable 意外继承 R-JORA 模块的问题。
# 版本: Fixed Config Inheritance

import yaml
import torch
import numpy as np
import copy
import os
import shutil
from tqdm import tqdm
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
        print("Loading datasets...")
        self.train_ds, self.test_ds = get_dataset(self.base_config['dataset'], self.base_config['data_root'])
        self.test_loader = DataLoader(self.test_ds, batch_size=1000, shuffle=False)
        self.data_ready = True

    def run_single_seed(self, config_override, seed, log_file):
        if os.path.exists(log_file):
            print(f"  [Skip] {log_file} already exists.")
            return 0.0

        self._prepare_data()
        config = copy.deepcopy(self.base_config)
        self._recursive_update(config, config_override)

        if torch.cuda.is_available():
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

        pbar = tqdm(range(config['num_rounds']), desc=f"Run: {config['scenario']} (S{seed})", unit="rnd")
        acc = 0
        for t in pbar:
            stats = server.run_round(t)
            loss, acc = server.evaluate(self.test_loader)
            stats.update({'accuracy': acc, 'loss': loss})
            meta = {'scenario': config['scenario'], 'seed': seed, 'beta': config['attack']['malicious_fraction'],
                    'sigma_z': stats['dp_sigma']}
            logger.log_round(t, stats, meta)
            pbar.set_postfix({'Acc': f"{acc:.2f}%", 'Loss': f"{loss:.2f}"})

        return acc

    def _recursive_update(self, d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = self._recursive_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    def _smart_reuse(self, src_exp, src_name, dst_exp, dst_name, seed, fallback_conf):
        src_file = f'logs/{src_exp}/{src_name}_seed{seed}.csv'
        dst_file = f'logs/{dst_exp}/{dst_name}_seed{seed}.csv'
        os.makedirs(os.path.dirname(dst_file), exist_ok=True)

        if os.path.exists(dst_file):
            print(f"  [Exists] {dst_name} (S{seed}) in {dst_exp}")
            return

        if not os.path.exists(src_file):
            print(f"  [Gen Source] Generating {src_name} for {src_exp}...")
            os.makedirs(os.path.dirname(src_file), exist_ok=True)
            self.run_single_seed(fallback_conf, 42 + seed, src_file)

        shutil.copy(src_file, dst_file)
        print(f"  [Copy] Copied {src_name} from {src_exp} to {dst_exp}")

    # --- 实验定义 ---

    def run_exp1_vulnerability(self, n_seeds=3):
        print(f"\n=== Experiment 1: Vulnerability (Seeds={n_seeds}) ===")
        # [Fix] 显式关闭 r_jora 以确保是纯净的基线
        base_off = {'enabled': False}
        scenarios = {
            'Ideal': {'scenario': 'Ideal', 'aggregator': 'FedAvg', 'attack': {'malicious_fraction': 0.0},
                      'r_jora': base_off},
            'Vulnerable': {'scenario': 'Vulnerable', 'aggregator': 'FedAvg', 'attack': {'malicious_fraction': 0.2},
                           'r_jora': base_off}
        }
        for name, conf in scenarios.items():
            for s in range(n_seeds):
                self.run_single_seed(conf, 42 + s, f'logs/exp1/{name}_seed{s}.csv')

    def run_exp2_efficacy(self, n_seeds=3):
        print(f"\n=== Experiment 2: Efficacy (Smart Reuse) ===")
        # 定义回退配置 (必须和 Exp1 一致，显式关闭 r_jora)
        base_off = {'enabled': False}
        fallback_configs = {
            'Ideal': {'scenario': 'Ideal', 'aggregator': 'FedAvg', 'attack': {'malicious_fraction': 0.0},
                      'r_jora': base_off},
            'Vulnerable': {'scenario': 'Vulnerable', 'aggregator': 'FedAvg', 'attack': {'malicious_fraction': 0.2},
                           'r_jora': base_off}
        }

        rjora_conf = {'scenario': 'R-JORA', 'aggregator': 'STGA', 'attack': {'malicious_fraction': 0.2},
                      'r_jora': {'enabled': True, 'enable_stga': True, 'enable_optimal_dp': True,
                                 'enable_secure_isac': True}}

        for s in range(n_seeds):
            for src, dst in [('Ideal', 'Ideal'), ('Vulnerable', 'Vulnerable')]:
                self._smart_reuse('exp1', src, 'exp2', dst, s, fallback_configs[src])
            self.run_single_seed(rjora_conf, 42 + s, f'logs/exp2/R-JORA_seed{s}.csv')

    def run_exp3_baselines(self, n_seeds=3):
        print(f"\n=== Experiment 3: Baselines (Smart Reuse) ===")
        betas = [0.1, 0.2, 0.3]
        modes = ['Vulnerable', 'Krum', 'Median', 'HighDP', 'R-JORA']

        # [Fix] 显式关闭 r_jora
        base_off = {'enabled': False}
        vuln_conf_base = {'scenario': 'Vulnerable', 'aggregator': 'FedAvg', 'attack': {'malicious_fraction': 0.2},
                          'r_jora': base_off}

        for beta in betas:
            for mode in modes:
                scenario_name = f'{mode}_beta{beta}'

                if mode == 'Vulnerable' and beta == 0.2:
                    for s in range(n_seeds):
                        self._smart_reuse('exp1', 'Vulnerable', 'exp3', scenario_name, s, vuln_conf_base)
                    continue

                # [Critical Fix] 这里的 conf 必须包含 r_jora: False，除非是 R-JORA 模式
                conf = {'attack': {'malicious_fraction': beta}, 'scenario': scenario_name}

                if mode == 'R-JORA':
                    conf['aggregator'] = 'STGA'
                    conf['r_jora'] = {'enabled': True, 'enable_stga': True, 'enable_optimal_dp': True,
                                      'enable_secure_isac': True}
                else:
                    # 对于所有基线，强制关闭 R-JORA 模块，防止 OptimalDP 自动介入
                    conf['r_jora'] = {'enabled': False}

                    if mode == 'Vulnerable':
                        conf['aggregator'] = 'FedAvg'
                    elif mode == 'Krum':
                        conf['aggregator'] = 'Krum'
                    elif mode == 'Median':
                        conf['aggregator'] = 'Median'
                    elif mode == 'HighDP':
                        conf['aggregator'] = 'FedAvg'
                        conf['dp_sigma_z'] = 0.1  # 这里手动设置才会生效

                for s in range(n_seeds):
                    self.run_single_seed(conf, 42 + s, f'logs/exp3/{scenario_name}_seed{s}.csv')

    def run_exp4_pru_tradeoff(self):
        print(f"\n=== Experiment 4: PRU Trade-off ===")
        sigmas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
        modes = ['Vulnerable', 'R-JORA']

        for sigma in sigmas:
            for mode in modes:
                conf = {'scenario': f'{mode}_sigma{sigma}', 'dp_sigma_z': sigma}
                if mode == 'Vulnerable':
                    conf['aggregator'] = 'FedAvg'
                    conf['r_jora'] = {'enabled': False}  # [Fix] 确保不开启 OptimalDP
                else:
                    conf['aggregator'] = 'STGA'
                    # 注意：这里 enable_optimal_dp 必须为 False，因为我们要手动扫描 sigma
                    conf['r_jora'] = {'enabled': True, 'enable_stga': True, 'enable_optimal_dp': False,
                                      'enable_secure_isac': True}

                self.run_single_seed(conf, 42, f'logs/exp4/{mode}_sigma{sigma}.csv')

    def run_exp5_ablation(self, n_seeds=3):
        print(f"\n=== Experiment 5: Ablation (Seeds={n_seeds}) ===")
        base = {'enabled': True, 'enable_stga': True, 'enable_optimal_dp': True, 'enable_secure_isac': True}
        rjora_conf = {'scenario': 'R-JORA', 'aggregator': 'STGA', 'attack': {'malicious_fraction': 0.2}, 'r_jora': base}

        configs = {'Full': base, 'No-STGA': {**base, 'enable_stga': False},
                   'No-OptDP': {**base, 'enable_optimal_dp': False}, 'No-ISAC': {**base, 'enable_secure_isac': False}}

        for name, rc in configs.items():
            if name == 'Full':
                for s in range(n_seeds):
                    self._smart_reuse('exp2', 'R-JORA', 'exp5', 'Full', s, rjora_conf)
            else:
                # 只有 Full 开启 STGA (如果 rc 中 enable_stga 为 True)
                # No-STGA 模式下聚合器应退回 FedAvg
                agg = 'STGA' if rc['enable_stga'] else 'FedAvg'
                conf = {'scenario': name, 'r_jora': rc, 'aggregator': agg}
                for s in range(n_seeds): self.run_single_seed(conf, 42 + s, f'logs/exp5/{name}_seed{s}.csv')