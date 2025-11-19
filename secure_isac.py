# 文件名: secure_isac.py
# 作用: R-JORA 子模块 3 - 安全感知调度器。生成动态掩码 Mask。
# 版本: Final

import torch
import numpy as np


class SecureISACScheduler:
    """
    [Academic Abstraction Note]:
    Abstracts PHY beamforming into Resource Blocks.
    Provides Dynamic Masking (Psi > 0).
    """

    def __init__(self, config):
        self.conf = config['r_jora']
        self.num_beams = self.conf['num_beams']
        self.visible_beams = self.conf['visible_beams']
        self.last_mask = None
        self.instability_metric = 0.0

    def update_schedule(self, client_ids, round_idx):
        num_clients = len(client_ids)

        if not self.conf['enabled'] or not self.conf['enable_secure_isac']:
            mask = torch.zeros(num_clients, dtype=torch.bool)
            mask[:int(num_clients * 0.8)] = True  # Static baseline
            self.instability_metric = 0.0
            return mask

        # Random Permutation (Entropy Injection)
        beam_allocation = np.random.randint(0, self.num_beams, size=num_clients)
        visible_set = set(range(self.visible_beams))

        mask_list = [b in visible_set for b in beam_allocation]
        mask_t = torch.tensor(mask_list, dtype=torch.bool)

        if self.last_mask is not None and len(self.last_mask) == len(mask_t):
            self.instability_metric = (mask_t ^ self.last_mask).float().mean().item()

        self.last_mask = mask_t
        return mask_t