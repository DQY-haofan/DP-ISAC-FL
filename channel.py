# 文件名: channel.py
# 作用: 模拟 ISAC 信道噪声。
# 版本: Final

import torch

class AbstractISACChannel:
    def __init__(self, config):
        self.sigma_ch = config['isac_sigma_ch']
        self.device = config['device']

    def forward(self, updates):
        noisy_updates = []
        for update in updates:
            noisy_update = {}
            for key, param in update.items():
                noise = torch.normal(mean=0.0, std=self.sigma_ch, size=param.size()).to(self.device)
                noisy_update[key] = param + noise
            noisy_updates.append(noisy_update)
        return noisy_updates