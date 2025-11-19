# 文件名: client.py
# 作用: 客户端逻辑。BenignClient 包含 DP 流程和显存优化；MaliciousClient 是攻击者的傀儡。
# 版本: Final (包含无状态显存优化)

import torch
import copy
from torch.utils.data import DataLoader


class BenignClient:
    def __init__(self, client_id, dataset, indices, model_fn, config):
        self.client_id = client_id
        self.conf = config
        self.device = config['device']
        self.model_fn = model_fn  # 保存构造函数，而不是模型实例 (Stateless)

        # Colab 优化: 使用多线程加载数据
        num_workers = 2 if self.device == 'cuda' else 0

        if indices is not None:
            self.loader = DataLoader(
                torch.utils.data.Subset(dataset, indices),
                batch_size=self.conf['batch_size'],
                shuffle=True,
                num_workers=num_workers,
                pin_memory=(self.device == 'cuda')
            )
        else:
            self.loader = None

    def local_train(self, global_params):
        # 1. 临时实例化模型 (节省显存)
        model = self.model_fn().to(self.device)
        model.load_state_dict(global_params)
        original_params = copy.deepcopy(global_params)

        # 2. 训练
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.conf['lr'])
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(self.conf['local_epochs']):
            if self.loader:
                for data, target in self.loader:
                    data, target = data.to(self.device), target.to(self.device)
                    optimizer.zero_grad()
                    loss = criterion(model(data), target)
                    loss.backward()
                    optimizer.step()

        # 3. 计算更新 (Delta)
        update_dict = {}
        trained_params = model.state_dict()
        all_update_tensors = []

        for key in trained_params:
            diff = trained_params[key] - original_params[key]
            update_dict[key] = diff
            if diff.dtype in [torch.float32, torch.float64]:
                all_update_tensors.append(diff.flatten())

        # 4. 裁剪 (Clipping)
        total_norm = torch.norm(torch.cat(all_update_tensors), p=2)
        clip_C = self.conf['clip_threshold']
        scale = min(1.0, clip_C / (total_norm.item() + 1e-6))

        # 5. DP 加噪
        final_update = {}
        for key in update_dict:
            clipped_tensor = update_dict[key] * scale
            if clipped_tensor.dtype in [torch.float32, torch.float64]:
                # 使用 self.conf 中的最新 sigma
                noise = torch.normal(mean=0.0, std=self.conf['dp_sigma_z'], size=clipped_tensor.size()).to(self.device)
                final_update[key] = clipped_tensor + noise
            else:
                final_update[key] = clipped_tensor

        # 6. 销毁模型
        del model
        return final_update


class MaliciousClient:
    """
    恶意客户端: 不训练，只向攻击者请求恶意更新。
    """

    def __init__(self, client_id, attacker, config):
        self.client_id = client_id
        self.attacker = attacker
        self.conf = config

    def local_train(self, global_params):
        # 伪装成 local_train 接口，实际上是调用攻击者生成
        return self.attacker.generate_malicious_update(global_params)