# 文件名: vgae_attacker.py
# 作用: 核心攻击逻辑。包含 VGAE 模型和基于梯度的恶意样本生成 (Input Optimization)。
# 版本: Final (Fix NaN using Cosine Similarity)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy


# --- GCN & VGAE 层定义 ---

class GraphConv(nn.Module):
    """
    简单的图卷积层 (GCN Layer)
    公式: H' = ReLU( D^-1/2 A D^-1/2 X W )
    """

    def __init__(self, in_features, out_features):
        super(GraphConv, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        support = self.linear(x)
        output = torch.matmul(adj, support)
        return output


class VGAE(nn.Module):
    """
    变分图自编码器
    """

    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VGAE, self).__init__()
        self.gc1 = GraphConv(input_dim, hidden_dim)
        self.gc_mu = GraphConv(hidden_dim, latent_dim)
        self.gc_logvar = GraphConv(hidden_dim, latent_dim)

    def encode(self, x, adj):
        hidden = F.relu(self.gc1(x, adj))
        return self.gc_mu(hidden, adj), self.gc_logvar(hidden, adj)

    def reparameterize(self, mu, logvar):
        if self.training:
            logvar = torch.clamp(logvar, max=10.0)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x, adj):
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


def inner_product_decoder(z):
    adj_logits = torch.matmul(z, z.t())
    probs = torch.sigmoid(adj_logits)
    probs = torch.clamp(probs, min=1e-7, max=1.0 - 1e-7)
    return probs, adj_logits


# --- 攻击者逻辑 ---

class VGAEAttacker:
    def __init__(self, config, input_dim):
        self.conf = config['attack']
        self.device = config['device']
        self.input_dim = input_dim

        self.vgae = VGAE(input_dim, 32, self.conf['latent_dim']).to(self.device)
        self.optimizer = optim.Adam(self.vgae.parameters(), lr=self.conf['vgae_lr'])

        self.f_polluted = None
        self.adj_norm = None
        self.adj_label = None

    def _flatten_updates(self, updates):
        flat_list = []
        for update in updates:
            vec = []
            for k in sorted(update.keys()):
                if update[k].dtype in [torch.float32, torch.float64]:
                    vec.append(update[k].view(-1))
            flat_list.append(torch.cat(vec))
        if not flat_list: return None
        return torch.stack(flat_list)

    def observe_dp_updates(self, dp_updates, round_idx, external_mask=None):
        num_updates = len(dp_updates)
        if num_updates == 0: return

        flat_updates = self._flatten_updates(dp_updates)

        if external_mask is not None:
            obs_indices = torch.where(external_mask)[0]
        else:
            num_obs = int(num_updates * self.conf['q_eaves'])
            obs_indices = torch.randperm(num_updates)[:num_obs].to(self.device)

        if len(obs_indices) == 0:
            self.f_polluted = None
            return

        noise = torch.normal(0, self.conf['eaves_sigma'], size=flat_updates.shape).to(self.device)
        polluted_updates = flat_updates + noise
        self.f_polluted = polluted_updates[obs_indices]

        self._build_graph(self.f_polluted)

    def _build_graph(self, features):
        features_norm = F.normalize(features, p=2, dim=1, eps=1e-8)
        sim_matrix = torch.mm(features_norm, features_norm.t())

        adj = (sim_matrix >= self.conf['tau_sim']).float()
        adj.fill_diagonal_(0)
        self.adj_label = adj

        adj_tilde = adj + torch.eye(adj.shape[0]).to(self.device)
        degree = torch.sum(adj_tilde, dim=1)
        d_inv_sqrt = torch.pow(degree + 1e-8, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)

        self.adj_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj_tilde), d_mat_inv_sqrt)

    def train_vgae_if_needed(self, round_idx):
        if round_idx % self.conf['t_vgae'] != 0 or self.f_polluted is None:
            return

        self.vgae.train()
        for epoch in range(self.conf['vgae_epochs']):
            self.optimizer.zero_grad()

            if torch.isnan(self.f_polluted).any():
                print(f"⚠️ [VGAE] NaN Input, Skip Round {round_idx}")
                return

            z, mu, logvar = self.vgae(self.f_polluted.detach(), self.adj_norm.detach())
            rec_adj, _ = inner_product_decoder(z)

            loss_rec = F.binary_cross_entropy(rec_adj, self.adj_label.detach())
            loss_kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

            loss = loss_rec + loss_kl

            if torch.isnan(loss):
                print(f"⚠️ [VGAE] NaN Loss, Stop Training")
                break

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.vgae.parameters(), max_norm=1.0)
            self.optimizer.step()

    def generate_malicious_update(self, benign_update_template):
        """
        基于梯度的恶意样本生成 (Stable Cosine Version)
        """
        if self.f_polluted is None:
            return self._generate_random_update(benign_update_template)

        self.vgae.eval()
        for p in self.vgae.parameters(): p.requires_grad = False

        with torch.no_grad():
            # 计算良性潜在中心
            _, mu_benign, _ = self.vgae(self.f_polluted, self.adj_norm)
            z_target = torch.mean(mu_benign, dim=0, keepdim=True)
            mean_benign_vec = torch.mean(self.f_polluted, dim=0)

        # 确定攻击方向: 反向于良性更新的平均方向
        attack_direction = -torch.sign(mean_benign_vec).detach()
        # 归一化攻击方向，确保计算 Cosine 时数值稳定
        # attack_direction = F.normalize(attack_direction, p=2, dim=0)
        # 上面这行不需要，因为 cosine_similarity 会自动处理归一化

        # 初始化恶意输入 w_mal
        w_mal = mean_benign_vec.clone().detach().unsqueeze(0)
        w_mal.requires_grad_(True)

        # [优化] 降低学习率，防止震荡
        input_optimizer = optim.Adam([w_mal], lr=0.01)
        lambda_reg = self.conf['lambda_attack']

        for i in range(30):
            input_optimizer.zero_grad()

            if torch.isnan(w_mal).any(): break

            h1 = F.relu(self.vgae.gc1.linear(w_mal))
            z_mu_approx = self.vgae.gc_mu.linear(h1)

            # Loss 1: 隐蔽性
            loss_latent = F.mse_loss(z_mu_approx, z_target)

            # Loss 2: 破坏性 (改为余弦相似度!)
            # 我们希望 w_mal 与 attack_direction 的相似度越大越好
            # Cosine Sim 范围 [-1, 1].
            # 我们最小化负的相似度 -> -1 (完全同向)
            # 注意：unsqueeze 确保维度匹配 [1, d] vs [1, d]
            cos_sim = F.cosine_similarity(w_mal, attack_direction.unsqueeze(0))
            loss_attack = -cos_sim.mean()

            # Total Loss
            total_loss = loss_latent + lambda_reg * loss_attack

            if torch.isnan(total_loss): break

            total_loss.backward()
            input_optimizer.step()

        for p in self.vgae.parameters(): p.requires_grad = True

        final_mal_vec = w_mal.detach().squeeze()

        if torch.isnan(final_mal_vec).any():
            print("⚠️ [Attack] Generated NaN, fallback to random.")
            return self._generate_random_update(benign_update_template)

        return self._vec_to_dict(final_mal_vec, benign_update_template)

    def _generate_random_update(self, template):
        res = {}
        for k, v in template.items():
            if v.dtype in [torch.float32, torch.float64]:
                res[k] = torch.randn_like(v) * self.conf['lambda_attack']
            else:
                res[k] = v
        return res

    def _vec_to_dict(self, vec, template):
        res = {}
        idx = 0
        for k in sorted(template.keys()):
            v = template[k]
            if v.dtype in [torch.float32, torch.float64]:
                numel = v.numel()
                res[k] = vec[idx:idx + numel].view(v.shape)
                idx += numel
            else:
                res[k] = copy.deepcopy(v)
        return res