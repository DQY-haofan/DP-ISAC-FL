# 文件名: vgae_attacker.py
# 作用: 核心攻击逻辑。包含 VGAE 模型和基于梯度的恶意样本生成 (Input Optimization)。
# 版本: Final (Gradient-based optimization & Stability Fixes)

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
    H' = ReLU( D^-1/2 A D^-1/2 X W )
    """

    def __init__(self, in_features, out_features):
        super(GraphConv, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        # x: [N, in_features]
        # adj: [N, N] (Normalized Adjacency)
        support = self.linear(x)
        output = torch.matmul(adj, support)
        return output


class VGAE(nn.Module):
    """
    变分图自编码器
    Encoder: GCN -> mu, logvar
    Decoder: Inner Product
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
            # [稳定性修复] 限制 logvar，防止 exp() 爆炸
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
    """解码器: A_hat = sigmoid(Z * Z^T)"""
    adj_logits = torch.matmul(z, z.t())
    # [稳定性修复] 限制 sigmoid 输出，防止 BCE Loss 计算 log(0) 崩溃
    # 理论上 sigmoid 输出 (0, 1)，但在 float32 下可能溢出为 0 或 1
    # 截断到 [eps, 1-eps] 是最安全的做法
    probs = torch.sigmoid(adj_logits)
    probs = torch.clamp(probs, min=1e-7, max=1.0 - 1e-7)
    return probs, adj_logits


# --- 攻击者逻辑 ---

class VGAEAttacker:
    def __init__(self, config, input_dim):
        self.conf = config['attack']
        self.device = config['device']
        self.input_dim = input_dim

        # 初始化 VGAE 模型
        self.vgae = VGAE(input_dim, 32, self.conf['latent_dim']).to(self.device)
        self.optimizer = optim.Adam(self.vgae.parameters(), lr=self.conf['vgae_lr'])

        # 缓存
        self.f_polluted = None  # 特征矩阵
        self.adj_norm = None  # 归一化邻接矩阵
        self.adj_label = None  # 原始邻接矩阵 (用于计算 Loss)

    def _flatten_updates(self, updates):
        """将参数字典列表展平为矩阵 [N, d]"""
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
        """
        窃听阶段:
        1. 接收良性客户端的 DP 更新
        2. 筛选可观测的客户端
        3. 叠加窃听信道噪声
        4. 构建 F_polluted
        """
        num_updates = len(dp_updates)
        if num_updates == 0: return

        flat_updates = self._flatten_updates(dp_updates)

        # 确定观测掩码 (Silo Effect)
        if external_mask is not None:
            # 使用 R-JORA 强制指定的掩码
            obs_indices = torch.where(external_mask)[0]
        else:
            # 回退到旧逻辑 (随机掩码)
            num_obs = int(num_updates * self.conf['q_eaves'])
            # 随机选择被窃听的客户端
            obs_indices = torch.randperm(num_updates)[:num_obs].to(self.device)

        # [安全检查] 确保有观测到的客户端，否则跳过
        if len(obs_indices) == 0:
            self.f_polluted = None
            return

        # 叠加窃听噪声
        # \tilde{\Delta w}_{i, eaves} = \tilde{\Delta w}_i + n_{eaves}
        noise = torch.normal(0, self.conf['eaves_sigma'], size=flat_updates.shape).to(self.device)
        polluted_updates = flat_updates + noise
        self.f_polluted = polluted_updates[obs_indices]

        # 构图
        self._build_graph(self.f_polluted)

    def _build_graph(self, features):
        """
        构建局部图 A_partial
        基于 Cosine Similarity 和 Tau_sim
        """
        # [稳定性] 加 eps 防止除零
        features_norm = F.normalize(features, p=2, dim=1, eps=1e-8)
        sim_matrix = torch.mm(features_norm, features_norm.t())

        # 阈值截断
        adj = (sim_matrix >= self.conf['tau_sim']).float()

        # 移除自环 (用于 Loss 计算)
        adj.fill_diagonal_(0)
        self.adj_label = adj

        # GCN 归一化: D^-1/2 A_tilde D^-1/2
        # 加自环用于卷积
        adj_tilde = adj + torch.eye(adj.shape[0]).to(self.device)
        degree = torch.sum(adj_tilde, dim=1)

        # [稳定性] 防止孤立点导致的除零
        d_inv_sqrt = torch.pow(degree + 1e-8, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)

        self.adj_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj_tilde), d_mat_inv_sqrt)

    def train_vgae_if_needed(self, round_idx):
        """
        按频率 T_vgae 训练 VGAE
        """
        if round_idx % self.conf['t_vgae'] != 0 or self.f_polluted is None:
            return

        self.vgae.train()
        for epoch in range(self.conf['vgae_epochs']):
            self.optimizer.zero_grad()

            # [稳定性] 检查输入是否包含 NaN
            if torch.isnan(self.f_polluted).any():
                print(f"⚠️ [VGAE] 警告: 输入特征包含 NaN，跳过本轮训练 (Round {round_idx})")
                return

            z, mu, logvar = self.vgae(self.f_polluted.detach(), self.adj_norm.detach())
            rec_adj, _ = inner_product_decoder(z)

            # [关键修复] 输入已经截断在 [eps, 1-eps]，这里不需要再截断

            loss_rec = F.binary_cross_entropy(rec_adj, self.adj_label.detach())

            # KL Divergence
            # [稳定性] 对求和项进行保护，虽然 reparameterize 已经 clamp 了 logvar
            loss_kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

            loss = loss_rec + loss_kl

            if torch.isnan(loss):
                print(f"⚠️ [VGAE] 警告: Loss 变成 NaN，停止训练 (Round {round_idx})")
                break

            loss.backward()

            # [稳定性] 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.vgae.parameters(), max_norm=1.0)

            self.optimizer.step()

    def generate_malicious_update(self, benign_update_template):
        """
        使用 Gradient-based Input Optimization 生成恶意样本
        策略:
        1. 固定 VGAE 参数
        2. 优化输入 w_mal，使其:
           a) 隐蔽性: 编码后的潜在向量 z_mal 靠近良性中心 z_target
           b) 破坏性: 在原始空间沿着攻击方向 (attack_direction) 投影最大化
        """
        if self.f_polluted is None:
            return self._generate_random_update(benign_update_template)

        self.vgae.eval()
        # 冻结 VGAE 参数
        for p in self.vgae.parameters(): p.requires_grad = False

        with torch.no_grad():
            # 计算良性潜在中心 (Target Latent State)
            _, mu_benign, _ = self.vgae(self.f_polluted, self.adj_norm)
            z_target = torch.mean(mu_benign, dim=0, keepdim=True)
            mean_benign_vec = torch.mean(self.f_polluted, dim=0)

        # 确定攻击方向: 反向于良性更新的平均方向 (Untargeted)
        attack_direction = -torch.sign(mean_benign_vec).detach()

        # 初始化恶意输入 w_mal (从良性均值开始)
        w_mal = mean_benign_vec.clone().detach().unsqueeze(0)
        w_mal.requires_grad_(True)

        # 定义优化器 (只优化 w_mal)
        input_optimizer = optim.Adam([w_mal], lr=0.1)
        lambda_reg = self.conf['lambda_attack']

        # 优化循环 (30步)
        for i in range(30):
            input_optimizer.zero_grad()

            # 检查 w_mal 是否异常
            if torch.isnan(w_mal).any(): break

            # 近似编码: 只计算 GCN 第一层投影
            # 假设 w_mal 处于图的 "平均位置"，忽略邻接矩阵对其影响的细节
            h1 = F.relu(self.vgae.gc1.linear(w_mal))
            z_mu_approx = self.vgae.gc_mu.linear(h1)

            # Loss 1: 隐蔽性 (Latent Loss)
            loss_latent = F.mse_loss(z_mu_approx, z_target)

            # Loss 2: 破坏性 (Attack Loss)
            # 最大化在攻击方向上的投影 -> 最小化 - dot(w, delta)
            loss_attack = -torch.mean(torch.matmul(w_mal, attack_direction.unsqueeze(1)))

            total_loss = loss_latent + lambda_reg * loss_attack

            if torch.isnan(total_loss): break

            total_loss.backward()
            input_optimizer.step()

        # 恢复 VGAE 梯度状态
        for p in self.vgae.parameters(): p.requires_grad = True

        final_mal_vec = w_mal.detach().squeeze()

        # [稳定性] 最终检查
        if torch.isnan(final_mal_vec).any():
            print("⚠️ [Attack] 生成的恶意更新包含 NaN，回退到随机噪声。")
            return self._generate_random_update(benign_update_template)

        return self._vec_to_dict(final_mal_vec, benign_update_template)

    def _generate_random_update(self, template):
        """生成简单随机噪声 (用于冷启动或回退)"""
        res = {}
        for k, v in template.items():
            if v.dtype in [torch.float32, torch.float64]:
                res[k] = torch.randn_like(v) * self.conf['lambda_attack']
            else:
                res[k] = v
        return res

    def _vec_to_dict(self, vec, template):
        """将扁平向量还原为参数字典"""
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