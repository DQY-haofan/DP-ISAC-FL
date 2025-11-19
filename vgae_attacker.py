# 文件名: vgae_attacker.py
# 作用: 核心攻击逻辑。包含 VGAE 模型和基于梯度的恶意样本生成。
# 版本: Fixed (Solved NaN explosion via Hard Scaling & Bilateral Clamping)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy


# --- GCN & VGAE 层定义 ---

class GraphConv(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConv, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        # [Fix 3] 在每一层卷积后加 LayerNorm，防止层间信号放大
        self.ln = nn.LayerNorm(out_features)

    def forward(self, x, adj):
        support = self.linear(x)
        output = torch.matmul(adj, support)
        return self.ln(output)


class VGAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VGAE, self).__init__()
        # [Fix 1] 移除输入处的 LayerNorm (计算太重且对高维稀疏向量不稳定)
        # 改用硬缩放 (Hard Scaling) 策略，见 forward

        self.gc1 = GraphConv(input_dim, hidden_dim)
        self.gc_mu = GraphConv(hidden_dim, latent_dim)
        self.gc_logvar = GraphConv(hidden_dim, latent_dim)

    def encode(self, x, adj):
        hidden = F.relu(self.gc1(x, adj))
        return self.gc_mu(hidden, adj), self.gc_logvar(hidden, adj)

    def reparameterize(self, mu, logvar):
        if self.training:
            # [Fix 2] 双边截断，防止 KL 散度中的 (1 + logvar) 导致 Loss 爆炸
            # logvar = -10 -> std = exp(-5) ~ 0.006 (足够小)
            # logvar = 10  -> std = exp(5) ~ 148 (足够大)
            logvar = torch.clamp(logvar, min=-10.0, max=10.0)
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
    # [Fix 4] Logits 截断，防止 sigmoid 梯度饱和
    adj_logits = torch.clamp(adj_logits, min=-10.0, max=10.0)
    probs = torch.sigmoid(adj_logits)
    return probs, adj_logits


# --- 攻击者逻辑 ---

class VGAEAttacker:
    def __init__(self, config, input_dim):
        self.conf = config['attack']
        self.device = config['device']
        self.input_dim = input_dim

        # [Fix 1] 硬缩放因子。将 20万维的梯度数值强行压缩，防止点积爆炸
        # 梯度通常在 1e-2 ~ 1e-4 量级，DP 噪声在 1e-2 量级。
        # 经过 Linear(200000) 后，方差会扩大 200000 倍。
        # 除以 100.0 是经验值，可以有效压制初始方差。
        self.input_scale = 100.0

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
            # 确保 mask 长度匹配
            if len(external_mask) != num_updates:
                # 如果长度不匹配，可能是只有部分 benign 客户端被选中
                # 这里做一个防御性编程，取前 N 个
                obs_indices = torch.where(external_mask[:num_updates])[0]
            else:
                obs_indices = torch.where(external_mask)[0]
        else:
            num_obs = int(num_updates * self.conf['q_eaves'])
            obs_indices = torch.randperm(num_updates)[:num_obs].to(self.device)

        if len(obs_indices) == 0:
            self.f_polluted = None
            return

        # 添加窃听噪声
        noise = torch.normal(0, self.conf['eaves_sigma'], size=flat_updates.shape).to(self.device)
        polluted_raw = flat_updates + noise

        # [Fix 5] 极值截断 (Sanity Check)
        polluted_raw = torch.clamp(polluted_raw, min=-100.0, max=100.0)

        self.f_polluted = polluted_raw[obs_indices]
        self._build_graph(self.f_polluted)

    def _build_graph(self, features):
        # 计算相似度时不需要 scaling，因为 normalize 会抵消常数因子
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

        # [Fix 1] 应用 Scaling
        inputs = self.f_polluted / self.input_scale
        adj = self.adj_norm.detach()
        label = self.adj_label.detach()

        for epoch in range(self.conf['vgae_epochs']):
            self.optimizer.zero_grad()

            z, mu, logvar = self.vgae(inputs, adj)

            if torch.isnan(z).any():
                # print(f"⚠️ [VGAE] Latent Z is NaN at Round {round_idx} (Epoch {epoch}). Aborting training.")
                break

            rec_adj, _ = inner_product_decoder(z)

            loss_rec = F.binary_cross_entropy(rec_adj, label)

            # KL Divergence 防溢出保护
            # clamp logvar 已经在 reparameterize 里做了，但为了计算 Loss 安全，再次 clamp
            mu_safe = torch.clamp(mu, min=-10, max=10)
            logvar_safe = torch.clamp(logvar, min=-10, max=10)

            loss_kl = -0.5 * torch.mean(torch.sum(1 + logvar_safe - mu_safe.pow(2) - logvar_safe.exp(), dim=1))

            loss = loss_rec + loss_kl

            if torch.isnan(loss):
                # print(f"⚠️ [VGAE] Loss NaN at Round {round_idx}")
                break

            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.vgae.parameters(), max_norm=1.0)
            self.optimizer.step()

    def generate_malicious_update(self, benign_update_template):
        if self.f_polluted is None:
            return self._generate_random_update(benign_update_template)

        self.vgae.eval()
        # 冻结参数
        for p in self.vgae.parameters(): p.requires_grad = False

        # [Fix 1] 应用 Scaling 准备输入
        inputs = self.f_polluted / self.input_scale

        with torch.no_grad():
            # 获取 benign 样本的潜在均值
            # 使用 encode (不采样) 来获取准确的 mu
            _, mu_benign, _ = self.vgae(inputs, self.adj_norm)

            # 目标：模仿 benign 样本的中心
            z_target = torch.mean(mu_benign, dim=0, keepdim=True)

            # 初始恶意更新：使用 benign 的平均值
            mean_benign_vec = torch.mean(self.f_polluted, dim=0)

        if torch.isnan(z_target).any():
            return self._generate_random_update(benign_update_template)

        # 攻击方向：反向
        attack_direction = -torch.sign(mean_benign_vec).detach()

        # 优化变量：w_mal (未缩放的原始空间)
        w_mal = mean_benign_vec.clone().detach().unsqueeze(0)
        w_mal.requires_grad_(True)

        # 攻击优化器
        input_optimizer = optim.Adam([w_mal], lr=0.05)  # 提高一点 LR
        lambda_reg = self.conf['lambda_attack']

        for i in range(30):
            input_optimizer.zero_grad()

            # [重要] 必须对 w_mal 进行同样的 Scaling 才能输入 VGAE
            w_scaled = w_mal / self.input_scale

            # 手动执行 GraphConv 前向过程 (模拟第一层)
            # 简化：我们直接通过 gc1 的 linear 层，忽略 adj (即 adj=I)
            # 这是一种近似，因为我们只想让它的 feature embedding 靠近 benign

            # Layer 1: GC1
            h1 = self.vgae.gc1.linear(w_scaled)
            h1 = self.vgae.gc1.ln(h1)
            h1 = F.relu(h1)

            # Layer 2: GC_MU
            z_mu_approx = self.vgae.gc_mu.linear(h1)
            z_mu_approx = self.vgae.gc_mu.ln(z_mu_approx)

            # Loss 1: 隐空间距离 (让恶意更新看起来像 benign)
            loss_latent = F.mse_loss(z_mu_approx, z_target)

            # Loss 2: 攻击目标 (余弦相似度反向)
            cos_sim = F.cosine_similarity(w_mal, attack_direction.unsqueeze(0))
            loss_attack = -cos_sim.mean()

            total_loss = loss_latent + lambda_reg * loss_attack

            if torch.isnan(total_loss):
                break

            total_loss.backward()
            input_optimizer.step()

        # 解冻
        for p in self.vgae.parameters(): p.requires_grad = True

        final_mal_vec = w_mal.detach().squeeze()
        if torch.isnan(final_mal_vec).any():
            return self._generate_random_update(benign_update_template)

        return self._vec_to_dict(final_mal_vec, benign_update_template)

    def _generate_random_update(self, template):
        res = {}
        for k, v in template.items():
            if v.dtype in [torch.float32, torch.float64]:
                # 生成与参数同样形状的噪声
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