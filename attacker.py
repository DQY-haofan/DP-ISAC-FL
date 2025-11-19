import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy


class GraphConv(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        return torch.matmul(adj, self.linear(x))


class VGAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.gc1 = GraphConv(input_dim, hidden_dim)
        self.gc_mu = GraphConv(hidden_dim, latent_dim)
        self.gc_logvar = GraphConv(hidden_dim, latent_dim)

    def encode(self, x, adj):
        h = F.relu(self.gc1(x, adj))
        return self.gc_mu(h, adj), self.gc_logvar(h, adj)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std
        return mu

    def forward(self, x, adj):
        mu, logvar = self.encode(x, adj)
        return self.reparameterize(mu, logvar), mu, logvar


def inner_product_decoder(z):
    adj_logits = torch.matmul(z, z.t())
    return torch.sigmoid(adj_logits), adj_logits


class VGAEAttacker:
    def __init__(self, config, input_dim):
        self.conf = config['attack']
        self.device = config['device']
        self.vgae = VGAE(input_dim, 32, self.conf['latent_dim']).to(self.device)
        self.optimizer = optim.Adam(self.vgae.parameters(), lr=self.conf['vgae_lr'])
        self.f_polluted = None
        self.adj_norm = None

    def _flatten_updates(self, updates):
        flat_list = []
        for update in updates:
            vec = []
            for k in sorted(update.keys()):
                if update[k].dtype in [torch.float32, torch.float64]: vec.append(update[k].view(-1))
            flat_list.append(torch.cat(vec))
        return torch.stack(flat_list) if flat_list else None

    def observe_dp_updates(self, dp_updates, round_idx, external_mask=None):
        if not dp_updates: return
        flat = self._flatten_updates(dp_updates)
        if external_mask is not None:
            obs_idx = torch.where(external_mask)[0]
        else:
            n_obs = int(len(dp_updates) * self.conf['q_eaves'])
            obs_idx = torch.randperm(len(dp_updates))[:n_obs].to(self.device)

        noise = torch.normal(0, self.conf['eaves_sigma'], size=flat.shape).to(self.device)
        self.f_polluted = (flat + noise)[obs_idx]
        self._build_graph(self.f_polluted)

    def _build_graph(self, feats):
        norm = F.normalize(feats, p=2, dim=1)
        sim = torch.mm(norm, norm.t())
        adj = (sim >= self.conf['tau_sim']).float()
        adj.fill_diagonal_(0)
        self.adj_label = adj

        adj_tilde = adj + torch.eye(adj.shape[0]).to(self.device)
        deg = torch.sum(adj_tilde, dim=1)
        d_inv = torch.diag(torch.pow(deg, -0.5))
        d_inv[torch.isinf(d_inv)] = 0.
        self.adj_norm = torch.mm(torch.mm(d_inv, adj_tilde), d_inv)

    def train_vgae_if_needed(self, round_idx):
        if round_idx % self.conf['t_vgae'] != 0 or self.f_polluted is None: return
        self.vgae.train()
        for _ in range(self.conf['vgae_epochs']):
            self.optimizer.zero_grad()
            z, mu, logvar = self.vgae(self.f_polluted.detach(), self.adj_norm.detach())
            rec, _ = inner_product_decoder(z)
            loss = F.binary_cross_entropy(rec, self.adj_label.detach()) - 0.5 * torch.mean(
                torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
            loss.backward()
            self.optimizer.step()

    def generate_malicious_update(self, template):
        if self.f_polluted is None: return self._rnd_update(template)

        self.vgae.eval()
        for p in self.vgae.parameters(): p.requires_grad = False

        with torch.no_grad():
            _, mu, _ = self.vgae(self.f_polluted, self.adj_norm)
            z_target = torch.mean(mu, dim=0, keepdim=True)
            mean_vec = torch.mean(self.f_polluted, dim=0)

        w_mal = mean_vec.clone().detach().unsqueeze(0).requires_grad_(True)
        opt = optim.Adam([w_mal], lr=0.1)
        target_dir = -torch.sign(mean_vec).detach()

        for _ in range(30):
            opt.zero_grad()
            h1 = F.relu(self.vgae.gc1.linear(w_mal))
            z_approx = self.vgae.gc_mu.linear(h1)
            loss = F.mse_loss(z_approx, z_target) - self.conf['lambda_attack'] * torch.mean(
                torch.matmul(w_mal, target_dir.unsqueeze(1)))
            loss.backward()
            opt.step()

        for p in self.vgae.parameters(): p.requires_grad = True
        return self._vec_to_dict(w_mal.detach().squeeze(), template)

    def _rnd_update(self, t):
        return {k: torch.randn_like(v) * self.conf['lambda_attack'] if v.dtype in [torch.float32, torch.float64] else v
                for k, v in t.items()}

    def _vec_to_dict(self, vec, t):
        res, idx = {}, 0
        for k in sorted(t.keys()):
            v = t[k]
            if v.dtype in [torch.float32, torch.float64]:
                n = v.numel()
                res[k] = vec[idx:idx + n].view(v.shape);
                idx += n
            else:
                res[k] = copy.deepcopy(v)
        return res