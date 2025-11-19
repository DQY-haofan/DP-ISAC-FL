import torch
import torch.nn.functional as F
from collections import deque


class STGAAggregator:
    def __init__(self, config):
        self.conf = config['r_jora']
        self.device = config['device']
        self.history = deque(maxlen=5)
        self.diagnostics = {'avg_trust_benign': 0.0, 'avg_trust_malicious': 0.0}

    def _flatten(self, u):
        return torch.cat([u[k].view(-1) for k in sorted(u.keys()) if u[k].dtype in [torch.float32, torch.float64]])

    def aggregate(self, updates, client_types=None):
        if not updates: return None
        if not self.conf['enabled'] or not self.conf['enable_stga']: return self._fedavg(updates)

        flat = torch.stack([self._flatten(u) for u in updates]).to(self.device)
        spatial_center = torch.median(flat, dim=0).values
        s_spat = F.cosine_similarity(flat, spatial_center.unsqueeze(0), dim=1)

        if self.history:
            exp_upd = self.history[-1].to(self.device)
            s_temp = F.cosine_similarity(flat, exp_upd.unsqueeze(0), dim=1)
        else:
            s_temp = torch.ones(len(updates)).to(self.device)

        trust = torch.exp((self.conf['stga_alpha'] * s_temp + (1 - self.conf['stga_alpha']) * s_spat) * 5)
        weights = trust / trust.sum()

        if client_types:
            b = [w.item() for w, t in zip(weights, client_types) if t == 'benign']
            m = [w.item() for w, t in zip(weights, client_types) if t != 'benign']
            self.diagnostics['avg_trust_benign'] = sum(b) / len(b) if b else 0
            self.diagnostics['avg_trust_malicious'] = sum(m) / len(m) if m else 0

        weighted_avg = torch.mv(flat.t(), weights)
        self.history.append(weighted_avg.detach().cpu())
        return self._unflatten(weighted_avg, updates[0])

    def _fedavg(self, updates):
        avg = {k: torch.zeros_like(v) for k, v in updates[0].items()}
        n = len(updates)
        for u in updates:
            for k, v in u.items(): avg[k] += v
        for k in avg: avg[k] /= n
        return avg

    def _unflatten(self, vec, t):
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