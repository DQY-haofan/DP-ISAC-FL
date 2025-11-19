# 文件名: calibration.py
# 作用: 离线校准脚本。估算 L 和 G^2 常数。
# 版本: Final

import yaml
import torch
import numpy as np
from server import Server
from datasets import get_dataset, partition_dataset_dirichlet


def calibrate():
    print("Running Calibration...")
    with open('config.yaml') as f:
        c = yaml.safe_load(f)
    # 强制无噪声环境
    c['scenario'], c['aggregator'], c['attack']['malicious_fraction'] = 'Ideal', 'FedAvg', 0.0
    c['dp_sigma_z'], c['isac_sigma_ch'] = 0.0, 0.0

    if torch.cuda.is_available(): c['device'] = 'cuda'

    ds, _ = get_dataset(c['dataset'], c['data_root'])
    idx = partition_dataset_dirichlet(ds, c['num_clients'], c['alpha'], 42)
    srv = Server(c, ds, idx)

    Ls, Gs = [], []
    w_prev, g_prev = None, None

    for t in range(10):  # 跑 10 轮估算
        w_curr = torch.cat([p.view(-1) for p in srv.global_model.parameters()])
        sel = srv.select_clients()
        # 获取纯净梯度 (Update / lr)
        ups = torch.stack([torch.cat(
            [v.view(-1) for k, v in sorted(c.local_train(srv.global_model.state_dict()).items()) if
             v.dtype == torch.float32]) for c in sel])

        # G^2 = Var(gradients)
        grads = -ups / c['lr']
        g_var = torch.var(grads, dim=0).mean().item()
        Gs.append(g_var)

        # L = ||g_t - g_{t-1}|| / ||w_t - w_{t-1}||
        g_curr = torch.mean(grads, dim=0)
        if w_prev is not None:
            wd = torch.norm(w_curr - w_prev).item()
            if wd > 1e-6:
                L_est = torch.norm(g_curr - g_prev).item() / wd
                Ls.append(L_est)

        w_prev, g_prev = w_curr, g_curr
        srv.run_round(t)
        print(f"Round {t}: G^2={g_var:.4f}, L={Ls[-1] if Ls else 0:.4f}")

    L = np.max(Ls) if Ls else 0.1
    print(f"\nSuggested: const_c_dp={L * 0.1:.4f}, const_c_attack={L:.4f}")


if __name__ == "__main__": calibrate()