# 文件名: main.py
# 作用: 快速单次测试脚本。
# 版本: Final

import yaml, torch
from datasets import get_dataset, partition_dataset_dirichlet
from server import Server
from torch.utils.data import DataLoader


def main():
    try:
        with open('config.yaml') as f:
            c = yaml.safe_load(f)
    except:
        print("Missing config.yaml"); return

    # 快速测试覆盖
    c['num_rounds'], c['num_clients'] = 5, 20
    if torch.cuda.is_available(): c['device'] = 'cuda'

    print("=== Quick Test Start ===")
    ds, tds = get_dataset(c['dataset'], c['data_root'])
    idx = partition_dataset_dirichlet(ds, c['num_clients'], c['alpha'])
    srv = Server(c, ds, idx)
    ldr = DataLoader(tds, batch_size=1000)

    for t in range(5):
        s = srv.run_round(t)
        _, acc = srv.evaluate(ldr)
        print(f"R{t}: Acc={acc:.1f}%, Sigma={s['dp_sigma']:.3f}")
    print("Test Passed.")


if __name__ == "__main__": main()