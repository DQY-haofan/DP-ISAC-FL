# 文件名: tune_attacker.py
# 作用: 对抗性搜索。寻找能攻破 Krum 的 Lambda 值。
# 版本: Final

from runner import SimulationRunner

def tune():
    print("Tuning Attacker...")
    lams = [1.0, 2.0, 3.0, 5.0]
    for l in lams:
        r = SimulationRunner('config.yaml')
        # 强制使用 Krum 聚合器和 Vulnerable 模式
        ovr = {
            'scenario': 'Tune',
            'num_rounds': 5,
            'aggregator': 'Krum',
            'attack': {'malicious_fraction': 0.2, 'lambda_attack': l}
        }
        acc = r.run_single_seed(ovr, 42, f'logs/tune_l{l}.csv')
        print(f"Lambda={l}, Acc={acc:.2f}% (Lower Accuracy = Better Attack)")

if __name__=="__main__": tune()