import yaml
import torch
import numpy as np
from runner import SimulationRunner


def diagnose():
    print("=== 🏥 Training Health Check ===")

    # 1. Check Ideal Scenario (No Attack)
    print("\n[Test 1] Ideal Scenario (No Attack)")
    runner = SimulationRunner('config.yaml')
    conf_ideal = {
        'scenario': 'Diag_Ideal',
        'num_rounds': 5,  # Quick check
        'aggregator': 'FedAvg',
        'attack': {'malicious_fraction': 0.0}
    }
    acc_ideal = runner.run_single_seed(conf_ideal, 42, 'logs/diag_ideal.csv')
    print(f"  -> Ideal Acc: {acc_ideal:.2f}%")

    if acc_ideal < 20.0:
        print("  ❌ ERROR: Benign training failed! Check LR or Model.")
        return  # Stop here if benign training is broken

    # 2. Check Krum with Zero Attack (Lambda=0)
    # 这一步是为了测试 Krum 在 Non-IID 下是否会误杀良性更新
    print("\n[Test 2] Vulnerable (Krum + Zero Attack)")
    conf_krum_zero = {
        'scenario': 'Diag_Krum_Zero',
        'num_rounds': 5,
        'aggregator': 'Krum',
        'attack': {
            'malicious_fraction': 0.2,
            'lambda_attack': 0.0  # No perturbation
        }
    }
    acc_krum = runner.run_single_seed(conf_krum_zero, 42, 'logs/diag_krum_zero.csv')
    print(f"  -> Krum (Zero Attack) Acc: {acc_krum:.2f}%")

    if acc_krum < acc_ideal - 20.0:
        print("  ⚠️ Warning: Krum is hurting performance significantly even without attack.")
        print("     (Common in Non-IID settings. Consider relaxing Non-IID alpha)")

    # 3. Check Weak Attack
    print("\n[Test 3] Vulnerable (Krum + Weak Attack Lambda=0.1)")
    conf_weak = {
        'scenario': 'Diag_Weak',
        'num_rounds': 5,
        'aggregator': 'Krum',
        'attack': {
            'malicious_fraction': 0.2,
            'lambda_attack': 0.1
        }
    }
    acc_weak = runner.run_single_seed(conf_weak, 42, 'logs/diag_weak.csv')
    print(f"  -> Weak Attack Acc: {acc_weak:.2f}%")


if __name__ == "__main__":
    diagnose()