import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # 强制同步报错

import torch
import numpy as np
from vgae_attacker import VGAEAttacker
from datasets import partition_dataset_dirichlet, get_dataset
from server import Server
import yaml


def test_data_partition():
    print("\n[1/4] Testing Data Partitioning...")
    try:
        train_ds, _ = get_dataset('mnist', './data')
        indices = partition_dataset_dirichlet(train_ds, 100, 0.3, 42)

        # Check indices
        total_samples = sum([len(v) for v in indices.values()])
        print(f"  - Total samples distributed: {total_samples}")
        print(f"  - Min samples per client: {min([len(v) for v in indices.values()])}")
        print(f"  - Max samples per client: {max([len(v) for v in indices.values()])}")

        # Check boundaries
        all_indices = np.concatenate(list(indices.values()))
        print(f"  - Max index value: {all_indices.max()}")
        print(f"  - Dataset length: {len(train_ds)}")

        if all_indices.max() >= len(train_ds):
            print("  ❌ ERROR: Index out of bounds in partition!")
        else:
            print("  ✅ Partition looks OK.")

    except Exception as e:
        print(f"  ❌ Crash in Data Partition: {e}")


def test_attacker_graph_build():
    print("\n[2/4] Testing VGAE Graph Building (CUDA)...")
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"  - Using device: {device}")

        # Mock config
        config = {
            'attack': {
                'latent_dim': 16, 'vgae_lr': 0.01, 'q_eaves': 0.8,
                'eaves_sigma': 0.01, 'tau_sim': 0.5, 't_vgae': 1, 'vgae_epochs': 1
            }
        }
        input_dim = 1000
        attacker = VGAEAttacker(config, input_dim)

        # Simulate updates
        num_updates = 10
        updates = []
        for _ in range(num_updates):
            updates.append({'w': torch.randn(input_dim).to(device)})

        print("  - Simulating observe_dp_updates...")
        attacker.observe_dp_updates(updates, 0)

        if attacker.f_polluted is not None:
            print(f"  - f_polluted shape: {attacker.f_polluted.shape}")
            print(f"  - adj_norm shape: {attacker.adj_norm.shape}")

        print("  - Simulating training step...")
        attacker.train_vgae_if_needed(0)

        print("  ✅ VGAE operations OK.")

    except Exception as e:
        print(f"  ❌ Crash in VGAE: {e}")
        import traceback
        traceback.print_exc()


def test_generation():
    print("\n[3/4] Testing Malicious Generation...")
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        config = {
            'attack': {
                'latent_dim': 16, 'vgae_lr': 0.01, 'q_eaves': 0.8,
                'eaves_sigma': 0.01, 'tau_sim': 0.5, 't_vgae': 1, 'vgae_epochs': 1,
                'lambda_attack': 3.0
            }
        }
        input_dim = 100
        attacker = VGAEAttacker(config, input_dim)

        # Mock state
        attacker.f_polluted = torch.randn(8, input_dim).to(device)
        attacker._build_graph(attacker.f_polluted)

        template = {'w': torch.randn(input_dim).to(device)}

        print("  - Generating update...")
        mal = attacker.generate_malicious_update(template)

        if torch.isnan(mal['w']).any():
            print("  ❌ ERROR: Generated update contains NaN!")
        else:
            print("  ✅ Generation OK.")

    except Exception as e:
        print(f"  ❌ Crash in Generation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("=== Starting Diagnostics ===")
    test_data_partition()
    test_attacker_graph_build()
    test_generation()
    print("\n=== Diagnostics Finished ===")