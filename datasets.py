# 文件名: datasets.py
# 作用: 数据集加载与 Non-IID 划分 (中文注释增强版)

import torch
import numpy as np
from torchvision import datasets, transforms


def get_dataset(dataset_name, data_root):
    """下载并加载数据集"""
    if dataset_name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST(data_root, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(data_root, train=False, download=True, transform=transform)
    elif dataset_name == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = datasets.CIFAR10(data_root, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(data_root, train=False, download=True, transform=transform)
    else:
        raise ValueError(f"未知数据集: {dataset_name}")
    return train_dataset, test_dataset


def partition_dataset_dirichlet(dataset, num_clients, alpha=0.3, seed=42):
    """
    使用 Dirichlet 分布进行 Non-IID 数据划分
    """
    np.random.seed(seed)

    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    else:
        labels = np.array(dataset.train_labels)

    num_classes = len(dataset.classes)
    min_size = 0
    client_idcs = {}

    # 安全计数器，防止死循环
    attempt = 0
    max_attempts = 100

    while min_size < 10:
        if attempt > max_attempts:
            print(f"警告: 尝试 {max_attempts} 次后仍无法满足最小样本数要求，强制继续。")
            break
        attempt += 1

        client_idcs = {i: [] for i in range(num_clients)}

        for k in range(num_classes):
            idx_k = np.where(labels == k)[0]
            np.random.shuffle(idx_k)

            # [修复] 如果该类样本太少，直接随机分配，不走 Dirichlet
            if len(idx_k) < num_clients:
                subset_clients = np.random.choice(num_clients, len(idx_k), replace=False)
                for i, client_idx in enumerate(subset_clients):
                    client_idcs[client_idx].append(idx_k[i])
                continue

            # 生成概率分布
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))

            # [关键修复] 防止除以零
            total_p = proportions.sum()
            if total_p < 1e-9:
                proportions = np.ones(num_clients) / num_clients
            else:
                proportions = proportions / total_p

            # 计算切分点 (防止越界)
            split_points = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

            # 分割并分配
            idx_batch = np.split(idx_k, split_points)
            for i in range(num_clients):
                client_idcs[i] += idx_batch[i].tolist()

        min_size = min([len(c) for c in client_idcs.values()])

    return client_idcs