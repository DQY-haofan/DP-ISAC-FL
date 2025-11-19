import torch
import numpy as np
from torchvision import datasets, transforms


def get_dataset(dataset_name, data_root):
    """下载并加载 MNIST 或 CIFAR10 数据集"""
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
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return train_dataset, test_dataset


def partition_dataset_dirichlet(dataset, num_clients, alpha=0.3, seed=42):
    """
    使用 Dirichlet 分布进行 Non-IID 数据划分 (增强鲁棒性版)
    """
    np.random.seed(seed)

    # 获取标签
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    else:
        labels = np.array(dataset.train_labels)

    num_classes = len(dataset.classes)
    min_size = 0
    client_idcs = {}

    # 尝试划分直到所有客户端都有至少一点数据
    # Add a safety counter to prevent infinite loops
    attempt = 0
    while min_size < 10:
        if attempt > 100:
            print(
                "Warning: Could not satisfy min_size < 10 requirement after 100 attempts. Proceeding with current partition.")
            break
        attempt += 1

        client_idcs = {i: [] for i in range(num_clients)}

        for k in range(num_classes):
            idx_k = np.where(labels == k)[0]
            np.random.shuffle(idx_k)

            # [Fix 1] 如果该类样本太少，不足以分配，则不使用 Dirichlet
            # 而是随机均匀分给所有客户端，或者只分给部分客户端
            if len(idx_k) < num_clients:
                # 样本太少，随机分给部分客户端
                subset_clients = np.random.choice(num_clients, len(idx_k), replace=False)
                for i, client_idx in enumerate(subset_clients):
                    client_idcs[client_idx].append(idx_k[i])
                continue

            # 生成 Dirichlet 分布
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))

            # 调整比例以匹配该类别的样本数

            # [Fix 2] 安全归一化
            total_p = proportions.sum()
            if total_p < 1e-6:
                # 如果随机出来的概率总和极小（极罕见），重置为均匀分布
                proportions = np.ones(num_clients) / num_clients
            else:
                proportions = proportions / total_p

            # 计算切分点
            # (np.cumsum... * len).astype(int) 会截断小数
            # 我们需要确保最后一个切分点是 len(idx_k)
            split_points = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

            # 分割
            idx_batch = np.split(idx_k, split_points)

            for i in range(num_clients):
                client_idcs[i] += idx_batch[i].tolist()

        min_size = min([len(c) for c in client_idcs.values()])

    return client_idcs