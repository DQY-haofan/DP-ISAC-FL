import torch
import numpy as np
from torchvision import datasets, transforms


def get_dataset(dataset_name, data_root):
    if dataset_name == 'mnist':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = datasets.MNIST(data_root, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(data_root, train=False, download=True, transform=transform)
    elif dataset_name == 'cifar10':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = datasets.CIFAR10(data_root, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(data_root, train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return train_dataset, test_dataset


def partition_dataset_dirichlet(dataset, num_clients, alpha=0.3, seed=42):
    np.random.seed(seed)
    num_classes = len(dataset.classes)
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    else:
        labels = np.array(dataset.train_labels)

    min_size = 0
    client_idcs = {}
    while min_size < 10:
        client_idcs = {i: [] for i in range(num_clients)}
        for k in range(num_classes):
            idx_k = np.where(labels == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = np.array([p * (len(idx_k) < num_clients / 10.0) for p in proportions])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = np.split(idx_k, proportions)
            for i in range(num_clients):
                client_idcs[i] += idx_batch[i].tolist()
        min_size = min([len(c) for c in client_idcs.values()])
    return client_idcs