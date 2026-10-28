from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split, TensorDataset
from disentanglement_datasets import DSprites
from sklearn.datasets import fetch_olivetti_faces, fetch_lfw_people
from sklearn.model_selection import train_test_split

import torch
import torch.nn.functional as F
import numpy as np


def load_fashion_mnist(batch_size=64, shuffle=True, class_subset=None):
    """
    Load FashionMNIST dataset with optional class subset filtering.
    Args:
        batch_size (int): Batch size for data loaders.
        shuffle (bool): Whether to shuffle the training data.
        class_subset (list, optional): List of class labels to include. If None, includes all classes.
    Returns:
        train_loader, test_loader: Data loaders for training and testing datasets.
    """

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1]
        ]
    )

    train_set = datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=transform
    )

    test_set = datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=transform
    )

    if class_subset is not None:
        # Filter indices where label is in the specified class subset
        train_idx = [
            i for i in range(len(train_set)) if train_set[i][1] in class_subset
        ]
        test_idx = [i for i in range(len(test_set)) if test_set[i][1] in class_subset]

        train_set = Subset(train_set, train_idx)
        test_set = Subset(test_set, test_idx)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=shuffle
    )

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


def load_mnist_digit(batch_size=64, shuffle=True, digit=None):
    """
    Load MNIST dataset with optional digit filtering.
    Args:
        batch_size (int): Batch size for data loaders.
        shuffle (bool): Whether to shuffle the training data.
        digit (int, optional): Digit label to include. If None, includes all digits.
    Returns:
        train_loader, test_loader: Data loaders for training and testing datasets.
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1]
        ]
    )

    train_set = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_set = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    if digit is not None:
        # Filter indices where label is the specified digit
        train_idx = [i for i in range(len(train_set)) if train_set[i][1] == digit]
        test_idx = [i for i in range(len(test_set)) if test_set[i][1] == digit]

        train_set = Subset(train_set, train_idx)
        test_set = Subset(test_set, test_idx)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=shuffle
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


def load_CIFAR10(batch_size=64, shuffle=True, class_subset=None):
    """
    Load FashionMNIST dataset with optional class subset filtering.
    Args:
        batch_size (int): Batch size for data loaders.
        shuffle (bool): Whether to shuffle the training data.
        class_subset (list, optional): List of class labels to include. If None, includes all classes.
    Returns:
        train_loader, test_loader: Data loaders for training and testing datasets.
    """

    transform = transforms.Compose([transforms.ToTensor()])

    train_set = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )

    test_set = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    if class_subset is not None:
        # Filter indices where label is in the specified class subset
        train_idx = [
            i for i in range(len(train_set)) if train_set[i][1] in class_subset
        ]
        test_idx = [i for i in range(len(test_set)) if test_set[i][1] in class_subset]

        train_set = Subset(train_set, train_idx)
        test_set = Subset(test_set, test_idx)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=shuffle
    )

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


def load_olivetti(batch_size: int = 64, shuffle: bool = True):
    """
    Load the Olivetti faces dataset.

    Args:
        batch_size (int): Batch size for the loaders.
        shuffle (bool): Shuffle the training set.

    Returns:
        train_loader, test_loader: DataLoaders for train / test splits.
    """
    data = fetch_olivetti_faces(data_home="./data", shuffle=True, random_state=0)
    images = data.images
    targets = data.target

    n_samples = images.shape[0]
    n_train = int(0.7 * n_samples)

    train_imgs, test_imgs = images[:n_train], images[n_train:]
    train_lbls, test_lbls = targets[:n_train], targets[n_train:]

    train_imgs = torch.from_numpy(train_imgs).unsqueeze(1).float()
    test_imgs = torch.from_numpy(test_imgs).unsqueeze(1).float()
    train_lbls = torch.from_numpy(train_lbls).long()
    test_lbls = torch.from_numpy(test_lbls).long()

    train_set = TensorDataset(train_imgs, train_lbls)
    test_set = TensorDataset(test_imgs, test_lbls)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)

    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def load_lfw(
    batch_size=64,
    shuffle=True,
    class_subset=None,
    image_size=128,
    test_size=0.2,
    random_state=42,
    sharpen=False,
):
    """
    Load LFW dataset with optional class subset filtering.
    Args:
        batch_size (int): Batch size for data loaders.
        shuffle (bool): Whether to shuffle the training data.
        class_subset (list or None): Optional list of person names to keep.
        sharpen (bool): Whether to apply a mild sharpening filter after resizing.
    """
    lfw = fetch_lfw_people(color=False, resize=0.5, min_faces_per_person=20)
    X = lfw.images
    y = lfw.target
    names = np.array(lfw.target_names)

    # Subset class selection
    if class_subset is not None:
        mask = np.isin(names[y], class_subset)
        X = X[mask]
        y = y[mask]
        kept_names = np.unique(names[y])
        name_to_idx = {name: i for i, name in enumerate(kept_names)}
        y = np.array([name_to_idx[names[i]] for i in y])
        class_names = kept_names
    else:
        class_names = names

    X = X.astype(np.float32) / 255.0
    X = torch.tensor(X).unsqueeze(1)
    X = F.interpolate(X, size=(image_size, image_size), mode="bilinear")

    if sharpen:
        kernel = torch.tensor(
            [[[[0, -1, 0], [-1, 5, -1], [0, -1, 0]]]], dtype=torch.float32
        )
        X = F.conv2d(X, kernel, padding=1)

    X = torch.clamp(X, 0.0, 1.0)
    y = torch.tensor(y, dtype=torch.long)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, class_names


def circle_g_peaks(
    n_peaks, points_per_gauss, var, batch_size, test_size=0.2, seed=0, shuffle=True
):
    rng = np.random.RandomState(seed)
    angles = np.linspace(0, 2 * np.pi, num=n_peaks)[:-1]
    means = np.stack((np.cos(angles), np.sin(angles)), axis=0)
    points = [
        rng.multivariate_normal(means[:, n], var * np.eye(2), size=(points_per_gauss,))
        for n in range(n_peaks - 1)
    ]
    out = torch.tensor(
        np.concat(points, axis=0)
    ).unsqueeze(1).unsqueeze(1).to(torch.float32)

    labels = torch.tensor(np.concat(
        [np.ones(shape=(points_per_gauss,)) * n for n in range(n_peaks - 1)], axis=0
        )
    ).to(torch.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        out, labels, test_size=test_size, random_state=seed
    )

    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
