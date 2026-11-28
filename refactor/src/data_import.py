import numpy as np
import torch
import torch.nn.functional as F

# from disentanglement_datasets import DSprites
from sklearn.datasets import fetch_lfw_people, fetch_olivetti_faces, load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset, random_split
from torchvision import datasets, transforms


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


def load_mnist_digit(batch_size=64, shuffle=True, class_subset=None):
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
    if class_subset is not None:
        # Filter indices where label is the specified digit
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


def load_olivetti(
    batch_size: int = 64,
    shuffle: bool = True,
    data_home: str = "./data",
    normalize: bool = True,
):
    """
    Load the Olivetti faces dataset.

    ``normalize=True`` maps the original [0, 1] pixels to [-1, 1], which is
    the range used by the Langevin sampler.
    """
    data = fetch_olivetti_faces(data_home=data_home, shuffle=True, random_state=0)
    images = data.images
    targets = data.target

    n_samples = images.shape[0]
    n_train = int(0.7 * n_samples)

    train_imgs, test_imgs = images[:n_train], images[n_train:]
    train_lbls, test_lbls = targets[:n_train], targets[n_train:]

    train_imgs = torch.from_numpy(train_imgs).unsqueeze(1).float()
    test_imgs = torch.from_numpy(test_imgs).unsqueeze(1).float()
    if normalize:
        train_imgs = train_imgs * 2.0 - 1.0
        test_imgs = test_imgs * 2.0 - 1.0
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
    data_home="./data",
    min_faces_per_person=20,
    max_samples=None,
    normalize=True,
):
    """
    Load LFW faces with optional class subset filtering.

    Images are mapped to [-1, 1] by default (``normalize=True``) to match the
    sampler's clamp range.  ``max_samples`` is useful for quick smoke runs.
    """
    lfw = fetch_lfw_people(
        color=False,
        resize=0.5,
        min_faces_per_person=min_faces_per_person,
        data_home=data_home,
    )
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

    if max_samples is not None:
        X, y = X[:max_samples], y[:max_samples]

    if normalize:
        X = X * 2.0 - 1.0

    counts = torch.bincount(y) if len(y) else torch.tensor([])
    stratify = y if len(counts) > 1 and int(counts.min()) >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
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
    out = (
        torch.tensor(np.concat(points, axis=0))
        .unsqueeze(1)
        .unsqueeze(1)
        .to(torch.float32)
    )

    labels = torch.tensor(
        np.concat(
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


def load_digits_data(batch_size: int = 64, shuffle: bool = True):
    """
    Load the sklearn digits dataset.

    Args:
        batch_size (int): Batch size for the loaders.
        shuffle (bool): Shuffle the training set.

    Returns:
        train_loader, test_loader: DataLoaders for train / test splits.
    """
    data = load_digits()
    images = data.images
    targets = data.target

    n_samples = images.shape[0]
    n_train = int(0.7 * n_samples)

    train_imgs, test_imgs = images[:n_train], images[n_train:]
    train_lbls, test_lbls = targets[:n_train], targets[n_train:]

    # Convert to tensors and add channel dimension (batch, 1, 8, 8)
    train_imgs = torch.from_numpy(train_imgs).unsqueeze(1).float()
    test_imgs = torch.from_numpy(test_imgs).unsqueeze(1).float()
    train_lbls = torch.from_numpy(train_lbls).long()
    test_lbls = torch.from_numpy(test_lbls).long()

    train_set = TensorDataset(train_imgs, train_lbls)
    test_set = TensorDataset(test_imgs, test_lbls)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
