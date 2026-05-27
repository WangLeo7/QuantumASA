import logging
from sklearn.datasets import fetch_openml
import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torchvision import transforms
from torchvision.datasets import FashionMNIST


def load_breast_cancer_data(imbalance_ratio=None, noise_level=0.0, random_state=0):
    """Load and preprocess breast cancer dataset."""
    cancer = datasets.load_breast_cancer()
    X = cancer.data
    y = cancer.target

    if imbalance_ratio is not None:
        logging.info(f"Applying Imbalance Ratio: {imbalance_ratio}")
        idx_0 = np.where(y == 0)[0]
        idx_1 = np.where(y == 1)[0]
        n_keep = int(len(idx_0) * imbalance_ratio)
        if n_keep < 5:
            n_keep = 5

        rng = np.random.RandomState(random_state)
        idx_0_kept = rng.choice(idx_0, n_keep, replace=False)

        idx_final = np.concatenate([idx_0_kept, idx_1])
        X = X[idx_final]
        y = y[idx_final]
        logging.info(f"After imbalance: Malignant={len(idx_0_kept)}, Benign={len(idx_1)}")

    total_features = X.shape[1]

    X_train_val, sample_test, y_train_val, label_test = train_test_split(
        X, y, test_size=0.20, shuffle=True,
        random_state=random_state, stratify=y
    )

    if noise_level > 0.0:
        logging.info(f"Injecting Noise Level: {noise_level}")
        rng = np.random.RandomState(random_state)

        stds = np.std(X_train_val, axis=0)

        noise_train = rng.normal(0, 1, X_train_val.shape) * stds * noise_level
        X_train_val = X_train_val + noise_train

        rng_test = np.random.RandomState(random_state + 1)
        noise_test = rng_test.normal(0, 1, sample_test.shape) * stds * noise_level
        sample_test = sample_test + noise_test

    sample_train, sample_valid, label_train, label_valid = train_test_split(
        X_train_val, y_train_val, test_size=0.25, shuffle=True,
        random_state=random_state, stratify=y_train_val
    )

    logging.info(
        f"Data Split: Train={sample_train.shape}, "
        f"Valid={sample_valid.shape}, Test={sample_test.shape}"
    )

    return (cancer, total_features,
            sample_train, sample_valid, label_train, label_valid,
            sample_test, label_test)


def load_fashion_mnist_data(
    n_components=50,
    classes=(0, 3),
    imbalance_ratio=None,
    noise_level=0.0,
    pca_random_state=42,
    random_state=0
):
    """Load and preprocess Fashion-MNIST dataset with PCA dimensionality reduction."""
    logging.info(f"Loading Fashion-MNIST dataset (classes {classes})...")
    print(f"Loading Fashion-MNIST...")
    transform = transforms.ToTensor()

    train_dataset = FashionMNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = FashionMNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    X_train = train_dataset.data.numpy()
    y_train = train_dataset.targets.numpy()

    X_test = test_dataset.data.numpy()
    y_test = test_dataset.targets.numpy()

    X_full = np.concatenate([X_train, X_test], axis=0)
    y_full = np.concatenate([y_train, y_test], axis=0)
    X_full = X_full.reshape(-1, 28 * 28)
    X_full = X_full.astype(np.float64) / 255.0
    y_full = y_full.astype(np.int64)

    print(f"Full dataset: {X_full.shape[0]} samples, {X_full.shape[1]} raw features")
    logging.info(f"Full dataset: {X_full.shape[0]} samples, {X_full.shape[1]} raw features")

    class_a, class_b = classes
    mask = (y_full == class_a) | (y_full == class_b)
    X = X_full[mask]
    y = y_full[mask]

    y_binary = np.where(y == class_a, 0, 1).astype(int)

    n0 = np.sum(y_binary == 0)
    n1 = np.sum(y_binary == 1)

    if n0 > n1:
        y_binary = 1 - y_binary
        swapped = True
        print(f"Swapped labels: class 0 ({classes[1]}) = minority ({n1}), class 1 ({classes[0]}) = majority ({n0})")
    else:
        swapped = False
        print(f"Class 0 ({class_a}): {n0} samples, Class 1 ({class_b}): {n1} samples")

    logging.info(f"Binary classes: Class 0={n0}, Class 1={n1}, swapped={swapped}")

    if imbalance_ratio is not None:
        logging.info(f"Applying Imbalance Ratio: {imbalance_ratio}")
        idx_0 = np.where(y_binary == 0)[0]
        idx_1 = np.where(y_binary == 1)[0]
        n_keep = int(len(idx_0) * imbalance_ratio)
        if n_keep < 5:
            n_keep = 5

        rng = np.random.RandomState(random_state)
        idx_0_kept = rng.choice(idx_0, n_keep, replace=False)

        idx_final = np.concatenate([idx_0_kept, idx_1])
        X = X[idx_final]
        y_binary = y_binary[idx_final]
        print(f"After imbalance: Class 0={len(idx_0_kept)}, Class 1={len(idx_1)}")

    X_train_val, sample_test, y_train_val, label_test = train_test_split(
        X, y_binary, test_size=0.20, shuffle=True,
        random_state=random_state, stratify=y_binary
    )

    print(f"Fitting StandardScaler + PCA on train_val set only (anti-leakage)...")
    scaler_for_pca = StandardScaler()
    X_train_val_scaled = scaler_for_pca.fit_transform(X_train_val)
    sample_test_scaled = scaler_for_pca.transform(sample_test)

    pca = PCA(n_components=n_components, random_state=pca_random_state)
    X_train_val_pca = pca.fit_transform(X_train_val_scaled)
    sample_test_pca = pca.transform(sample_test_scaled)

    explained_var = pca.explained_variance_ratio_.sum()
    print(f"PCA reduced {X.shape[1]}D -> {n_components}D (explained variance: {explained_var:.4f})")
    logging.info(f"PCA: {X.shape[1]}D -> {n_components}D, explained variance ratio = {explained_var:.4f}")

    total_features = n_components

    if noise_level > 0.0:
        logging.info(f"Injecting Noise Level: {noise_level}")
        rng = np.random.RandomState(random_state)
        stds = np.std(X_train_val_pca, axis=0)

        noise_train = rng.normal(0, 1, X_train_val_pca.shape) * stds * noise_level
        X_train_val_pca = X_train_val_pca + noise_train

        rng_test = np.random.RandomState(random_state + 1)
        noise_test = rng_test.normal(0, 1, sample_test_pca.shape) * stds * noise_level
        sample_test_pca = sample_test_pca + noise_test

    sample_train, sample_valid, label_train, label_valid = train_test_split(
        X_train_val_pca, y_train_val, test_size=0.25, shuffle=True,
        random_state=random_state, stratify=y_train_val
    )

    logging.info(
        f"Data Split: Train={sample_train.shape}, Valid={sample_valid.shape}, Test={sample_test_pca.shape} (Locked)"
    )
    print(f"Split sizes: Train={sample_train.shape}, Valid={sample_valid.shape}, Test={sample_test_pca.shape}")

    class FashionMNISTInfo:
        def __init__(self):
            self.feature_names = [f'PC_{i+1}' for i in range(n_components)]
            self.DESCR = f"Fashion-MNIST binary (classes {classes}), PCA-{n_components}, no data leakage"
            self.pca = pca
            self.scaler_for_pca = scaler_for_pca

    fm_data = FashionMNISTInfo()

    return fm_data, total_features, sample_train, sample_valid, label_train, label_valid, sample_test_pca, label_test


def load_ionosphere_data(imbalance_ratio=None, noise_level=0.0, random_state=0):
    """Load and preprocess Ionosphere dataset."""
    logging.info("Loading Ionosphere dataset from OpenML...")
    ionosphere = fetch_openml('ionosphere', version=1, as_frame=False, parser='auto')
    X = ionosphere.data.astype(np.float64)
    y_raw = np.array(ionosphere.target)

    y = np.where(y_raw == 'b', 0, 1).astype(int)

    n0 = np.sum(y == 0)
    n1 = np.sum(y == 1)
    logging.info(f"Ionosphere loaded: Bad(0)={n0}, Good(1)={n1}")

    if imbalance_ratio is not None:
        logging.info(f"Applying Imbalance Ratio: {imbalance_ratio}")
        idx_0 = np.where(y == 0)[0]
        idx_1 = np.where(y == 1)[0]
        n_keep = int(len(idx_0) * imbalance_ratio)
        if n_keep < 5:
            n_keep = 5

        rng = np.random.RandomState(random_state)
        idx_0_kept = rng.choice(idx_0, n_keep, replace=False)

        idx_final = np.concatenate([idx_0_kept, idx_1])
        X = X[idx_final]
        y = y[idx_final]
        logging.info(f"After imbalance: Bad(0)={len(idx_0_kept)}, Good(1)={len(idx_1)}")

    total_features = X.shape[1]

    X_train_val, sample_test, y_train_val, label_test = train_test_split(
        X, y, test_size=0.20, shuffle=True,
        random_state=random_state, stratify=y
    )

    if noise_level > 0.0:
        logging.info(f"Injecting Noise Level: {noise_level}")
        rng = np.random.RandomState(random_state)

        stds = np.std(X_train_val, axis=0)

        noise_train = rng.normal(0, 1, X_train_val.shape) * stds * noise_level
        X_train_val = X_train_val + noise_train

        rng_test = np.random.RandomState(random_state + 1)
        noise_test = rng_test.normal(0, 1, sample_test.shape) * stds * noise_level
        sample_test = sample_test + noise_test

    sample_train, sample_valid, label_train, label_valid = train_test_split(
        X_train_val, y_train_val, test_size=0.25, shuffle=True,
        random_state=random_state, stratify=y_train_val
    )

    logging.info(
        f"Data Split: Train={sample_train.shape}, "
        f"Valid={sample_valid.shape}, Test={sample_test.shape}"
    )

    return (ionosphere, total_features,
            sample_train, sample_valid, label_train, label_valid,
            sample_test, label_test)


def load_parkinsons_data(imbalance_ratio=None, noise_level=0.0, random_state=0):
    """Load and preprocess Parkinsons dataset."""
    logging.info("Loading Parkinsons dataset from OpenML...")
    parkinsons = fetch_openml('parkinsons', version=1, as_frame=False, parser='auto')
    X = parkinsons.data.astype(np.float64)
    y_raw = np.array(parkinsons.target)

    y = (y_raw.astype(str) == '2').astype(int)

    n0 = np.sum(y == 0)
    n1 = np.sum(y == 1)

    logging.info(f"Parkinsons loaded: Healthy(0)={n0}, Parkinson(1)={n1}")

    if imbalance_ratio is not None:
        logging.info(f"Applying Imbalance Ratio: {imbalance_ratio}")
        idx_0 = np.where(y == 0)[0]
        idx_1 = np.where(y == 1)[0]
        n_keep = int(len(idx_0) * imbalance_ratio)
        if n_keep < 5:
            n_keep = 5

        rng = np.random.RandomState(random_state)
        idx_0_kept = rng.choice(idx_0, n_keep, replace=False)

        idx_final = np.concatenate([idx_0_kept, idx_1])
        X = X[idx_final]
        y = y[idx_final]
        logging.info(f"After imbalance: Healthy(0)={len(idx_0_kept)}, Parkinson(1)={len(idx_1)}")

    total_features = X.shape[1]

    X_train_val, sample_test, y_train_val, label_test = train_test_split(
        X, y, test_size=0.20, shuffle=True,
        random_state=random_state, stratify=y
    )

    if noise_level > 0.0:
        logging.info(f"Injecting Noise Level: {noise_level}")
        rng = np.random.RandomState(random_state)

        stds = np.std(X_train_val, axis=0)

        noise_train = rng.normal(0, 1, X_train_val.shape) * stds * noise_level
        X_train_val = X_train_val + noise_train

        rng_test = np.random.RandomState(random_state + 1)
        noise_test = rng_test.normal(0, 1, sample_test.shape) * stds * noise_level
        sample_test = sample_test + noise_test

    sample_train, sample_valid, label_train, label_valid = train_test_split(
        X_train_val, y_train_val, test_size=0.25, shuffle=True,
        random_state=random_state, stratify=y_train_val
    )

    logging.info(
        f"Data Split: Train={sample_train.shape}, "
        f"Valid={sample_valid.shape}, Test={sample_test.shape}"
    )

    return (parkinsons, total_features,
            sample_train, sample_valid, label_train, label_valid,
            sample_test, label_test)


def load_fashion_mnist_tshirt_vs_dress(n_components=50, **kwargs):
    """Fashion-MNIST: T-shirt/top (class 0) vs Dress (class 3)."""
    return load_fashion_mnist_data(n_components=n_components, classes=(0, 3), **kwargs)


def load_fashion_mnist_shirt_vs_dress(n_components=50, **kwargs):
    """Fashion-MNIST: Shirt (class 6) vs Dress (class 3)."""
    return load_fashion_mnist_data(n_components=n_components, classes=(6, 3), **kwargs)


def load_fashion_mnist_shirt_vs_skirt(n_components=50, **kwargs):
    """Fashion-MNIST: Shirt (class 6) vs Skirt (class 8)."""
    return load_fashion_mnist_data(n_components=n_components, classes=(6, 8), **kwargs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    print("Testing data loaders...")
    data, tf, st, sv, lt, lv, ste, lte = load_breast_cancer_data(random_state=0)
    print(f"Breast cancer: {tf} features, train={st.shape}")
