import logging

import numpy as np


def create_target_kernel(labels):
    """Create ideal kernel (Label Kernel). K_y[i, j] = 1 if y[i] == y[j], else -1."""
    n = len(labels)
    K_y = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K_y[i, j] = 1.0 if labels[i] == labels[j] else -1.0
    K_y_c = _center_kernel(K_y)
    return K_y_c


def _center_kernel(K):
    """Center kernel matrix (O(N^2) efficient implementation)."""
    n = K.shape[0]
    if n == 0:
        return K

    K = K.astype(float)
    row_means = np.sum(K, axis=1, keepdims=True) / n
    col_means = np.sum(K, axis=0, keepdims=True) / n
    total_mean = np.mean(K)

    K_c = K - row_means - col_means + total_mean
    return K_c


def compute_sample_weights(labels):
    """Compute sample weights s_i = 1 / N_{y_i} for energy balancing."""
    classes, counts = np.unique(labels, return_counts=True)
    count_dict = dict(zip(classes, counts))

    weights = np.array([1.0 / count_dict[l] for l in labels])

    return weights


def calculate_batch_wka_score(K_q, labels):
    """Calculate Batch Weighted Kernel Alignment score."""
    n = len(labels)

    y_sign = np.where(labels == 1, 1.0, -1.0)
    K_y = np.outer(y_sign, y_sign)

    s = compute_sample_weights(labels)

    weight_matrix = np.outer(s, s)

    K_q_weighted = K_q * weight_matrix
    K_y_weighted = K_y * weight_matrix

    K_q_final = _center_kernel(K_q_weighted)
    K_y_final = _center_kernel(K_y_weighted)

    score = np.sum(K_q_final * K_y_final)
    norm_q = np.sqrt(np.sum(K_q_final * K_q_final))
    norm_y = np.sqrt(np.sum(K_y_final * K_y_final))

    if norm_q * norm_y < 1e-9:
        return 0.0

    return score / (norm_q * norm_y)


def calculate_center_kernel_alignment(K_q, K_y_c):
    """Calculate Centered Kernel Alignment score. Assumes K_y_c is already centered."""
    try:
        K_q_c = _center_kernel(K_q)

        alignment_score = np.sum(K_q_c * K_y_c)

        norm_q = np.sqrt(np.sum(K_q_c * K_q_c))
        norm_y = np.sqrt(np.sum(K_y_c * K_y_c))

        if norm_q * norm_y < 1e-9:
            return 0.0

        ka_score = alignment_score / (norm_q * norm_y)
        return ka_score

    except Exception as e:
        logging.exception(f"Error in calculate_kernel_alignment: {e}")
        return 0.0


def create_kta_target_kernel(labels):
    """Create ideal kernel for KTA without centering."""
    n = len(labels)
    y_sign = np.where(labels == 1, 1.0, -1.0)
    K_y = np.outer(y_sign, y_sign)
    return K_y


def calculate_kta_score(K_q, labels):
    """Calculate Kernel Target Alignment score (no centering)."""
    try:
        K_y = create_kta_target_kernel(labels)

        alignment_score = np.sum(K_q * K_y)

        norm_q = np.sqrt(np.sum(K_q * K_q))
        norm_y = np.sqrt(np.sum(K_y * K_y))

        if norm_q * norm_y < 1e-9:
            return 0.0

        kta_score = alignment_score / (norm_q * norm_y)
        return kta_score

    except Exception as e:
        logging.exception(f"Error in calculate_kta: {e}")
        return 0.0


if __name__ == "__main__":
    print(1)
