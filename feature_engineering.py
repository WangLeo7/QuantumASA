import random
from itertools import combinations

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def feature_selection_givencolumns(split_data, columns):
    """Feature selection, standardization, and normalization based on column indices."""
    sample_train, sample_test, label_train, label_test = split_data[0], split_data[1], split_data[2], split_data[3]

    cols = np.array(columns, dtype=int).flatten()

    if cols.shape[0] != sample_train.shape[1]:
        raise ValueError(
            f"Length of 'columns' ({cols.shape[0]}) does not match number of features in sample_train ({sample_train.shape[1]})")

    selected_mask = (cols == 1)

    if not np.any(selected_mask):
        sample_train_selected = sample_train[:, selected_mask]
        sample_test_selected = sample_test[:, selected_mask] if sample_test is not None else None
    else:
        sample_train_selected = sample_train[:, selected_mask]
        sample_test_selected = sample_test[:, selected_mask] if sample_test is not None else None

    std_scale = StandardScaler().fit(sample_train_selected)
    sample_train_scaled = std_scale.transform(sample_train_selected)
    sample_test_scaled = std_scale.transform(sample_test_selected) if sample_test_selected is not None else None

    if sample_test_scaled is not None:
        if sample_test_scaled.shape[0] > 0:
            samples = np.append(sample_train_scaled, sample_test_scaled, axis=0)
        else:
            samples = sample_train_scaled
    else:
        samples = sample_train_scaled

    minmax_scale = MinMaxScaler((-1, 1)).fit(sample_train_scaled)

    sample_train_norm = minmax_scale.transform(sample_train_scaled)
    sample_test_norm = minmax_scale.transform(sample_test_scaled) if sample_test_scaled is not None else None

    return [sample_train_norm, sample_test_norm, label_train,
            label_test], std_scale, minmax_scale


def adjust_feature_dimensions(sample_train, sample_test, target_qubits, seed=None):
    """Adjust feature dimensions to match the number of qubits."""
    if seed is not None:
        if isinstance(seed, (np.integer, np.int64, np.int32)):
            seed = int(seed)
        elif not isinstance(seed, int):
            raise TypeError(f"Seed must be int, got {type(seed)}")
        rng = random.Random(seed)
    else:
        rng = random

    k = sample_train.shape[1]
    sample_train1 = sample_train.copy()
    sample_test1 = sample_test.copy() if sample_test is not None else None

    deleted_indices = []
    filled_pairs = []

    if k > target_qubits:
        num_to_delete = k - target_qubits
        del_idx = rng.sample(range(0, k), num_to_delete)
        del_idx_sorted = sorted(del_idx, reverse=True)
        for idx in del_idx_sorted:
            sample_train1 = np.delete(sample_train1, idx, axis=1)
            if sample_test1 is not None:
                sample_test1 = np.delete(sample_test1, idx, axis=1)
            deleted_indices.append(idx)

    elif k < target_qubits:
        if k == 0:
            train_shape = (sample_train.shape[0], target_qubits)
            test_shape = (sample_test.shape[0], target_qubits) if sample_test is not None else None

            sample_train1 = np.zeros(train_shape)
            sample_test1 = np.zeros(test_shape) if sample_test is not None else None

            return sample_train1, sample_test1, [], []

        num_to_fill = target_qubits - k

        for i in range(num_to_fill):
            orig = rng.randint(0, k - 1)
            col_train = sample_train[:, orig].copy().reshape(-1, 1)
            sample_train1 = np.hstack([sample_train1, col_train])

            if sample_test1 is not None:
                col_test = sample_test[:, orig].copy().reshape(-1, 1)
                sample_test1 = np.hstack([sample_test1, col_test])

            new_index = sample_train1.shape[1] - 1
            filled_pairs.append((orig, new_index))

    return sample_train1, sample_test1, deleted_indices, filled_pairs


def decode_individual(individual, total_features, qubits):
    """Decode individual chromosome into feature selection and circuit configuration."""
    feature_selection = individual[:total_features]
    selected_features = [i for i, val in enumerate(feature_selection) if val == 1]
    n_selected = len(selected_features)

    circuit_encoding = individual[total_features:]

    rx_gates = circuit_encoding[:qubits]

    gate_type_bits = circuit_encoding[qubits:qubits + 2]
    if gate_type_bits[0] == 1 and gate_type_bits[1] == 0:
        gate_type = "RX"
    elif gate_type_bits[0] == 0 and gate_type_bits[1] == 1:
        gate_type = "RY"
    elif gate_type_bits[0] == 1 and gate_type_bits[1] == 1:
        gate_type = "RZ"
    else:
        gate_type = "None"

    qub_comb = list(combinations(range(qubits), 2))
    entanglement_start = qubits + 2
    entanglement_end = entanglement_start + len(qub_comb)
    entanglement_gates = circuit_encoding[entanglement_start:entanglement_end]
    entangled_pairs = [qub_comb[i] for i, val in enumerate(entanglement_gates) if val == 1]

    rep_bits = circuit_encoding[-2:]
    repeats = int(''.join(str(x) for x in rep_bits), 2) + 1

    feature_info = {
        'selected_features': selected_features,
        'n_selected_features': n_selected,
        'feature_indices': ','.join(map(str, selected_features))
    }

    circuit_info = {
        'gate_type': gate_type,
        'rx_gates': ','.join(map(str, rx_gates)),
        'entangled_pairs': ';'.join([f'({p[0]},{p[1]})' for p in entangled_pairs]),
        'n_entanglements': len(entangled_pairs),
        'repeats': repeats
    }

    return feature_info, circuit_info


def get_feature_to_qubit_mapping(individual, total_features, qubits, sample_train, feature_seed):
    """Get feature-to-qubit mapping for reporting."""
    feature_selection = individual[:total_features]
    selected_indices = [i for i, val in enumerate(feature_selection) if val == 1]

    if not selected_indices:
        return "No features selected"

    cols = np.array(feature_selection, dtype=int).flatten()
    selected_mask = (cols == 1)
    sample_selected = sample_train[:, selected_mask]

    k = sample_selected.shape[1]

    if k == 0:
        return "No features selected (k=0)"

    if isinstance(feature_seed, (np.integer, np.int64, np.int32)):
        feature_seed = int(feature_seed)

    rng = random.Random(feature_seed)

    mapping = {}

    if k > qubits:
        num_to_delete = k - qubits
        del_idx = sorted(rng.sample(range(0, k), num_to_delete))
        qubit_idx = 0
        for feat_idx in range(k):
            if feat_idx not in del_idx:
                mapping[qubit_idx] = f"F{selected_indices[feat_idx]}"
                qubit_idx += 1

    elif k < qubits:
        num_to_fill = qubits - k
        for i in range(k):
            mapping[i] = f"F{selected_indices[i]}"
        for i in range(num_to_fill):
            orig = rng.randint(0, k - 1)
            mapping[k + i] = f"F{selected_indices[orig]}(copy)"

    else:
        for i in range(k):
            mapping[i] = f"F{selected_indices[i]}"

    mapping_str = ';'.join([f"Q{q}:{feat}" for q, feat in sorted(mapping.items())])
    return mapping_str


def adjust_feature_dimensions_test_only(sample_test, target_qubits, deleted_indices, filled_pairs):
    """Adjust test set dimensions using recorded operations from training set."""
    sample_test1 = sample_test.copy()
    k = sample_test.shape[1]

    if k > target_qubits:
        del_idx_sorted = sorted(deleted_indices, reverse=True)
        for idx in del_idx_sorted:
            sample_test1 = np.delete(sample_test1, idx, axis=1)

    elif k < target_qubits:
        for orig, new_idx in filled_pairs:
            col_test = sample_test[:, orig].copy().reshape(-1, 1)
            sample_test1 = np.hstack([sample_test1, col_test])

    return sample_test1


if __name__ == "__main__":
    print(1)
