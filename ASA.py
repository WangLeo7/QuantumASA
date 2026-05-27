import csv
import json
import logging
import os
import random
import sys
import time

import h5py
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import kendalltau
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import SVC

from KA import *
from load_data import *
from nsga3_operators import *
from qsvm import *


# Parameters
RANDOM_SEED = int(sys.argv[1])
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
num_generations = 100
save_files = 1
pop_size = 100
rate_crossover = 20
rate_mutation = 20
N_JOB = -1
all_qubits = [8]
data_number = 1  # 1=Breast Cancer, 2=Fashion-MNIST, 3=Ionosphere, 4=Parkinsons
KA_MODE = "WKA"
KA_TOP_PERCENT = 0
KA_BATCH_RATIO = 0.5
now_dir = 'results_' + KA_MODE + '_robust_' + str(RANDOM_SEED)
current_script_path = os.path.dirname(os.path.abspath(__file__))
all_dir = os.path.join(current_script_path, now_dir)

scenarios = [
    {"name": "Baseline (Clean)", "imb": None, "noise": 0.0},
    {"name": "Robustness (Noise 0.5)", "imb": None, "noise": 0.5},
    {"name": "Robustness (Imbalance 20%)", "imb": 0.2, "noise": 0.0}
]


# Logging configuration
LOG_DIR = all_dir
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
log_file = os.path.join(LOG_DIR, 'optimization_log_' + KA_MODE + '_Percent.txt')
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')


def _init_seed(job_id):
    """Initialize unique random seed for parallel jobs."""
    seed = RANDOM_SEED + job_id * 9973
    seed_int = int(seed)
    random.seed(seed_int)
    np.random.seed(seed_int)
    return seed_int


def _eval_ka_prelim(local_i, x,
                    total_features, params_qubits,
                    sample_train, label_train,
                    sample_indices,
                    feature_map_seed,
                    target_kernel_batch):
    """Stage 1: Cheap evaluation (KA score + cost)."""
    penalty_objs_on_error = [1.0, 1000.0, 1000.0, float(total_features)]
    empty_timing = {'feat_time': 0.0, 'quantum_time': 0.0, 'wka_time': 0.0}

    try:
        seed = feature_map_seed
        x = x.astype(int)
        circuit_solution = x[total_features:]

        count = np.sum(x[:total_features])

        if count == 0:
            return local_i, 0.0, penalty_objs_on_error, empty_timing

        t_feat_start = time.time()

        split_data = [sample_train, None, label_train, None]
        selected_data, _, _ = feature_selection_givencolumns(split_data, x[:total_features])
        sample_train_selected = selected_data[0]

        obj4 = float(count)

        sample_train_batch_selected = sample_train_selected[sample_indices]

        qsvm = Cqsvm(params_qubits, "qsvm")
        kernel_instance = qsvm.get_kernel(circuit_solution)
        obj2 = float(qsvm.number_of_lgates)
        obj3 = float(qsvm.number_of_cnot)

        sample_train_batch_adj, _, _, _ = adjust_feature_dimensions(
            sample_train_batch_selected, None, params_qubits, seed=seed
        )

        t_feat_end = time.time()
        feat_time = t_feat_end - t_feat_start

        t_quantum_start = time.time()

        matrix_train_batch = kernel_instance.evaluate(x_vec=sample_train_batch_adj)

        t_quantum_end = time.time()
        quantum_time = t_quantum_end - t_quantum_start

        t_wka_start = time.time()
        current_batch_labels = label_train[sample_indices]

        if KA_MODE == "CKA":
            ka_score = calculate_center_kernel_alignment(matrix_train_batch, target_kernel_batch)
        elif KA_MODE == "KTA":
            ka_score = calculate_kta_score(matrix_train_batch, current_batch_labels)
        else:
            ka_score = calculate_batch_wka_score(matrix_train_batch, current_batch_labels)

        t_wka_end = time.time()
        wka_time = t_wka_end - t_wka_start

        prelim_objs = [1.0, obj2, obj3, obj4]
        timing_dict = {
            'feat_time': feat_time,
            'quantum_time': quantum_time,
            'wka_time': wka_time,
        }
        return local_i, ka_score, prelim_objs, timing_dict

    except Exception as e:
        logging.exception(f"Error in _eval_ka_prelim (job {local_i}): {e}")
        return local_i, 0.0, penalty_objs_on_error, empty_timing


def _eval_full_accuracy(local_i, x,
                        total_features, params_qubits,
                        sample_train, sample_test, label_train, label_test,
                        feature_map_seed):
    """Stage 2: Expensive evaluation (compute true accuracy)."""
    empty_timing = {'feat_time': 0.0, 'quantum_time': 0.0, 'svm_time': 0.0}

    try:
        seed = feature_map_seed
        x = x.astype(int)
        circuit_solution = x[total_features:]

        t_feat_start = time.time()

        split_data = [sample_train, sample_test, label_train, label_test]
        selected_data, _, _ = feature_selection_givencolumns(split_data, x[:total_features])

        sample_train_selected, sample_test_selected = selected_data[0], selected_data[1]

        qsvm = Cqsvm(params_qubits, "qsvm")
        kernel_instance = qsvm.get_kernel(circuit_solution)

        sample_train_adj, sample_test_adj, _, _ = adjust_feature_dimensions(
            sample_train_selected, sample_test_selected, params_qubits, seed=seed
        )

        t_feat_end = time.time()
        feat_time = t_feat_end - t_feat_start

        t_quantum_start = time.time()

        matrix_train = kernel_instance.evaluate(x_vec=sample_train_adj)
        matrix_test = kernel_instance.evaluate(x_vec=sample_test_adj, y_vec=sample_train_adj)

        t_quantum_end = time.time()
        quantum_time = t_quantum_end - t_quantum_start

        t_svm_start = time.time()

        adhoc_svc = SVC(kernel='precomputed', class_weight='balanced')
        adhoc_svc.fit(matrix_train, label_train)

        y_pred = adhoc_svc.predict(matrix_test)

        score = balanced_accuracy_score(label_test, y_pred)
        score = round(score, 4)

        t_svm_end = time.time()
        svm_time = t_svm_end - t_svm_start

        true_obj1 = 1.0 - score

        timing_dict = {
            'feat_time': feat_time,
            'quantum_time': quantum_time,
            'svm_time': svm_time,
        }
        return local_i, true_obj1, timing_dict

    except Exception as e:
        logging.exception(f"Error in full eval: {e}")
        return local_i, 1.0, empty_timing


def evaluation(pop,
               total_features, params_qubits,
               sample_train, sample_test, label_train, label_test,
               sample_indices,
               feature_map_seed,
               ka_top_percent,
               target_kernel_batch,
               n_jobs=-1,
               force_full_evaluation=False,
               gen_id=None):
    """Coordinate two-stage evaluation with dynamic screening ratio."""
    pop_size = pop.shape[0]
    if pop_size == 0:
        return np.array([]).reshape(0, 5), 0.0

    full_penalty_objs = [1.0, 1000.0, 1000.0, float(total_features)]

    # Stage 1: Parallel KA score computation
    prelim_tasks = [
        delayed(_eval_ka_prelim)(
            i, pop[i],
            total_features, params_qubits,
            sample_train, label_train,
            sample_indices,
            feature_map_seed,
            target_kernel_batch
        ) for i in range(pop_size)
    ]
    prelim_results = Parallel(n_jobs=n_jobs, prefer="processes")(prelim_tasks)

    all_prelim_results = {}
    prelim_timing_list = []

    for res in prelim_results:
        if res is None:
            continue
        local_i_pop_index, ka_score, prelim_objs, timing_dict = res
        all_prelim_results[local_i_pop_index] = (ka_score, prelim_objs, timing_dict)
        prelim_timing_list.append(timing_dict)

    # Stage 2: Dynamic screening
    selected_indices_set = set()

    if force_full_evaluation:
        selected_indices_set = set(all_prelim_results.keys())
    else:
        ka_scores_for_sampling = []
        for local_i, (ka_score, prelim_objs, _) in all_prelim_results.items():
            is_penalized = (prelim_objs[1] == 1000.0)
            if not is_penalized:
                ka_weight = max(ka_score, 1e-9)
                ka_scores_for_sampling.append((local_i, ka_weight))

        if ka_scores_for_sampling:
            num_to_select = int(np.floor(pop_size * (ka_top_percent / 100.0)))
            num_to_select = max(5, num_to_select)
            num_to_select = min(len(ka_scores_for_sampling), num_to_select)

            indices_to_choose_from = np.array([item[0] for item in ka_scores_for_sampling])
            weights = np.array([item[1] for item in ka_scores_for_sampling])
            total_weight = np.sum(weights)

            if total_weight < 1e-8:
                ka_scores_for_sampling.sort(key=lambda item: item[1], reverse=True)
                selected_indices_set = set(item[0] for item in ka_scores_for_sampling[:num_to_select])
            else:
                probabilities = weights / total_weight
                try:
                    selected_indices = np.random.choice(
                        indices_to_choose_from, size=num_to_select, replace=False, p=probabilities
                    )
                    selected_indices_set = set(selected_indices)
                except Exception:
                    ka_scores_for_sampling.sort(key=lambda item: item[1], reverse=True)
                    selected_indices_set = set(item[0] for item in ka_scores_for_sampling[:num_to_select])

    # Stage 3: Expensive evaluation
    full_eval_tasks = []
    for local_i in selected_indices_set:
        full_eval_tasks.append(
            delayed(_eval_full_accuracy)(
                local_i, pop[local_i],
                total_features, params_qubits,
                sample_train, sample_test, label_train, label_test,
                feature_map_seed
            )
        )

    true_obj1_results = {}
    full_timing_list = []
    if full_eval_tasks:
        full_results = Parallel(n_jobs=n_jobs, prefer="processes")(full_eval_tasks)
        for res in full_results:
            if res is None:
                continue
            local_i, true_obj1, timing_dict = res
            true_obj1_results[local_i] = (true_obj1, timing_dict)
            full_timing_list.append(timing_dict)

    # Stage 4: Build fitness values
    fitness_values = np.tile(full_penalty_objs, (pop_size, 1))

    corr_ka_list = []
    corr_acc_list = []

    proxy_obj1_list = []
    true_obj1_list = []

    for local_i, (ka_score, prelim_objs, _) in all_prelim_results.items():
        fitness_values[local_i, 1:] = prelim_objs[1:]

        if local_i in true_obj1_results:
            true_obj1, _ = true_obj1_results[local_i]
            accuracy = 1.0 - true_obj1
            fitness_values[local_i, 0] = true_obj1
            true_obj1_list.append((local_i, true_obj1))

            if prelim_objs[1] < 1000.0:
                corr_ka_list.append(ka_score)
                corr_acc_list.append(accuracy)
        else:
            is_penalized = (prelim_objs[1] == 1000.0)
            if is_penalized:
                fitness_values[local_i, 0] = 1.0
            else:
                proxy_obj1 = 1.0 - ka_score
                fitness_values[local_i, 0] = proxy_obj1
                proxy_obj1_list.append((local_i, proxy_obj1))

    # Stage 5: Rescaling
    if force_full_evaluation:
        logging.info("Skipping proxy rescaling (full evaluation).")
    elif not proxy_obj1_list:
        logging.info("No proxy values to rescale.")
    elif not true_obj1_list:
        logging.warning("No true values found. Cannot rescale.")
    else:
        try:
            true_vals = np.array([v for i, v in true_obj1_list])
            min_true = np.min(true_vals)
            max_true = np.max(true_vals)
            range_true = max_true - min_true

            proxy_vals = np.array([v for i, v in proxy_obj1_list])
            min_proxy = np.min(proxy_vals)
            max_proxy = np.max(proxy_vals)
            range_proxy = max_proxy - min_proxy

            if range_proxy > 1e-9:
                for local_i, proxy_obj1 in proxy_obj1_list:
                    normalized_proxy = (proxy_obj1 - min_proxy) / range_proxy
                    rescaled_proxy = min_true + (normalized_proxy * range_true)
                    fitness_values[local_i, 0] = rescaled_proxy
            else:
                for local_i, proxy_obj1 in proxy_obj1_list:
                    fitness_values[local_i, 0] = max_true
        except Exception as e:
            logging.error(f"Rescaling failed: {e}.")

    if not force_full_evaluation and proxy_obj1_list and true_obj1_list:
        try:
            true_vals = np.array([v for i, v in true_obj1_list])
            min_true, max_true = np.min(true_vals), np.max(true_vals)
            range_true = max_true - min_true
            proxy_vals = np.array([v for i, v in proxy_obj1_list])
            min_proxy, max_proxy = np.min(proxy_vals), np.max(proxy_vals)
            range_proxy = max_proxy - min_proxy
            if range_proxy > 1e-9:
                for local_i, proxy_obj1 in proxy_obj1_list:
                    normalized_proxy = (proxy_obj1 - min_proxy) / range_proxy
                    fitness_values[local_i, 0] = min_true + (normalized_proxy * range_true)
            else:
                for local_i, proxy_obj1 in proxy_obj1_list:
                    fitness_values[local_i, 0] = max_true
        except Exception:
            pass

    # Compute Kendall's Tau correlation
    tau = 0.0
    if len(corr_ka_list) >= 5:
        try:
            tau, _ = kendalltau(corr_ka_list, corr_acc_list)
            if np.isnan(tau):
                tau = 0.0
        except:
            tau = 0.0

    if force_full_evaluation:
        tau = 1.0

    # Timing summary
    if prelim_timing_list:
        p1_feat = np.array([d['feat_time'] for d in prelim_timing_list])
        p1_quantum = np.array([d['quantum_time'] for d in prelim_timing_list])
        p1_wka = np.array([d['wka_time'] for d in prelim_timing_list])

        p1_feat_sum = np.sum(p1_feat)
        p1_feat_mean = np.mean(p1_feat)
        p1_quantum_sum = np.sum(p1_quantum)
        p1_quantum_mean = np.mean(p1_quantum)
        p1_quantum_max = np.max(p1_quantum)
        p1_wka_sum = np.sum(p1_wka)
        p1_wka_mean = np.mean(p1_wka)
        p1_total = p1_feat_sum + p1_quantum_sum + p1_wka_sum
    else:
        p1_feat_sum = p1_feat_mean = p1_quantum_sum = p1_quantum_mean = p1_quantum_max = p1_wka_sum = p1_wka_mean = p1_total = 0.0

    if full_timing_list:
        p2_feat = np.array([d['feat_time'] for d in full_timing_list])
        p2_quantum = np.array([d['quantum_time'] for d in full_timing_list])
        p2_svm = np.array([d['svm_time'] for d in full_timing_list])

        p2_feat_sum = np.sum(p2_feat)
        p2_feat_mean = np.mean(p2_feat)
        p2_quantum_sum = np.sum(p2_quantum)
        p2_quantum_mean = np.mean(p2_quantum)
        p2_quantum_max = np.max(p2_quantum)
        p2_svm_sum = np.sum(p2_svm)
        p2_svm_mean = np.mean(p2_svm)
        p2_total = p2_feat_sum + p2_quantum_sum + p2_svm_sum
        eval_count = len(full_timing_list)
    else:
        p2_feat_sum = p2_feat_mean = p2_quantum_sum = p2_quantum_mean = p2_quantum_max = p2_svm_sum = p2_svm_mean = p2_total = 0.0
        eval_count = 0

    timing_log_path = os.path.join(all_dir, 'timing_breakdown_WKA_Percent.txt')
    gen_label = gen_id if gen_id is not None else -1
    record = {
        "gen": gen_label,
        "pop_size": pop_size,
        "eval_count": eval_count,
        "phase1_feat_sum": round(p1_feat_sum, 6),
        "phase1_feat_mean": round(p1_feat_mean, 6),
        "phase1_quantum_sum": round(p1_quantum_sum, 6),
        "phase1_quantum_mean": round(p1_quantum_mean, 6),
        "phase1_quantum_max": round(p1_quantum_max, 6),
        "phase1_wka_sum": round(p1_wka_sum, 6),
        "phase1_wka_mean": round(p1_wka_mean, 6),
        "phase1_total": round(p1_total, 6),
        "phase2_feat_sum": round(p2_feat_sum, 6),
        "phase2_feat_mean": round(p2_feat_mean, 6),
        "phase2_quantum_sum": round(p2_quantum_sum, 6),
        "phase2_quantum_mean": round(p2_quantum_mean, 6),
        "phase2_quantum_max": round(p2_quantum_max, 6),
        "phase2_svm_sum": round(p2_svm_sum, 6),
        "phase2_svm_mean": round(p2_svm_mean, 6),
        "phase2_total": round(p2_total, 6),
        "grand_total": round(p1_total + p2_total, 6),
    }
    with open(timing_log_path, 'a', encoding='utf-8') as f_timing:
        f_timing.write(json.dumps(record) + '\n')

    return fitness_values, tau


def run_optimization(params_qubits, params_num_generations, imbalance_ratio=None, noise_level=0.0):
    """Main optimization function."""
    logging.info(f"Started optimization: qubits={params_qubits}, imbalance={imbalance_ratio}, noise={noise_level}")

    h5_path = os.path.join(all_dir, 'generations.h5')
    h5_file = h5py.File(h5_path, 'w')

    # Load data
    if data_number == 1:
        _, total_features, sample_train, sample_valid, label_train, label_valid, sample_test, label_test = \
            load_breast_cancer_data(imbalance_ratio=imbalance_ratio, noise_level=noise_level, random_state=RANDOM_SEED)
    elif data_number == 2:
        _, total_features, sample_train, sample_valid, label_train, label_valid, sample_test, label_test = \
            load_fashion_mnist_data(n_components=50, imbalance_ratio=imbalance_ratio, noise_level=noise_level, random_state=RANDOM_SEED)
    elif data_number == 3:
        _, total_features, sample_train, sample_valid, label_train, label_valid, sample_test, label_test = \
            load_ionosphere_data(imbalance_ratio=imbalance_ratio, noise_level=noise_level, random_state=RANDOM_SEED)
    elif data_number == 4:
        _, total_features, sample_train, sample_valid, label_train, label_valid, sample_test, label_test = \
            load_parkinsons_data(imbalance_ratio=imbalance_ratio, noise_level=noise_level, random_state=RANDOM_SEED)

    # Create KA batch
    current_batch_size = int(len(label_train) * KA_BATCH_RATIO)
    if current_batch_size < 1:
        current_batch_size = 1

    feature_map_seed = _init_seed(RANDOM_SEED)
    rng = random.Random(RANDOM_SEED)
    sample_indices = rng.sample(range(len(label_train)), current_batch_size)
    target_kernel_batch = create_target_kernel(label_train[sample_indices])

    # Initialize population
    qub_comb = list(combinations(range(params_qubits), 2))
    n_var = total_features + params_qubits + 2 + len(qub_comb) + 2

    print(f"Initializing population (size {pop_size})...")
    pop = random_population(n_var, pop_size)

    current_ka_percent = KA_TOP_PERCENT
    params_min_percent = 20
    params_max_percent = 80

    # Evaluate initial population
    fitness_values, correlation = evaluation(
        pop,
        total_features, params_qubits,
        sample_train, sample_valid, label_train, label_valid,
        sample_indices,
        feature_map_seed=feature_map_seed,
        ka_top_percent=current_ka_percent,
        target_kernel_batch=target_kernel_batch,
        n_jobs=N_JOB,
        gen_id=0
    )

    grp0 = h5_file.create_group('gen_000')
    grp0.create_dataset('population', data=pop, compression='gzip')
    grp0.create_dataset('fitness', data=fitness_values, compression='gzip')

    # Main evolution loop
    for i in range(params_num_generations):
        gen_start_time = time.time()

        # Adaptive strategy
        if i > 0:
            clipped_tau = max(0.0, min(1.0, correlation))
            target_percent = params_max_percent - (clipped_tau * (params_max_percent - params_min_percent))
            current_ka_percent = 0.7 * target_percent + 0.3 * current_ka_percent
            current_ka_percent = int(max(params_min_percent, min(params_max_percent, current_ka_percent)))

        print(f'Gen {i + 1}/{params_num_generations} | WKA w/ ASA | Eval: {current_ka_percent}% (Tau={correlation:.3f})')

        # Create offspring
        n_crossover = compute_offspring_count(rate_crossover, pop_size)
        n_mutation = compute_offspring_count(rate_mutation, pop_size)
        children = []
        if n_crossover > 0:
            children.append(crossover(pop, n_crossover))
        if n_mutation > 0:
            children.append(mutation(pop, n_mutation))

        if children:
            children = np.vstack(children)
            fitness_children, child_correlation = evaluation(
                children,
                total_features, params_qubits,
                sample_train, sample_valid, label_train, label_valid,
                sample_indices,
                target_kernel_batch=target_kernel_batch,
                feature_map_seed=feature_map_seed,
                ka_top_percent=current_ka_percent,
                n_jobs=N_JOB,
                gen_id=i + 1
            )
            correlation = child_correlation

            combined_pop = np.vstack([pop, children])
            combined_fitness = np.vstack([fitness_values, fitness_children])
            pop, selected_idx = selection_nsga3(combined_pop, combined_fitness, pop_size)
            fitness_values = combined_fitness[selected_idx]

        gen_time = time.time() - gen_start_time
        valid_fitness = fitness_values[fitness_values[:, 0] < 1.0]
        best_valid_acc = 1.0 - np.min(valid_fitness[:, 0]) if len(valid_fitness) > 0 else 0
        logging.info(f"Gen {i + 1}: Tau={correlation:.4f}, EvalRate={current_ka_percent}, BestValidAcc={best_valid_acc:.4f}, Time={gen_time:.2f}s")

        grp = h5_file.create_group(f'gen_{i+1:03d}')
        grp.create_dataset('population', data=pop, compression='gzip')
        grp.create_dataset('fitness', data=fitness_values, compression='gzip')

    # Final evaluation
    print("Optimization loop ended. Running final pareto evaluation on VALIDATION set...")
    pop = np.unique(pop, axis=0)

    valid_fitness_values, _ = evaluation(
        pop,
        total_features, params_qubits,
        sample_train, sample_valid, label_train, label_valid,
        sample_indices,
        target_kernel_batch=target_kernel_batch,
        feature_map_seed=feature_map_seed,
        ka_top_percent=100,
        n_jobs=N_JOB,
        force_full_evaluation=True,
        gen_id=-1
    )

    pareto_front_index = pareto_front_finding(valid_fitness_values, np.arange(len(pop)))
    pareto_pop = pop[pareto_front_index]
    pareto_valid_fitness = valid_fitness_values[pareto_front_index]

    # Final evaluation on locked test set
    print(f"\n{'=' * 60}")
    print(f"FINAL STEP: Evaluating Pareto Solutions on LOCKED TEST SET")
    print(f"{'=' * 60}")

    test_fitness_values, _ = evaluation(
        pareto_pop,
        total_features, params_qubits,
        sample_train, sample_test, label_train, label_test,
        sample_indices,
        target_kernel_batch=target_kernel_batch,
        feature_map_seed=feature_map_seed,
        ka_top_percent=100,
        n_jobs=N_JOB,
        force_full_evaluation=True,
        gen_id=-2
    )

    final_valid_acc = 1.0 - pareto_valid_fitness[:, 0]
    final_test_acc = 1.0 - test_fitness_values[:, 0]
    best_valid_idx = np.argmax(final_valid_acc)
    final_report_acc = final_test_acc[best_valid_idx]

    print(f"Best Model Selected by Validation Balanced Accuracy: {final_valid_acc[best_valid_idx]:.4f}")
    print(f">>> FINAL REPORT (Locked Test Balanced Accuracy): {final_report_acc:.4f}")

    h5_file.close()

    if save_files:
        output_dir = all_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        base_filename = f"{output_dir}/{params_qubits}bits_WKA_{str(imbalance_ratio)}_{str(noise_level)}"

        np.save(f'{base_filename}_solutions.npy', pop)
        np.save(f'{base_filename}_valid_fitness.npy', pareto_valid_fitness)
        np.save(f'{base_filename}_test_fitness.npy', test_fitness_values)

        gene_headers = [f'gene_{i + 1}' for i in range(pareto_pop.shape[1])]
        feature_headers = ['selected_feature_indices', 'n_selected_features']
        circuit_headers = ['gate_type', 'rx_gates_config', 'entangled_pairs',
                           'n_entanglements', 'circuit_repeats', 'feature_to_qubit_mapping']
        objective_headers = ['valid_balanced_accuracy', 'test_balanced_accuracy (unseen)', 'n_logic_gates', 'n_cnot_gates', 'n_features']

        all_headers = gene_headers + feature_headers + circuit_headers + objective_headers

        output_path_csv = f"{base_filename}_output.csv"

        with open(output_path_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(all_headers)

            for i in range(len(pareto_pop)):
                chrom = pareto_pop[i]
                v_fit = pareto_valid_fitness[i]
                v_acc = 1.0 - v_fit[0]
                t_acc = final_test_acc[i]

                obj_data = [round(v_acc, 5), round(t_acc, 5)] + [float(x) for x in v_fit[1:]]

                feature_info, circuit_info = decode_individual(chrom, total_features, params_qubits)
                mapping_str = get_feature_to_qubit_mapping(
                    chrom, total_features, params_qubits, sample_train,
                    feature_map_seed
                )
                row = (
                    list(chrom) +
                    [feature_info['feature_indices'], feature_info['n_selected_features']] +
                    [circuit_info['gate_type'], circuit_info['rx_gates'],
                     circuit_info['entangled_pairs'], circuit_info['n_entanglements'],
                     circuit_info['repeats'], mapping_str] +
                    obj_data
                )
                writer.writerow(row)

        print(f"Results saved to {output_path_csv}")

    return pareto_pop, test_fitness_values


def main():
    """Main entry point."""
    logging.info('WKA_robust')
    logging.info(f"Dataset={data_number}, N_JOB={N_JOB}, Seed={RANDOM_SEED}, Generations={num_generations}, PopSize={pop_size}, Crossover={rate_crossover}, Mutation={rate_mutation}, KA_Top={KA_TOP_PERCENT}, KA_Batch={KA_BATCH_RATIO}")

    for target_qubits in all_qubits:
        for scen in scenarios:
            print(f"\n{'#'*60}")
            logging.info(f"RUNNING SCENARIO: {scen['name']}")
            logging.info(f"Settings: Imbalance={scen['imb']}, Noise={scen['noise']}")
            print(f"{'#'*60}")

            pop, fit = run_optimization(
                params_qubits=target_qubits,
                params_num_generations=num_generations,
                imbalance_ratio=scen['imb'],
                noise_level=scen['noise']
            )

            best_error = np.min(fit[:, 0])
            best_score = 1.0 - best_error

            logging.info(f"SCENARIO {scen['name']} FINISHED. Best Balanced Accuracy: {best_score:.4f}")
            print(f">>> Scenario {scen['name']} Result: Best Balanced Acc = {best_score:.4f}")


if __name__ == "__main__":
    main()
