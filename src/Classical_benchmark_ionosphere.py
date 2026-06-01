"""
============================================================================
Classical Benchmark for Fair Comparison with QSVM
============================================================================
Experimental Protocol (Responding to Reviewer Comments 3.3 & 3.4):

References:
  - Nested CV: Cawley & Talbot, JMLR 2010
  - Evolutionary FS: Xue et al., IEEE TEVC 2016
  - Bayesian FS: Swersky et al., NeurIPS 2013; Baptista & Poloczek, NeurIPS 2018
============================================================================
"""

import numpy as np
import pandas as pd
import random
import time
import logging
import warnings
from itertools import product
import os
from joblib import Parallel, delayed
from datetime import datetime
import sys
from sklearn import datasets
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import (
    RFE,
    SelectKBest,
    SelectFromModel,
    mutual_info_classif,
    f_classif,
    VarianceThreshold
)
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

from xgboost import XGBClassifier
from skrebate import ReliefF
from pymrmr import mRMR
from skopt import gp_minimize
from skopt.space import Real
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint, loguniform
from sklearn.exceptions import ConvergenceWarning
import warnings as _warnings_module
from load_data import *




_warnings_module.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore')
N_TOTAL_CORES = os.cpu_count()
# =============================================================================
# Global experiment configuration
# =============================================================================

RANDOM_SEEDS = [0, 10, 20, 30, 40]      # 5 random seeds for result stability
N_OUTER_FOLDS = 5                       # Outer CV folds (performance estimation)
N_INNER_FOLDS = 3                       # Inner CV folds (hyperparameter tuning)
# Feature counts to evaluate
FEATURE_COUNTS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 'ALL']
N_RANDOM_ITER = 10                      # Random parameter sets per inner CV
# Parallel strategy: task-level parallelism instead of seed-level
# Flatten all (seed × fs_method × model) into task pool, N_CORES-2 workers
N_PARALLEL_WORKERS = max(1, N_TOTAL_CORES - 2)  # Reserve 2 cores for system/IO
INNER_N_JOBS = 1                        # Single thread per task (parallelism at upper level)
data_number = 3  # 1=Breast Cancer, 2=Fashion-MNIST, 3=Ionosphere, 4=Parkinsons
# --- Experiment scenario definitions ---
scenarios = [
  {"name": "Baseline (Clean)", "imb": None, "noise": 0.0}#,
  #{"name": "Imbalance 20%", "imb": 0.2, "noise": 0.0},
  #{"name": "Noise 0.5", "imb": None, "noise": 0.5}
    ]

# Classifier hyperparameter search space - ensure classical methods are well-tuned
PARAM_DISTRIBUTIONS = {
    "SVM-RBF": {
        'C': loguniform(0.01, 100),         # 从对数均匀分布采样
        'gamma': loguniform(0.001, 0.1)     # 从对数均匀分布采样
    },
    "Random Forest": {
        'n_estimators': randint(50, 300),   # 整数均匀分布
        'max_depth': [None, 5, 10, 15, 20],
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10)
    },
    "XGBoost": {
        'n_estimators': randint(50, 300),
        'learning_rate': loguniform(0.01, 0.3),
        'max_depth': randint(2, 10),
        'subsample': uniform(0.6, 0.4),     # [0.6, 1.0]
        'colsample_bytree': uniform(0.6, 0.4)
    },
    "Logistic Regression": {
        'C': loguniform(0.001, 100),
        'solver': ['liblinear', 'saga']
    },
    "Gradient Boosting": {
        'n_estimators': randint(50, 300),
        'learning_rate': loguniform(0.01, 0.3),
        'max_depth': randint(2, 10),
        'min_samples_split': randint(2, 20)
    }
}

# =============================================================================
# Utility functions
# =============================================================================
def _init_seed(seed):
    """Set global random seed for reproducibility."""
    seed_int = int(seed)
    random.seed(seed_int)
    np.random.seed(seed_int)
    os.environ['PYTHONHASHSEED'] = str(seed_int)
    return seed_int
class Tee:
    """Output to both console and file (supports stdout and stderr)."""

    def __init__(self, filepath, stream, mode='a'):
        self.file = open(filepath, mode, encoding='utf-8')
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.file.write(data)
        self.file.flush()

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def close(self):
        self.file.close()

    def __getattr__(self, name):
        """Proxy other attributes (e.g., isatty) to original stream."""
        return getattr(self.stream, name)

# =============================================================================
# Data loading and preprocessing
# =============================================================================

def load_data_and_split(scenario, random_state):
    """Load and split data according to experiment scenario."""
    if data_number==1:
        feature, total_features, sample_train, sample_valid, label_train, label_valid, sample_test, label_test = \
            load_breast_cancer_data(imbalance_ratio=scenario.get('imb'), noise_level=scenario.get('noise', 0.0), random_state=random_state)
    elif data_number==2:
        feature, total_features, sample_train, sample_valid, label_train, label_valid, sample_test, label_test = \
            load_fashion_mnist_data(n_components=50,  imbalance_ratio=scenario.get('imb'), noise_level=scenario.get('noise', 0.0), random_state=random_state)
    elif data_number==3:
        feature, total_features, sample_train, sample_valid, label_train, label_valid, sample_test, label_test = \
            load_ionosphere_data(imbalance_ratio=scenario.get('imb'), noise_level=scenario.get('noise', 0.0), random_state=random_state)
    elif data_number==4:
        feature, total_features, sample_train, sample_valid, label_train, label_valid, sample_test, label_test = \
            load_parkinsons_data(imbalance_ratio=scenario.get('imb'), noise_level=scenario.get('noise', 0.0), random_state=random_state)
    return {
        'train': (sample_train, label_train),
        'valid': (sample_valid, label_valid),
        'test': (sample_test, label_test),
        'feature_names': feature.feature_names,
        'total_features': total_features,
        # Merge train+valid for nested CV
        'X_full': np.vstack([sample_train, sample_valid]),
        'y_full': np.concatenate([label_train, label_valid])
    }


def apply_scaling(X_train, X_test):
    """
    Two-stage normalization:
    1. StandardScaler: zero mean, unit variance
    2. MinMaxScaler: scale to [-1, 1]

    Note: fit on training set only, transform applied to test set.
    """
    std_scaler = StandardScaler().fit(X_train)
    X_train_std = std_scaler.transform(X_train)
    X_test_std = std_scaler.transform(X_test)

    minmax_scaler = MinMaxScaler((-1, 1)).fit(X_train_std)
    X_train_norm = minmax_scaler.transform(X_train_std)
    X_test_norm = minmax_scaler.transform(X_test_std)

    return X_train_norm, X_test_norm


# =============================================================================
# Feature Selection: Evolutionary Algorithm (Genetic Algorithm)
# =============================================================================
def evolutionary_feature_selection(X, y, n_features_to_select, random_state=None,
                                   n_gen=50, pop_size=60, cx_prob=0.8,
                                   mut_prob=0.03, tournament_size=3, elite_size=2):
    """
    Genetic Algorithm-based Feature Selection (Binary GA).

    Algorithm details:
    - Encoding: Binary vector, 1=selected, 0=not selected
    - Constraint: Repair operator ensures exactly n_features_to_select features
    - Fitness: 3-fold stratified CV balanced accuracy (lightweight RF evaluator)
    - Selection: Tournament selection (tournament_size=3)
    - Crossover: Uniform crossover (better for feature selection)
    - Mutation: Bit-flip mutation, repair to target feature count after mutation
    - Elitism: Preserve top elite_size individuals
    - Convergence: Track best fitness per generation

    References:
    - Xue, B. et al. "A Survey on Evolutionary Computation Approaches to
      Feature Selection." IEEE TEVC, 2016.
    - Yang, J. & Honavar, V. "Feature subset selection using a genetic
      algorithm." IEEE Intelligent Systems, 1998.

    Args:
        X: np.ndarray, shape (n_samples, n_features), feature matrix
        y: np.ndarray, shape (n_samples,), label vector
        n_features_to_select: int, target number of features
        random_state: int or None, random seed
        n_gen: int, number of generations (default=50)
        pop_size: int, population size (default=60)
        cx_prob: float, crossover probability (default=0.8)
        mut_prob: float, per-bit mutation probability (default=0.03)
        tournament_size: int, tournament size (default=3)
        elite_size: int, number of elite individuals (default=2)

    Returns:
        selected: np.ndarray of bool, boolean mask of selected features
        feature_scores: np.ndarray, feature importance scores (based on population frequency)
    """
    rng = np.random.RandomState(random_state)
    n_total_features = X.shape[1]

    # If selecting all features, return immediately
    if n_features_to_select >= n_total_features:
        return np.ones(n_total_features, dtype=bool), np.ones(n_total_features)

    # --- Repair operator: ensure individual has exactly k ones ---
    def repair(individual, k):
        """Repair individual to select exactly k features."""
        n_selected = np.sum(individual)
        if n_selected == k:
            return individual
        elif n_selected > k:
            # Randomly turn off excess features
            ones_idx = np.where(individual == 1)[0]
            turn_off = rng.choice(ones_idx, int(n_selected - k), replace=False)
            individual[turn_off] = 0
        else:
            # Randomly turn on missing features
            zeros_idx = np.where(individual == 0)[0]
            turn_on = rng.choice(zeros_idx, int(k - n_selected), replace=False)
            individual[turn_on] = 1
        return individual

    # --- Fitness evaluation: internal 3-fold CV ---
    fitness_cache = {}  # Cache evaluated individuals to avoid recomputation

    def evaluate_fitness(individual):
        """
        Fitness function: 3-fold stratified CV balanced accuracy.
        Uses lightweight RandomForest as evaluator.
        """
        # Convert individual to hashable tuple for caching
        key = tuple(individual)
        if key in fitness_cache:
            return fitness_cache[key]

        selected_mask = individual.astype(bool)
        n_selected = np.sum(selected_mask)

        if n_selected == 0:
            fitness_cache[key] = 0.0
            return 0.0

        X_sel = X[:, selected_mask]

        # Lightweight RF evaluator (fewer estimators for speed)
        clf = RandomForestClassifier(
            n_estimators=50, max_depth=10,
            class_weight='balanced',
            random_state=random_state, n_jobs=1
        )

        try:
            # 3-fold stratified cross-validation
            inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
            scores = cross_val_score(
                clf, X_sel, y, cv=inner_cv,
                scoring='balanced_accuracy', n_jobs=INNER_N_JOBS
            )
            fitness = np.mean(scores)
        except Exception:
            fitness = 0.0

        fitness_cache[key] = fitness
        return fitness

    # --- Initialize population: each individual has exactly n_features_to_select ones ---
    population = np.zeros((pop_size, n_total_features), dtype=int)
    for i in range(pop_size):
        selected_idx = rng.choice(n_total_features, n_features_to_select, replace=False)
        population[i, selected_idx] = 1

    # --- Main evolution loop ---
    best_fitness_history = []  # Track best fitness per generation (convergence analysis)

    for gen in range(n_gen):
        # Evaluate fitness for all individuals
        fitness_values = np.array([evaluate_fitness(ind) for ind in population])

        # Record best of current generation
        best_gen_fitness = np.max(fitness_values)
        best_fitness_history.append(best_gen_fitness)

        # --- Elitism: pass best individuals directly to next generation ---
        elite_indices = np.argsort(fitness_values)[::-1][:elite_size]
        elites = population[elite_indices].copy()

        # --- Selection + Crossover + Mutation to generate new population ---
        new_population = []

        while len(new_population) < pop_size - elite_size:
            # Tournament selection: select two parents
            def tournament_select():
                candidates = rng.choice(pop_size, tournament_size, replace=False)
                winner = candidates[np.argmax(fitness_values[candidates])]
                return population[winner].copy()

            parent1 = tournament_select()
            parent2 = tournament_select()

            # Uniform Crossover
            if rng.rand() < cx_prob:
                mask = rng.randint(0, 2, n_total_features)
                child1 = np.where(mask, parent1, parent2)
                child2 = np.where(mask, parent2, parent1)
            else:
                child1 = parent1.copy()
                child2 = parent2.copy()

            # Bit-flip Mutation
            for child in [child1, child2]:
                mutation_mask = rng.rand(n_total_features) < mut_prob
                child[mutation_mask] = 1 - child[mutation_mask]

            # Repair constraint: ensure exactly k features selected
            child1 = repair(child1, n_features_to_select)
            child2 = repair(child2, n_features_to_select)

            new_population.append(child1)
            if len(new_population) < pop_size - elite_size:
                new_population.append(child2)

        # Merge elites and newly generated individuals
        population = np.vstack([elites, np.array(new_population)])

        # Early stopping: if no improvement for 10 consecutive generations
        if len(best_fitness_history) > 15:
            recent = best_fitness_history[-10:]
            if max(recent) - min(recent) < 1e-4:
                logging.info(f"  Evolutionary FS converged at generation {gen}")
                break

    # --- Final evaluation: select best individual ---
    final_fitness = np.array([evaluate_fitness(ind) for ind in population])
    best_individual = population[np.argmax(final_fitness)]
    selected = best_individual.astype(bool)

    # --- Compute feature importance scores: based on population frequency ---
    # Use top 50% fitness individuals for frequency calculation (quality-weighted)
    top_half = np.argsort(final_fitness)[::-1][:pop_size // 2]
    feature_scores = np.mean(population[top_half], axis=0)

    logging.info(
        f"  Evolutionary FS: best_fitness={np.max(final_fitness):.4f}, "
        f"generations={len(best_fitness_history)}, "
        f"cache_size={len(fitness_cache)}"
    )

    return selected, feature_scores


# =============================================================================
# Feature Selection: Bayesian Optimization
# =============================================================================
def bayesian_feature_selection(X, y, n_features_to_select, random_state=None, n_calls=30):
    """
    Bayesian Optimization-based Feature Selection (Continuous Relaxation + GP Surrogate).

    Algorithm details:
    - Idea: Relax discrete feature selection to continuous optimization
    - Search space: Assign continuous weight w_i ∈ [0, 1] to each feature
    - Dimensionality reduction: Since original 30D is GP-unfriendly, use "group weight" strategy:
      Group features by preliminary importance, optimize group weights + intra-group threshold
    - Objective: 3-fold CV balanced accuracy on selected feature subset
    - Surrogate model: Gaussian Process (Matérn 5/2 kernel)
    - Acquisition function: Expected Improvement (EI)
    - Final selection: Select top-k features based on optimal weight vector

    References:
    - Baptista, R. & Poloczek, M. "Bayesian Optimization of Combinatorial
      Structures." NeurIPS, 2018.
    - Swersky, K. et al. "Multi-task Bayesian Optimization." NeurIPS, 2013.
    - Shahriari, B. et al. "Taking the Human Out of the Loop." IEEE, 2016.

    Args:
        X: np.ndarray, shape (n_samples, n_features), feature matrix
        y: np.ndarray, shape (n_samples,), label vector
        n_features_to_select: int, target number of features
        random_state: int or None, random seed
        n_calls: int, BO iterations (default=30)

    Returns:
        selected: np.ndarray of bool, boolean mask of selected features
        feature_scores: np.ndarray, feature importance scores
    """
    rng = np.random.RandomState(random_state)
    n_total_features = X.shape[1]

    # If selecting all features, return immediately
    if n_features_to_select >= n_total_features:
        return np.ones(n_total_features, dtype=bool), np.ones(n_total_features)

    # --- Phase 1: Compute preliminary feature importance (as prior information) ---
    # Use fusion of multiple metrics: MI + ANOVA F-value + RF importance
    # This provides an informed starting point for BO

    # Mutual Information
    mi_scores = mutual_info_classif(X, y, random_state=random_state)
    mi_scores_norm = (mi_scores - mi_scores.min()) / (mi_scores.max() - mi_scores.min() + 1e-10)

    # ANOVA F-value
    f_scores, _ = f_classif(X, y)
    f_scores = np.nan_to_num(f_scores, nan=0.0)
    f_scores_norm = (f_scores - f_scores.min()) / (f_scores.max() - f_scores.min() + 1e-10)

    # Random Forest importance
    rf_temp = RandomForestClassifier(
        n_estimators=100, class_weight='balanced',
        random_state=random_state, n_jobs=1
    )
    rf_temp.fit(X, y)
    rf_scores = rf_temp.feature_importances_
    rf_scores_norm = (rf_scores - rf_scores.min()) / (rf_scores.max() - rf_scores.min() + 1e-10)

    # --- Phase 2: Define BO search space ---
    # Strategy: Optimize 3 weight parameters (for fusing three importance metrics) + 1 threshold parameter
    # Final score = w1*MI + w2*ANOVA + w3*RF, select top-k features by score
    # This reduces 30D discrete problem to 4D continuous problem, GP-friendly

    def objective(params):
        """
        BO objective function: Given weight parameters, compute fusion score,
        select top-k features, then evaluate with 3-fold CV.
        """
        w_mi, w_anova, w_rf, perturbation = params

        # Normalize weights
        w_sum = abs(w_mi) + abs(w_anova) + abs(w_rf) + 1e-10
        w_mi_n = abs(w_mi) / w_sum
        w_anova_n = abs(w_anova) / w_sum
        w_rf_n = abs(w_rf) / w_sum

        # Compute fusion score
        combined_scores = (w_mi_n * mi_scores_norm +
                           w_anova_n * f_scores_norm +
                           w_rf_n * rf_scores_norm)

        # Add small random perturbation to explore different feature subsets
        noise = rng.normal(0, perturbation, n_total_features)
        perturbed_scores = combined_scores + noise

        # Select top-k features by score
        top_indices = np.argsort(perturbed_scores)[::-1][:n_features_to_select]
        selected_mask = np.zeros(n_total_features, dtype=bool)
        selected_mask[top_indices] = True

        X_sel = X[:, selected_mask]

        # 3-fold stratified CV evaluation
        clf = RandomForestClassifier(
            n_estimators=50, max_depth=10,
            class_weight='balanced',
            random_state=random_state, n_jobs=1
        )
        try:
            inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
            scores = cross_val_score(
                clf, X_sel, y, cv=inner_cv,
                scoring='balanced_accuracy', n_jobs=INNER_N_JOBS
            )
            return -np.mean(scores)  # gp_minimize minimizes, so negate
        except Exception:
            return 0.0  # Return poor result

    # --- Phase 3: Run Bayesian Optimization ---
    search_space = [
        Real(0.0, 1.0, name='w_mi'),         # MI weight
        Real(0.0, 1.0, name='w_anova'),       # ANOVA weight
        Real(0.0, 1.0, name='w_rf'),          # RF importance weight
        Real(0.0, 0.3, name='perturbation')   # Perturbation strength (exploration parameter)
    ]

    try:
        result = gp_minimize(
            func=objective,
            dimensions=search_space,
            n_calls=n_calls,
            n_initial_points=10,  # Initial random exploration points
            acq_func='EI',        # Expected Improvement acquisition function
            random_state=random_state,
            noise=1e-10           # Assume objective function is nearly noiseless
        )

        # Use optimal parameters to determine final feature selection
        best_w_mi, best_w_anova, best_w_rf, _ = result.x

        # Normalize optimal weights
        w_sum = abs(best_w_mi) + abs(best_w_anova) + abs(best_w_rf) + 1e-10
        feature_scores = (abs(best_w_mi) / w_sum * mi_scores_norm +
                          abs(best_w_anova) / w_sum * f_scores_norm +
                          abs(best_w_rf) / w_sum * rf_scores_norm)

        # Select top-k features by score
        top_indices = np.argsort(feature_scores)[::-1][:n_features_to_select]
        selected = np.zeros(n_total_features, dtype=bool)
        selected[top_indices] = True

        logging.info(
            f"  Bayesian FS: best_score={-result.fun:.4f}, "
            f"weights=[MI:{best_w_mi:.2f}, ANOVA:{best_w_anova:.2f}, RF:{best_w_rf:.2f}]"
        )

    except Exception as e:
        # Fallback strategy: use equal-weight fusion
        logging.warning(f"  Bayesian FS failed ({e}), falling back to equal-weight fusion")
        feature_scores = (mi_scores_norm + f_scores_norm + rf_scores_norm) / 3.0
        top_indices = np.argsort(feature_scores)[::-1][:n_features_to_select]
        selected = np.zeros(n_total_features, dtype=bool)
        selected[top_indices] = True

    return selected, feature_scores


def run_holdout_test(X_full, y_full, X_test, y_test, fs_method, model_name,
                     n_features, random_state, feature_names):
    """
    Train on full training set (train+valid), evaluate on locked test set.

    This is independent of nested CV - simulates QSVM hold-out experimental conditions:
    1. Feature selection + hyperparameter tuning on entire train+valid
    2. Evaluate generalization on unseen test set

    Returns: dict containing test balanced accuracy, accuracy, and selected feature info.
    """
    n_actual = X_full.shape[1] if n_features == 'ALL' else n_features

    # Normalization (fit on X_full only)
    std_scaler = StandardScaler().fit(X_full)
    X_full_std = std_scaler.transform(X_full)
    X_test_std = std_scaler.transform(X_test)
    minmax_scaler = MinMaxScaler((-1, 1)).fit(X_full_std)
    X_full_norm = minmax_scaler.transform(X_full_std)
    X_test_norm = minmax_scaler.transform(X_test_std)

    # Feature selection (on entire training set)
    selector = FeatureSelector(fs_method, n_actual, random_state)
    X_train_sel = selector.fit_transform(X_full_norm, y_full)
    X_test_sel = selector.transform(X_test_norm)

    if X_train_sel.shape[1] == 0:
        return None

    # Hyperparameter tuning (inner CV on full training set)
    param_grid = PARAM_DISTRIBUTIONS.get(model_name)
    base_model = create_base_model(model_name, random_state)
    tuned_model, best_params = tune_hyperparameters(
        base_model, X_train_sel, y_full, param_grid, random_state
    )

    # Final training + test set evaluation
    fit_params = {}
    if 'XGB' in str(type(tuned_model)):
        weights = compute_sample_weight(class_weight='balanced', y=y_full)
        fit_params = {'sample_weight': weights}
    tuned_model.fit(X_train_sel, y_full, **fit_params)
    y_pred = tuned_model.predict(X_test_sel)

    test_ba = balanced_accuracy_score(y_test, y_pred)
    test_acc = accuracy_score(y_test, y_pred)

    return {
        'n_features_selected': X_train_sel.shape[1],
        'selected_features': ';'.join(
            name for name, keep in zip(feature_names, selector.support_)
            if keep),
        'best_params': str(best_params),
        'Test_Balanced_Accuracy': test_ba,
        'Test_Accuracy': test_acc
    }


def run_one_fs_model(seed, scen_name, X_full, y_full, X_test, y_test,
                     feature_names, fs_method, model_name, rng):
    """
    Process all feature count experiments for a single (seed, fs_method, model_name) combination.
    Designed as stateless, independently parallelizable task unit.

    Returns: (nested_cv_results, holdout_test_results)
        - nested_cv_results: list of dict, each dict corresponds to one nested CV fold
        - holdout_test_results: list of dict, each dict corresponds to one feature count's holdout test result
    """
    nested_results = []
    holdout_results = []
    for n_feat in FEATURE_COUNTS:
        n_actual = X_full.shape[1] if n_feat == 'ALL' else n_feat

        # Nested CV evaluation (on train+valid)
        nested_result = run_nested_cv(
            X_full, y_full, fs_method, model_name,
            n_actual, rng, feature_names
        )
        if nested_result is not None:
            for fold in nested_result['fold_results']:
                nested_results.append({
                    'Scenario': scen_name,
                    'Seed': seed,
                    'Feature_Selection': fs_method,
                    'Model': model_name,
                    'N_Features': n_actual,
                    'Outer_Fold': fold['fold'],
                    'Balanced_Accuracy': fold['balanced_accuracy'],
                    'Accuracy': fold['accuracy'],
                    'N_Features_Selected': fold['n_features_selected'],
                    'Selected_Features': ';'.join(fold['selected_features']),
                    'Best_Params': str(fold['best_params'])
                })

        # Holdout test evaluation (train on full train+valid, evaluate on locked test set)
        holdout_result = run_holdout_test(
            X_full, y_full, X_test, y_test,
            fs_method, model_name, n_actual, rng, feature_names
        )
        if holdout_result is not None:
            holdout_results.append({
                'Scenario': scen_name,
                'Seed': seed,
                'Feature_Selection': fs_method,
                'Model': model_name,
                'N_Features': n_actual,
                **holdout_result
            })

    return nested_results, holdout_results
# =============================================================================
# Unified Feature Selector Interface
# =============================================================================


class FeatureSelector:
    """
    Unified feature selection interface, wrapping all feature selection methods.

    Supported methods:
    - Filter: 'MI', 'ANOVA', 'Variance', 'ReliefF', 'mRMR'
    - Wrapper: 'RFE-RF', 'RFE-SVM', 'RFE-XGB'
    - Embedded: 'L1-SVM', 'L1-LR'
    - Meta-heuristic: 'Evolutionary', 'Bayesian'

    Usage:
        selector = FeatureSelector(method='MI', n_features=5, random_state=42)
        X_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
    """

    def __init__(self, method, n_features, random_state):
        self.method = method
        self.n_features = n_features
        self.random_state = random_state
        self.support_ = None
        self.selector_ = None
        self.scores_ = None
        self.n_features_actual = None

    def fit(self, X, y):
        """Execute feature selection on training data."""
        n_total = X.shape[1]

        # Determine actual number of features to select
        if self.n_features == 'ALL':
            self.n_features_actual = n_total
        else:
            self.n_features_actual = min(self.n_features, n_total)

        # If selecting all features and not a score-computing method, return all selected
        if self.n_features_actual == n_total and self.method not in ['Evolutionary', 'Bayesian']:
            self.support_ = np.ones(n_total, dtype=bool)
            self.scores_ = np.ones(n_total)
            return self

        # ====================== Filter Methods ======================
        if self.method == 'MI':
            # Mutual Information: measures nonlinear dependency between feature and target
            self.selector_ = SelectKBest(
                score_func=mutual_info_classif,
                k=self.n_features_actual
            )
            self.selector_.fit(X, y)
            self.support_ = self.selector_.get_support()
            self.scores_ = self.selector_.scores_

        elif self.method == 'ANOVA':
            # ANOVA F-test: measures linear relationship between feature and target
            self.selector_ = SelectKBest(
                score_func=f_classif,
                k=self.n_features_actual
            )
            self.selector_.fit(X, y)
            self.support_ = self.selector_.get_support()
            self.scores_ = self.selector_.scores_

        elif self.method == 'Variance':
            # Variance threshold: remove low-variance features (low information)
            variances = np.var(X, axis=0)
            top_indices = np.argsort(variances)[::-1][:self.n_features_actual]
            self.support_ = np.zeros(n_total, dtype=bool)
            self.support_[top_indices] = True
            self.scores_ = variances

        elif self.method == 'ReliefF':
            # ReliefF: nearest-neighbor based feature weight estimation
            n_neighbors = min(10, len(y) - 1)
            relief = ReliefF(
                n_features_to_select=self.n_features_actual,
                n_neighbors=n_neighbors
            )
            relief.fit(X, y)
            top_indices = np.argsort(relief.feature_importances_)[::-1][:self.n_features_actual]
            self.support_ = np.zeros(n_total, dtype=bool)
            self.support_[top_indices] = True
            self.scores_ = relief.feature_importances_

        elif self.method == 'mRMR':
            # Maximum Relevance Minimum Redundancy
            # Note: pymrmr requires DataFrame's first column to be the target variable
            # (pymrmr C++ implementation only recognizes column position, not column name)
            df = pd.DataFrame(X, columns=[str(i) for i in range(n_total)])
            df.insert(0, 'target', y)  # Insert target at column 0
            selected = mRMR(df, 'MIQ', self.n_features_actual)
            # Filter out non-numeric strings (e.g., 'target'), keep valid feature indices
            selected_indices = []
            for s in selected:
                try:
                    idx = int(s)
                    # Ensure index is in valid range
                    if 0 <= idx < n_total:
                        selected_indices.append(idx)
                except (ValueError, TypeError):
                    # Skip values that can't be converted to int (e.g., 'target')
                    continue
            # If no features selected after filtering, use all features
            if len(selected_indices) == 0:
                selected_indices = list(range(min(self.n_features_actual, n_total)))
            self.support_ = np.zeros(n_total, dtype=bool)
            self.support_[selected_indices] = True
            # mRMR ranking is itself a score (earlier = more important)
            self.scores_ = np.zeros(n_total)
            for rank, idx in enumerate(selected_indices):
                self.scores_[idx] = self.n_features_actual - rank

        # ====================== Wrapper Methods ======================
        elif self.method == 'RFE-RF':
            # Recursive Feature Elimination with Random Forest
            estimator = RandomForestClassifier(
                n_estimators=100, class_weight='balanced',
                random_state=self.random_state, n_jobs=1
            )
            self.selector_ = RFE(
                estimator, n_features_to_select=self.n_features_actual, step=1
            )
            self.selector_.fit(X, y)
            self.support_ = self.selector_.support_
            self.scores_ = self.selector_.ranking_

        elif self.method == 'RFE-SVM':
            # Recursive Feature Elimination with Linear SVM
            # Note: RFE requires linear kernel (needs coef_ attribute)
            estimator = LinearSVC(
                penalty='l2', dual=False,  # For n_samples > n_features
                class_weight='balanced',
                random_state=self.random_state,
                max_iter=10000
            )
            self.selector_ = RFE(
                estimator, n_features_to_select=self.n_features_actual, step=1
            )
            self.selector_.fit(X, y)
            self.support_ = self.selector_.support_
            self.scores_ = self.selector_.ranking_

        elif self.method == 'RFE-XGB':
            # Recursive Feature Elimination with XGBoost
            estimator = XGBClassifier(
                random_state=self.random_state,
                eval_metric='logloss', n_jobs=1, verbosity=0
            )
            weights = compute_sample_weight(class_weight='balanced', y=y)
            self.selector_ = RFE(
                estimator, n_features_to_select=self.n_features_actual, step=1
            )
            self.selector_.fit(X, y, sample_weight=weights)
            self.support_ = self.selector_.support_
            self.scores_ = self.selector_.ranking_

        # ====================== Embedded Methods ======================
        elif self.method == 'L1-SVM':
            # ===== 1. Internal hyperparameter tuning for C =====
            param_grid = {
                'C': np.logspace(-4, 2, 10)
            }

            base_model = LinearSVC(
                penalty='l1',
                dual=False,
                class_weight='balanced',
                random_state=self.random_state,
                max_iter=10000
            )

            inner_cv = StratifiedKFold(
                n_splits=3,
                shuffle=True,
                random_state=self.random_state
            )

            grid = RandomizedSearchCV(
                base_model,
                param_grid,
                cv=inner_cv,
                scoring='balanced_accuracy',
                n_jobs=INNER_N_JOBS,
                n_iter=min(N_RANDOM_ITER, 10),  # Randomly select from 10 C values
                random_state=self.random_state
            )

            grid.fit(X, y)
            best_model = grid.best_estimator_

            # ===== 2. Use best C to train and select features =====
            coef = np.abs(best_model.coef_).ravel()

            n_nonzero = np.sum(coef > 1e-10)

            if n_nonzero >= self.n_features_actual:
                # Sufficient non-zero features, select top k
                top_indices = np.argsort(coef)[::-1][:self.n_features_actual]
            else:
                # Insufficient non-zero features, supplement with zero-coefficient features
                non_zero_indices = np.where(coef > 1e-10)[0]
                zero_indices = np.where(coef <= 1e-10)[0]
                top_indices = list(non_zero_indices)
                if len(zero_indices) > 0:
                    supplement_count = self.n_features_actual - n_nonzero
                    rng = np.random.RandomState(self.random_state)
                    supplement = rng.choice(zero_indices, min(supplement_count, len(zero_indices)), replace=False)
                    top_indices.extend(supplement)
                top_indices = np.array(top_indices)

            self.support_ = np.zeros(n_total, dtype=bool)
            self.support_[top_indices] = True
            self.scores_ = coef

        elif self.method == 'L1-LR':
            param_grid = {
                'C': np.logspace(-4, 2, 10)
            }

            base_model = LogisticRegression(
                penalty='l1',
                solver='liblinear',
                class_weight='balanced',
                random_state=self.random_state,
                max_iter=5000
            )

            inner_cv = StratifiedKFold(
                n_splits=3,
                shuffle=True,
                random_state=self.random_state
            )

            grid = RandomizedSearchCV(
                base_model,
                param_grid,
                cv=inner_cv,
                scoring='balanced_accuracy',
                n_jobs=INNER_N_JOBS,
                n_iter=min(N_RANDOM_ITER, 10),
                random_state=self.random_state
            )

            grid.fit(X, y)
            best_model = grid.best_estimator_

            coef = np.abs(best_model.coef_).ravel()
            n_nonzero = np.sum(coef > 1e-10)

            if n_nonzero >= self.n_features_actual:
                # Sufficient non-zero features, select top k
                top_indices = np.argsort(coef)[::-1][:self.n_features_actual]
            else:
                # Insufficient non-zero features, supplement with zero-coefficient features
                non_zero_indices = np.where(coef > 1e-10)[0]
                zero_indices = np.where(coef <= 1e-10)[0]
                top_indices = list(non_zero_indices)
                if len(zero_indices) > 0:
                    supplement_count = self.n_features_actual - n_nonzero
                    rng = np.random.RandomState(self.random_state)
                    supplement = rng.choice(zero_indices, min(supplement_count, len(zero_indices)), replace=False)
                    top_indices.extend(supplement)
                top_indices = np.array(top_indices)

            self.support_ = np.zeros(n_total, dtype=bool)
            self.support_[top_indices] = True
            self.scores_ = coef
        # ====================== Meta-heuristic Methods ======================
        elif self.method == 'Evolutionary':
            # Genetic Algorithm feature selection
            self.support_, self.scores_ = evolutionary_feature_selection(
                X, y, self.n_features_actual,
                random_state=self.random_state,
                n_gen=50,
                pop_size=60,
                cx_prob=0.8, mut_prob=0.03,
                tournament_size=3, elite_size=2
            )

        elif self.method == 'Bayesian':
            # Bayesian Optimization feature selection
            self.support_, self.scores_ = bayesian_feature_selection(
                X, y, self.n_features_actual,
                random_state=self.random_state,
                n_calls=30
            )

        else:
            raise ValueError(f"Unknown feature selection method: {self.method}")

        return self

    def transform(self, X):
        """Apply feature selection (keep only selected feature columns)."""
        if self.support_ is None:
            raise RuntimeError("FeatureSelector has not been fitted yet.")
        return X[:, self.support_]

    def fit_transform(self, X, y):
        """Fit on training data and transform."""
        self.fit(X, y)
        return self.transform(X)


# =============================================================================
# Classifier creation and hyperparameter tuning
# =============================================================================
def create_base_model(model_name, random_state):
    """Create base classifier instance (with default hyperparameters)."""
    if model_name == "SVM-RBF":
        return SVC(
            kernel='rbf', class_weight='balanced',
            random_state=random_state
        )
    elif model_name == "Random Forest":
        return RandomForestClassifier(
            class_weight='balanced',
            random_state=random_state, n_jobs=1
        )
    elif model_name == "XGBoost":
        return XGBClassifier(
            random_state=random_state,
            eval_metric='logloss', n_jobs=1, verbosity=0
        )
    elif model_name == "Logistic Regression":
        return LogisticRegression(
            class_weight='balanced',
            random_state=random_state, max_iter=5000
        )
    elif model_name == "Gradient Boosting":
        return GradientBoostingClassifier(random_state=random_state)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def tune_hyperparameters(model, X, y, param_grid, random_state):
    """
    Hyperparameter search using inner cross-validation (RandomizedSearchCV).

    This corresponds to the inner loop of nested validation:
    - N_INNER_FOLDS-fold CV on given training fold
    - Use balanced_accuracy as evaluation metric
    - Use sample_weight for XGBoost to handle class imbalance

    Args:
        model: Base classifier instance
        X: Training features
        y: Training labels
        param_grid: Hyperparameter search grid
        random_state: Random seed

    Returns:
        best_model: Tuned best model
    """
    if param_grid is None:
        return model

    inner_cv = StratifiedKFold(
        n_splits=N_INNER_FOLDS, shuffle=True, random_state=random_state
    )

    fit_params = {}
    if 'XGB' in str(type(model)):
        weights = compute_sample_weight(class_weight='balanced', y=y)
        fit_params = {'sample_weight': weights}

    random_search = RandomizedSearchCV(
        model, param_grid, cv=inner_cv,
        scoring='balanced_accuracy',
        n_jobs=INNER_N_JOBS,
        verbose=0, refit=True,
        n_iter=N_RANDOM_ITER,
        random_state=random_state
    )
    random_search.fit(X, y, **fit_params)
    return random_search.best_estimator_, random_search.best_params_


# =============================================================================
# Evaluation Protocol 1: Nested Cross-Validation (Nested CV)
# =============================================================================
def run_nested_cv(X, y, feature_selection_method, model_name, n_features,
                  random_state, feature_names):
    """
    Complete nested cross-validation evaluation.

    Protocol description (responding to reviewer 3.4):
    +---------------------------------------------------------------------+
    | Outer loop (5-fold): Provides unbiased generalization estimate      |
    |   +-- Data normalization (fit on outer-train, transform outer-test) |
    |   +-- Feature selection (fit on outer-train only)                   |
    |   +-- Inner loop (3-fold RandomizedSearchCV): Hyperparameter tuning |
    |   |   +-- 3-fold CV on outer-train to select optimal hyperparams    |
    |   +-- Retrain on entire outer-train with optimal hyperparams        |
    |   +-- Evaluate on outer-test -> unbiased performance estimate       |
    +---------------------------------------------------------------------+

    Key: Feature selection is performed independently within each outer fold,
         no information leakage.

    Args:
        X: Complete feature matrix (merged train+valid)
        y: Complete label vector
        feature_selection_method: str, feature selection method name
        model_name: str, classifier name
        n_features: int or 'ALL', target number of features
        random_state: int, random seed
        feature_names: array, list of feature names

    Returns:
        dict: Contains per-fold results and summary statistics
    """
    outer_cv = StratifiedKFold(
        n_splits=N_OUTER_FOLDS, shuffle=True, random_state=random_state
    )

    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        X_outer_train, X_outer_test = X[train_idx], X[test_idx]
        y_outer_train, y_outer_test = y[train_idx], y[test_idx]

        # --- Step 1: Data normalization (fit on outer training fold) ---
        X_outer_train_norm, X_outer_test_norm = apply_scaling(
            X_outer_train, X_outer_test
        )

        # --- Step 2: Feature selection (only on outer training fold, no leakage) ---
        n_feat_actual = X_outer_train_norm.shape[1] if n_features == 'ALL' else n_features
        selector = FeatureSelector(
            feature_selection_method, n_feat_actual, random_state
        )
        X_train_sel = selector.fit_transform(X_outer_train_norm, y_outer_train)
        X_test_sel = selector.transform(X_outer_test_norm)

        if X_train_sel.shape[1] == 0:
            continue

        # --- Step 3: Inner CV hyperparameter tuning ---
        param_grid = PARAM_DISTRIBUTIONS.get(model_name)
        base_model = create_base_model(model_name, random_state)
        tuned_model, best_params = tune_hyperparameters(
            base_model, X_train_sel, y_outer_train, param_grid, random_state
        )

        # --- Step 4: Retrain on outer training fold, evaluate on outer test fold ---
        fit_params = {}
        if 'XGB' in str(type(tuned_model)):
            weights = compute_sample_weight(class_weight='balanced', y=y_outer_train)
            fit_params = {'sample_weight': weights}

        tuned_model.fit(X_train_sel, y_outer_train, **fit_params)
        y_pred = tuned_model.predict(X_test_sel)

        b_acc = balanced_accuracy_score(y_outer_test, y_pred)
        acc = accuracy_score(y_outer_test, y_pred)

        fold_results.append({
            'fold': fold_idx,
            'balanced_accuracy': b_acc,
            'accuracy': acc,
            'n_features_selected': X_train_sel.shape[1],
            'selected_features': [
                name for name, keep in zip(feature_names, selector.support_)
                if keep],
            'best_params': str(best_params)
        })

    if not fold_results:
        return None

    b_accs = [r['balanced_accuracy'] for r in fold_results]
    accs = [r['accuracy'] for r in fold_results]

    return {
        'fold_results': fold_results,
        'nested_cv_balanced_accuracy_mean': np.mean(b_accs),
        'nested_cv_balanced_accuracy_std': np.std(b_accs),
        'nested_cv_accuracy_mean': np.mean(accs),
        'nested_cv_accuracy_std': np.std(accs)
    }


# =============================================================================
# Main experiment workflow
# =============================================================================
def run_full_experiment():
    """Main experiment function."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filepath, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )



    # --- Feature selection methods (responding to reviewer 3.3) ---
    feature_selection_methods = [
        'RFE-RF',         # Wrapper: RFE with Random Forest
        'RFE-SVM',        # Wrapper: RFE with Linear SVM
        'RFE-XGB',        # Wrapper: RFE with XGBoost
        'L1-SVM',         # Embedded: L1-regularized SVM
        'L1-LR',          # Embedded: L1-regularized Logistic Regression
        'MI',             # Filter: Mutual Information
        'ANOVA',          # Filter: ANOVA F-test
        'Variance',       # Filter: Variance Threshold
        'ReliefF',        # Filter: ReliefF (nearest-neighbor based)
        'mRMR',           # Filter: Maximum Relevance Minimum Redundancy
        'Evolutionary',   # Meta-heuristic: Genetic Algorithm
        'Bayesian'        # Meta-heuristic: Bayesian Optimization
    ]

    # --- Classifier list ---
    models = [
        "SVM-RBF",
        "Random Forest",
        "Logistic Regression",
        "XGBoost",
        "Gradient Boosting"
    ]

    # --- Print experiment configuration ---
    logging.info("=" * 120)
    logging.info("Enhanced Classical Benchmark - Fair Comparison with QSVM")
    logging.info("=" * 120)
    logging.info(f"\nExperiment Configuration (Responding to Reviewer 3.3 & 3.4):")
    logging.info(f"  - Random Seeds: {RANDOM_SEEDS} (5 repetitions)")
    logging.info(f"  - Nested CV: {N_OUTER_FOLDS}-fold outer (performance estimation)"
                 f" × {N_INNER_FOLDS}-fold inner (hyperparameter tuning)")
    logging.info(f"  - Metric: Balanced Accuracy")
    logging.info(f"  - Feature Selection Methods ({len(feature_selection_methods)}): "
                 f"{feature_selection_methods}")
    logging.info(f"  - Classifiers ({len(models)}): {models}")
    logging.info(f"  - Feature Counts: {FEATURE_COUNTS}")
    logging.info("=" * 120)

    all_nested_results = []   # Nested CV per-fold results
    all_holdout_results = []  # Holdout test results
    base_output_dir = "results"
    os.makedirs(base_output_dir, exist_ok=True)

    for scen in scenarios:
        logging.info(f"\n{'='*100}")
        logging.info(f"SCENARIO: {scen['name']}")
        logging.info(f"{'='*100}")

        # -- Step 1: Preload data for each seed (serial, small data) --
        seeds_data = {}
        for seed in RANDOM_SEEDS:
            rng = _init_seed(seed)
            seeds_data[seed] = load_data_and_split(scen, rng)
            logging.info(f"  [Seed={seed}] Data loaded: "
                         f"Train+Valid={seeds_data[seed]['X_full'].shape[0]} samples, "
                         f"Test={seeds_data[seed]['test'][0].shape[0]} samples")

        # -- Step 2: Build flat task list --
        # Each task = (seed, fs_method, model) tuple
        tasks = []
        for seed in RANDOM_SEEDS:
            data = seeds_data[seed]
            X_full = data['X_full']
            y_full = data['y_full']
            X_test, y_test = data['test']  # Locked test set
            feature_names = data['feature_names']
            rng = _init_seed(seed)
            for fs_method in feature_selection_methods:
                for model_name in models:
                    tasks.append(
                        delayed(run_one_fs_model)(
                            seed, scen['name'], X_full, y_full, X_test, y_test,
                            feature_names, fs_method, model_name, rng
                        )
                    )

        # -- Step 3: Task-level parallel execution --
        logging.info(f"  Dispatching {len(tasks)} tasks across {N_PARALLEL_WORKERS} workers...")
        t_start = time.time()

        task_results = Parallel(n_jobs=N_PARALLEL_WORKERS, verbose=10)(
            tasks
        )

        elapsed = time.time() - t_start
        logging.info(f"  Scenario '{scen['name']}' completed in {elapsed/60:.1f} minutes")

        # -- Step 4: Collect results (nested CV + holdout test) --
        for nested_res, holdout_res in task_results:
            if nested_res:
                all_nested_results.extend(nested_res)
            if holdout_res:
                all_holdout_results.extend(holdout_res)

        # Save intermediate results for each seed (nested CV per-fold)
        for seed in RANDOM_SEEDS:
            seed_dir = os.path.join(base_output_dir, f"seed_{seed}")
            os.makedirs(seed_dir, exist_ok=True)
            seed_nested = [r for r in all_nested_results
                           if r['Seed'] == seed and r['Scenario'] == scen['name']]
            if seed_nested:
                pd.DataFrame(seed_nested).to_csv(
                    os.path.join(seed_dir, f"nested_cv_seed{seed}.csv"),
                    index=False)
            seed_holdout = [r for r in all_holdout_results
                            if r['Seed'] == seed and r['Scenario'] == scen['name']]
            if seed_holdout:
                pd.DataFrame(seed_holdout).to_csv(
                    os.path.join(seed_dir, f"holdout_test_seed{seed}.csv"),
                    index=False)

        logging.info(f"  Scenario '{scen['name']}' results saved.")
        logging.info("-" * 100)

    # =================================================================
    # Save results
    # =================================================================

    # Save nested CV per-fold raw results
    df_nested = pd.DataFrame(all_nested_results)
    nested_file = "classical_benchmark_nested_cv_raw.csv"
    df_nested.to_csv(nested_file, index=False)
    logging.info(f"Nested CV raw results (per-fold) saved to: {nested_file}")

    # Generate nested CV summary by seed
    if not df_nested.empty:
        summary_nested = df_nested.groupby(
            ['Scenario', 'Feature_Selection', 'Model', 'N_Features', 'Seed']
        ).agg({
            'Balanced_Accuracy': ['mean', 'std', 'count'],
            'Accuracy': ['mean', 'std']
        }).round(4)
        summary_nested.columns = ['NestedCV_BA_mean', 'NestedCV_BA_std',
                                  'N_Folds', 'NestedCV_Acc_mean', 'NestedCV_Acc_std']
        summary_nested = summary_nested.reset_index()
        summary_nested.to_csv("classical_benchmark_nested_cv_summary.csv", index=False)
        logging.info("Nested CV summary by seed saved.")

    # ---- Holdout Test Results ----
    df_holdout = pd.DataFrame(all_holdout_results)
    holdout_file = "classical_benchmark_holdout_test.csv"
    df_holdout.to_csv(holdout_file, index=False)
    logging.info(f"Holdout test results saved to: {holdout_file}")

    # Generate holdout test summary by seed and across seeds
    df_holdout_summary = pd.DataFrame()
    if not df_holdout.empty:
        # Summary by seed
        holdout_by_seed = df_holdout.groupby(
            ['Scenario', 'Feature_Selection', 'Model', 'N_Features', 'Seed']
        ).agg({
            'Test_Balanced_Accuracy': 'mean',
            'Test_Accuracy': 'mean'
        }).round(4).reset_index()
        holdout_by_seed.to_csv("classical_benchmark_holdout_by_seed.csv", index=False)

        # Summary across seeds
        df_holdout_summary = df_holdout.groupby(
            ['Scenario', 'Feature_Selection', 'Model', 'N_Features']
        ).agg({
            'Test_Balanced_Accuracy': ['mean', 'std'],
            'Test_Accuracy': ['mean', 'std']
        }).round(4)
        df_holdout_summary.columns = ['Test_BA_mean', 'Test_BA_std', 'Test_Acc_mean', 'Test_Acc_std']
        df_holdout_summary = df_holdout_summary.reset_index()
        df_holdout_summary.to_csv("classical_benchmark_holdout_summary.csv", index=False)
        logging.info("Holdout test summaries saved.")

    # Generate cross-seed summary (Nested CV aggregation): by (Scenario, FS, Model, N_Features)
    df_nested_summary = pd.DataFrame()
    if not df_nested.empty:
        df_nested_summary = df_nested.groupby(
            ['Scenario', 'Feature_Selection', 'Model', 'N_Features']
        ).agg({
            'Balanced_Accuracy': ['mean', 'std'],
            'Accuracy': ['mean', 'std']
        }).round(4)
        df_nested_summary.columns = ['NestedCV_BA_mean', 'NestedCV_BA_std', 'NestedCV_Acc_mean', 'NestedCV_Acc_std']
        df_nested_summary = df_nested_summary.reset_index()

    # =================================================================
    # Print final report
    # =================================================================
    logging.info("\n" + "=" * 120)
    logging.info("REVIEWER RESPONSE SUMMARY")
    logging.info("=" * 120)

    logging.info("\n[3.3] Feature Selection Methods Coverage:")
    logging.info(f"  Total methods: {len(feature_selection_methods)}")
    logging.info(f"  Filter: MI, ANOVA, Variance, ReliefF, mRMR")
    logging.info(f"  Wrapper: RFE-RF, RFE-SVM, RFE-XGB")
    logging.info(f"  Embedded: L1-SVM, L1-LR")
    logging.info(f"  Meta-heuristic: Evolutionary (GA), Bayesian (BO)")

    logging.info("\n[3.4] Fairness Guarantees:")
    logging.info(f"  + Nested CV ({N_OUTER_FOLDS}-fold outer x {N_INNER_FOLDS}-fold inner)")
    logging.info(f"  + Feature selection inside each outer fold (no leakage)")
    logging.info(f"  + Comprehensive hyperparameter tuning via RandomizedSearchCV")
    logging.info(f"  + Identical splits across all methods (same random seeds)")
    logging.info(f"  + Both nested CV and hold-out test reported for completeness")

    # Best combination ranking (Nested CV)
    if not df_nested_summary.empty:
        logging.info("\n" + "-" * 80)
        logging.info("TOP 10 by Nested CV Balanced Accuracy (averaged over seeds):")
        logging.info("-" * 80)
        top_nested = df_nested_summary.groupby(
            ['Scenario', 'Feature_Selection', 'Model', 'N_Features']
        )['NestedCV_BA_mean'].mean().sort_values(ascending=False).head(10)
        logging.info(top_nested.to_string())

    # Best combination ranking (Holdout Test)
    if not df_holdout_summary.empty:
        logging.info("\n" + "-" * 80)
        logging.info("TOP 10 by Holdout Test Balanced Accuracy (averaged over seeds):")
        logging.info("-" * 80)
        top_holdout = df_holdout_summary.groupby(
            ['Scenario', 'Feature_Selection', 'Model', 'N_Features']
        )['Test_BA_mean'].mean().sort_values(ascending=False).head(10)
        logging.info(top_holdout.to_string())

    return df_nested_summary, df_holdout_summary


# =============================================================================
# Program entry point
# =============================================================================
if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filepath = f"results/experiment_log_{timestamp}.txt"

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # Prevent duplicate handlers

    # File output
    file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    # Console output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Log format
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Use Tee class to capture both stdout and stderr to log file
    # This way Joblib progress bars, mRMR prints, ConvergenceWarning etc. will be logged
    _original_stdout = sys.stdout
    _original_stderr = sys.stderr
    sys.stdout = Tee(log_filepath, _original_stdout, mode='a')
    sys.stderr = Tee(log_filepath, _original_stderr, mode='a')

    try:
        run_full_experiment()
    finally:
        # Restore original streams and close Tee
        sys.stdout = _original_stdout
        sys.stderr = _original_stderr
        logging.info(f"Experiment finished, log saved to: {log_filepath}")
