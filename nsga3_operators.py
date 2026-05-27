import numpy as np


def compute_offspring_count(rate_or_count, parent_pop_size):
    """Convert rate (0,1] to offspring count, or use explicit count if >= 1."""
    try:
        val = float(rate_or_count)
    except Exception:
        val = 0.0
    if val <= 0:
        return 0
    if 0 < val < 1:
        n = int(np.floor(val * parent_pop_size))
    else:
        n = int(np.floor(val))
    if n % 2 == 1:
        n = max(2, n - 1)
    return n


def random_population(n_var, n_sol):
    """Initialize random population with binary values."""
    pop = np.random.randint(0, 2, size=(n_sol, n_var), dtype=int)
    return pop


def crossover(pop, crossover_rate):
    """Crossover operation."""
    parent_size, n_genes = pop.shape
    n_offspring = compute_offspring_count(crossover_rate, parent_size)
    if n_offspring == 0:
        return np.zeros((0, n_genes), dtype=int)

    offspring = np.zeros((n_offspring, n_genes), dtype=int)
    for i in range(n_offspring // 2):
        r1 = np.random.randint(0, parent_size)
        r2 = np.random.randint(0, parent_size)
        while r2 == r1:
            r2 = np.random.randint(0, parent_size)
        cutting_point = np.random.randint(1, n_genes)
        offspring[2 * i, :cutting_point] = pop[r1, :cutting_point]
        offspring[2 * i, cutting_point:] = pop[r2, cutting_point:]
        offspring[2 * i + 1, :cutting_point] = pop[r2, :cutting_point]
        offspring[2 * i + 1, cutting_point:] = pop[r1, cutting_point:]
    return offspring


def mutation(pop, mutation_rate):
    """Mutation operation via bit flip."""
    parent_size, n_genes = pop.shape
    n_offspring = compute_offspring_count(mutation_rate, parent_size)
    if n_offspring == 0:
        return np.zeros((0, n_genes), dtype=int)

    offspring = np.zeros((n_offspring, n_genes), dtype=int)
    for i in range(n_offspring // 2):
        r1 = np.random.randint(0, parent_size)
        r2 = np.random.randint(0, parent_size)
        while r2 == r1:
            r2 = np.random.randint(0, parent_size)
        cutting_point = np.random.randint(0, n_genes)
        offspring[2 * i] = pop[r1].copy()
        offspring[2 * i, cutting_point] = 1 - offspring[2 * i, cutting_point]
        offspring[2 * i + 1] = pop[r2].copy()
        offspring[2 * i + 1, cutting_point] = 1 - offspring[2 * i + 1, cutting_point]
    return offspring


def generate_reference_points(n_obj, n_partitions=5):
    """Generate Das-Dennis reference points."""
    def recursive_generate(n_obj, left, total, depth, current):
        points = []
        if depth == n_obj - 1:
            current[depth] = left / total
            points.append(current.copy())
        else:
            for i in range(left + 1):
                current[depth] = i / total
                points.extend(recursive_generate(n_obj, left - i, total, depth + 1, current))
        return points

    current = np.zeros(n_obj)
    points = recursive_generate(n_obj, n_partitions, n_partitions, 0, current)
    return np.array(points)


def normalize_objectives(fitness_values, ideal_point=None, nadir_point=None):
    """Normalize objective values."""
    if ideal_point is None:
        ideal_point = np.min(fitness_values, axis=0)

    if nadir_point is None:
        nadir_point = np.max(fitness_values, axis=0)

    denominator = nadir_point - ideal_point
    denominator = np.where(denominator == 0, 1e-10, denominator)

    normalized_fitness = (fitness_values - ideal_point) / denominator

    return normalized_fitness, ideal_point, nadir_point


def associate_to_reference_points(normalized_fitness, reference_points):
    """Associate solutions to nearest reference point using perpendicular distance."""
    associations = []
    distances = []

    for i in range(normalized_fitness.shape[0]):
        x = normalized_fitness[i]
        min_dist = np.inf
        min_idx = -1

        for j, z in enumerate(reference_points):
            z_norm = np.linalg.norm(z)
            if z_norm == 0:
                z_norm = 1e-10
            proj = np.dot(x, z) / (z_norm ** 2)
            d = np.linalg.norm(x - proj * z)
            if d < min_dist:
                min_dist = d
                min_idx = j

        associations.append(min_idx)
        distances.append(min_dist)

    return np.array(associations), np.array(distances)


def niching(fitness_values, n_survive, reference_points):
    """NSGA-III niching selection mechanism."""
    pop_size = fitness_values.shape[0]

    normalized_fitness, ideal_point, nadir_point = normalize_objectives(fitness_values)

    associations, distances = associate_to_reference_points(normalized_fitness, reference_points)

    n_ref_points = reference_points.shape[0]
    niche_counts = np.zeros(n_ref_points, dtype=float)

    for assoc in associations:
        niche_counts[assoc] += 1

    selected_indices = []
    remaining_indices = list(range(pop_size))

    while len(selected_indices) < n_survive and len(remaining_indices) > 0:
        min_niche = np.min(niche_counts)
        min_niche_indices = np.where(niche_counts == min_niche)[0]

        selected_niche = np.random.choice(min_niche_indices)

        candidates = [i for i in remaining_indices if associations[i] == selected_niche]

        if len(candidates) > 0:
            candidate_distances = [distances[i] for i in candidates]
            best_candidate = candidates[np.argmin(candidate_distances)]

            selected_indices.append(best_candidate)
            remaining_indices.remove(best_candidate)
            niche_counts[selected_niche] += 1
        else:
            niche_counts[selected_niche] = np.inf

    return np.array(selected_indices, dtype=int)


def non_dominated_sorting(fitness_values):
    """Fast non-dominated sorting."""
    pop_size = fitness_values.shape[0]
    domination_counts = np.zeros(pop_size, dtype=int)
    dominated_solutions = [[] for _ in range(pop_size)]

    fronts = [[]]

    for i in range(pop_size):
        for j in range(i + 1, pop_size):
            if all(fitness_values[i] <= fitness_values[j]) and any(fitness_values[i] < fitness_values[j]):
                dominated_solutions[i].append(j)
                domination_counts[j] += 1
            elif all(fitness_values[j] <= fitness_values[i]) and any(fitness_values[j] < fitness_values[i]):
                dominated_solutions[j].append(i)
                domination_counts[i] += 1

        if domination_counts[i] == 0:
            fronts[0].append(i)

    k = 0
    while len(fronts[k]) > 0:
        next_front = []
        for i in fronts[k]:
            for j in dominated_solutions[i]:
                domination_counts[j] -= 1
                if domination_counts[j] == 0:
                    next_front.append(j)
        k += 1
        fronts.append(next_front)

    return fronts[:-1]


def pareto_front_finding(fitness_values, pop_index):
    """Find Pareto front indices."""
    pop_size = fitness_values.shape[0]
    pareto_front = np.ones(pop_size, dtype=bool)
    for i in range(pop_size):
        for j in range(pop_size):
            if all(fitness_values[j] <= fitness_values[i]) and any(fitness_values[j] < fitness_values[i]):
                pareto_front[i] = 0
                break

    return pop_index[pareto_front]


def selection_nsga3(pop, fitness_values, pop_size, n_partitions=5):
    """NSGA-III selection operation."""
    n_obj = fitness_values.shape[1]

    reference_points = generate_reference_points(n_obj, n_partitions)

    fronts = non_dominated_sorting(fitness_values)

    selected_indices = []
    k = 0

    while k < len(fronts) and len(selected_indices) + len(fronts[k]) <= pop_size:
        selected_indices.extend(fronts[k])
        k += 1

    if k < len(fronts) and len(selected_indices) < pop_size:
        last_front_indices = np.array(fronts[k], dtype=int)
        last_front_fitness = fitness_values[last_front_indices]

        n_remaining = pop_size - len(selected_indices)

        selected_from_last = niching(last_front_fitness, n_remaining, reference_points)
        selected_indices.extend(list(last_front_indices[selected_from_last]))

    selected_indices = list(dict.fromkeys(selected_indices))
    if len(selected_indices) < pop_size:
        all_indices = list(range(pop.shape[0]))
        for idx in all_indices:
            if idx not in selected_indices:
                selected_indices.append(idx)
            if len(selected_indices) == pop_size:
                break

    selected_indices = np.array(selected_indices[:pop_size], dtype=int)
    selected_pop = pop[selected_indices]

    return selected_pop, selected_indices


if __name__ == "__main__":
    print(1)
