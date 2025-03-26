import config
import cvrp
import greedy_cvrp
import random_cvrp
import genetic_cvrp
import tabu_cvrp
import time

cvrp_problem = cvrp.CVRP(num_cities=config.CVRP_NUMBER_OF_CITIES, depot=config.CVRP_DEPOT_POSITION, distance=config.CVRP_DISTANCE_BETWEEN_CITIES, capacity=config.CVRP_CAPACITY)
greedy_solver = greedy_cvrp.GreedyCVRP(cvrp_problem)
random_solver = random_cvrp.RandomCVRP(cvrp_problem)
ga_solver = genetic_cvrp.GeneticCVRP(
    cvrp_problem,
    population_size=config.GA_POPULATION_SIZE,
    generations=config.GA_GENERATIONS,
    crossover_rate=config.GA_CROSSOVER_RATE,
    mutation_rate=config.GA_MUTATION_RATE,
    tournament_size=config.GA_TOURNAMENT_SIZE,
    elitism_count=config.GA_ELITISM_SIZE,
    improved_evaluation=config.GA_IMPROVED_EVALUATION,
)
ts_solver = tabu_cvrp.TabuSearchCVRP(
    cvrp_problem,
    max_iterations=config.TS_MAX_ITERATIONS,
    tabu_size=config.TS_TABU_SIZE,
    neighborhood_size=config.TS_NEIGHBORHOOD_SIZE
)

# Greedy
print("1. Greedy Algorithm:")
start_time = time.time()
greedy_path = greedy_solver.find_path()
greedy_time = time.time() - start_time
greedy_movements, greedy_valid = cvrp_problem.is_valid_path(greedy_path)
print(f"Valid: {greedy_valid}, Movements: {greedy_movements}, Time: {greedy_time:.4f}s")
print(f"Path: {greedy_path}")
print()

# Random
print("2. Random Algorithm:")
start_time = time.time()
random_path = random_solver.random_path()
random_time = time.time() - start_time
random_movements, random_valid = cvrp_problem.is_valid_path(random_path)
print(f"Valid: {random_valid}, Movement: {random_movements}, Time: {random_time:.4f}s")
print(f"Path: {random_path}")
print()

# Tabu Search
print("4. Tabu Search Algorithm:")
start_time = time.time()
ts_path = ts_solver.find_path()
ts_time = time.time() - start_time

if ts_path is not None:
    ts_movements, ts_valid = cvrp_problem.is_valid_path(ts_path)
    print(f"Valid: {ts_valid}, Movements: {ts_movements}, Time: {ts_time:.4f}s")
    print(f"Path: {ts_path}")
else:
    print(f"Couldn't find a solution, Time: {ts_time:.4f}s")

# Genetic Algorithm
print("3. Genetic Algorithm:")
start_time = time.time()
ga_path = ga_solver.solve()
ga_time = time.time() - start_time

if ga_path is not None:
    ga_movements, ga_valid = cvrp_problem.is_valid_path(ga_path)
    print(f"Valid: {ga_valid}, Movements: {ga_movements}, Time: {ga_time:.4f}s")
    print(f"Path: {ga_path}")
else:
    print(f"Couldn't find a solution, Time: {ga_time:.4f}s")