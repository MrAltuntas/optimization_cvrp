import config
import cvrp
import greedy_cvrp
import random_cvrp
import genetic_cvrp
import time

# CVRP problemi oluştur
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
    elitism_count=config.GA_ELITISM_SIZE
)


# 1. Greedy algoritma
print("1. Greedy Algoritma:")
start_time = time.time()
greedy_path = greedy_solver.find_path()
greedy_time = time.time() - start_time
greedy_movements, greedy_valid = cvrp_problem.is_valid_path(greedy_path)
print(f"Geçerli: {greedy_valid}, Hareketler: {greedy_movements}, Süre: {greedy_time:.4f}s")
print(f"Rota: {greedy_path}")
print()

# 2. Random algoritma
print("2. Random Algoritma:")
start_time = time.time()
random_path = random_solver.random_path()
random_time = time.time() - start_time
random_movements, random_valid = cvrp_problem.is_valid_path(random_path)
print(f"Geçerli: {random_valid}, Hareketler: {random_movements}, Süre: {random_time:.4f}s")
print(f"Rota: {random_path}")
print()

# 3. Genetik algoritma
print("3. Genetik Algoritma:")
start_time = time.time()
ga_path = ga_solver.solve()
ga_time = time.time() - start_time

if ga_path is not None:
    ga_movements, ga_valid = cvrp_problem.is_valid_path(ga_path)
    print(f"Geçerli: {ga_valid}, Hareketler: {ga_movements}, Süre: {ga_time:.4f}s")
    print(f"Rota: {ga_path}")
else:
    print("Geçerli çözüm bulunamadı, Süre: {ga_time:.4f}s")