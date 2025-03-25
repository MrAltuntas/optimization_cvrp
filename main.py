import cvrp
import greedy_cvrp
import random_cvrp
import genetic_cvrp
import time

cvrp = cvrp.CVRP(num_cities=5, depot=0, distance=5, capacity=25)

#print(cvrp.is_valid_path([1,2,0,3,4]))  # 5, True
#print(cvrp.is_valid_path([1, 2, 1, 3]))  # 2, False
#print(cvrp.is_valid_path([1, 2, 3]))     # 3, False

greedy_solver = greedy_cvrp.GreedyCVRP(cvrp)
greedy_path = greedy_solver.find_path()

random_solver = random_cvrp.RandomCVRP(cvrp)
random_path = random_solver.random_path()



print("Greedy path:", cvrp.is_valid_path(greedy_path), greedy_path)
print("Random path:", cvrp.is_valid_path(random_path), random_path)


#### tüm genetic için olan işlemleri burada yap

print("Running Genetic Algorithm:")
start_time = time.time()
ga_solver = genetic_cvrp.GeneticCVRP(
    cvrp,
    population_size=100,
    generations=50,
    crossover_rate=0.7,
    mutation_rate=0.1,
    tournament_size=5,
    elitism_count=2
)
ga_path = ga_solver.solve()
ga_time = time.time() - start_time
print("Genetic algorithm path:", ga_path)
movements, is_valid = cvrp.is_valid_path(ga_path)
print(f"Valid: {is_valid}, Movements: {movements}")
print(f"Time taken: {ga_time:.4f} seconds")
print("-" * 50)