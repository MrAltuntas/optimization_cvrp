import cvrp
import greedy_cvrp
import random_cvrp


cvrp = cvrp.CVRP(num_cities=5, depot=0, distance=5, capacity=10)

#print(cvrp.is_valid_path([1,2,0,3,4]))  # 5, True
#print(cvrp.is_valid_path([1, 2, 1, 3]))  # 2, False
#print(cvrp.is_valid_path([1, 2, 3]))     # 3, False

greedy_solver = greedy_cvrp.GreedyCVRP(cvrp)
greedy_path = greedy_solver.find_path()

random_solver = random_cvrp.RandomCVRP(cvrp)
random_path = random_solver.random_path()



print("Greedy path:", greedy_path)
print("Random path:", random_path)