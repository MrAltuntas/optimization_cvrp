import cvrp
import greedyCvrp



cvrp = cvrp.CVRP(num_cities=5, depot=0, distance=5, capacity=25)

#print(cvrp.is_valid_path([1,2,0,3,4]))  # 5, True
#print(cvrp.is_valid_path([1, 2, 1, 3]))  # 2, False
#print(cvrp.is_valid_path([1, 2, 3]))     # 3, False

greedy_solver = greedyCvrp.GreedyCVRP(cvrp)

path = greedy_solver.find_path()
print("Greedy path:", path)