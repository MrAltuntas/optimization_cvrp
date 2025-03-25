class CVRP:
    def __init__(self, num_cities: int, depot: int, distance: float, capacity: float):
        """
        :param num_cities: city count (except depot)
        :param depot: depot city (0)
        :param distance: distance between cities static
        :param capacity: vehicle capacity
        """
        self.num_cities = num_cities
        self.depot = depot
        self.distance = distance
        self.capacity = capacity
        self.all_cities = set(range(num_cities)) - {depot}

    def is_valid_path(self, path: list[int]) -> bool:
        """
        :param path: path of cities it will always start at depot [1, 2, 3, 4]
        :return: movement count, boolean if all the cities visited
        """
        visited = set()
        fuel = self.capacity
        movement = 0

        for city in path:

            if fuel < self.distance:
                return movement, False

            if city == self.depot:
                movement += 1
                current = city
                fuel = self.capacity
                print(current, fuel)
                continue

            if city in visited:
                return movement, False  # city must not visit again

            fuel -= self.distance
            movement += 1
            current = city
            visited.add(city)
            print(current, fuel)


        return movement, visited == self.all_cities


cvrp = CVRP(num_cities=5, depot=0, distance=5, capacity=15)

print(cvrp.is_valid_path([1,2,0,3,4]))  # 5, True
print(cvrp.is_valid_path([1, 2, 1, 3]))  # 2, False
print(cvrp.is_valid_path([1, 2, 3]))     # 3, False
