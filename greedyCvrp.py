import cvrp as CVRP

class GreedyCVRP:
    def __init__(self, cvrp: CVRP):
        self.problem = cvrp

    def find_path(self) -> list[int]:
        unvisited = sorted(self.problem.all_cities.copy())
        path = []
        fuel = self.problem.capacity
        current = self.problem.depot

        while unvisited:
            next_city = None

            if fuel >= 2 * self.problem.distance:
                next_city = unvisited[0]

            if next_city is None:
                # can't go anywhere fuel is not enough
                path.append(self.problem.depot)
                fuel = self.problem.capacity
                current = self.problem.depot
                continue

            path.append(next_city)
            fuel -= self.problem.distance
            unvisited.remove(next_city)
            current = next_city

        return path
