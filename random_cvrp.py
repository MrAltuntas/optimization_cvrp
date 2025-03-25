import random
import cvrp as CVRP

class RandomCVRP:

    def __init__(self, cvrp: CVRP):
        self.problem = cvrp

    def random_path(self) -> list[int]:
        """
        CVRP nesnesi kullanılarak rastgele geçerli bir yol üretir.
        Depoya dönme ve yakıt sıfırlama işlemleri yapılır.
        """
        unvisited = list(self.problem.all_cities)
        random.shuffle(unvisited)
        path = []
        fuel = self.problem.capacity
        current = self.problem.depot

        for city in unvisited:
            if fuel < 2 * self.problem.distance:
                # fuel is almost empty
                path.append(self.problem.depot)
                fuel = self.problem.capacity
                current = self.problem.depot

            path.append(city)
            fuel -= self.problem.distance
            current = city

        return path



