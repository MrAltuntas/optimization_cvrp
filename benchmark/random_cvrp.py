import random
import time

class RandomCVRP:
    def __init__(self, cvrp):
        """
        Initialize the Random CVRP solver.

        Args:
            cvrp: CVRP problem instance
        """
        self.problem = cvrp

    def generate_random_solution(self):
        """
        Generate a random solution (permutation of cities).

        Returns:
            A random permutation of cities (excluding depot)
        """
        cities = list(range(self.problem.num_cities))
        cities.remove(self.problem.depot)
        random.shuffle(cities)
        return cities

    def decode_to_routes(self, individual):
        """
        Decode a permutation of cities into feasible routes.

        Args:
            individual: Permutation of cities (excluding depot)

        Returns:
            List of routes, each route is a list of city indices
        """
        routes = []
        route = []
        capacity_remaining = self.problem.capacity

        for city in individual:
            demand = self.problem.demands[city]

            # If adding this city exceeds capacity, start a new route
            if demand > capacity_remaining:
                if route:  # Only add non-empty routes
                    routes.append(route)
                route = [city]
                capacity_remaining = self.problem.capacity - demand
            else:
                route.append(city)
                capacity_remaining -= demand

        # Add the last route if not empty
        if route:
            routes.append(route)

        return routes

    def calculate_fitness(self, individual):
        """
        Calculate the fitness of an individual.

        Fitness is the total distance of all routes.

        Args:
            individual: Permutation of cities (excluding depot)

        Returns:
            Fitness value (lower is better)
        """
        routes = self.decode_to_routes(individual)
        return self.problem.get_total_distance(routes)

    def solve(self, num_iterations=10000):
        """
        Solve the CVRP using random search.

        Args:
            num_iterations: Number of random solutions to generate

        Returns:
            best_routes: List of routes in the best solution
            best_fitness: Fitness of the best solution
        """
        start_time = time.time()

        best_solution = None
        best_fitness = float('inf')

        for i in range(num_iterations):
            # Generate a random solution
            solution = self.generate_random_solution()

            # Calculate fitness
            fitness = self.calculate_fitness(solution)

            # Update best solution if improved
            if fitness < best_fitness:
                best_solution = solution
                best_fitness = fitness

            # Optional: print progress every 1000 iterations
            if (i + 1) % 1000 == 0:
                print(f"Iteration {i + 1}/{num_iterations}: Best fitness = {best_fitness}")

        # Decode the best solution to get routes
        best_routes = self.decode_to_routes(best_solution)

        end_time = time.time()
        execution_time = end_time - start_time

        print(f"Random Search completed in {execution_time:.2f} seconds")
        print(f"Best solution has fitness {best_fitness}")

        return best_routes, best_fitness