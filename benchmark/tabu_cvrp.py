import random
import time
import copy
import numpy as np
import matplotlib.pyplot as plt

class TabuSearchCVRP:
    def __init__(self, cvrp, max_iterations=100, tabu_size=10, neighborhood_size=20):
        """
        Initialize the Tabu Search for CVRP.

        Args:
            cvrp: CVRP problem instance
            max_iterations: Maximum number of iterations
            tabu_size: Size of the tabu list
            neighborhood_size: Number of neighbors to explore in each iteration
        """
        self.problem = cvrp
        self.max_iterations = max_iterations
        self.tabu_size = tabu_size
        self.neighborhood_size = neighborhood_size

        # Store the best solution found
        self.best_solution = None
        self.best_fitness = float('inf')
        self.best_iteration = 0

        # Track progress for visualization
        self.iteration_stats = []

    def generate_initial_solution(self):
        """
        Generate an initial solution using a simple construction heuristic.

        Returns:
            An individual as a permutation of cities (excluding depot)
        """
        # Use a simple greedy approach
        cities = list(range(self.problem.num_cities))
        cities.remove(self.problem.depot)

        # Sort cities by their distance from the depot
        depot = self.problem.depot
        cities.sort(key=lambda city: self.problem.distance_matrix[depot, city])

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

    def generate_neighbors(self, solution):
        """
        Generate a set of neighboring solutions.

        Neighbors are created by:
        1. Swap: Exchange the positions of two cities
        2. Insert: Move a city to a different position
        3. 2-opt: Reverse a segment of the route

        Args:
            solution: Current solution

        Returns:
            List of neighboring solutions
        """
        neighbors = []
        size = len(solution)

        for _ in range(self.neighborhood_size):
            # Randomly choose a move type
            move_type = random.choice(['swap', 'insert', '2-opt'])

            # Create a copy of the solution
            neighbor = solution.copy()

            if move_type == 'swap':
                # Swap two random cities
                i, j = random.sample(range(size), 2)
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]

            elif move_type == 'insert':
                # Move a city to a different position
                i = random.randint(0, size - 1)
                j = random.randint(0, size - 1)
                if i != j:
                    city = neighbor.pop(i)
                    neighbor.insert(j, city)

            elif move_type == '2-opt':
                # Reverse a segment of the route
                i, j = sorted(random.sample(range(size), 2))
                neighbor[i:j+1] = reversed(neighbor[i:j+1])

            # Add the neighbor to the list if it's not already there
            if neighbor not in neighbors:
                neighbors.append(neighbor)

        return neighbors

    def is_tabu(self, move, tabu_list):
        """
        Check if a move is in the tabu list.

        Args:
            move: A tuple (move_type, i, j) or the solution itself
            tabu_list: List of tabu moves

        Returns:
            True if the move is tabu, False otherwise
        """
        return move in tabu_list

    def calculate_hash(self, solution):
        """
        Calculate a hash value for a solution.

        Args:
            solution: A permutation of cities

        Returns:
            A hash value
        """
        return hash(tuple(solution))

    def solve(self):
        """
        Solve the CVRP using Tabu Search.

        Returns:
            best_routes: List of routes in the best solution
            stats: Statistics about the search process
        """
        start_time = time.time()

        # Generate initial solution
        current_solution = self.generate_initial_solution()
        current_fitness = self.calculate_fitness(current_solution)

        # Initialize best solution
        self.best_solution = current_solution.copy()
        self.best_fitness = current_fitness
        self.best_iteration = 0

        # Initialize tabu list (store solution hashes)
        tabu_list = []

        # Store initial stats
        self.iteration_stats.append((0, current_fitness))

        # Main search loop
        for iteration in range(1, self.max_iterations + 1):
            # Generate neighbors
            neighbors = self.generate_neighbors(current_solution)

            # Evaluate neighbors
            neighbor_fitnesses = [self.calculate_fitness(neighbor) for neighbor in neighbors]

            # Find the best non-tabu neighbor
            best_neighbor = None
            best_neighbor_fitness = float('inf')

            for i, neighbor in enumerate(neighbors):
                neighbor_hash = self.calculate_hash(neighbor)
                fitness = neighbor_fitnesses[i]

                # Accept if not tabu or if better than best solution (aspiration criterion)
                if (not self.is_tabu(neighbor_hash, tabu_list) or fitness < self.best_fitness):
                    if fitness < best_neighbor_fitness:
                        best_neighbor = neighbor
                        best_neighbor_fitness = fitness

            # If no non-tabu neighbor is found, choose the best neighbor
            if best_neighbor is None:
                best_idx = neighbor_fitnesses.index(min(neighbor_fitnesses))
                best_neighbor = neighbors[best_idx]
                best_neighbor_fitness = neighbor_fitnesses[best_idx]

            # Update current solution
            current_solution = best_neighbor
            current_fitness = best_neighbor_fitness

            # Update tabu list
            tabu_list.append(self.calculate_hash(current_solution))
            if len(tabu_list) > self.tabu_size:
                tabu_list.pop(0)  # Remove oldest move

            # Update best solution if improved
            if current_fitness < self.best_fitness:
                self.best_solution = current_solution.copy()
                self.best_fitness = current_fitness
                self.best_iteration = iteration

            # Store iteration stats
            self.iteration_stats.append((iteration, current_fitness))

            # Optional: print progress every few iterations
            if iteration % 10 == 0:
                print(f"Iteration {iteration}: Current fitness = {current_fitness}, Best fitness = {self.best_fitness}")

        # Decode the best solution to get routes
        best_routes = self.decode_to_routes(self.best_solution)

        end_time = time.time()
        execution_time = end_time - start_time

        print(f"Tabu Search completed in {execution_time:.2f} seconds")
        print(f"Best solution found in iteration {self.best_iteration} with fitness {self.best_fitness}")

        return best_routes, self.iteration_stats

    def plot_progress(self):
        """
        Plot the evolution of fitness over iterations.

        Returns:
            plt.Figure: The matplotlib figure
        """
        if not self.iteration_stats:
            print("No data to plot. Run solve() first.")
            return None

        iterations, fitnesses = zip(*self.iteration_stats)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(iterations, fitnesses, 'b-')

        # Mark the best solution
        ax.plot(self.best_iteration, self.best_fitness, 'ro', markersize=8,
                label=f'Best: {self.best_fitness:.2f} at iteration {self.best_iteration}')

        ax.set_xlabel('Iteration')
        ax.set_ylabel('Fitness (Total Distance)')
        ax.set_title('Tabu Search Progress')
        ax.legend()

        plt.grid(True)
        plt.tight_layout()

        return fig