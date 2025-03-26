import cvrp as CVRP
import random
import time

class TabuSearchCVRP:
    def __init__(self, cvrp: CVRP.CVRP, max_iterations=100, tabu_size=10, neighborhood_size=20):
        """
        Initialize the Tabu Search solver for CVRP.

        Args:
            cvrp: CVRP problem instance
            max_iterations: Maximum number of iterations
            tabu_size: Size of the tabu list
            neighborhood_size: Number of neighbors to generate in each iteration
        """
        self.problem = cvrp
        self.max_iterations = max_iterations
        self.tabu_size = tabu_size
        self.neighborhood_size = neighborhood_size
        self.best_solution = None
        self.best_movements = float('inf')
        self.best_valid = False

    def generate_initial_solution(self):
        """
        Generate an initial solution using a greedy approach.
        Visits cities in order, returning to depot when fuel runs low.
        """
        unvisited = list(self.problem.all_cities)
        path = []
        fuel = self.problem.capacity

        while unvisited:
            # If not enough fuel to visit next city and return, go back to depot
            if fuel < 2 * self.problem.distance:
                path.append(self.problem.depot)
                fuel = self.problem.capacity
                continue

            # Pick the next city
            next_city = unvisited[0]
            path.append(next_city)
            fuel -= self.problem.distance
            unvisited.remove(next_city)

        return path

    def evaluate_solution(self, solution):
        """
        Evaluate a solution and return (movements, valid).
        """
        movements, valid = self.problem.is_valid_path(solution)
        return movements, valid

    def generate_neighbors(self, solution):
        """
        Generate a list of neighboring solutions by applying small modifications.
        """
        neighbors = []

        for _ in range(self.neighborhood_size):
            # Create a copy to avoid modifying the original
            neighbor = solution.copy()

            # Randomly select a move type
            move_type = random.choice(['swap', 'insert_depot', 'remove_depot', 'relocate'])

            if move_type == 'swap' and len(neighbor) >= 2:
                # Swap two positions
                idx1, idx2 = random.sample(range(len(neighbor)), 2)
                neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]

            elif move_type == 'insert_depot':
                # Insert a depot at a random position
                pos = random.randint(0, len(neighbor))
                neighbor.insert(pos, self.problem.depot)

            elif move_type == 'remove_depot' and neighbor.count(self.problem.depot) > 1:
                # Remove a depot if there are multiple
                depot_positions = [i for i, city in enumerate(neighbor) if city == self.problem.depot]
                if depot_positions:
                    pos = random.choice(depot_positions)
                    neighbor.pop(pos)

            elif move_type == 'relocate' and len(neighbor) > 1:
                # Move a city from one position to another
                from_pos = random.randint(0, len(neighbor) - 1)
                city = neighbor.pop(from_pos)
                to_pos = random.randint(0, len(neighbor))
                neighbor.insert(to_pos, city)

            # Ensure all cities are still visited
            if set(neighbor) - {self.problem.depot} == self.problem.all_cities:
                neighbors.append(neighbor)

        # If no valid neighbors were generated, return the original solution
        # to avoid getting stuck
        if not neighbors:
            neighbors.append(solution.copy())

        return neighbors

    def is_tabu(self, solution, tabu_list):
        """
        Check if a solution is in the tabu list.
        AI suggest me to use a hash for efficiency.
        """
        solution_hash = hash(tuple(solution))
        return solution_hash in tabu_list

    def find_path(self):
        """
        Run the Tabu Search algorithm.
        This method name matches the interface of other solvers.

        Returns:
            The best solution found.
        """
        # Generate initial solution
        current_solution = self.generate_initial_solution()
        current_movements, current_valid = self.evaluate_solution(current_solution)

        # Initialize best solution
        self.best_solution = current_solution.copy()
        self.best_movements, self.best_valid = current_movements, current_valid

        # Initialize tabu list (using solution hashes)
        tabu_list = []

        # Main loop
        iteration = 0
        no_improvement = 0
        max_no_improvement = self.max_iterations // 4  # Stopping criterion

        start_time = time.time()

        while iteration < self.max_iterations and no_improvement < max_no_improvement:
            # Generate neighbors
            neighbors = self.generate_neighbors(current_solution)

            # Find best admissible neighbor
            best_neighbor = None
            best_neighbor_movements = float('inf')
            best_neighbor_valid = False

            for neighbor in neighbors:
                movements, valid = self.evaluate_solution(neighbor)

                # Check if this move is admissible (not tabu or meets aspiration criterion)
                is_admissible = (
                        not self.is_tabu(neighbor, tabu_list) or  # Not tabu
                        (valid and movements < self.best_movements)  # Aspiration criterion
                )

                if is_admissible:
                    # Prefer valid solutions
                    if valid and (not best_neighbor_valid or movements < best_neighbor_movements):
                        best_neighbor = neighbor
                        best_neighbor_movements = movements
                        best_neighbor_valid = True
                    # If no valid solution found yet, consider invalid ones
                    elif not best_neighbor_valid and (not best_neighbor or movements < best_neighbor_movements):
                        best_neighbor = neighbor
                        best_neighbor_movements = movements
                        best_neighbor_valid = valid

            # If no admissible neighbor found, break
            if best_neighbor is None:
                break

            # Update current solution
            current_solution = best_neighbor
            current_movements = best_neighbor_movements
            current_valid = best_neighbor_valid

            # Update tabu list
            tabu_list.append(hash(tuple(current_solution)))
            if len(tabu_list) > self.tabu_size:
                tabu_list.pop(0)  # Remove oldest entry

            # Update best solution if needed
            if current_valid and (not self.best_valid or current_movements < self.best_movements):
                self.best_solution = current_solution.copy()
                self.best_movements = current_movements
                self.best_valid = current_valid
                no_improvement = 0
            else:
                no_improvement += 1

            # Log progress periodically
            if iteration % 10 == 0:
                elapsed_time = time.time() - start_time
                print(f"Iteration {iteration}/{self.max_iterations}, "
                      f"Best: {self.best_movements}, "
                      f"Current: {current_movements}, "
                      f"Valid: {current_valid}, "
                      f"Time: {elapsed_time:.2f}s")

            iteration += 1

        # Final log
        elapsed_time = time.time() - start_time
        print(f"Tabu Search completed after {iteration} iterations and {elapsed_time:.2f} seconds")
        print(f"Best solution has {self.best_movements} movements, Valid: {self.best_valid}")

        return self.best_solution