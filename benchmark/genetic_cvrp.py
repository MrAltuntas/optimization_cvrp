import random
import numpy as np
import time
import matplotlib.pyplot as plt

class GeneticCVRP:
    def __init__(self, cvrp, population_size=100, generations=100,
                 crossover_rate=0.7, mutation_rate=0.1, tournament_size=5,
                 elitism_size=2):
        """
        Initialize the Genetic Algorithm for CVRP.

        Args:
            cvrp: CVRP problem instance
            population_size: Size of the population
            generations: Number of generations
            crossover_rate: Probability of crossover
            mutation_rate: Probability of mutation
            tournament_size: Size of tournament selection
            elitism_size: Number of best individuals to preserve
        """
        self.problem = cvrp
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elitism_size = elitism_size

        # Store the best solution found
        self.best_individual = None
        self.best_fitness = float('inf')
        self.best_generation = 0

        # Track progress for visualization
        self.generation_stats = []

    def create_individual(self):
        """
        Create a random individual (solution).

        An individual is represented as a permutation of cities (excluding depot).
        The actual routes are decoded later considering capacity constraints.

        Returns:
            A list of cities in visit order (excluding depot)
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

    def tournament_selection(self, population, fitnesses):
        """
        Select an individual using tournament selection.

        Args:
            population: List of individuals
            fitnesses: List of fitness values

        Returns:
            Selected individual
        """
        # Select tournament_size individuals randomly
        tournament_indices = random.sample(range(len(population)), self.tournament_size)
        tournament_fitness = [fitnesses[i] for i in tournament_indices]

        # Return the best individual in the tournament
        best_idx = tournament_indices[tournament_fitness.index(min(tournament_fitness))]
        return population[best_idx]

    def ordered_crossover(self, parent1, parent2):
        """
        Apply Ordered Crossover (OX) operator.

        Args:
            parent1, parent2: Parents to crossover

        Returns:
            child: Offspring
        """
        if random.random() > self.crossover_rate:
            return parent1.copy()

        # Choose crossover points
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))

        # Initialize the child with the segment from parent1
        child = [None] * size
        for i in range(start, end + 1):
            child[i] = parent1[i]

        # Fill the remaining positions with cities from parent2
        p2_idx = 0
        for i in range(size):
            if child[i] is None:
                # Find next city in parent2 that is not already in child
                while parent2[p2_idx] in child:
                    p2_idx = (p2_idx + 1) % size
                child[i] = parent2[p2_idx]
                p2_idx = (p2_idx + 1) % size

        return child

    def partially_matched_crossover(self, parent1, parent2):
        """
        Apply Partially Matched Crossover (PMX) operator with a more robust implementation.

        Args:
            parent1, parent2: Parents to crossover

        Returns:
            child: Offspring
        """
        if random.random() > self.crossover_rate:
            return parent1.copy()

        size = len(parent1)

        # Step 1: Choose crossover segment
        start, end = sorted(random.sample(range(size), 2))

        # Step 2: Create child with None values
        child = [None] * size

        # Step 3: Copy segment from parent1
        for i in range(start, end + 1):
            child[i] = parent1[i]

        # Step 4: Fill remaining positions using permutation from parent2
        p2_idx = 0
        for i in range(size):
            if child[i] is None:  # This position needs to be filled
                # Find a value from parent2 that isn't already in child
                while parent2[p2_idx] in child:
                    p2_idx = (p2_idx + 1) % size

                # Use this value
                child[i] = parent2[p2_idx]
                p2_idx = (p2_idx + 1) % size

        return child

    def swap_mutation(self, individual):
        """
        Apply swap mutation operator.

        Args:
            individual: Individual to mutate

        Returns:
            Mutated individual
        """
        if random.random() > self.mutation_rate:
            return individual

        # Create a copy to avoid modifying the original
        mutated = individual.copy()

        # Choose two random positions and swap them
        idx1, idx2 = random.sample(range(len(mutated)), 2)
        mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]

        return mutated

    def inversion_mutation(self, individual):
        """
        Apply inversion mutation operator.

        Args:
            individual: Individual to mutate

        Returns:
            Mutated individual
        """
        if random.random() > self.mutation_rate:
            return individual

        # Create a copy to avoid modifying the original
        mutated = individual.copy()

        # Choose two random positions
        idx1, idx2 = sorted(random.sample(range(len(mutated)), 2))

        # Invert the segment between the two positions
        mutated[idx1:idx2+1] = reversed(mutated[idx1:idx2+1])

        return mutated

    def solve(self):
        """
        Solve the CVRP using the Genetic Algorithm.

        Returns:
            best_routes: List of routes in the best solution
            stats: Statistics about the evolution process
        """
        start_time = time.time()

        # Initialize population
        population = [self.create_individual() for _ in range(self.population_size)]

        # Calculate initial fitness
        fitnesses = [self.calculate_fitness(ind) for ind in population]

        # Keep track of the best solution
        best_idx = fitnesses.index(min(fitnesses))
        self.best_individual = population[best_idx].copy()
        self.best_fitness = fitnesses[best_idx]
        self.best_generation = 0

        # Store initial stats
        gen_best = min(fitnesses)
        gen_avg = sum(fitnesses) / len(fitnesses)
        gen_worst = max(fitnesses)
        self.generation_stats.append((0, gen_best, gen_avg, gen_worst))

        # Main evolution loop
        for generation in range(1, self.generations + 1):
            # Sort population by fitness
            sorted_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i])
            sorted_population = [population[i] for i in sorted_indices]

            # Create new population
            new_population = []

            # Elitism: directly copy the best individuals
            new_population.extend(sorted_population[:self.elitism_size])

            # Create the rest of the population through selection, crossover, mutation
            while len(new_population) < self.population_size:
                # Select parents
                parent1 = self.tournament_selection(population, fitnesses)
                parent2 = self.tournament_selection(population, fitnesses)

                # Create offspring through crossover
                if random.random() < 0.5:  # Randomly choose crossover operator
                    offspring = self.ordered_crossover(parent1, parent2)
                else:
                    offspring = self.partially_matched_crossover(parent1, parent2)

                # Apply mutation
                if random.random() < 0.5:  # Randomly choose mutation operator
                    offspring = self.swap_mutation(offspring)
                else:
                    offspring = self.inversion_mutation(offspring)

                new_population.append(offspring)

            # Update population
            population = new_population[:self.population_size]  # Ensure population size is maintained

            # Calculate fitness for new population
            fitnesses = [self.calculate_fitness(ind) for ind in population]

            # Update best solution if improved
            best_idx = fitnesses.index(min(fitnesses))
            if fitnesses[best_idx] < self.best_fitness:
                self.best_individual = population[best_idx].copy()
                self.best_fitness = fitnesses[best_idx]
                self.best_generation = generation

            # Store generation stats
            gen_best = min(fitnesses)
            gen_avg = sum(fitnesses) / len(fitnesses)
            gen_worst = max(fitnesses)
            self.generation_stats.append((generation, gen_best, gen_avg, gen_worst))

            # Optional: print progress every few generations
            if generation % 10 == 0:
                print(f"Generation {generation}: Best fitness = {self.best_fitness}")

        # Decode the best individual to get routes
        best_routes = self.decode_to_routes(self.best_individual)

        end_time = time.time()
        execution_time = end_time - start_time

        print(f"Genetic Algorithm completed in {execution_time:.2f} seconds")
        print(f"Best solution found in generation {self.best_generation} with fitness {self.best_fitness}")

        return best_routes, self.generation_stats

    def plot_progress(self):
        """
        Plot the evolution of fitness over generations.

        Returns:
            plt.Figure: The matplotlib figure
        """
        if not self.generation_stats:
            print("No data to plot. Run solve() first.")
            return None

        generations, best, avg, worst = zip(*self.generation_stats)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(generations, best, 'g-', label='Best')
        ax.plot(generations, avg, 'b-', label='Average')
        ax.plot(generations, worst, 'r-', label='Worst')

        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness (Total Distance)')
        ax.set_title('Fitness Evolution')
        ax.legend()

        plt.grid(True)
        plt.tight_layout()

        return fig