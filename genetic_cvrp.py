import random
import cvrp as CVRP

class GeneticCVRP:
    def __init__(self, cvrp: CVRP.CVRP, population_size=100, generations=100,
                 crossover_rate=0.7, mutation_rate=0.1, tournament_size=5,
                 elitism_count=2, improved_evaluation=False):
        self.problem = cvrp
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elitism_count = elitism_count
        self.best_solution = None
        self.best_fitness = float('inf')
        self.improved_evaluation = improved_evaluation

    def create_individual(self):
        """
        Creates a random individual that can include the depot (0) multiple times.
        Cities may appear multiple times in this representation.
        Maximum length is twice the number of cities.
        """
        # Determine a random length between number of cities and twice the number
        num_cities = len(self.problem.all_cities)
        max_length = 2 * num_cities
        length = random.randint(num_cities, max_length)

        # Include all cities at least once to ensure a valid solution
        cities = list(self.problem.all_cities)
        individual = []

        # Add random cities (including depot) to fill up to desired length
        all_possible_cities = list(self.problem.all_cities) + [self.problem.depot]
        for _ in range(length):
            individual.append(random.choice(all_possible_cities))

        return individual

    def calculate_fitness_improved(self, individual):
        """
        Evaluates fitness based on:
        1. Whether all cities are visited
        2. Total route length
        3. Respect for capacity constraints
        """

        # Check if all cities are visited
        visited = set(individual) - {self.problem.depot}
        if visited != self.problem.all_cities:
            # For missing cities, we'll still use a high penalty but based on how many cities are missing
            missing_cities = len(self.problem.all_cities) - len(visited)
            return 10000 + (missing_cities * 1000)  # High penalty scaled by missing cities

        # Simulate the journey to validate capacity constraints
        path = individual.copy()

        # Validate the path
        movements, is_valid = self.problem.is_valid_path(path)

        if not is_valid:
            # Instead of infinity, return a high but finite value that rewards progress
            # The more movements completed, the better (lower) the fitness value
            return 5000 - movements  # This creates a gradient for invalid solutions

        return movements  # Return total movements as fitness for valid solutions

    def calculate_fitness(self, individual):
        """
        Evaluates fitness based on:
        1. Whether all cities are visited
        2. Total route length
        3. Respect for capacity constraints
        """

        # Check if all cities are visited
        visited = set(individual) - {self.problem.depot}
        if visited != self.problem.all_cities:
            return float('inf')  # Invalid solution

        # Simulate the journey to validate capacity constraints
        path = individual.copy()

        # Validate the path
        movements, is_valid = self.problem.is_valid_path(path)

        if not is_valid:
            return float('inf')

        return movements  # Return total movements as fitness

    def tournament_selection(self, population):
        """
        Selects an individual using tournament selection.
        """
        tournament = random.sample(population, self.tournament_size)

        if self.improved_evaluation:
            return min(tournament, key=lambda x: self.calculate_fitness_improved(x))
        else:
            return min(tournament, key=lambda x: self.calculate_fitness(x))



    def crossover(self, parent1, parent2):
        """
        Modified crossover that works with the new representation
        where cities and depot can appear multiple times.

        here is a big problem when we combine two parent they usually tend to have the same cities (cousine marrigment :D)
        """
        if random.random() > self.crossover_rate:
            return parent1.copy()

        # Choose random crossover points
        point1 = random.randint(0, len(parent1))
        point2 = random.randint(0, len(parent2))

        # Create child by combining segments
        child = parent1[:point1] + parent2[point2:]

        # Ensure we don't exceed maximum length
        max_length = 2 * len(self.problem.all_cities)
        if len(child) > max_length:
            child = child[:max_length]

        return child

    def mutate(self, individual):
        """
        Modified mutation that works with the new representation.
        Can change, add, or remove cities.
        """
        if random.random() > self.mutation_rate:
            return individual

        # Copy to avoid modifying original
        individual = individual.copy()

        # Choose mutation type
        mutation_type = random.choice(['change', 'add', 'remove'])

        if mutation_type == 'change' and individual:
            # Change a random position to a random city or depot
            pos = random.randint(0, len(individual) - 1)
            all_cities = list(self.problem.all_cities) + [self.problem.depot]
            individual[pos] = random.choice(all_cities)

        elif mutation_type == 'add' and len(individual) < 2 * len(self.problem.all_cities):
            # Add a random city or depot
            pos = random.randint(0, len(individual))
            all_cities = list(self.problem.all_cities) + [self.problem.depot]
            individual.insert(pos, random.choice(all_cities))

        elif mutation_type == 'remove' and len(individual) > 1:
            # Remove a random city, ensuring we don't remove the last instance of a city
            removable_indices = []
            for i, city in enumerate(individual):
                # Can remove if it's depot or if the city appears more than once
                if city == self.problem.depot or individual.count(city) > 1:
                    removable_indices.append(i)

            if removable_indices:
                pos = random.choice(removable_indices)
                individual.pop(pos)

        return individual

    def evolve(self):
        """
        Runs the genetic algorithm and returns the best solution found.
        """
        # Create initial population
        population = [self.create_individual() for _ in range(self.population_size)]

        # Evolution loop
        for generation in range(self.generations):
            # Evaluate and sort population
            if self.improved_evaluation:
                population.sort(key=lambda x: self.calculate_fitness_improved(x))
            else:
                population.sort(key=lambda x: self.calculate_fitness(x))



            # Update best solution if better one found
            current_best = population[0]
            if self.improved_evaluation:
                current_best_fitness = self.calculate_fitness_improved(current_best)
            else:
                current_best_fitness = self.calculate_fitness(current_best)



            if current_best_fitness < self.best_fitness:
                self.best_solution = current_best.copy()
                self.best_fitness = current_best_fitness

            # Create new population
            new_population = []

            # Elitism - copy best individuals directly
            new_population.extend(population[:self.elitism_count])

            # Create rest of population through selection, crossover, mutation
            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection(population)
                parent2 = self.tournament_selection(population)

                offspring = self.crossover(parent1, parent2)
                mutated_offspring = self.mutate(offspring)

                new_population.append(parent1)
                new_population.append(parent2)
                new_population.append(offspring)
                new_population.append(mutated_offspring)

            population = new_population

            # Log progress periodically
            if generation % 10 == 0:
                print(f"Generation {generation}: Best fitness = {self.best_fitness}")

        if self.best_solution is None:
            print("Warning: No valid solution found")
            return None

        # Prepare final solution with depot added if needed
        solution = self.best_solution.copy()
        return solution

    def solve(self):
        """
        Solves the CVRP problem and returns the best path found.
        """
        path = self.evolve()
        return path