# CVRP Settings
CVRP_NUMBER_OF_CITIES = 32
CVRP_DEPOT_POSITION = 0 # do not change it i didn't checked how it's effects
CVRP_DISTANCE_BETWEEN_CITIES = 10
CVRP_CAPACITY = 100


# Genetic Algorithm Settings
GA_POPULATION_SIZE = 10000
GA_GENERATIONS = 300
GA_CROSSOVER_RATE = 0.7
GA_MUTATION_RATE = 0.1
GA_TOURNAMENT_SIZE = 5
GA_ELITISM_SIZE = 2
# basicaly it's about evaluation func(fitness func) if it's false when we can't find a valid path we are returning infinite penalty but if it's true we will return a penatly according to how many city able to visit
GA_IMPROVED_EVALUATION = True

# Tabu Search Settings
TS_MAX_ITERATIONS = 100
TS_TABU_SIZE = 10
TS_NEIGHBORHOOD_SIZE = 20


# Run 10 times as per the exercise requirements
RUN_TEST_COUNT = 10

# Simplified configuration focusing on GA parameter effects
RUN_TEST = [
    # Test case for A-n32-k5 with varying GA parameters
    {
        "NAME": "A-n32-k5",
        "CVRP_NUMBER_OF_CITIES": 32,
        "CVRP_DEPOT_POSITION": 0,
        "CVRP_DISTANCE_BETWEEN_CITIES": 10,
        "CVRP_CAPACITY": 100,
        "GA_SETTINGS": [
            # Base configuration - proven to work
            {
                "GA_POPULATION_SIZE": 10000,
                "GA_GENERATIONS": 500,
                "GA_CROSSOVER_RATE": 0.7,
                "GA_MUTATION_RATE": 0.1,
                "GA_TOURNAMENT_SIZE": 5,
                "GA_ELITISM_SIZE": 2,
                "GA_IMPROVED_EVALUATION": True,
            },
            # Testing mutation rate effect - low
            {
                "GA_POPULATION_SIZE": 10000,
                "GA_GENERATIONS": 500,
                "GA_CROSSOVER_RATE": 0.7,
                "GA_MUTATION_RATE": 0.01,
                "GA_TOURNAMENT_SIZE": 5,
                "GA_ELITISM_SIZE": 2,
                "GA_IMPROVED_EVALUATION": True,
            },
            # Testing mutation rate effect - high
            {
                "GA_POPULATION_SIZE": 10000,
                "GA_GENERATIONS": 500,
                "GA_CROSSOVER_RATE": 0.7,
                "GA_MUTATION_RATE": 0.3,
                "GA_TOURNAMENT_SIZE": 5,
                "GA_ELITISM_SIZE": 2,
                "GA_IMPROVED_EVALUATION": True,
            },
            # Testing tournament size effect - small (low selection pressure)
            {
                "GA_POPULATION_SIZE": 10000,
                "GA_GENERATIONS": 500,
                "GA_CROSSOVER_RATE": 0.7,
                "GA_MUTATION_RATE": 0.1,
                "GA_TOURNAMENT_SIZE": 1,
                "GA_ELITISM_SIZE": 2,
                "GA_IMPROVED_EVALUATION": True,
            },
            # Testing tournament size effect - large (high selection pressure)
            {
                "GA_POPULATION_SIZE": 10000,
                "GA_GENERATIONS": 500,
                "GA_CROSSOVER_RATE": 0.7,
                "GA_MUTATION_RATE": 0.1,
                "GA_TOURNAMENT_SIZE": 20,
                "GA_ELITISM_SIZE": 2,
                "GA_IMPROVED_EVALUATION": True,
            },
            # Testing crossover rate effect - low
            {
                "GA_POPULATION_SIZE": 10000,
                "GA_GENERATIONS": 500,
                "GA_CROSSOVER_RATE": 0.3,
                "GA_MUTATION_RATE": 0.1,
                "GA_TOURNAMENT_SIZE": 5,
                "GA_ELITISM_SIZE": 2,
                "GA_IMPROVED_EVALUATION": True,
            },
            # Testing crossover rate effect - high
            {
                "GA_POPULATION_SIZE": 10000,
                "GA_GENERATIONS": 500,
                "GA_CROSSOVER_RATE": 0.9,
                "GA_MUTATION_RATE": 0.1,
                "GA_TOURNAMENT_SIZE": 5,
                "GA_ELITISM_SIZE": 2,
                "GA_IMPROVED_EVALUATION": True,
            },
            # Testing elitism effect - none
            {
                "GA_POPULATION_SIZE": 10000,
                "GA_GENERATIONS": 500,
                "GA_CROSSOVER_RATE": 0.7,
                "GA_MUTATION_RATE": 0.1,
                "GA_TOURNAMENT_SIZE": 5,
                "GA_ELITISM_SIZE": 0,
                "GA_IMPROVED_EVALUATION": True,
            },
            # Testing elitism effect - high
            {
                "GA_POPULATION_SIZE": 10000,
                "GA_GENERATIONS": 500,
                "GA_CROSSOVER_RATE": 0.7,
                "GA_MUTATION_RATE": 0.1,
                "GA_TOURNAMENT_SIZE": 5,
                "GA_ELITISM_SIZE": 10,
                "GA_IMPROVED_EVALUATION": True,
            },
            # Testing improved evaluation vs standard evaluation
            {
                "GA_POPULATION_SIZE": 10000,
                "GA_GENERATIONS": 500,
                "GA_CROSSOVER_RATE": 0.7,
                "GA_MUTATION_RATE": 0.1,
                "GA_TOURNAMENT_SIZE": 5,
                "GA_ELITISM_SIZE": 2,
                "GA_IMPROVED_EVALUATION": False,
            },
            # Testing population size effect
            {
                "GA_POPULATION_SIZE": 5000,
                "GA_GENERATIONS": 500,
                "GA_CROSSOVER_RATE": 0.7,
                "GA_MUTATION_RATE": 0.1,
                "GA_TOURNAMENT_SIZE": 5,
                "GA_ELITISM_SIZE": 2,
                "GA_IMPROVED_EVALUATION": True,
            },
            # Testing generation count effect
            {
                "GA_POPULATION_SIZE": 10000,
                "GA_GENERATIONS": 250,
                "GA_CROSSOVER_RATE": 0.7,
                "GA_MUTATION_RATE": 0.1,
                "GA_TOURNAMENT_SIZE": 5,
                "GA_ELITISM_SIZE": 2,
                "GA_IMPROVED_EVALUATION": True,
            },
        ],
        "TS_SETTINGS": [
            # Base TS configuration
            {
                "TS_MAX_ITERATIONS": 100,
                "TS_TABU_SIZE": 10,
                "TS_NEIGHBORHOOD_SIZE": 20,
            },
            # Testing larger tabu size
            {
                "TS_MAX_ITERATIONS": 100,
                "TS_TABU_SIZE": 20,
                "TS_NEIGHBORHOOD_SIZE": 20,
            },
        ]
    },

    # One larger instance to test scalability
    {
        "NAME": "A-n48-k7",
        "CVRP_NUMBER_OF_CITIES": 48,
        "CVRP_DEPOT_POSITION": 0,
        "CVRP_DISTANCE_BETWEEN_CITIES": 10,
        "CVRP_CAPACITY": 100,
        "GA_SETTINGS": [
            # Base configuration with larger population for larger instance
            {
                "GA_POPULATION_SIZE": 15000,
                "GA_GENERATIONS": 600,
                "GA_CROSSOVER_RATE": 0.7,
                "GA_MUTATION_RATE": 0.1,
                "GA_TOURNAMENT_SIZE": 5,
                "GA_ELITISM_SIZE": 2,
                "GA_IMPROVED_EVALUATION": True,
            },
            # Testing optimal combined parameters
            {
                "GA_POPULATION_SIZE": 15000,
                "GA_GENERATIONS": 600,
                "GA_CROSSOVER_RATE": 0.8,
                "GA_MUTATION_RATE": 0.15,
                "GA_TOURNAMENT_SIZE": 7,
                "GA_ELITISM_SIZE": 3,
                "GA_IMPROVED_EVALUATION": True,
            },
        ],
        "TS_SETTINGS": [
            # Base TS configuration
            {
                "TS_MAX_ITERATIONS": 150,
                "TS_TABU_SIZE": 15,
                "TS_NEIGHBORHOOD_SIZE": 25,
            },
        ]
    }
]
