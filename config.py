# CVRP Settings
CVRP_NUMBER_OF_CITIES = 5
CVRP_DEPOT_POSITION = 0 # do not change it i didn't checked how it's effects
CVRP_DISTANCE_BETWEEN_CITIES = 5
CVRP_CAPACITY = 10


# Genetic Algorithm Settings
GA_POPULATION_SIZE = 1000
GA_GENERATIONS = 50
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

# Configuration for the benchmark instances
RUN_TEST = [
    # Basic instances (5)

    # A-n32-k5: 31 clients + 1 depot, 5 vehicles
    {
        "NAME": "A-n32-k5",
        "CVRP_NUMBER_OF_CITIES": 32,
        "CVRP_DEPOT_POSITION": 0,
        "CVRP_DISTANCE_BETWEEN_CITIES": 10,  # Adjusted for larger problem
        "CVRP_CAPACITY": 100,  # Standard capacity in benchmarks
        "GA_SETTINGS": [
            # Standard parameter configuration
            {
                "GA_POPULATION_SIZE": 100,
                "GA_GENERATIONS": 100,
                "GA_CROSSOVER_RATE": 0.7,
                "GA_MUTATION_RATE": 0.1,
                "GA_TOURNAMENT_SIZE": 5,
                "GA_ELITISM_SIZE": 2,
                "GA_IMPROVED_EVALUATION": True,
            },
            # Testing increased population size
            {
                "GA_POPULATION_SIZE": 200,
                "GA_GENERATIONS": 100,
                "GA_CROSSOVER_RATE": 0.7,
                "GA_MUTATION_RATE": 0.1,
                "GA_TOURNAMENT_SIZE": 5,
                "GA_ELITISM_SIZE": 2,
                "GA_IMPROVED_EVALUATION": True,
            },
            # Testing increased mutation rate
            {
                "GA_POPULATION_SIZE": 100,
                "GA_GENERATIONS": 100,
                "GA_CROSSOVER_RATE": 0.7,
                "GA_MUTATION_RATE": 0.2,
                "GA_TOURNAMENT_SIZE": 5,
                "GA_ELITISM_SIZE": 2,
                "GA_IMPROVED_EVALUATION": True,
            },
        ],
        "TS_SETTINGS": [
            # Standard TS configuration
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

    # A-n37-k6: 36 clients + 1 depot, 6 vehicles
    {
        "NAME": "A-n37-k6",
        "CVRP_NUMBER_OF_CITIES": 37,
        "CVRP_DEPOT_POSITION": 0,
        "CVRP_DISTANCE_BETWEEN_CITIES": 10,
        "CVRP_CAPACITY": 100,
        "GA_SETTINGS": [
            # Standard parameter configuration
            {
                "GA_POPULATION_SIZE": 100,
                "GA_GENERATIONS": 100,
                "GA_CROSSOVER_RATE": 0.7,
                "GA_MUTATION_RATE": 0.1,
                "GA_TOURNAMENT_SIZE": 5,
                "GA_ELITISM_SIZE": 2,
                "GA_IMPROVED_EVALUATION": True,
            },
            # Testing larger tournament size (selection pressure)
            {
                "GA_POPULATION_SIZE": 100,
                "GA_GENERATIONS": 100,
                "GA_CROSSOVER_RATE": 0.7,
                "GA_MUTATION_RATE": 0.1,
                "GA_TOURNAMENT_SIZE": 10,
                "GA_ELITISM_SIZE": 2,
                "GA_IMPROVED_EVALUATION": True,
            },
        ],
        "TS_SETTINGS": [
            # Standard TS configuration
            {
                "TS_MAX_ITERATIONS": 100,
                "TS_TABU_SIZE": 10,
                "TS_NEIGHBORHOOD_SIZE": 20,
            },
            # Testing larger neighborhood
            {
                "TS_MAX_ITERATIONS": 100,
                "TS_TABU_SIZE": 10,
                "TS_NEIGHBORHOOD_SIZE": 40,
            },
        ]
    },

    # A-n39-k5: 38 clients + 1 depot, 5 vehicles
    {
        "NAME": "A-n39-k5",
        "CVRP_NUMBER_OF_CITIES": 39,
        "CVRP_DEPOT_POSITION": 0,
        "CVRP_DISTANCE_BETWEEN_CITIES": 10,
        "CVRP_CAPACITY": 100,
        "GA_SETTINGS": [
            # Standard parameter configuration
            {
                "GA_POPULATION_SIZE": 100,
                "GA_GENERATIONS": 100,
                "GA_CROSSOVER_RATE": 0.7,
                "GA_MUTATION_RATE": 0.1,
                "GA_TOURNAMENT_SIZE": 5,
                "GA_ELITISM_SIZE": 2,
                "GA_IMPROVED_EVALUATION": True,
            },
            # Testing different crossover rate
            {
                "GA_POPULATION_SIZE": 100,
                "GA_GENERATIONS": 100,
                "GA_CROSSOVER_RATE": 0.5,
                "GA_MUTATION_RATE": 0.1,
                "GA_TOURNAMENT_SIZE": 5,
                "GA_ELITISM_SIZE": 2,
                "GA_IMPROVED_EVALUATION": True,
            },
        ],
        "TS_SETTINGS": [
            # Standard TS configuration
            {
                "TS_MAX_ITERATIONS": 100,
                "TS_TABU_SIZE": 10,
                "TS_NEIGHBORHOOD_SIZE": 20,
            },
            # Testing more iterations
            {
                "TS_MAX_ITERATIONS": 200,
                "TS_TABU_SIZE": 10,
                "TS_NEIGHBORHOOD_SIZE": 20,
            },
        ]
    },

    # A-n45-k6: 44 clients + 1 depot, 6 vehicles
    {
        "NAME": "A-n45-k6",
        "CVRP_NUMBER_OF_CITIES": 45,
        "CVRP_DEPOT_POSITION": 0,
        "CVRP_DISTANCE_BETWEEN_CITIES": 10,
        "CVRP_CAPACITY": 100,
        "GA_SETTINGS": [
            # Standard parameter configuration
            {
                "GA_POPULATION_SIZE": 100,
                "GA_GENERATIONS": 100,
                "GA_CROSSOVER_RATE": 0.7,
                "GA_MUTATION_RATE": 0.1,
                "GA_TOURNAMENT_SIZE": 5,
                "GA_ELITISM_SIZE": 2,
                "GA_IMPROVED_EVALUATION": True,
            },
            # Testing more generations
            {
                "GA_POPULATION_SIZE": 100,
                "GA_GENERATIONS": 200,
                "GA_CROSSOVER_RATE": 0.7,
                "GA_MUTATION_RATE": 0.1,
                "GA_TOURNAMENT_SIZE": 5,
                "GA_ELITISM_SIZE": 2,
                "GA_IMPROVED_EVALUATION": True,
            },
        ],
        "TS_SETTINGS": [
            # Standard TS configuration
            {
                "TS_MAX_ITERATIONS": 100,
                "TS_TABU_SIZE": 10,
                "TS_NEIGHBORHOOD_SIZE": 20,
            },
        ]
    },

    # A-n48-k7: 47 clients + 1 depot, 7 vehicles
    {
        "NAME": "A-n48-k7",
        "CVRP_NUMBER_OF_CITIES": 48,
        "CVRP_DEPOT_POSITION": 0,
        "CVRP_DISTANCE_BETWEEN_CITIES": 10,
        "CVRP_CAPACITY": 100,
        "GA_SETTINGS": [
            # Standard parameter configuration
            {
                "GA_POPULATION_SIZE": 100,
                "GA_GENERATIONS": 100,
                "GA_CROSSOVER_RATE": 0.7,
                "GA_MUTATION_RATE": 0.1,
                "GA_TOURNAMENT_SIZE": 5,
                "GA_ELITISM_SIZE": 2,
                "GA_IMPROVED_EVALUATION": True,
            },
            # Testing different elitism size
            {
                "GA_POPULATION_SIZE": 100,
                "GA_GENERATIONS": 100,
                "GA_CROSSOVER_RATE": 0.7,
                "GA_MUTATION_RATE": 0.1,
                "GA_TOURNAMENT_SIZE": 5,
                "GA_ELITISM_SIZE": 5,
                "GA_IMPROVED_EVALUATION": True,
            },
        ],
        "TS_SETTINGS": [
            # Standard TS configuration
            {
                "TS_MAX_ITERATIONS": 100,
                "TS_TABU_SIZE": 10,
                "TS_NEIGHBORHOOD_SIZE": 20,
            },
        ]
    },

    # Hard instances (2)

    # A-n54-k7: 53 clients + 1 depot, 7 vehicles
    {
        "NAME": "A-n54-k7",
        "CVRP_NUMBER_OF_CITIES": 54,
        "CVRP_DEPOT_POSITION": 0,
        "CVRP_DISTANCE_BETWEEN_CITIES": 10,
        "CVRP_CAPACITY": 100,
        "GA_SETTINGS": [
            # Standard parameter configuration
            {
                "GA_POPULATION_SIZE": 100,
                "GA_GENERATIONS": 100,
                "GA_CROSSOVER_RATE": 0.7,
                "GA_MUTATION_RATE": 0.1,
                "GA_TOURNAMENT_SIZE": 5,
                "GA_ELITISM_SIZE": 2,
                "GA_IMPROVED_EVALUATION": True,
            },
            # Testing evaluation method
            {
                "GA_POPULATION_SIZE": 100,
                "GA_GENERATIONS": 100,
                "GA_CROSSOVER_RATE": 0.7,
                "GA_MUTATION_RATE": 0.1,
                "GA_TOURNAMENT_SIZE": 5,
                "GA_ELITISM_SIZE": 2,
                "GA_IMPROVED_EVALUATION": False,  # Testing without improved evaluation
            },
        ],
        "TS_SETTINGS": [
            # Standard TS configuration
            {
                "TS_MAX_ITERATIONS": 100,
                "TS_TABU_SIZE": 10,
                "TS_NEIGHBORHOOD_SIZE": 20,
            },
        ]
    },

    # A-n60-k9: 59 clients + 1 depot, 9 vehicles
    {
        "NAME": "A-n60-k9",
        "CVRP_NUMBER_OF_CITIES": 60,
        "CVRP_DEPOT_POSITION": 0,
        "CVRP_DISTANCE_BETWEEN_CITIES": 10,
        "CVRP_CAPACITY": 100,
        "GA_SETTINGS": [
            # Standard parameter configuration
            {
                "GA_POPULATION_SIZE": 100,
                "GA_GENERATIONS": 100,
                "GA_CROSSOVER_RATE": 0.7,
                "GA_MUTATION_RATE": 0.1,
                "GA_TOURNAMENT_SIZE": 5,
                "GA_ELITISM_SIZE": 2,
                "GA_IMPROVED_EVALUATION": True,
            },
            # Testing larger population and more generations
            {
                "GA_POPULATION_SIZE": 200,
                "GA_GENERATIONS": 200,
                "GA_CROSSOVER_RATE": 0.7,
                "GA_MUTATION_RATE": 0.1,
                "GA_TOURNAMENT_SIZE": 5,
                "GA_ELITISM_SIZE": 2,
                "GA_IMPROVED_EVALUATION": True,
            },
        ],
        "TS_SETTINGS": [
            # Standard TS configuration
            {
                "TS_MAX_ITERATIONS": 100,
                "TS_TABU_SIZE": 10,
                "TS_NEIGHBORHOOD_SIZE": 20,
            },
            # Testing comprehensive parameter changes
            {
                "TS_MAX_ITERATIONS": 200,
                "TS_TABU_SIZE": 20,
                "TS_NEIGHBORHOOD_SIZE": 40,
            },
        ]
    }
]
