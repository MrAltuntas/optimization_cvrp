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

# Configuration for the benchmark instances
RUN_TEST = [
    # Basic instances (5)

    # A-n32-k5: 31 clients + 1 depot, 5 vehicles
    {
        "NAME": "A-n32-k5",
        "CVRP_NUMBER_OF_CITIES": 32,
        "CVRP_DEPOT_POSITION": 0,
        "CVRP_DISTANCE_BETWEEN_CITIES": 10,
        "CVRP_CAPACITY": 100,
        "GA_SETTINGS": [
            # Base configuration - based on your successful settings
            {
                "GA_POPULATION_SIZE": 10000,
                "GA_GENERATIONS": 500,
                "GA_CROSSOVER_RATE": 0.7,
                "GA_MUTATION_RATE": 0.1,
                "GA_TOURNAMENT_SIZE": 5,
                "GA_ELITISM_SIZE": 2,
                "GA_IMPROVED_EVALUATION": True,
            },
            # Testing different tournament size
            {
                "GA_POPULATION_SIZE": 10000,
                "GA_GENERATIONS": 500,
                "GA_CROSSOVER_RATE": 0.7,
                "GA_MUTATION_RATE": 0.1,
                "GA_TOURNAMENT_SIZE": 10,
                "GA_ELITISM_SIZE": 2,
                "GA_IMPROVED_EVALUATION": True,
            },
            # Testing higher mutation rate
            {
                "GA_POPULATION_SIZE": 10000,
                "GA_GENERATIONS": 500,
                "GA_CROSSOVER_RATE": 0.7,
                "GA_MUTATION_RATE": 0.2,
                "GA_ELITISM_SIZE": 2,
                "GA_TOURNAMENT_SIZE": 5,
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
                "TS_MAX_ITERATIONS": 200,
                "TS_TABU_SIZE": 20,
                "TS_NEIGHBORHOOD_SIZE": 30,
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
            # Base configuration
            {
                "GA_POPULATION_SIZE": 10000,
                "GA_GENERATIONS": 500,
                "GA_CROSSOVER_RATE": 0.7,
                "GA_MUTATION_RATE": 0.1,
                "GA_TOURNAMENT_SIZE": 5,
                "GA_ELITISM_SIZE": 2,
                "GA_IMPROVED_EVALUATION": True,
            },
            # Testing larger population
            {
                "GA_POPULATION_SIZE": 15000,
                "GA_GENERATIONS": 500,
                "GA_CROSSOVER_RATE": 0.7,
                "GA_MUTATION_RATE": 0.1,
                "GA_TOURNAMENT_SIZE": 5,
                "GA_ELITISM_SIZE": 2,
                "GA_IMPROVED_EVALUATION": True,
            },
            # Testing different crossover rate
            {
                "GA_POPULATION_SIZE": 10000,
                "GA_GENERATIONS": 500,
                "GA_CROSSOVER_RATE": 0.5,
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
            # Base configuration
            {
                "GA_POPULATION_SIZE": 10000,
                "GA_GENERATIONS": 500,
                "GA_CROSSOVER_RATE": 0.7,
                "GA_MUTATION_RATE": 0.1,
                "GA_TOURNAMENT_SIZE": 5,
                "GA_ELITISM_SIZE": 2,
                "GA_IMPROVED_EVALUATION": True,
            },
            # Testing elitism effect
            {
                "GA_POPULATION_SIZE": 10000,
                "GA_GENERATIONS": 500,
                "GA_CROSSOVER_RATE": 0.7,
                "GA_MUTATION_RATE": 0.1,
                "GA_TOURNAMENT_SIZE": 5,
                "GA_ELITISM_SIZE": 5,
                "GA_IMPROVED_EVALUATION": True,
            },
            # Testing without improved evaluation
            {
                "GA_POPULATION_SIZE": 10000,
                "GA_GENERATIONS": 500,
                "GA_CROSSOVER_RATE": 0.7,
                "GA_MUTATION_RATE": 0.1,
                "GA_TOURNAMENT_SIZE": 5,
                "GA_ELITISM_SIZE": 2,
                "GA_IMPROVED_EVALUATION": False,
            },
        ],
        "TS_SETTINGS": [
            # Base TS configuration
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
            # Base configuration
            {
                "GA_POPULATION_SIZE": 12000,
                "GA_GENERATIONS": 600,
                "GA_CROSSOVER_RATE": 0.7,
                "GA_MUTATION_RATE": 0.1,
                "GA_TOURNAMENT_SIZE": 5,
                "GA_ELITISM_SIZE": 2,
                "GA_IMPROVED_EVALUATION": True,
            },
            # Testing increased generations
            {
                "GA_POPULATION_SIZE": 12000,
                "GA_GENERATIONS": 800,
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
            # Testing combined parameter changes
            {
                "TS_MAX_ITERATIONS": 150,
                "TS_TABU_SIZE": 15,
                "TS_NEIGHBORHOOD_SIZE": 30,
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
            # Base configuration
            {
                "GA_POPULATION_SIZE": 15000,
                "GA_GENERATIONS": 600,
                "GA_CROSSOVER_RATE": 0.7,
                "GA_MUTATION_RATE": 0.1,
                "GA_TOURNAMENT_SIZE": 5,
                "GA_ELITISM_SIZE": 2,
                "GA_IMPROVED_EVALUATION": True,
            },
            # Testing combined parameter optimizations
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
            # Testing more aggressive settings
            {
                "TS_MAX_ITERATIONS": 200,
                "TS_TABU_SIZE": 20,
                "TS_NEIGHBORHOOD_SIZE": 40,
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
            # Base configuration with larger population for hard instance
            {
                "GA_POPULATION_SIZE": 20000,
                "GA_GENERATIONS": 800,
                "GA_CROSSOVER_RATE": 0.7,
                "GA_MUTATION_RATE": 0.1,
                "GA_TOURNAMENT_SIZE": 5,
                "GA_ELITISM_SIZE": 2,
                "GA_IMPROVED_EVALUATION": True,
            },
            # Testing very large population
            {
                "GA_POPULATION_SIZE": 25000,
                "GA_GENERATIONS": 800,
                "GA_CROSSOVER_RATE": 0.7,
                "GA_MUTATION_RATE": 0.1,
                "GA_TOURNAMENT_SIZE": 5,
                "GA_ELITISM_SIZE": 2,
                "GA_IMPROVED_EVALUATION": True,
            },
        ],
        "TS_SETTINGS": [
            # Base TS configuration for hard instance
            {
                "TS_MAX_ITERATIONS": 200,
                "TS_TABU_SIZE": 15,
                "TS_NEIGHBORHOOD_SIZE": 30,
            },
            # Testing more intensive search
            {
                "TS_MAX_ITERATIONS": 300,
                "TS_TABU_SIZE": 20,
                "TS_NEIGHBORHOOD_SIZE": 50,
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
            # Base configuration with larger population for hard instance
            {
                "GA_POPULATION_SIZE": 25000,
                "GA_GENERATIONS": 1000,
                "GA_CROSSOVER_RATE": 0.7,
                "GA_MUTATION_RATE": 0.1,
                "GA_TOURNAMENT_SIZE": 5,
                "GA_ELITISM_SIZE": 2,
                "GA_IMPROVED_EVALUATION": True,
            },
            # Testing different parameter balance
            {
                "GA_POPULATION_SIZE": 20000,
                "GA_GENERATIONS": 1200,
                "GA_CROSSOVER_RATE": 0.75,
                "GA_MUTATION_RATE": 0.12,
                "GA_TOURNAMENT_SIZE": 6,
                "GA_ELITISM_SIZE": 3,
                "GA_IMPROVED_EVALUATION": True,
            },
        ],
        "TS_SETTINGS": [
            # Base TS configuration for hard instance
            {
                "TS_MAX_ITERATIONS": 250,
                "TS_TABU_SIZE": 20,
                "TS_NEIGHBORHOOD_SIZE": 40,
            },
            # Intensive search for hard instance
            {
                "TS_MAX_ITERATIONS": 400,
                "TS_TABU_SIZE": 25,
                "TS_NEIGHBORHOOD_SIZE": 60,
            },
        ]
    }
]
