# CVRP Settings
CVRP_NUMBER_OF_CITIES = 32  # Number of cities in the problem (including depot)
CVRP_DEPOT_POSITION = 1  # Position of depot in the data file (usually 1)
CVRP_CAPACITY = 100  # Vehicle capacity

# Genetic Algorithm Settings
GA_POPULATION_SIZE = 100
GA_GENERATIONS = 100
GA_CROSSOVER_RATE = 0.7
GA_MUTATION_RATE = 0.1
GA_TOURNAMENT_SIZE = 5
GA_ELITISM_SIZE = 2

# Tabu Search Settings
TS_MAX_ITERATIONS = 100
TS_TABU_SIZE = 10
TS_NEIGHBORHOOD_SIZE = 20

# Run 10 times as per the exercise requirements
RUN_TEST_COUNT = 10

# Instance file paths
INSTANCE_FILES = [
    "instances/A-n32-k5.vrp",
    "instances/A-n37-k6.vrp",
    "instances/A-n39-k5.vrp",
    "instances/A-n45-k6.vrp",
    "instances/A-n48-k7.vrp",
    "instances/A-n54-k7.vrp",
    "instances/A-n60-k9.vrp"
]

# Test configurations for studying parameter effects
PARAMETER_TESTS = [
    # Testing mutation rate effect
    {
        "name": "Mutation Rate",
        "values": [0.01, 0.05, 0.1, 0.2, 0.3],
        "parameter": "GA_MUTATION_RATE",
        "default_params": {
            "GA_POPULATION_SIZE": 100,
            "GA_GENERATIONS": 100,
            "GA_CROSSOVER_RATE": 0.7,
            "GA_TOURNAMENT_SIZE": 5,
            "GA_ELITISM_SIZE": 2
        }
    },
    # Testing crossover rate effect
    {
        "name": "Crossover Rate",
        "values": [0.3, 0.5, 0.7, 0.9],
        "parameter": "GA_CROSSOVER_RATE",
        "default_params": {
            "GA_POPULATION_SIZE": 100,
            "GA_GENERATIONS": 100,
            "GA_MUTATION_RATE": 0.1,
            "GA_TOURNAMENT_SIZE": 5,
            "GA_ELITISM_SIZE": 2
        }
    },
    # Testing tournament size effect
    {
        "name": "Tournament Size",
        "values": [1, 3, 5, 10, 20],
        "parameter": "GA_TOURNAMENT_SIZE",
        "default_params": {
            "GA_POPULATION_SIZE": 100,
            "GA_GENERATIONS": 100,
            "GA_CROSSOVER_RATE": 0.7,
            "GA_MUTATION_RATE": 0.1,
            "GA_ELITISM_SIZE": 2
        }
    },
    # Testing elitism size effect
    {
        "name": "Elitism Size",
        "values": [0, 1, 2, 5, 10],
        "parameter": "GA_ELITISM_SIZE",
        "default_params": {
            "GA_POPULATION_SIZE": 100,
            "GA_GENERATIONS": 100,
            "GA_CROSSOVER_RATE": 0.7,
            "GA_MUTATION_RATE": 0.1,
            "GA_TOURNAMENT_SIZE": 5
        }
    },
    # Testing population size effect
    {
        "name": "Population Size",
        "values": [50, 100, 200, 500],
        "parameter": "GA_POPULATION_SIZE",
        "default_params": {
            "GA_GENERATIONS": 100,
            "GA_CROSSOVER_RATE": 0.7,
            "GA_MUTATION_RATE": 0.1,
            "GA_TOURNAMENT_SIZE": 5,
            "GA_ELITISM_SIZE": 2
        }
    },
    # Testing generations effect
    {
        "name": "Generations",
        "values": [50, 100, 200, 500],
        "parameter": "GA_GENERATIONS",
        "default_params": {
            "GA_POPULATION_SIZE": 100,
            "GA_CROSSOVER_RATE": 0.7,
            "GA_MUTATION_RATE": 0.1,
            "GA_TOURNAMENT_SIZE": 5,
            "GA_ELITISM_SIZE": 2
        }
    }
]

# Tabu Search parameter tests
TS_PARAMETER_TESTS = [
    # Testing tabu size effect
    {
        "name": "Tabu Size",
        "values": [5, 10, 15, 20],
        "parameter": "TS_TABU_SIZE",
        "default_params": {
            "TS_MAX_ITERATIONS": 100,
            "TS_NEIGHBORHOOD_SIZE": 20
        }
    },
    # Testing neighborhood size effect
    {
        "name": "Neighborhood Size",
        "values": [10, 20, 30, 40],
        "parameter": "TS_NEIGHBORHOOD_SIZE",
        "default_params": {
            "TS_MAX_ITERATIONS": 100,
            "TS_TABU_SIZE": 10
        }
    },
    # Testing max iterations effect
    {
        "name": "Max Iterations",
        "values": [50, 100, 200, 300],
        "parameter": "TS_MAX_ITERATIONS",
        "default_params": {
            "TS_TABU_SIZE": 10,
            "TS_NEIGHBORHOOD_SIZE": 20
        }
    }
]