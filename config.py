# CVRP Settings
CVRP_NUMBER_OF_CITIES = 5
CVRP_DEPOT_POSITION = 0
CVRP_DISTANCE_BETWEEN_CITIES = 5
CVRP_CAPACITY = 10


# Genetic Algorithm Settings
GA_POPULATION_SIZE = 1000
GA_GENERATIONS = 50
GA_CROSSOVER_RATE = 0.7
GA_MUTATION_RATE = 0.1
GA_TOURNAMENT_SIZE = 5
GA_ELITISM_SIZE = 2

RUN_TEST = [
    {
        "CVRP_NUMBER_OF_CITIES": 5,
        "CVRP_DEPOT_POSITION": 0,
        "CVRP_DISTANCE_BETWEEN_CITIES": 5,
        "CVRP_CAPACITY": 10,
        "GA_SETTINGS": [
            {
                "GA_POPULATION_SIZE": 1000,
                "GA_GENERATIONS": 50,
                "GA_CROSSOVER_RATE": 0.7,
                "GA_MUTATION_RATE": 0.1,
                "GA_TOURNAMENT_SIZE": 5,
                "GA_ELITISM_SIZE": 2
            },
            {
                "GA_POPULATION_SIZE": 2000,
                "GA_GENERATIONS": 70,
                "GA_CROSSOVER_RATE": 0.5,
                "GA_MUTATION_RATE": 0.2,
                "GA_TOURNAMENT_SIZE": 10,
                "GA_ELITISM_SIZE": 3
            },
            {
                "GA_POPULATION_SIZE": 2000,
                "GA_GENERATIONS": 70,
                "GA_CROSSOVER_RATE": 0.3,
                "GA_MUTATION_RATE": 0.3,
                "GA_TOURNAMENT_SIZE": 7,
                "GA_ELITISM_SIZE": 1
            },
        ]
    },
    {
        "CVRP_NUMBER_OF_CITIES": 10,
        "CVRP_DEPOT_POSITION": 0,
        "CVRP_DISTANCE_BETWEEN_CITIES": 7,
        "CVRP_CAPACITY": 30,
        "GA_SETTINGS": [
            {
                "GA_POPULATION_SIZE": 10000,
                "GA_GENERATIONS": 50,
                "GA_CROSSOVER_RATE": 0.7,
                "GA_MUTATION_RATE": 0.1,
                "GA_TOURNAMENT_SIZE": 5,
                "GA_ELITISM_SIZE": 2
            },
            {
                "GA_POPULATION_SIZE": 20000,
                "GA_GENERATIONS": 70,
                "GA_CROSSOVER_RATE": 0.5,
                "GA_MUTATION_RATE": 0.2,
                "GA_TOURNAMENT_SIZE": 10,
                "GA_ELITISM_SIZE": 3
            },
            {
                "GA_POPULATION_SIZE": 20000,
                "GA_GENERATIONS": 70,
                "GA_CROSSOVER_RATE": 0.3,
                "GA_MUTATION_RATE": 0.3,
                "GA_TOURNAMENT_SIZE": 7,
                "GA_ELITISM_SIZE": 1
            },
        ]
    }
]

