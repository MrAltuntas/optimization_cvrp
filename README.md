# CVRP Solver

## Overview

This repository implements and compares different optimization algorithms for solving the Capacitated Vehicle Routing Problem (CVRP):

- Genetic Algorithm (GA)
- Tabu Search (TS)
- Greedy Algorithm
- Random Search Algorithm

## Getting Started

### Clone the repository

```bash
git clone https://github.com/MrAltuntas/optimization_cvrp.git
cd optimization_cvrp
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Requirements

- Python 3.6+
- NumPy
- Pandas

## How to Use

The project is configured through the `config.py` file, which controls both problem parameters and algorithm settings.

### Basic Usage

1. Run a simple test with default settings:

```bash
python main.py
```

2. Run the test suite to compare multiple configurations:

```bash
python test.py
```

## Configuration Guide

The `config.py` file is the central control point for all settings. Here's how to use it:

### CVRP Problem Settings

```python
# Basic CVRP settings
CVRP_NUMBER_OF_CITIES = 32        # Number of cities to visit
CVRP_DEPOT_POSITION = 0           # Depot position (keep at 0)
CVRP_DISTANCE_BETWEEN_CITIES = 10 # Distance between cities
CVRP_CAPACITY = 100               # Vehicle capacity
```

### Algorithm Settings

```python
# Genetic Algorithm settings
GA_POPULATION_SIZE = 10000        # Size of population
GA_GENERATIONS = 300              # Number of generations to run
GA_CROSSOVER_RATE = 0.7           # Probability of crossover
GA_MUTATION_RATE = 0.1            # Probability of mutation
GA_TOURNAMENT_SIZE = 5            # Tournament selection size
GA_ELITISM_SIZE = 2               # Number of elite solutions preserved
GA_IMPROVED_EVALUATION = True     # Use improved fitness function

# Tabu Search settings
TS_MAX_ITERATIONS = 100           # Maximum iterations
TS_TABU_SIZE = 10                 # Size of tabu list
TS_NEIGHBORHOOD_SIZE = 20         # Neighborhood size for move generation
```

### Test Configuration

For comparing multiple algorithm configurations, use the `RUN_TEST` array:

```python
RUN_TEST_COUNT = 10  # Number of runs per configuration

RUN_TEST = [
   {
      "NAME": "A-n32-k5",
      "CVRP_NUMBER_OF_CITIES": 32,
      "CVRP_DEPOT_POSITION": 0,
      "CVRP_DISTANCE_BETWEEN_CITIES": 10,
      "CVRP_CAPACITY": 100,
      "GA_SETTINGS": [
         # First GA configuration
         {
            "GA_POPULATION_SIZE": 10000,
            "GA_GENERATIONS": 500,
            "GA_CROSSOVER_RATE": 0.7,
            "GA_MUTATION_RATE": 0.1,
            "GA_TOURNAMENT_SIZE": 5,
            "GA_ELITISM_SIZE": 2,
            "GA_IMPROVED_EVALUATION": True,
         },
         # You can add more GA configurations here
      ],
      "TS_SETTINGS": [
         # Tabu Search configuration
         {
            "TS_MAX_ITERATIONS": 100,
            "TS_TABU_SIZE": 10,
            "TS_NEIGHBORHOOD_SIZE": 20,
         }
      ]
   }
]
```

## Understanding Results

After running tests, a CSV file with comparison results will be generated in the `results` directory. The file contains:

- Performance metrics for each algorithm
- Solution quality (movements - lower is better)
- Execution time
- For GA: generation where best solution was found

## Recommended Settings

For best results:

- Use `GA_IMPROVED_EVALUATION = True` for better solution guidance
- Try population sizes of 10,000+ for complex problems
- Tournament size of 5 provides good selection pressure
- Mutation rate of 0.1 offers good exploration/exploitation balance

## Notes

- The Genetic Algorithm implementation tracks which generation found the best solution
- The improved fitness function was crucial for finding valid solutions
- Tabu Search generally provides faster results with good quality