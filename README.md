# CVRP Solver - Readme

## Overview

This project implements and compares different optimization algorithms for solving the Capacitated Vehicle Routing Problem (CVRP). The implemented algorithms include:

- Greedy Algorithm
- Random Search Algorithm
- Genetic Algorithm (with configurable parameters)

The system allows you to configure problem parameters and algorithm settings through a central configuration file (`config.py`), run comprehensive tests, and generate detailed statistical analysis of the results.

## Configuration File (`config.py`)

The `config.py` file serves as the central configuration for both the CVRP problem definition and algorithm parameters. Below is an explanation of each setting.

### CVRP Settings

```python
# CVRP Settings
CVRP_NUMBER_OF_CITIES = 5         # Total number of cities (excluding depot)
CVRP_DEPOT_POSITION = 0           # Position/index of the depot
CVRP_DISTANCE_BETWEEN_CITIES = 5  # Unit distance between cities
CVRP_CAPACITY = 10                # Vehicle capacity (fuel/resources)
```

- **CVRP_NUMBER_OF_CITIES**: Specifies how many cities (excluding the depot) are in the problem. Higher numbers create more complex problems.
- **CVRP_DEPOT_POSITION**: Defines which index represents the depot (usually 0). Vehicles must start and return to this position.
- **CVRP_DISTANCE_BETWEEN_CITIES**: The standard distance between cities. This impacts how quickly vehicles use their capacity.
- **CVRP_CAPACITY**: The maximum capacity of each vehicle before it needs to return to the depot.

### Genetic Algorithm Settings

```python
# Genetic Algorithm Settings
GA_POPULATION_SIZE = 1000         # Number of individuals in each generation
GA_GENERATIONS = 50               # Number of evolution cycles
GA_CROSSOVER_RATE = 0.7           # Probability of crossover occurring
GA_MUTATION_RATE = 0.1            # Probability of mutation occurring
GA_TOURNAMENT_SIZE = 5            # Number of individuals in each tournament
GA_ELITISM_SIZE = 2               # Number of best individuals preserved in each generation
GA_IMPROVED_EVALUATION = True     # Use enhanced fitness evaluation function
```

- **GA_POPULATION_SIZE**: The number of individual solutions maintained in each generation. Larger populations increase diversity but require more computation.
- **GA_GENERATIONS**: The number of evolutionary cycles to run. More generations allow for further optimization but increase runtime.
- **GA_CROSSOVER_RATE**: Probability (0.0-1.0) that crossover will occur for a selected pair of individuals. Higher values promote genetic exchange.
- **GA_MUTATION_RATE**: Probability (0.0-1.0) that mutation will occur for each individual. Higher values increase diversity and exploration.
- **GA_TOURNAMENT_SIZE**: Number of individuals randomly selected for each tournament selection. Larger values increase selection pressure.
- **GA_ELITISM_SIZE**: Number of best individuals automatically preserved in each new generation. Ensures that high-quality solutions aren't lost.
- **GA_IMPROVED_EVALUATION**: When `True`, uses a gradient-based fitness function that provides better guidance for invalid solutions by considering partial city visits. When `False`, simply assigns infinite penalties to invalid solutions.

### Test Configuration

The `RUN_TEST` array allows you to define multiple test scenarios, each with its own CVRP problem parameters and GA configurations:

```python
RUN_TEST = [
    {
        # First test scenario
        "CVRP_NUMBER_OF_CITIES": 5,
        "CVRP_DEPOT_POSITION": 0,
        "CVRP_DISTANCE_BETWEEN_CITIES": 5,
        "CVRP_CAPACITY": 10,
        "GA_SETTINGS": [
            # First GA configuration
            {
                "GA_POPULATION_SIZE": 1000,
                "GA_GENERATIONS": 50,
                "GA_CROSSOVER_RATE": 0.7,
                "GA_MUTATION_RATE": 0.1,
                "GA_TOURNAMENT_SIZE": 5,
                "GA_ELITISM_SIZE": 2,
                "GA_IMPROVED_EVALUATION": True,
            },
            # Additional GA configurations...
        ]
    },
    # Additional test scenarios...
]
```

Each test scenario defines:
1. A CVRP problem instance (number of cities, distance, capacity)
2. Multiple GA configurations to test on that problem

This structure allows for systematic testing of different algorithm parameters across different problem complexities.

## Running Tests

To run the tests defined in your configuration file:

```bash
python test_runner.py
```

The test runner will:
1. Process each test scenario from the `RUN_TEST` configuration
2. Run the Greedy algorithm (deterministic, runs once)
3. Run the Random algorithm multiple times to get statistical measures
4. Run each GA configuration multiple times to get statistical results
5. Generate comprehensive CSV reports with the results

## Understanding Result Files

After running tests, three CSV files will be generated in the `results` directory:

### 1. Detailed Results (`detailed_results_TIMESTAMP.csv`)

Contains raw data for every algorithm run, including:
- Problem parameters (cities, distance, capacity)
- Algorithm type and configuration
- Performance metrics (valid solution, movements, execution time)
- GA-specific parameters when applicable

This file is useful for detailed analysis of individual algorithm runs.

### 2. Algorithm Comparison (`algorithm_comparison_TIMESTAMP.csv`)

Provides a direct comparison between all algorithms and configurations:
- Best movement count for each algorithm
- Average execution time and standard deviation
- Success rate (percentage of runs finding valid solutions)
- Performance relative to the Greedy algorithm (percentage improvement/degradation)
- Time efficiency metrics

This file helps identify which algorithms perform best for each test scenario.

### 3. Results Summary (`results_summary_TIMESTAMP.csv`)

High-level findings for each test case:
- Which algorithm performed best overall
- The best movement count achieved
- Improvement percentage compared to the Greedy baseline
- Whether GA was successful at finding valid solutions

This file provides a quick overview of test outcomes.

## Example: Modifying Configuration

To test a new CVRP problem configuration:

1. Edit the `config.py` file to add a new test scenario:

```python
RUN_TEST.append({
    "CVRP_NUMBER_OF_CITIES": 15,
    "CVRP_DEPOT_POSITION": 0,
    "CVRP_DISTANCE_BETWEEN_CITIES": 8,
    "CVRP_CAPACITY": 40,
    "GA_SETTINGS": [
        {
            "GA_POPULATION_SIZE": 5000,
            "GA_GENERATIONS": 100,
            "GA_CROSSOVER_RATE": 0.8,
            "GA_MUTATION_RATE": 0.1,
            "GA_TOURNAMENT_SIZE": 5,
            "GA_ELITISM_SIZE": 3,
            "GA_IMPROVED_EVALUATION": True,
        }
    ]
})
```

2. Run the tests:

```bash
python test_runner.py
```

3. Analyze the results in the generated CSV files.

## Tips for Effective Configuration

1. **For smaller problems** (5-10 cities):
    - Population sizes of 1,000-5,000 should be sufficient
    - 50-100 generations usually allows for convergence
    - Start with crossover rate ~0.7 and mutation rate ~0.1

2. **For larger problems** (15+ cities):
    - Increase population size to 10,000+ to maintain diversity
    - Increase generations to 100+ to allow more time for convergence
    - Consider increasing mutation rate slightly (0.1-0.2) to promote exploration
    - Always enable improved evaluation for better guidance

3. **Tournament size** affects selection pressure:
    - Smaller values (3-5) maintain diversity but slow convergence
    - Larger values (7-10) promote faster convergence but may lead to premature convergence

4. **Elitism** preserves best solutions:
    - Values of 1-3 typically work well
    - Too high values can inhibit exploration

## Interpreting Results

- **Movement count** is the primary optimization objective (lower is better)
- **Valid solution** indicates whether all cities were visited within capacity constraints
- **Success rate** shows reliability of stochastic algorithms (Random, GA)
- **Efficiency** measures quality of solution relative to computational cost
- **Movement improvement vs. Greedy** indicates relative performance compared to the baseline

By comparing these metrics across algorithms and configurations, you can determine which approach works best for particular CVRP problem instances.