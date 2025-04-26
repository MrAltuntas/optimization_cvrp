import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from datetime import datetime

import config
from cvrp import CVRP
from greedy_cvrp import GreedyCVRP
from random_cvrp import RandomCVRP
from genetic_cvrp import GeneticCVRP
from tabu_cvrp import TabuSearchCVRP

def ensure_dir(directory):
    """Ensure directory exists, create if it doesn't."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def run_greedy(problem):
    """Run greedy algorithm once."""
    print("\n==== Running Greedy Algorithm ====")
    start_time = time.time()

    solver = GreedyCVRP(problem)
    routes = solver.solve()

    end_time = time.time()
    execution_time = end_time - start_time

    total_distance = problem.get_total_distance(routes)
    is_valid, message = problem.is_valid_solution(routes)

    print(f"Valid solution: {is_valid}")
    print(f"Total distance: {total_distance}")
    print(f"Execution time: {execution_time:.4f} seconds")
    print(f"Routes: {routes}")

    return {
        'algorithm': 'Greedy',
        'distance': total_distance,
        'time': execution_time,
        'valid': is_valid,
        'routes': routes
    }

def run_random(problem, iterations=10000):
    """Run random algorithm."""
    print(f"\n==== Running Random Algorithm ({iterations} iterations) ====")

    solver = RandomCVRP(problem)
    routes, best_fitness = solver.solve(num_iterations=iterations)

    is_valid, message = problem.is_valid_solution(routes)

    print(f"Valid solution: {is_valid}")
    print(f"Total distance: {best_fitness}")
    print(f"Routes: {routes}")

    return {
        'algorithm': 'Random',
        'distance': best_fitness,
        'valid': is_valid,
        'routes': routes
    }

def run_genetic(problem, **params):
    """Run genetic algorithm with specified parameters."""
    # Set default parameters if not provided
    population_size = params.get('population_size', config.GA_POPULATION_SIZE)
    generations = params.get('generations', config.GA_GENERATIONS)
    crossover_rate = params.get('crossover_rate', config.GA_CROSSOVER_RATE)
    mutation_rate = params.get('mutation_rate', config.GA_MUTATION_RATE)
    tournament_size = params.get('tournament_size', config.GA_TOURNAMENT_SIZE)
    elitism_size = params.get('elitism_size', config.GA_ELITISM_SIZE)

    print("\n==== Running Genetic Algorithm ====")
    print(f"Parameters: pop_size={population_size}, generations={generations}, Px={crossover_rate}, Pm={mutation_rate}, Tour={tournament_size}, Elite={elitism_size}")

    solver = GeneticCVRP(
        problem,
        population_size=population_size,
        generations=generations,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        tournament_size=tournament_size,
        elitism_size=elitism_size
    )

    routes, stats = solver.solve()

    total_distance = problem.get_total_distance(routes)
    is_valid, message = problem.is_valid_solution(routes)

    print(f"Valid solution: {is_valid}")
    print(f"Total distance: {total_distance}")
    print(f"Routes: {routes}")

    # Plot progress
    fig = solver.plot_progress()

    return {
        'algorithm': 'Genetic',
        'distance': total_distance,
        'valid': is_valid,
        'routes': routes,
        'stats': stats,
        'figure': fig,
        'best_generation': solver.best_generation,
        'parameters': {
            'population_size': population_size,
            'generations': generations,
            'crossover_rate': crossover_rate,
            'mutation_rate': mutation_rate,
            'tournament_size': tournament_size,
            'elitism_size': elitism_size
        }
    }

def run_tabu(problem, **params):
    """Run tabu search algorithm with specified parameters."""
    # Set default parameters if not provided
    max_iterations = params.get('max_iterations', config.TS_MAX_ITERATIONS)
    tabu_size = params.get('tabu_size', config.TS_TABU_SIZE)
    neighborhood_size = params.get('neighborhood_size', config.TS_NEIGHBORHOOD_SIZE)

    print("\n==== Running Tabu Search Algorithm ====")
    print(f"Parameters: max_iterations={max_iterations}, tabu_size={tabu_size}, neighborhood_size={neighborhood_size}")

    solver = TabuSearchCVRP(
        problem,
        max_iterations=max_iterations,
        tabu_size=tabu_size,
        neighborhood_size=neighborhood_size
    )

    routes, stats = solver.solve()

    total_distance = problem.get_total_distance(routes)
    is_valid, message = problem.is_valid_solution(routes)

    print(f"Valid solution: {is_valid}")
    print(f"Total distance: {total_distance}")
    print(f"Routes: {routes}")

    # Plot progress
    fig = solver.plot_progress()

    return {
        'algorithm': 'Tabu Search',
        'distance': total_distance,
        'valid': is_valid,
        'routes': routes,
        'stats': stats,
        'figure': fig,
        'best_iteration': solver.best_iteration,
        'parameters': {
            'max_iterations': max_iterations,
            'tabu_size': tabu_size,
            'neighborhood_size': neighborhood_size
        }
    }

def run_parameter_test(problem, instance_name):
    """Run tests with different parameter settings."""
    results_dir = f"results/{instance_name}/parameter_tests"
    ensure_dir(results_dir)

    # Test different GA parameters
    ga_results = []

    for test in config.PARAMETER_TESTS:
        parameter = test['parameter']
        values = test['values']
        default_params = test['default_params'].copy()

        print(f"\n==== Testing {test['name']} ====")

        for value in values:
            print(f"\nTesting {parameter}={value}")

            params = default_params.copy()
            params[parameter.lower().split('ga_')[-1]] = value

            # Run genetic algorithm with these parameters
            result = run_genetic(problem, **params)

            # Add test info
            result['test_parameter'] = parameter
            result['test_value'] = value

            ga_results.append(result)

            # Save the figure
            if 'figure' in result:
                fig_path = f"{results_dir}/GA_{parameter}_{value}.png"
                result['figure'].savefig(fig_path, dpi=300)
                plt.close(result['figure'])

    # Test different Tabu Search parameters
    ts_results = []

    for test in config.TS_PARAMETER_TESTS:
        parameter = test['parameter']
        values = test['values']
        default_params = test['default_params'].copy()

        print(f"\n==== Testing {test['name']} ====")

        for value in values:
            print(f"\nTesting {parameter}={value}")

            params = default_params.copy()
            params[parameter.lower().split('ts_')[-1]] = value

            # Run tabu search with these parameters
            result = run_tabu(problem, **params)

            # Add test info
            result['test_parameter'] = parameter
            result['test_value'] = value

            ts_results.append(result)

            # Save the figure
            if 'figure' in result:
                fig_path = f"{results_dir}/TS_{parameter}_{value}.png"
                result['figure'].savefig(fig_path, dpi=300)
                plt.close(result['figure'])

    # Create summary tables
    ga_summary = []
    for result in ga_results:
        ga_summary.append({
            'Parameter': result['test_parameter'],
            'Value': result['test_value'],
            'Distance': result['distance'],
            'Valid': result['valid'],
            'Generation': result.get('best_generation', 'N/A')
        })

    ts_summary = []
    for result in ts_results:
        ts_summary.append({
            'Parameter': result['test_parameter'],
            'Value': result['test_value'],
            'Distance': result['distance'],
            'Valid': result['valid'],
            'Iteration': result.get('best_iteration', 'N/A')
        })

    # Convert to DataFrames and save
    ga_df = pd.DataFrame(ga_summary)
    ts_df = pd.DataFrame(ts_summary)

    ga_df.to_csv(f"{results_dir}/GA_parameter_summary.csv", index=False)
    ts_df.to_csv(f"{results_dir}/TS_parameter_summary.csv", index=False)

    # Plot parameter effect on distance
    for param_name in set(ga_df['Parameter']):
        param_data = ga_df[ga_df['Parameter'] == param_name]
        plt.figure(figsize=(10, 6))
        plt.plot(param_data['Value'], param_data['Distance'], 'bo-', markersize=8)
        plt.title(f'Effect of {param_name} on Solution Quality')
        plt.xlabel(param_name)
        plt.ylabel('Total Distance (lower is better)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{results_dir}/GA_{param_name}_effect.png", dpi=300)
        plt.close()

    for param_name in set(ts_df['Parameter']):
        param_data = ts_df[ts_df['Parameter'] == param_name]
        plt.figure(figsize=(10, 6))
        plt.plot(param_data['Value'], param_data['Distance'], 'ro-', markersize=8)
        plt.title(f'Effect of {param_name} on Solution Quality')
        plt.xlabel(param_name)
        plt.ylabel('Total Distance (lower is better)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{results_dir}/TS_{param_name}_effect.png", dpi=300)
        plt.close()

    return ga_results, ts_results

def run_algorithm_comparison(problem, instance_name):
    """Run all algorithms and compare their performance."""
    results_dir = f"results/{instance_name}"
    ensure_dir(results_dir)

    all_results = []

    # Run greedy algorithm (deterministic, run once)
    greedy_result = run_greedy(problem)
    all_results.append(greedy_result)

    # Random search
    random_result = run_random(problem, iterations=config.GA_POPULATION_SIZE * config.GA_GENERATIONS)
    all_results.append(random_result)

    # Genetic algorithm
    ga_results = []
    for i in range(config.RUN_TEST_COUNT):
        print(f"\n==== Genetic Algorithm Run {i+1}/{config.RUN_TEST_COUNT} ====")
        result = run_genetic(problem)
        ga_results.append(result)

    # Use best GA result for comparison
    best_ga = min(ga_results, key=lambda x: x['distance'])
    all_results.append(best_ga)

    # Calculate GA statistics
    ga_distances = [r['distance'] for r in ga_results]
    ga_stats = {
        'algorithm': 'Genetic [Stats]',
        'best': min(ga_distances),
        'worst': max(ga_distances),
        'avg': np.mean(ga_distances),
        'std': np.std(ga_distances),
        'success_rate': sum(1 for r in ga_results if r['valid']) / len(ga_results)
    }

    # Tabu search
    ts_results = []
    for i in range(config.RUN_TEST_COUNT):
        print(f"\n==== Tabu Search Run {i+1}/{config.RUN_TEST_COUNT} ====")
        result = run_tabu(problem)
        ts_results.append(result)

    # Use best TS result for comparison
    best_ts = min(ts_results, key=lambda x: x['distance'])
    all_results.append(best_ts)

    # Calculate TS statistics
    ts_distances = [r['distance'] for r in ts_results]
    ts_stats = {
        'algorithm': 'Tabu Search [Stats]',
        'best': min(ts_distances),
        'worst': max(ts_distances),
        'avg': np.mean(ts_distances),
        'std': np.std(ts_distances),
        'success_rate': sum(1 for r in ts_results if r['valid']) / len(ts_results)
    }

    # Create comparison table
    comparison = []
    for result in all_results:
        comparison.append({
            'Algorithm': result['algorithm'],
            'Distance': result['distance'],
            'Valid': result['valid']
        })

    # Add statistics rows
    comparison.append({
        'Algorithm': ga_stats['algorithm'],
        'Distance': f"{ga_stats['avg']:.2f} ± {ga_stats['std']:.2f}",
        'Valid': f"{ga_stats['success_rate']*100:.1f}%"
    })

    comparison.append({
        'Algorithm': ts_stats['algorithm'],
        'Distance': f"{ts_stats['avg']:.2f} ± {ts_stats['std']:.2f}",
        'Valid': f"{ts_stats['success_rate']*100:.1f}%"
    })

    # Convert to DataFrame and save
    comparison_df = pd.DataFrame(comparison)
    comparison_df.to_csv(f"{results_dir}/algorithm_comparison.csv", index=False)

    # Plot comparison
    algorithms = [r['Algorithm'] for r in comparison if not '[Stats]' in r['Algorithm']]
    distances = [r['Distance'] for r in comparison if not '[Stats]' in r['Algorithm']]

    plt.figure(figsize=(12, 8))
    bars = plt.bar(algorithms, distances)

    # Color bars based on algorithm
    colors = ['green', 'gray', 'blue', 'red']
    for i, bar in enumerate(bars):
        bar.set_color(colors[i % len(colors)])

    plt.title(f'Algorithm Performance Comparison - {instance_name}')
    plt.ylabel('Total Distance (lower is better)')
    plt.xticks(rotation=45, ha='right')

    # Add value labels above bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                 f'{height:.1f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(f"{results_dir}/algorithm_comparison.png", dpi=300)
    plt.close()

    return all_results, ga_stats, ts_stats

def main():
    parser = argparse.ArgumentParser(description='CVRP Solver')
    parser.add_argument('--instance', type=str, default='A-n32-k5',
                        help='Instance name (default: A-n32-k5)')
    parser.add_argument('--paramtest', action='store_true',
                        help='Run parameter testing')
    parser.add_argument('--compare', action='store_true',
                        help='Run algorithm comparison')

    args = parser.parse_args()

    instance_name = args.instance
    instance_file = f"instances/{instance_name}.vrp"

    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/{instance_name}"
    ensure_dir(results_dir)

    print(f"Loading CVRP instance: {instance_file}")
    problem = CVRP(instance_file)
    print(f"Loaded: {problem}")

    if args.paramtest:
        # Run parameter testing
        ga_results, ts_results = run_parameter_test(problem, instance_name)

    if args.compare:
        # Run algorithm comparison
        all_results, ga_stats, ts_stats = run_algorithm_comparison(problem, instance_name)

    # If no specific mode is selected, run both
    if not args.paramtest and not args.compare:
        ga_results, ts_results = run_parameter_test(problem, instance_name)
        all_results, ga_stats, ts_stats = run_algorithm_comparison(problem, instance_name)

    print("\nAll tests completed.")

if __name__ == "__main__":
    main()