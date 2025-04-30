import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import multiprocessing
from tqdm import tqdm

import config
from cvrp import CVRP
from greedy_cvrp import GreedyCVRP
from random_cvrp import RandomCVRP
from genetic_cvrp import GeneticCVRP
from tabu_cvrp import TabuSearchCVRP
from visualization import Visualization

def ensure_dir(directory):
    """Ensure directory exists, create if it doesn't."""
    if not os.path.exists(directory):
        os.makedirs(directory)

class TestRunner:
    def __init__(self, instance_files=None, num_runs=10, output_dir="results"):
        """
        Initialize the test runner.

        Args:
            instance_files: List of instance file paths
            num_runs: Number of runs for each algorithm
            output_dir: Output directory for results
        """
        self.instance_files = instance_files or config.INSTANCE_FILES
        self.num_runs = num_runs
        self.output_dir = output_dir

        ensure_dir(output_dir)

        self.visualizer = Visualization(base_dir=output_dir)

    def run_instance(self, instance_file):
        """
        Run all algorithms on a single instance.

        Args:
            instance_file: Path to instance file

        Returns:
            Dictionary with results
        """
        instance_name = os.path.basename(instance_file).split('.')[0]
        print(f"\n{'='*50}")
        print(f"Running tests for instance: {instance_name}")
        print(f"{'='*50}")

        instance_dir = os.path.join(self.output_dir, instance_name)
        ensure_dir(instance_dir)

        # Load problem
        problem = CVRP(instance_file)
        print(f"Loaded problem: {problem}")

        instance_results = {
            'instance': instance_name,
            'greedy': self.run_greedy(problem, instance_dir),
            'random': self.run_random(problem, instance_dir),
            'genetic': self.run_genetic(problem, instance_dir),
            'tabu': self.run_tabu(problem, instance_dir)
        }

        # Create comparison table and plot
        all_results = [
            instance_results['greedy'],
            instance_results['random']['best'],
            instance_results['genetic']['best'],
            instance_results['tabu']['best']
        ]

        comparison_table = self.visualizer.create_comparison_table(
            all_results,
            save_path=os.path.join(instance_dir, 'algorithm_comparison.csv')
        )

        # Plot all algorithms comparison
        self.visualizer.plot_all_algorithms_comparison(
            {
                'Greedy': instance_results['greedy'],
                'Random': instance_results['random']['best'],
                'Genetic': instance_results['genetic']['best'],
                'Tabu Search': instance_results['tabu']['best']
            },
            title=f'Algorithm Performance Comparison - {instance_name}',
            save_path=os.path.join(instance_dir, 'algorithm_comparison.png')
        )

        # Save the best solution visualization
        best_algorithm = min(all_results, key=lambda r: r['distance'])
        best_routes = best_algorithm['routes']

        self.visualizer.plot_cvrp_solution(
            problem, best_routes,
            title=f'Best Solution ({best_algorithm["algorithm"]}) - {instance_name}',
            save_path=os.path.join(instance_dir, 'best_solution.png')
        )

        return instance_results

    def run_greedy(self, problem, output_dir):
        """Run greedy algorithm."""
        print("\nRunning Greedy Algorithm...")
        start_time = time.time()

        solver = GreedyCVRP(problem)
        routes = solver.solve()

        end_time = time.time()
        execution_time = end_time - start_time

        total_distance = problem.get_total_distance(routes)
        is_valid, message = problem.is_valid_solution(routes)

        print(f"  Distance: {total_distance}, Valid: {is_valid}, Time: {execution_time:.2f}s")

        # Visualize solution
        fig = self.visualizer.plot_cvrp_solution(
            problem, routes,
            title=f'Greedy Solution - Distance: {total_distance}',
            save_path=os.path.join(output_dir, 'greedy_solution.png')
        )
        plt.close(fig)

        return {
            'algorithm': 'Greedy',
            'distance': total_distance,
            'time': execution_time,
            'valid': is_valid,
            'routes': routes
        }

    def run_random(self, problem, output_dir):
        """Run random algorithm multiple times."""
        print("\nRunning Random Algorithm...")

        # Number of iterations for random search
        iterations = config.GA_POPULATION_SIZE * config.GA_GENERATIONS

        results = []
        for i in range(self.num_runs):
            print(f"  Run {i+1}/{self.num_runs}")

            start_time = time.time()
            solver = RandomCVRP(problem)
            routes, distance = solver.solve(num_iterations=iterations // self.num_runs)
            end_time = time.time()

            execution_time = end_time - start_time
            is_valid, message = problem.is_valid_solution(routes)

            print(f"    Distance: {distance}, Valid: {is_valid}, Time: {execution_time:.2f}s")

            results.append({
                'algorithm': 'Random',
                'distance': distance,
                'time': execution_time,
                'valid': is_valid,
                'routes': routes
            })

        # Find best result
        best_result = min(results, key=lambda r: r['distance'])

        # Calculate statistics
        distances = [r['distance'] for r in results]
        times = [r['time'] for r in results]

        stats = {
            'best': min(distances),
            'worst': max(distances),
            'avg': np.mean(distances),
            'std': np.std(distances),
            'avg_time': np.mean(times)
        }

        # Visualize best solution
        fig = self.visualizer.plot_cvrp_solution(
            problem, best_result['routes'],
            title=f'Best Random Solution - Distance: {best_result["distance"]}',
            save_path=os.path.join(output_dir, 'random_best_solution.png')
        )
        plt.close(fig)

        return {
            'runs': results,
            'best': best_result,
            'stats': stats
        }

    def run_genetic(self, problem, output_dir):
        """Run genetic algorithm multiple times."""
        print("\nRunning Genetic Algorithm...")

        results = []
        for i in range(self.num_runs):
            print(f"  Run {i+1}/{self.num_runs}")

            start_time = time.time()
            solver = GeneticCVRP(
                problem,
                population_size=config.GA_POPULATION_SIZE,
                generations=config.GA_GENERATIONS,
                crossover_rate=config.GA_CROSSOVER_RATE,
                mutation_rate=config.GA_MUTATION_RATE,
                tournament_size=config.GA_TOURNAMENT_SIZE,
                elitism_size=config.GA_ELITISM_SIZE
            )

            routes, stats = solver.solve()
            end_time = time.time()

            execution_time = end_time - start_time
            distance = problem.get_total_distance(routes)
            is_valid, message = problem.is_valid_solution(routes)

            print(f"    Distance: {distance}, Valid: {is_valid}, Time: {execution_time:.2f}s")

            results.append({
                'algorithm': 'Genetic',
                'distance': distance,
                'time': execution_time,
                'valid': is_valid,
                'routes': routes,
                'stats': stats,
                'best_generation': solver.best_generation
            })

            # Plot progress for the first run
            if i == 0:
                fig = solver.plot_progress()
                plt.savefig(os.path.join(output_dir, 'genetic_progress.png'), dpi=300)
                plt.close(fig)

        # Find best result
        best_result = min(results, key=lambda r: r['distance'])

        # Calculate statistics
        distances = [r['distance'] for r in results]
        times = [r['time'] for r in results]
        best_generations = [r['best_generation'] for r in results]

        stats = {
            'best': min(distances),
            'worst': max(distances),
            'avg': np.mean(distances),
            'std': np.std(distances),
            'avg_time': np.mean(times),
            'avg_best_generation': np.mean(best_generations)
        }

        # Visualize best solution
        fig = self.visualizer.plot_cvrp_solution(
            problem, best_result['routes'],
            title=f'Best Genetic Solution - Distance: {best_result["distance"]}',
            save_path=os.path.join(output_dir, 'genetic_best_solution.png')
        )
        plt.close(fig)

        return {
            'runs': results,
            'best': best_result,
            'stats': stats
        }

    def run_tabu(self, problem, output_dir):
        """Run tabu search algorithm multiple times."""
        print("\nRunning Tabu Search Algorithm...")

        results = []
        for i in range(self.num_runs):
            print(f"  Run {i+1}/{self.num_runs}")

            start_time = time.time()
            solver = TabuSearchCVRP(
                problem,
                max_iterations=config.TS_MAX_ITERATIONS,
                tabu_size=config.TS_TABU_SIZE,
                neighborhood_size=config.TS_NEIGHBORHOOD_SIZE
            )

            routes, stats = solver.solve()
            end_time = time.time()

            execution_time = end_time - start_time
            distance = problem.get_total_distance(routes)
            is_valid, message = problem.is_valid_solution(routes)

            print(f"    Distance: {distance}, Valid: {is_valid}, Time: {execution_time:.2f}s")

            results.append({
                'algorithm': 'Tabu Search',
                'distance': distance,
                'time': execution_time,
                'valid': is_valid,
                'routes': routes,
                'stats': stats,
                'best_iteration': solver.best_iteration
            })

            # Plot progress for the first run
            if i == 0:
                fig = solver.plot_progress()
                plt.savefig(os.path.join(output_dir, 'tabu_progress.png'), dpi=300)
                plt.close(fig)

        # Find best result
        best_result = min(results, key=lambda r: r['distance'])

        # Calculate statistics
        distances = [r['distance'] for r in results]
        times = [r['time'] for r in results]
        best_iterations = [r['best_iteration'] for r in results]

        stats = {
            'best': min(distances),
            'worst': max(distances),
            'avg': np.mean(distances),
            'std': np.std(distances),
            'avg_time': np.mean(times),
            'avg_best_iteration': np.mean(best_iterations)
        }

        # Visualize best solution
        fig = self.visualizer.plot_cvrp_solution(
            problem, best_result['routes'],
            title=f'Best Tabu Search Solution - Distance: {best_result["distance"]}',
            save_path=os.path.join(output_dir, 'tabu_best_solution.png')
        )
        plt.close(fig)

        return {
            'runs': results,
            'best': best_result,
            'stats': stats
        }

    def run_parameter_tests(self, instance_file):
        """
        Test different parameter settings for GA and TS on a single instance.

        Args:
            instance_file: Path to instance file

        Returns:
            Dictionary with parameter test results
        """
        instance_name = os.path.basename(instance_file).split('.')[0]
        print(f"\n{'='*50}")
        print(f"Running parameter tests for instance: {instance_name}")
        print(f"{'='*50}")

        param_dir = os.path.join(self.output_dir, instance_name, 'parameter_tests')
        ensure_dir(param_dir)

        # Load problem
        problem = CVRP(instance_file)

        # Test GA parameters
        ga_results = self.test_ga_parameters(problem, param_dir)

        # Test TS parameters
        ts_results = self.test_ts_parameters(problem, param_dir)

        return {
            'instance': instance_name,
            'ga_params': ga_results,
            'ts_params': ts_results
        }

    def test_ga_parameters(self, problem, output_dir):
        """Test different GA parameters."""
        print("\nTesting GA parameters...")

        ga_param_dir = os.path.join(output_dir, 'genetic')
        ensure_dir(ga_param_dir)

        # Parameter test results
        param_results = {}

        for test in config.PARAMETER_TESTS:
            param_name = test['name']
            parameter = test['parameter']
            values = test['values']
            default_params = test['default_params'].copy()

            print(f"  Testing {param_name}...")
            param_results[parameter] = []

            for value in values:
                print(f"    Value: {value}")

                # Set parameter value
                params = default_params.copy()

                # Handle special case for selection pressure
                if parameter == "SELECTION_PRESSURE" and "special_params" in test:
                    # Get the specific parameters for this selection pressure value
                    special_param = test["special_params"].get(value, {})
                    # Update params with the special parameter values
                    params.update(special_param)
                else:
                    # Normal parameter handling
                    params[parameter] = value

                # Convert keys to lowercase without the GA_ prefix for proper parameter passing
                params_lower = {k.lower().split('ga_')[-1]: v for k, v in params.items()
                                if k.lower().startswith('ga_')}  # Only include GA parameters

                # Run genetic algorithm
                solver = GeneticCVRP(problem, **params_lower)
                routes, stats = solver.solve()

                distance = problem.get_total_distance(routes)
                is_valid, message = problem.is_valid_solution(routes)

                print(f"      Distance: {distance}, Valid: {is_valid}")

                # Store results
                result = {
                    'parameter': parameter,
                    'value': value,
                    'distance': distance,
                    'valid': is_valid,
                    'routes': routes,
                    'stats': stats,
                    'best_generation': solver.best_generation
                }

                param_results[parameter].append(result)

                # Plot progress
                fig = solver.plot_progress()
                plt.savefig(os.path.join(ga_param_dir, f'{parameter}_{value}_progress.png'), dpi=300)
                plt.close(fig)

            # Plot parameter effect
            self.visualizer.plot_parameter_effect(
                param_results[parameter],
                parameter,
                title=f'Effect of {param_name} on Solution Quality',
                save_path=os.path.join(ga_param_dir, f'{parameter}_effect.png')
            )

        # Create summary table
        summary = []
        for parameter, results in param_results.items():
            for result in results:
                summary.append({
                    'Parameter': parameter,
                    'Value': result['value'],
                    'Distance': result['distance'],
                    'Valid': result['valid'],
                    'Best Generation': result['best_generation']
                })

        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(os.path.join(ga_param_dir, 'parameter_summary.csv'), index=False)

        return param_results

    def test_ts_parameters(self, problem, output_dir):
        """Test different TS parameters."""
        print("\nTesting TS parameters...")

        ts_param_dir = os.path.join(output_dir, 'tabu')
        ensure_dir(ts_param_dir)

        # Parameter test results
        param_results = {}

        for test in config.TS_PARAMETER_TESTS:
            param_name = test['name']
            parameter = test['parameter']
            values = test['values']
            default_params = test['default_params'].copy()

            print(f"  Testing {param_name}...")
            param_results[parameter] = []

            for value in values:
                print(f"    Value: {value}")

                # Set parameter value
                params = default_params.copy()
                params_lower = {k.lower().split('ts_')[-1]: v for k, v in params.items()}
                params_lower[parameter.lower().split('ts_')[-1]] = value

                # Run tabu search
                solver = TabuSearchCVRP(problem, **params_lower)
                routes, stats = solver.solve()

                distance = problem.get_total_distance(routes)
                is_valid, message = problem.is_valid_solution(routes)

                print(f"      Distance: {distance}, Valid: {is_valid}")

                # Store results
                result = {
                    'parameter': parameter,
                    'value': value,
                    'distance': distance,
                    'valid': is_valid,
                    'routes': routes,
                    'stats': stats,
                    'best_iteration': solver.best_iteration
                }

                param_results[parameter].append(result)

                # Plot progress
                fig = solver.plot_progress()
                plt.savefig(os.path.join(ts_param_dir, f'{parameter}_{value}_progress.png'), dpi=300)
                plt.close(fig)

            # Plot parameter effect
            self.visualizer.plot_parameter_effect(
                param_results[parameter],
                parameter,
                title=f'Effect of {param_name} on Solution Quality',
                save_path=os.path.join(ts_param_dir, f'{parameter}_effect.png')
            )

        # Create summary table
        summary = []
        for parameter, results in param_results.items():
            for result in results:
                summary.append({
                    'Parameter': parameter,
                    'Value': result['value'],
                    'Distance': result['distance'],
                    'Valid': result['valid'],
                    'Best Iteration': result['best_iteration']
                })

        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(os.path.join(ts_param_dir, 'parameter_summary.csv'), index=False)

        return param_results

    def run_all_tests(self):
        """Run all tests on all instances."""
        print(f"Running tests on {len(self.instance_files)} instances...")

        all_results = []
        all_param_results = []

        for instance_file in self.instance_files:
            # Run algorithm comparison
            result = self.run_instance(instance_file)
            all_results.append(result)

            # Run parameter tests
            param_result = self.run_parameter_tests(instance_file)
            all_param_results.append(param_result)

        # Create final summary
        self.create_summary_table(all_results)

        return all_results, all_param_results

    def create_summary_table(self, all_results):
        """Create a summary table of results across all instances."""
        print("\nCreating summary table...")

        summary = []

        for result in all_results:
            instance = result['instance']

            # Greedy
            greedy = result['greedy']

            # Random
            random_best = result['random']['best']
            random_stats = result['random']['stats']

            # Genetic
            genetic_best = result['genetic']['best']
            genetic_stats = result['genetic']['stats']

            # Tabu
            tabu_best = result['tabu']['best']
            tabu_stats = result['tabu']['stats']

            summary.append({
                'Instance': instance,
                'Greedy Distance': greedy['distance'],
                'Greedy Time': greedy['time'],
                'Greedy Best Solution': self._format_routes(greedy['routes']),
                'Random Best': random_stats['best'],
                'Random Avg': random_stats['avg'],
                'Random Std': random_stats['std'],
                'Random Time': random_stats['avg_time'],
                'Random Best Solution': self._format_routes(random_best['routes']),
                'Genetic Best': genetic_stats['best'],
                'Genetic Avg': genetic_stats['avg'],
                'Genetic Std': genetic_stats['std'],
                'Genetic Time': genetic_stats['avg_time'],
                'Genetic Best Gen': genetic_stats['avg_best_generation'],
                'Genetic Best Solution': self._format_routes(genetic_best['routes']),
                'Tabu Best': tabu_stats['best'],
                'Tabu Avg': tabu_stats['avg'],
                'Tabu Std': tabu_stats['std'],
                'Tabu Time': tabu_stats['avg_time'],
                'Tabu Best Iter': tabu_stats['avg_best_iteration'],
                'Tabu Best Solution': self._format_routes(tabu_best['routes'])
            })

        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(os.path.join(self.output_dir, 'all_results_summary.csv'), index=False)

        # Also create HTML version for better visualization
        html_path = os.path.join(self.output_dir, 'all_results_summary.html')
        with open(html_path, 'w') as f:
            f.write('<html><head><title>CVRP Results Summary</title>')
            f.write('<style>table {border-collapse: collapse; width: 100%;} ')
            f.write('th, td {border: 1px solid black; padding: 8px; text-align: center;} ')
            f.write('th {background-color: #f2f2f2;} ')
            f.write('tr:nth-child(even) {background-color: #f9f9f9;} ')
            f.write('.route-cell {max-width: 250px; overflow: auto; white-space: nowrap;} ')
            f.write('</style></head><body>')
            f.write('<h1>CVRP Algorithms Performance Summary</h1>')

            # Convert DataFrame to HTML with custom formatting for route cells
            html_table = summary_df.to_html(index=False)
            # Add CSS class to route cells
            for col in ['Greedy Best Solution', 'Random Best Solution', 'Genetic Best Solution', 'Tabu Best Solution']:
                html_table = html_table.replace(f'<th>{col}</th>', f'<th class="route-cell">{col}</th>')
                html_table = html_table.replace(f'<td>{col}</td>', f'<td class="route-cell">{col}</td>')

            f.write(html_table)
            f.write('</body></html>')

        print(f"Summary saved to: {html_path}")

    def _format_routes(self, routes):
        """Format routes as a concise string representation."""
        # Format each route as a comma-separated list of cities
        route_strings = []
        for i, route in enumerate(routes):
            route_str = f"R{i+1}: {' → '.join(map(str, route))}"
            route_strings.append(route_str)

        # Join the routes with line breaks
        return " | ".join(route_strings)


def main():
    parser = argparse.ArgumentParser(description='CVRP Test Runner')
    parser.add_argument('--instances', nargs='+', help='Instance files to test')
    parser.add_argument('--num-runs', type=int, default=config.RUN_TEST_COUNT,
                        help='Number of runs for each algorithm')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory for results')

    args = parser.parse_args()

    # Use specified instances or all from config
    instance_files = args.instances or config.INSTANCE_FILES

    # Create test runner
    runner = TestRunner(
        instance_files=instance_files,
        num_runs=args.num_runs,
        output_dir=args.output_dir
    )

    # Run all tests
    results, param_results = runner.run_all_tests()

    print("\nAll tests completed.")


if __name__ == "__main__":
    main()