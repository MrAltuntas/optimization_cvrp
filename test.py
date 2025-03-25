import config
import cvrp
import greedy_cvrp
import random_cvrp
import genetic_cvrp
import time
import pandas as pd
import os
import numpy as np
from datetime import datetime

class TestRunner:
    def __init__(self, test_configs=None, num_runs=10):
        """
        Initialize the test runner.

        Args:
            test_configs: List of test configurations. If None, uses config.RUN_TEST
            num_runs: Number of times to run each test for statistical analysis
        """
        self.test_configs = test_configs if test_configs else config.RUN_TEST
        self.num_runs = num_runs
        self.results = []

    def run_tests(self):
        """Run all tests with specified configurations"""
        for test_idx, test_config in enumerate(self.test_configs):
            print(f"\n\n{'='*40} TEST {test_idx+1} {'='*40}")
            print(f"CVRP Configuration:")
            print(f"  Number of Cities: {test_config['CVRP_NUMBER_OF_CITIES']}")
            print(f"  Depot Position: {test_config['CVRP_DEPOT_POSITION']}")
            print(f"  Distance Between Cities: {test_config['CVRP_DISTANCE_BETWEEN_CITIES']}")
            print(f"  Capacity: {test_config['CVRP_CAPACITY']}")
            print(f"{'-'*90}")

            # Create CVRP problem
            cvrp_problem = cvrp.CVRP(
                num_cities=test_config['CVRP_NUMBER_OF_CITIES'],
                depot=test_config['CVRP_DEPOT_POSITION'],
                distance=test_config['CVRP_DISTANCE_BETWEEN_CITIES'],
                capacity=test_config['CVRP_CAPACITY']
            )

            test_results = {
                'test_id': test_idx + 1,
                'cvrp_config': test_config,
                'methods': {}
            }

            # Run Greedy algorithm (deterministic, only need to run once)
            print("\n1. Greedy Algorithm:")
            greedy_solver = greedy_cvrp.GreedyCVRP(cvrp_problem)
            greedy_times = []

            # Still run multiple times to get timing statistics
            for run in range(self.num_runs):
                start_time = time.time()
                greedy_path = greedy_solver.find_path()
                greedy_times.append(time.time() - start_time)

                if run == 0:  # Only need to evaluate path once since it's deterministic
                    greedy_movements, greedy_valid = cvrp_problem.is_valid_path(greedy_path)

            avg_greedy_time = np.mean(greedy_times)
            std_greedy_time = np.std(greedy_times)

            print(f"Valid: {greedy_valid}, Movements: {greedy_movements}, "
                  f"Avg Time: {avg_greedy_time:.4f}s, Std: {std_greedy_time:.4f}s")
            print(f"Path: {greedy_path}")

            test_results['methods']['greedy'] = {
                'valid': greedy_valid,
                'movements': greedy_movements,
                'avg_time': avg_greedy_time,
                'min_time': min(greedy_times),
                'max_time': max(greedy_times),
                'std_time': std_greedy_time,
                'path': greedy_path
            }

            # Run Random algorithm multiple times
            print("\n2. Random Algorithm:")
            random_solver = random_cvrp.RandomCVRP(cvrp_problem)
            random_results = []

            for run in range(self.num_runs):
                start_time = time.time()
                random_path = random_solver.random_path()
                run_time = time.time() - start_time
                movements, valid = cvrp_problem.is_valid_path(random_path)

                random_results.append({
                    'run': run + 1,
                    'valid': valid,
                    'movements': movements,
                    'time': run_time,
                    'path': random_path
                })

            # Calculate statistics for random algorithm
            random_times = [r['time'] for r in random_results]
            random_movements = [r['movements'] for r in random_results]

            avg_random_time = np.mean(random_times)
            std_random_time = np.std(random_times)
            avg_random_movements = np.mean(random_movements)
            std_random_movements = np.std(random_movements)

            # Find best random solution
            best_random = min(random_results, key=lambda x: x['movements'])

            print(f"Best Valid: {best_random['valid']}, Movements: {best_random['movements']}, "
                  f"Avg Movements: {avg_random_movements:.2f} (±{std_random_movements:.2f})")
            print(f"Avg Time: {avg_random_time:.4f}s, Std: {std_random_time:.4f}s")
            print(f"Best Path: {best_random['path']}")

            test_results['methods']['random'] = {
                'best_valid': best_random['valid'],
                'best_movements': best_random['movements'],
                'avg_movements': avg_random_movements,
                'min_movements': min(random_movements),
                'max_movements': max(random_movements),
                'std_movements': std_random_movements,
                'avg_time': avg_random_time,
                'min_time': min(random_times),
                'max_time': max(random_times),
                'std_time': std_random_time,
                'best_path': best_random['path'],
                'all_runs': random_results
            }

            # Run Genetic algorithm with different settings
            test_results['methods']['genetic'] = []

            for ga_idx, ga_setting in enumerate(test_config['GA_SETTINGS']):
                print(f"\n3. Genetic Algorithm (Config {ga_idx+1}):")
                print(f"  Population Size: {ga_setting['GA_POPULATION_SIZE']}")
                print(f"  Generations: {ga_setting['GA_GENERATIONS']}")
                print(f"  Crossover Rate: {ga_setting['GA_CROSSOVER_RATE']}")
                print(f"  Mutation Rate: {ga_setting['GA_MUTATION_RATE']}")
                print(f"  Tournament Size: {ga_setting['GA_TOURNAMENT_SIZE']}")
                print(f"  Elitism Size: {ga_setting['GA_ELITISM_SIZE']}")
                print(f"  Improved Evaluation: {ga_setting['GA_IMPROVED_EVALUATION']}")

                ga_results = []

                for run in range(self.num_runs):
                    print(f"  Run {run+1}/{self.num_runs}...")

                    ga_solver = genetic_cvrp.GeneticCVRP(
                        cvrp_problem,
                        population_size=ga_setting['GA_POPULATION_SIZE'],
                        generations=ga_setting['GA_GENERATIONS'],
                        crossover_rate=ga_setting['GA_CROSSOVER_RATE'],
                        mutation_rate=ga_setting['GA_MUTATION_RATE'],
                        tournament_size=ga_setting['GA_TOURNAMENT_SIZE'],
                        elitism_count=ga_setting['GA_ELITISM_SIZE'],
                        improved_evaluation=ga_setting['GA_IMPROVED_EVALUATION'],
                    )

                    start_time = time.time()
                    ga_path = ga_solver.solve()
                    run_time = time.time() - start_time

                    run_result = {
                        'run': run + 1,
                        'time': run_time,
                        'best_fitness': ga_solver.best_fitness
                    }

                    if ga_path is not None:
                        movements, valid = cvrp_problem.is_valid_path(ga_path)
                        run_result.update({
                            'valid': valid,
                            'movements': movements,
                            'path': ga_path
                        })
                    else:
                        run_result.update({
                            'valid': False,
                            'movements': float('inf'),
                            'path': None
                        })

                    ga_results.append(run_result)

                # Calculate statistics for this GA configuration
                valid_runs = [r for r in ga_results if r['valid']]

                if valid_runs:
                    ga_times = [r['time'] for r in ga_results]
                    ga_movements = [r['movements'] for r in valid_runs] if valid_runs else [float('inf')]

                    avg_ga_time = np.mean(ga_times)
                    std_ga_time = np.std(ga_times)
                    avg_ga_movements = np.mean(ga_movements) if ga_movements else float('inf')
                    std_ga_movements = np.std(ga_movements) if len(ga_movements) > 1 else 0

                    # Find best GA solution
                    best_ga = min(valid_runs, key=lambda x: x['movements']) if valid_runs else None

                    if best_ga:
                        print(f"  Best Valid: {best_ga['valid']}, Movements: {best_ga['movements']}, "
                              f"Avg Movements: {avg_ga_movements:.2f} (±{std_ga_movements:.2f})")
                        print(f"  Avg Time: {avg_ga_time:.4f}s, Std: {std_ga_time:.4f}s")
                        print(f"  Success Rate: {len(valid_runs)}/{self.num_runs} ({len(valid_runs)/self.num_runs*100:.1f}%)")
                        print(f"  Best Path: {best_ga['path']}")

                        ga_config_result = {
                            'config_id': ga_idx + 1,
                            'settings': ga_setting,
                            'best_valid': best_ga['valid'],
                            'best_movements': best_ga['movements'],
                            'avg_movements': avg_ga_movements,
                            'min_movements': min(ga_movements),
                            'max_movements': max(ga_movements),
                            'std_movements': std_ga_movements,
                            'avg_time': avg_ga_time,
                            'min_time': min(ga_times),
                            'max_time': max(ga_times),
                            'std_time': std_ga_time,
                            'success_rate': len(valid_runs) / self.num_runs,
                            'best_path': best_ga['path'],
                            'all_runs': ga_results
                        }
                    else:
                        print(f"  No valid solutions found in {self.num_runs} runs")
                        ga_config_result = {
                            'config_id': ga_idx + 1,
                            'settings': ga_setting,
                            'best_valid': False,
                            'avg_time': avg_ga_time,
                            'min_time': min(ga_times),
                            'max_time': max(ga_times),
                            'std_time': std_ga_time,
                            'success_rate': 0,
                            'all_runs': ga_results
                        }
                else:
                    ga_times = [r['time'] for r in ga_results]
                    avg_ga_time = np.mean(ga_times)
                    std_ga_time = np.std(ga_times)

                    print(f"  No valid solutions found in {self.num_runs} runs")
                    ga_config_result = {
                        'config_id': ga_idx + 1,
                        'settings': ga_setting,
                        'best_valid': False,
                        'avg_time': avg_ga_time,
                        'min_time': min(ga_times),
                        'max_time': max(ga_times),
                        'std_time': std_ga_time,
                        'success_rate': 0,
                        'all_runs': ga_results
                    }

                test_results['methods']['genetic'].append(ga_config_result)

            self.results.append(test_results)

            # Print summary for this test
            self._print_test_summary(test_results)

        return self.results

    def _print_test_summary(self, test_results):
        """Print a summary of the test results"""
        test_id = test_results['test_id']
        cvrp_config = test_results['cvrp_config']

        print(f"\n{'-'*40} SUMMARY FOR TEST {test_id} {'-'*40}")
        print(f"CVRP: Cities={cvrp_config['CVRP_NUMBER_OF_CITIES']}, "
              f"Distance={cvrp_config['CVRP_DISTANCE_BETWEEN_CITIES']}, "
              f"Capacity={cvrp_config['CVRP_CAPACITY']}")

        # Greedy results
        greedy = test_results['methods']['greedy']
        print(f"Greedy: Valid={greedy['valid']}, Movements={greedy['movements']}, "
              f"Avg Time={greedy['avg_time']:.4f}s (±{greedy['std_time']:.4f}s)")

        # Random results
        random = test_results['methods']['random']
        print(f"Random: Best Valid={random['best_valid']}, Best Movements={random['best_movements']}, "
              f"Avg Movements={random['avg_movements']:.2f} (±{random['std_movements']:.2f}), "
              f"Avg Time={random['avg_time']:.4f}s (±{random['std_time']:.4f}s)")

        # Genetic algorithm results
        for ga_result in test_results['methods']['genetic']:
            config_id = ga_result['config_id']
            if ga_result['best_valid']:
                ga_settings = ga_result['settings']
                print(f"GA (Config {config_id}): Valid=True, Best Movements={ga_result['best_movements']}, "
                      f"Avg Movements={ga_result['avg_movements']:.2f} (±{ga_result['std_movements']:.2f}), "
                      f"Success Rate={ga_result['success_rate']*100:.1f}%, "
                      f"Avg Time={ga_result['avg_time']:.4f}s (±{ga_result['std_time']:.4f}s)")
                print(f"  Parameters: Pop={ga_settings['GA_POPULATION_SIZE']}, "
                      f"Gen={ga_settings['GA_GENERATIONS']}, "
                      f"Cx={ga_settings['GA_CROSSOVER_RATE']}, "
                      f"Mut={ga_settings['GA_MUTATION_RATE']}, "
                      f"Tour={ga_settings['GA_TOURNAMENT_SIZE']}, "
                      f"Elite={ga_settings['GA_ELITISM_SIZE']}")
            else:
                print(f"GA (Config {config_id}): No valid solution found in {self.num_runs} runs, "
                      f"Avg Time={ga_result['avg_time']:.4f}s (±{ga_result['std_time']:.4f}s)")

    def export_to_csv(self, output_dir="results"):
        """Export all results to CSV files"""
        if not self.results:
            print("No results to export. Run tests first.")
            return

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Generate timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. Create detailed results dataframe
        detailed_rows = []

        for test_result in self.results:
            test_id = test_result['test_id']
            cities = test_result['cvrp_config']['CVRP_NUMBER_OF_CITIES']
            distance = test_result['cvrp_config']['CVRP_DISTANCE_BETWEEN_CITIES']
            capacity = test_result['cvrp_config']['CVRP_CAPACITY']

            # Greedy results
            greedy = test_result['methods']['greedy']
            detailed_rows.append({
                'test_id': test_id,
                'algorithm': 'Greedy',
                'config': 'N/A',
                'cities': cities,
                'distance': distance,
                'capacity': capacity,
                'valid': greedy['valid'],
                'movements': greedy['movements'],
                'avg_time': greedy['avg_time'],
                'min_time': greedy['min_time'],
                'max_time': greedy['max_time'],
                'std_time': greedy['std_time'],
                'success_rate': 1.0 if greedy['valid'] else 0.0,
                'pop_size': 'N/A',
                'generations': 'N/A',
                'crossover_rate': 'N/A',
                'mutation_rate': 'N/A',
                'tournament_size': 'N/A',
                'elitism_size': 'N/A'
            })

            # Random results
            random = test_result['methods']['random']
            detailed_rows.append({
                'test_id': test_id,
                'algorithm': 'Random',
                'config': 'N/A',
                'cities': cities,
                'distance': distance,
                'capacity': capacity,
                'valid': random['best_valid'],
                'movements': random['best_movements'],
                'avg_movements': random['avg_movements'],
                'min_movements': random['min_movements'],
                'max_movements': random['max_movements'],
                'std_movements': random['std_movements'],
                'avg_time': random['avg_time'],
                'min_time': random['min_time'],
                'max_time': random['max_time'],
                'std_time': random['std_time'],
                'success_rate': sum(1 for r in random['all_runs'] if r['valid']) / len(random['all_runs']),
                'pop_size': 'N/A',
                'generations': 'N/A',
                'crossover_rate': 'N/A',
                'mutation_rate': 'N/A',
                'tournament_size': 'N/A',
                'elitism_size': 'N/A'
            })

            # GA results
            for ga_result in test_result['methods']['genetic']:
                config_id = ga_result['config_id']
                settings = ga_result['settings']

                ga_row = {
                    'test_id': test_id,
                    'algorithm': 'Genetic Algorithm',
                    'config': f"Config {config_id}",
                    'cities': cities,
                    'distance': distance,
                    'capacity': capacity,
                    'avg_time': ga_result['avg_time'],
                    'min_time': ga_result['min_time'],
                    'max_time': ga_result['max_time'],
                    'std_time': ga_result['std_time'],
                    'success_rate': ga_result['success_rate'],
                    'pop_size': settings['GA_POPULATION_SIZE'],
                    'generations': settings['GA_GENERATIONS'],
                    'crossover_rate': settings['GA_CROSSOVER_RATE'],
                    'mutation_rate': settings['GA_MUTATION_RATE'],
                    'tournament_size': settings['GA_TOURNAMENT_SIZE'],
                    'elitism_size': settings['GA_ELITISM_SIZE']
                }

                if ga_result['best_valid']:
                    ga_row.update({
                        'valid': ga_result['best_valid'],
                        'movements': ga_result['best_movements'],
                        'avg_movements': ga_result['avg_movements'],
                        'min_movements': ga_result['min_movements'],
                        'max_movements': ga_result['max_movements'],
                        'std_movements': ga_result['std_movements']
                    })
                else:
                    ga_row.update({
                        'valid': False,
                        'movements': 'N/A',
                        'avg_movements': 'N/A',
                        'min_movements': 'N/A',
                        'max_movements': 'N/A',
                        'std_movements': 'N/A'
                    })

                detailed_rows.append(ga_row)

        detailed_df = pd.DataFrame(detailed_rows)
        detailed_file = f"{output_dir}/detailed_results_{timestamp}.csv"
        detailed_df.to_csv(detailed_file, index=False)
        print(f"Detailed results saved to: {detailed_file}")

        # 2. Create comparison table
        comparison_rows = []

        for test_result in self.results:
            test_id = test_result['test_id']
            cities = test_result['cvrp_config']['CVRP_NUMBER_OF_CITIES']
            distance = test_result['cvrp_config']['CVRP_DISTANCE_BETWEEN_CITIES']
            capacity = test_result['cvrp_config']['CVRP_CAPACITY']

            # Greedy as baseline
            greedy = test_result['methods']['greedy']
            greedy_movements = greedy['movements']

            # Add greedy row
            comparison_rows.append({
                'test_id': test_id,
                'algorithm': 'Greedy Algorithm',
                'cities': cities,
                'distance': distance,
                'capacity': capacity,
                'best_movements': greedy_movements,
                'avg_time': greedy['avg_time'],
                'std_time': greedy['std_time'],
                'success_rate': '100%',
                'movement_vs_greedy': '0%',
                'time_efficiency': 1 / greedy['avg_time'] if greedy['avg_time'] > 0 else float('inf')
            })

            # Add random row
            random = test_result['methods']['random']
            random_movement_diff = ((random['best_movements'] - greedy_movements) / greedy_movements) * 100

            comparison_rows.append({
                'test_id': test_id,
                'algorithm': 'Random Algorithm',
                'cities': cities,
                'distance': distance,
                'capacity': capacity,
                'best_movements': random['best_movements'],
                'avg_time': random['avg_time'],
                'std_time': random['std_time'],
                'success_rate': f"{sum(1 for r in random['all_runs'] if r['valid']) / len(random['all_runs']) * 100:.1f}%",
                'movement_vs_greedy': f"{random_movement_diff:.1f}%",
                'time_efficiency': 1 / random['avg_time'] if random['avg_time'] > 0 else float('inf')
            })

            # Add GA rows
            for ga_result in test_result['methods']['genetic']:
                config_id = ga_result['config_id']
                settings = ga_result['settings']

                if ga_result['best_valid']:
                    ga_movement_diff = ((ga_result['best_movements'] - greedy_movements) / greedy_movements) * 100

                    params = f"Pop={settings['GA_POPULATION_SIZE']}, Gen={settings['GA_GENERATIONS']}, " \
                             f"Cx={settings['GA_CROSSOVER_RATE']}, Mut={settings['GA_MUTATION_RATE']}, " \
                             f"Tour={settings['GA_TOURNAMENT_SIZE']}, Elite={settings['GA_ELITISM_SIZE']}"

                    comparison_rows.append({
                        'test_id': test_id,
                        'algorithm': f"GA (Config {config_id})",
                        'cities': cities,
                        'distance': distance,
                        'capacity': capacity,
                        'parameters': params,
                        'best_movements': ga_result['best_movements'],
                        'avg_time': ga_result['avg_time'],
                        'std_time': ga_result['std_time'],
                        'success_rate': f"{ga_result['success_rate'] * 100:.1f}%",
                        'movement_vs_greedy': f"{ga_movement_diff:.1f}%",
                        'time_efficiency': 1 / ga_result['avg_time'] if ga_result['avg_time'] > 0 else float('inf')
                    })
                else:
                    params = f"Pop={settings['GA_POPULATION_SIZE']}, Gen={settings['GA_GENERATIONS']}, " \
                             f"Cx={settings['GA_CROSSOVER_RATE']}, Mut={settings['GA_MUTATION_RATE']}, " \
                             f"Tour={settings['GA_TOURNAMENT_SIZE']}, Elite={settings['GA_ELITISM_SIZE']}"

                    comparison_rows.append({
                        'test_id': test_id,
                        'algorithm': f"GA (Config {config_id})",
                        'cities': cities,
                        'distance': distance,
                        'capacity': capacity,
                        'parameters': params,
                        'best_movements': 'N/A',
                        'avg_time': ga_result['avg_time'],
                        'std_time': ga_result['std_time'],
                        'success_rate': '0.0%',
                        'movement_vs_greedy': 'N/A',
                        'time_efficiency': 0
                    })

        comparison_df = pd.DataFrame(comparison_rows)
        comparison_file = f"{output_dir}/algorithm_comparison_{timestamp}.csv"
        comparison_df.to_csv(comparison_file, index=False)
        print(f"Algorithm comparison saved to: {comparison_file}")

        # 3. Create summary statistics
        summary_rows = []

        for test_result in self.results:
            test_id = test_result['test_id']
            cities = test_result['cvrp_config']['CVRP_NUMBER_OF_CITIES']

            # Find best algorithm for this test
            greedy_movements = test_result['methods']['greedy']['movements']
            random_movements = test_result['methods']['random']['best_movements']

            valid_ga_results = [ga for ga in test_result['methods']['genetic'] if ga['best_valid']]
            ga_movements = [ga['best_movements'] for ga in valid_ga_results] if valid_ga_results else []

            all_movements = [greedy_movements, random_movements] + ga_movements
            best_movement = min(all_movements)

            if best_movement == greedy_movements:
                best_algorithm = "Greedy Algorithm"
            elif best_movement == random_movements:
                best_algorithm = "Random Algorithm"
            else:
                for ga in valid_ga_results:
                    if ga['best_movements'] == best_movement:
                        best_algorithm = f"GA (Config {ga['config_id']})"
                        break
                else:
                    best_algorithm = "Unknown"

            # Calculate improvement percentage
            improvement = ((greedy_movements - best_movement) / greedy_movements) * 100

            summary_rows.append({
                'test_id': test_id,
                'cities': cities,
                'best_algorithm': best_algorithm,
                'best_movements': best_movement,
                'greedy_movements': greedy_movements,
                'improvement_vs_greedy': f"{improvement:.2f}%" if improvement > 0 else "0.00%",
                'ga_success': "Yes" if any(ga['best_valid'] for ga in test_result['methods']['genetic']) else "No"
            })

        summary_df = pd.DataFrame(summary_rows)
        summary_file = f"{output_dir}/results_summary_{timestamp}.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"Results summary saved to: {summary_file}")

        return {
            'detailed': detailed_file,
            'comparison': comparison_file,
            'summary': summary_file
        }

if __name__ == "__main__":
    # Run with 10 repetitions for statistical significance
    runner = TestRunner(num_runs=10)
    results = runner.run_tests()
    runner.export_to_csv()