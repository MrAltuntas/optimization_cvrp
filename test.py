import config
import cvrp
import greedy_cvrp
import random_cvrp
import genetic_cvrp
import tabu_cvrp
import time
import pandas as pd
import os
import numpy as np
from datetime import datetime

class TestRunner:
    def __init__(self, test_configs=None):
        """
        Initialize the test runner with simplified parameters.
        """
        self.test_configs = test_configs if test_configs else config.RUN_TEST
        self.num_runs = config.RUN_TEST_COUNT  # Get test count from config
        self.results = []

    def run_tests(self):
        """Run tests with focus on algorithm comparison"""
        for test_idx, test_config in enumerate(self.test_configs):
            print(f"\n\n{'='*40} TEST {test_idx+1} {'='*40}")
            print(f"CVRP: Cities={test_config['CVRP_NUMBER_OF_CITIES']}, "
                  f"Distance={test_config['CVRP_DISTANCE_BETWEEN_CITIES']}, "
                  f"Capacity={test_config['CVRP_CAPACITY']}")

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

            # Run Greedy algorithm since greedy will always give the same result we won't run it for many times
            print("\n1. Greedy Algorithm:")
            greedy_solver = greedy_cvrp.GreedyCVRP(cvrp_problem)
            start_time = time.time()
            greedy_path = greedy_solver.find_path()
            greedy_time = time.time() - start_time
            greedy_movements, greedy_valid = cvrp_problem.is_valid_path(greedy_path)

            print(f"Valid: {greedy_valid}, Movements: {greedy_movements}, Time: {greedy_time:.4f}s")

            test_results['methods']['greedy'] = {
                'valid': greedy_valid,
                'movements': greedy_movements,
                'time': greedy_time,
                'path': greedy_path
            }

            # Run Random algorithm
            print("\n2. Random Algorithm:")
            random_solver = random_cvrp.RandomCVRP(cvrp_problem)
            random_results = []

            for run in range(self.num_runs):
                start_time = time.time()
                random_path = random_solver.random_path()
                run_time = time.time() - start_time
                movements, valid = cvrp_problem.is_valid_path(random_path)
                random_results.append({
                    'valid': valid,
                    'movements': movements,
                    'time': run_time,
                    'path': random_path
                })

            # Find best random solution
            best_random = min(random_results, key=lambda x: x['movements'])
            avg_random_time = np.mean([r['time'] for r in random_results])

            print(f"Best Valid: {best_random['valid']}, Best Movements: {best_random['movements']}, "
                  f"Avg Time: {avg_random_time:.4f}s")

            test_results['methods']['random'] = {
                'best_valid': best_random['valid'],
                'best_movements': best_random['movements'],
                'avg_time': avg_random_time,
                'all_runs': random_results
            }

            # Run Genetic algorithm with different settings
            test_results['methods']['genetic'] = []

            for ga_idx, ga_setting in enumerate(test_config['GA_SETTINGS']):
                ga_name = f"GA Config {ga_idx+1}"
                print(f"\n3. {ga_name}:")
                print(f"  Pop={ga_setting['GA_POPULATION_SIZE']}, "
                      f"Gen={ga_setting['GA_GENERATIONS']}, "
                      f"Cx={ga_setting['GA_CROSSOVER_RATE']}, "
                      f"Mut={ga_setting['GA_MUTATION_RATE']},  "
                      f"Eva_Imp={ga_setting['GA_IMPROVED_EVALUATION']}")

                ga_results = []

                for run in range(self.num_runs):
                    print(f"  Run {run+1}/{self.num_runs}", end="\r")

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

                    if ga_path is not None:
                        movements, valid = cvrp_problem.is_valid_path(ga_path)
                        ga_results.append({
                            'valid': valid,
                            'movements': movements,
                            'time': run_time,
                            'path': ga_path
                        })
                    else:
                        ga_results.append({
                            'valid': False,
                            'movements': float('inf'),
                            'time': run_time,
                            'path': None
                        })

                print(" " * 30, end="\r")  # Clear the progress line

                # Calculate statistics
                valid_runs = [r for r in ga_results if r['valid']]

                if valid_runs:
                    best_ga = min(valid_runs, key=lambda x: x['movements'])
                    avg_ga_time = np.mean([r['time'] for r in ga_results])

                    print(f"  Valid: {len(valid_runs)}/{self.num_runs}, "
                          f"Best Movements: {best_ga['movements']}, "
                          f"Avg Time: {avg_ga_time:.4f}s")

                    ga_config_result = {
                        'config_id': ga_idx + 1,
                        'settings': ga_setting,
                        'best_valid': best_ga['valid'],
                        'best_movements': best_ga['movements'],
                        'avg_time': avg_ga_time,
                        'success_rate': len(valid_runs) / self.num_runs,
                        'all_runs': ga_results
                    }
                else:
                    avg_ga_time = np.mean([r['time'] for r in ga_results])
                    print(f"  No valid solutions found, Avg Time: {avg_ga_time:.4f}s")

                    ga_config_result = {
                        'config_id': ga_idx + 1,
                        'settings': ga_setting,
                        'best_valid': False,
                        'avg_time': avg_ga_time,
                        'success_rate': 0,
                        'all_runs': ga_results
                    }

                test_results['methods']['genetic'].append(ga_config_result)

            # Run Tabu Search algorithm
            test_results['methods']['tabu_search'] = []

            for ts_idx, ts_setting in enumerate(test_config['TS_SETTINGS']):
                ts_name = f"TS Config {ts_idx+1}"
                print(f"\n4. {ts_name}:")
                print(f"  Iterations={ts_setting['TS_MAX_ITERATIONS']}, "
                      f"TabuSize={ts_setting['TS_TABU_SIZE']}, "
                      f"NeighSize={ts_setting['TS_NEIGHBORHOOD_SIZE']}")

                ts_results = []

                for run in range(self.num_runs):
                    print(f"  Run {run+1}/{self.num_runs}", end="\r")

                    ts_solver = tabu_cvrp.TabuSearchCVRP(
                        cvrp_problem,
                        max_iterations=ts_setting['TS_MAX_ITERATIONS'],
                        tabu_size=ts_setting['TS_TABU_SIZE'],
                        neighborhood_size=ts_setting['TS_NEIGHBORHOOD_SIZE']
                    )

                    start_time = time.time()
                    ts_path = ts_solver.find_path()
                    run_time = time.time() - start_time

                    if ts_path is not None:
                        movements, valid = cvrp_problem.is_valid_path(ts_path)
                        ts_results.append({
                            'valid': valid,
                            'movements': movements,
                            'time': run_time,
                            'path': ts_path
                        })
                    else:
                        ts_results.append({
                            'valid': False,
                            'movements': float('inf'),
                            'time': run_time,
                            'path': None
                        })

                print(" " * 30, end="\r")  # Clear the progress line

                # Calculate statistics
                valid_runs = [r for r in ts_results if r['valid']]

                if valid_runs:
                    best_ts = min(valid_runs, key=lambda x: x['movements'])
                    avg_ts_time = np.mean([r['time'] for r in ts_results])

                    print(f"  Valid: {len(valid_runs)}/{self.num_runs}, "
                          f"Best Movements: {best_ts['movements']}, "
                          f"Avg Time: {avg_ts_time:.4f}s")

                    ts_config_result = {
                        'config_id': ts_idx + 1,
                        'settings': ts_setting,
                        'best_valid': best_ts['valid'],
                        'best_movements': best_ts['movements'],
                        'avg_time': avg_ts_time,
                        'success_rate': len(valid_runs) / self.num_runs,
                        'all_runs': ts_results
                    }
                else:
                    avg_ts_time = np.mean([r['time'] for r in ts_results])
                    print(f"  No valid solutions found, Avg Time: {avg_ts_time:.4f}s")

                    ts_config_result = {
                        'config_id': ts_idx + 1,
                        'settings': ts_setting,
                        'best_valid': False,
                        'avg_time': avg_ts_time,
                        'success_rate': 0,
                        'all_runs': ts_results
                    }

                test_results['methods']['tabu_search'].append(ts_config_result)

            self.results.append(test_results)

            # Print comparison summary
            self._print_comparison_summary(test_results)

        return self.results

    def _print_comparison_summary(self, test_results):
        """Print a focused comparison summary of algorithm performance"""
        print(f"\n{'-'*40} ALGORITHM COMPARISON {'-'*40}")

        # Get greedy as baseline
        greedy = test_results['methods']['greedy']
        greedy_movements = greedy['movements']

        print(f"Greedy: Movements={greedy_movements}, Time={greedy['time']:.4f}s")

        # Random algorithm
        random = test_results['methods']['random']

        print(f"Random: Movements={random['best_movements']}, "
              f"Time={random['avg_time']:.4f}s")

        # GA algorithms
        for ga in test_results['methods']['genetic']:
            config_id = ga['config_id']

            if ga['best_valid']:
                print(f"GA Config {config_id}: Movements={ga['best_movements']}, "
                      f"Time={ga['avg_time']:.4f}s, Success={ga['success_rate']*100:.1f}%")
            else:
                print(f"GA Config {config_id}: No valid solution, Time={ga['avg_time']:.4f}s")

        # TS algorithms
        for ts in test_results['methods']['tabu_search']:
            config_id = ts['config_id']

            if ts['best_valid']:
                print(f"TS Config {config_id}: Movements={ts['best_movements']}, "
                      f"Time={ts['avg_time']:.4f}s, Success={ts['success_rate']*100:.1f}%")
            else:
                print(f"TS Config {config_id}: No valid solution, Time={ts['avg_time']:.4f}s")

    def export_comparison_csv(self, output_dir="results"):
        """Export only the algorithm comparison table to CSV"""
        if not self.results:
            print("No results to export. Run tests first.")
            return

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create comparison table
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
                'time': greedy['time'],
                'success_rate': '100%',
                'parameters': 'N/A'
            })

            # Add random row
            random = test_result['methods']['random']

            comparison_rows.append({
                'test_id': test_id,
                'algorithm': 'Random Algorithm',
                'cities': cities,
                'distance': distance,
                'capacity': capacity,
                'best_movements': random['best_movements'],
                'time': random['avg_time'],
                'success_rate': f"{sum(1 for r in random['all_runs'] if r['valid']) / len(random['all_runs']) * 100:.1f}%",
                'parameters': 'N/A'
            })

            # Add GA rows
            for ga_result in test_result['methods']['genetic']:
                config_id = ga_result['config_id']
                settings = ga_result['settings']

                if ga_result['best_valid']:

                    params = f"Pop={settings['GA_POPULATION_SIZE']}, " \
                             f"Gen={settings['GA_GENERATIONS']}, " \
                             f"Cx={settings['GA_CROSSOVER_RATE']}, " \
                             f"Mut={settings['GA_MUTATION_RATE']}, " \
                             f"Tour={settings['GA_TOURNAMENT_SIZE']}, " \
                             f"Elite={settings['GA_ELITISM_SIZE']}, " \
                             f"Eval_Imp={settings['GA_IMPROVED_EVALUATION']}"

                    comparison_rows.append({
                        'test_id': test_id,
                        'algorithm': f"GA (Config {config_id})",
                        'cities': cities,
                        'distance': distance,
                        'capacity': capacity,
                        'best_movements': ga_result['best_movements'],
                        'time': ga_result['avg_time'],
                        'success_rate': f"{ga_result['success_rate'] * 100:.1f}%",
                        'parameters': params
                    })
                else:
                    params = f"Pop={settings['GA_POPULATION_SIZE']}, " \
                             f"Gen={settings['GA_GENERATIONS']}, " \
                             f"Cx={settings['GA_CROSSOVER_RATE']}, " \
                             f"Mut={settings['GA_MUTATION_RATE']}, " \
                             f"Tour={settings['GA_TOURNAMENT_SIZE']}, " \
                             f"Elite={settings['GA_ELITISM_SIZE']}, " \
                             f"Eval_Imp={settings['GA_IMPROVED_EVALUATION']}"

                    comparison_rows.append({
                        'test_id': test_id,
                        'algorithm': f"GA (Config {config_id})",
                        'cities': cities,
                        'distance': distance,
                        'capacity': capacity,
                        'best_movements': 'N/A',
                        'time': ga_result['avg_time'],
                        'success_rate': '0.0%',
                        'parameters': params
                    })

            # Add TS rows
            for ts_result in test_result['methods']['tabu_search']:
                config_id = ts_result['config_id']
                settings = ts_result['settings']

                if ts_result['best_valid']:

                    params = f"Iter={settings['TS_MAX_ITERATIONS']}, " \
                             f"TabuSize={settings['TS_TABU_SIZE']}, " \
                             f"NeighSize={settings['TS_NEIGHBORHOOD_SIZE']}"

                    comparison_rows.append({
                        'test_id': test_id,
                        'algorithm': f"TS (Config {config_id})",
                        'cities': cities,
                        'distance': distance,
                        'capacity': capacity,
                        'best_movements': ts_result['best_movements'],
                        'time': ts_result['avg_time'],
                        'success_rate': f"{ts_result['success_rate'] * 100:.1f}%",
                        'parameters': params
                    })
                else:
                    params = f"Iter={settings['TS_MAX_ITERATIONS']}, " \
                             f"TabuSize={settings['TS_TABU_SIZE']}, " \
                             f"NeighSize={settings['TS_NEIGHBORHOOD_SIZE']}"

                    comparison_rows.append({
                        'test_id': test_id,
                        'algorithm': f"TS (Config {config_id})",
                        'cities': cities,
                        'distance': distance,
                        'capacity': capacity,
                        'best_movements': 'N/A',
                        'time': ts_result['avg_time'],
                        'success_rate': '0.0%',
                        'parameters': params
                    })

        comparison_df = pd.DataFrame(comparison_rows)
        comparison_file = f"{output_dir}/algorithm_comparison_{timestamp}.csv"
        comparison_df.to_csv(comparison_file, index=False)
        print(f"Algorithm comparison saved to: {comparison_file}")

        return comparison_file

if __name__ == "__main__":
    runner = TestRunner()
    results = runner.run_tests()
    runner.export_comparison_csv()