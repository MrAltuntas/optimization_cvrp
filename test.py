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
from visualization import VisualizationTools

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
            print(f"\n\n{'='*40} TEST {test_idx+1} TEST NAME {test_config['NAME']} {'='*40}")
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
                'test_name': test_config['NAME'],
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

            print(f"Valid: {greedy_valid}, Movements: {greedy_movements}, Best Path: {greedy_path}, Time: {greedy_time:.4f}s")

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

            # Calculate statistics for random algorithm
            valid_random_runs = [r for r in random_results if r['valid']]
            if valid_random_runs:
                best_random = min(valid_random_runs, key=lambda x: x['movements'])
                worst_random = max(valid_random_runs, key=lambda x: x['movements'])
                avg_random_movements = np.mean([r['movements'] for r in valid_random_runs])
                std_random_movements = np.std([r['movements'] for r in valid_random_runs])
                avg_random_time = np.mean([r['time'] for r in random_results])

                print(f"Best Valid: {best_random['valid']}, "
                      f"Best Movements: {best_random['movements']}, "
                      f"Worst Movements: {worst_random['movements']}, "
                      f"Avg Movements: {avg_random_movements:.2f}, "
                      f"Std Movements: {std_random_movements:.2f}, "
                      f"Best Path: {best_random['path']}, "
                      f"Avg Time: {avg_random_time:.4f}s")

                test_results['methods']['random'] = {
                    'best_valid': best_random['valid'],
                    'best_path': best_random['path'],
                    'best_movements': best_random['movements'],
                    'worst_movements': worst_random['movements'],
                    'avg_movements': avg_random_movements,
                    'std_movements': std_random_movements,
                    'avg_time': avg_random_time,
                    'all_runs': random_results
                }
            else:
                avg_random_time = np.mean([r['time'] for r in random_results])
                print(f"No valid solutions found, Avg Time: {avg_random_time:.4f}s")

                test_results['methods']['random'] = {
                    'best_valid': False,
                    'best_path': None,
                    'best_movements': float('inf'),
                    'worst_movements': float('inf'),
                    'avg_movements': float('inf'),
                    'std_movements': 0,
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
                    # Update this line to handle the additional return value:
                    ga_path, solution_generation_count, generation_fitness = ga_solver.solve()
                    run_time = time.time() - start_time

                    if ga_path is not None:
                        movements, valid = cvrp_problem.is_valid_path(ga_path)
                        ga_results.append({
                            'valid': valid,
                            'movements': movements,
                            'time': run_time,
                            'path': ga_path,
                            'solution_generation_count': solution_generation_count,
                            'generation_fitness': generation_fitness,  # Add this line
                        })
                    else:
                        ga_results.append({
                            'valid': False,
                            'movements': float('inf'),
                            'time': run_time,
                            'path': None,
                            'solution_generation_count': solution_generation_count,
                            'generation_fitness': generation_fitness,  # Add this line
                        })

                print(" " * 30, end="\r")  # Clear the progress line

                # Calculate statistics
                valid_runs = [r for r in ga_results if r['valid']]

                if valid_runs:
                    best_ga = min(valid_runs, key=lambda x: x['movements'])
                    worst_ga = max(valid_runs, key=lambda x: x['movements'])
                    best_gen_count = min(valid_runs, key=lambda x: x['solution_generation_count'])
                    worst_gen_count = max(valid_runs, key=lambda x: x['solution_generation_count'])
                    avg_gen_count = np.mean([r['solution_generation_count'] for r in valid_runs])
                    std_gen_count = np.std([r['solution_generation_count'] for r in valid_runs])
                    avg_ga_movements = np.mean([r['movements'] for r in valid_runs])
                    std_ga_movements = np.std([r['movements'] for r in valid_runs])
                    avg_ga_time = np.mean([r['time'] for r in ga_results])

                    print(f"  Valid: {len(valid_runs)}/{self.num_runs}, "
                          f"Best Movements: {best_ga['movements']}, "
                          f"Worst Movements: {worst_ga['movements']}, "
                          f"Avg Movements: {avg_ga_movements:.2f}, "
                          f"Std Movements: {std_ga_movements:.2f}, "
                          f"Best Path: {best_ga['path']}, "
                          f"Best Generation Count: {best_gen_count['solution_generation_count']}, "
                          f"Worst Generation Count: {worst_gen_count['solution_generation_count']}, "
                          f"Avg Generation Count: {avg_gen_count}, "
                          f"Std Generation Count: {std_gen_count}, "
                          f"Avg Time: {avg_ga_time:.4f}s")

                    ga_config_result = {
                        'config_id': ga_idx + 1,
                        'best_path': best_ga['path'],
                        'settings': ga_setting,
                        'best_valid': best_ga['valid'],
                        'best_movements': best_ga['movements'],
                        'worst_movements': worst_ga['movements'],
                        'avg_movements': avg_ga_movements,
                        'std_movements': std_ga_movements,
                        'best_gen_count': best_gen_count['solution_generation_count'],
                        'worst_gen_count': worst_gen_count['solution_generation_count'],
                        'avg_gen_count': avg_gen_count,
                        'std_gen_count': std_gen_count,
                        'avg_time': avg_ga_time,
                        'success_rate': len(valid_runs) / self.num_runs,
                        'all_runs': ga_results
                    }
                else:
                    avg_ga_time = np.mean([r['time'] for r in ga_results])
                    print(f"  No valid solutions found, Avg Time: {avg_ga_time:.4f}s")

                    ga_config_result = {
                        'config_id': ga_idx + 1,
                        'best_path': None,
                        'settings': ga_setting,
                        'best_valid': False,
                        'best_movements': float('inf'),
                        'worst_movements': float('inf'),
                        'avg_movements': float('inf'),
                        'std_movements': 0,
                        'best_gen_count': 0,
                        'worst_gen_count': 0,
                        'avg_gen_count': 0,
                        'std_gen_count': 0,
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
                    worst_ts = max(valid_runs, key=lambda x: x['movements'])
                    avg_ts_movements = np.mean([r['movements'] for r in valid_runs])
                    std_ts_movements = np.std([r['movements'] for r in valid_runs])
                    avg_ts_time = np.mean([r['time'] for r in ts_results])

                    print(f"  Valid: {len(valid_runs)}/{self.num_runs}, "
                          f"Best Movements: {best_ts['movements']}, "
                          f"Worst Movements: {worst_ts['movements']}, "
                          f"Avg Movements: {avg_ts_movements:.2f}, "
                          f"Std Movements: {std_ts_movements:.2f}, "
                          f"Best Path: {best_ts['path']}, "
                          f"Avg Time: {avg_ts_time:.4f}s")

                    ts_config_result = {
                        'config_id': ts_idx + 1,
                        'settings': ts_setting,
                        'best_path': best_ts['path'],
                        'best_valid': best_ts['valid'],
                        'best_movements': best_ts['movements'],
                        'worst_movements': worst_ts['movements'],
                        'avg_movements': avg_ts_movements,
                        'std_movements': std_ts_movements,
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
                        'best_path': None,
                        'best_valid': False,
                        'best_movements': float('inf'),
                        'worst_movements': float('inf'),
                        'avg_movements': float('inf'),
                        'std_movements': 0,
                        'avg_time': avg_ts_time,
                        'success_rate': 0,
                        'all_runs': ts_results
                    }

                test_results['methods']['tabu_search'].append(ts_config_result)

            self.results.append(test_results)

            # Print comparison summary
            self._print_comparison_summary(test_results)

            self.export_fitness_data(test_results)

        return self.results

    def _print_comparison_summary(self, test_results):
        """Print a focused comparison summary of algorithm performance"""
        print(f"\n{'-'*40} ALGORITHM COMPARISON {'-'*40}")

        # Get greedy as baseline
        greedy = test_results['methods']['greedy']
        greedy_movements = greedy['movements']
        greedy_path = greedy['path']

        print(f"Greedy: Movements={greedy_movements}, Best path={greedy_path}, Time={greedy['time']:.4f}s")

        # Random algorithm
        random = test_results['methods']['random']

        if random['best_valid']:
            print(f"Random: Best={random['best_movements']}, "
                  f"Worst={random['worst_movements']}, "
                  f"Avg={random['avg_movements']:.2f}, "
                  f"Std={random['std_movements']:.2f}, "
                  f"Best path={random['best_path']}, "
                  f"Time={random['avg_time']:.4f}s")
        else:
            print(f"Random: No valid solution, Time={random['avg_time']:.4f}s")

        # GA algorithms
        for ga in test_results['methods']['genetic']:
            config_id = ga['config_id']

            if ga['best_valid']:
                print(f"GA Config {config_id}: Best={ga['best_movements']}, "
                      f"Worst={ga['worst_movements']}, "
                      f"Avg={ga['avg_movements']:.2f}, "
                      f"Std={ga['std_movements']:.2f}, "
                      f"Best path={ga['best_path']}, "
                      f"Best Generation Count={ga['best_gen_count']}, "
                      f"Worst Generation Count={ga['worst_gen_count']}, "
                      f"Avg Generation Count={ga['avg_gen_count']}, "
                      f"Std Generation Count={ga['std_gen_count']}, "
                      f"Time={ga['avg_time']:.4f}s, "
                      f"Success={ga['success_rate']*100:.1f}%")
            else:
                print(f"GA Config {config_id}: No valid solution, Time={ga['avg_time']:.4f}s")

        # TS algorithms
        for ts in test_results['methods']['tabu_search']:
            config_id = ts['config_id']

            if ts['best_valid']:
                print(f"TS Config {config_id}: Best={ts['best_movements']}, "
                      f"Worst={ts['worst_movements']}, "
                      f"Avg={ts['avg_movements']:.2f}, "
                      f"Std={ts['std_movements']:.2f}, "
                      f"Best path={ts['best_path']}, "
                      f"Time={ts['avg_time']:.4f}s, "
                      f"Success={ts['success_rate']*100:.1f}%")
            else:
                print(f"TS Config {config_id}: No valid solution, Time={ts['avg_time']:.4f}s")

    def export_fitness_data(self, test_results, output_dir="results"):
        """
        Export the fitness data for each GA configuration to CSV files
        for later visualization.
        """
        import os
        import pandas as pd
        import numpy as np

        os.makedirs(output_dir, exist_ok=True)

        test_id = test_results['test_id']

        # Process genetic algorithm results
        for config_idx, ga_config in enumerate(test_results['methods']['genetic']):
            config_id = ga_config['config_id']

            # Collect fitness data from all valid runs
            all_fitness_data = {}

            for run in ga_config['all_runs']:
                if 'generation_fitness' in run and run['valid']:
                    for gen, fitness in run['generation_fitness']:
                        if gen not in all_fitness_data:
                            all_fitness_data[gen] = []
                        all_fitness_data[gen].append(fitness)

            # If we have fitness data to save
            if all_fitness_data:
                # Convert to dataframe
                generations = sorted(all_fitness_data.keys())
                avg_fitness = [np.mean(all_fitness_data[gen]) for gen in generations]
                std_fitness = [np.std(all_fitness_data[gen]) for gen in generations]
                min_fitness = [min(all_fitness_data[gen]) for gen in generations]
                max_fitness = [max(all_fitness_data[gen]) for gen in generations]

                data = {
                    'generation': generations,
                    'avg_fitness': avg_fitness,
                    'std_fitness': std_fitness,
                    'min_fitness': min_fitness,
                    'max_fitness': max_fitness
                }

                fitness_df = pd.DataFrame(data)

                # Save to file
                filename = f"{output_dir}/fitness_data_test{test_id}_config{config_id}.csv"
                fitness_df.to_csv(filename, index=False)
                print(f"Fitness data for GA Config {config_id} saved to: {filename}")

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
            test_name = test_result['test_name']
            cities = test_result['cvrp_config']['CVRP_NUMBER_OF_CITIES']
            distance = test_result['cvrp_config']['CVRP_DISTANCE_BETWEEN_CITIES']
            capacity = test_result['cvrp_config']['CVRP_CAPACITY']

            # Greedy as baseline
            greedy = test_result['methods']['greedy']
            greedy_movements = greedy['movements']

            # Add greedy row
            comparison_rows.append({
                'test_id': test_id,
                'test_name': test_name,
                'algorithm': 'Greedy Algorithm',
                'cities': cities,
                'distance': distance,
                'capacity': capacity,
                'best_movements': greedy_movements,
                'worst_movements': 'N/A',  # Deterministic, no worst
                'avg_movements': greedy_movements,  # Same as best for greedy
                'std_movements': 0,  # No variation for greedy
                'time': greedy['time'],
                'best_gen_count': 'N/A',
                'worst_gen_count': 'N/A',
                'avg_gen_count': 'N/A',
                'std_gen_count': 'N/A',
                'best_path': greedy['path'],
                'success_rate': '100%',
                'parameters': 'N/A'
            })

            # Add random row
            random = test_result['methods']['random']

            if random['best_valid']:
                comparison_rows.append({
                    'test_id': test_id,
                    'test_name': test_name,
                    'algorithm': 'Random Algorithm',
                    'cities': cities,
                    'distance': distance,
                    'capacity': capacity,
                    'best_movements': random['best_movements'],
                    'worst_movements': random['worst_movements'],
                    'avg_movements': random['avg_movements'],
                    'std_movements': random['std_movements'],
                    'best_gen_count': 'N/A',
                    'worst_gen_count': 'N/A',
                    'avg_gen_count': 'N/A',
                    'std_gen_count': 'N/A',
                    'best_path': random['best_path'],
                    'time': random['avg_time'],
                    'success_rate': f"{sum(1 for r in random['all_runs'] if r['valid']) / len(random['all_runs']) * 100:.1f}%",
                    'parameters': 'N/A'
                })
            else:
                comparison_rows.append({
                    'test_id': test_id,
                    'test_name': test_name,
                    'algorithm': 'Random Algorithm',
                    'cities': cities,
                    'distance': distance,
                    'capacity': capacity,
                    'best_movements': 'N/A',
                    'worst_movements': 'N/A',
                    'avg_movements': 'N/A',
                    'std_movements': 'N/A',
                    'best_gen_count': 'N/A',
                    'worst_gen_count': 'N/A',
                    'avg_gen_count': 'N/A',
                    'std_gen_count': 'N/A',
                    'best_path': 'N/A',
                    'time': random['avg_time'],
                    'success_rate': '0.0%',
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
                        'test_name': test_name,
                        'algorithm': f"GA (Config {config_id})",
                        'cities': cities,
                        'distance': distance,
                        'capacity': capacity,
                        'best_movements': ga_result['best_movements'],
                        'worst_movements': ga_result['worst_movements'],
                        'avg_movements': ga_result['avg_movements'],
                        'std_movements': ga_result['std_movements'],
                        'best_gen_count': ga_result['best_gen_count'],
                        'worst_gen_count': ga_result['worst_gen_count'],
                        'avg_gen_count': ga_result['avg_gen_count'],
                        'std_gen_count': ga_result['std_gen_count'],
                        'best_path': ga_result['best_path'],
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
                        'test_name': test_name,
                        'algorithm': f"GA (Config {config_id})",
                        'cities': cities,
                        'distance': distance,
                        'capacity': capacity,
                        'best_movements': 'N/A',
                        'worst_movements': 'N/A',
                        'avg_movements': 'N/A',
                        'std_movements': 'N/A',
                        'best_gen_count': 'N/A',
                        'worst_gen_count': 'N/A',
                        'avg_gen_count': 'N/A',
                        'std_gen_count': 'N/A',
                        'best_path': 'N/A',
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
                        'test_name': test_name,
                        'algorithm': f"TS (Config {config_id})",
                        'cities': cities,
                        'distance': distance,
                        'capacity': capacity,
                        'best_movements': ts_result['best_movements'],
                        'worst_movements': ts_result['worst_movements'],
                        'avg_movements': ts_result['avg_movements'],
                        'std_movements': ts_result['std_movements'],
                        'best_path': ts_result['best_path'],
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
                        'test_name': test_name,
                        'algorithm': f"TS (Config {config_id})",
                        'cities': cities,
                        'distance': distance,
                        'capacity': capacity,
                        'best_movements': 'N/A',
                        'worst_movements': 'N/A',
                        'avg_movements': 'N/A',
                        'std_movements': 'N/A',
                        'best_path': 'N/A',
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
    csv_file_path = runner.export_comparison_csv()

    if csv_file_path:
        print(f"\n{'-'*20} GENERATING VISUALIZATIONS {'-'*20}")
        print(f"Reading data from: {csv_file_path}")

        # Initialize visualization tool with the generated CSV file
        viz_tool = VisualizationTools(csv_file_path)

        # Generate all charts
        plots_dir = viz_tool.generate_all_charts()
        print(f"All visualizations saved to: {plots_dir}")

        print(f"\n{'-'*20} VISUALIZATION COMPLETE {'-'*20}")
    else:
        print("No CSV file was generated. Cannot create visualizations.")