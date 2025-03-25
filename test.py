import config
import cvrp
import greedy_cvrp
import random_cvrp
import genetic_cvrp
import time
import pandas as pd

def run_test():
    """Run tests based on the RUN_TEST configuration"""
    results = []

    for test_idx, test_config in enumerate(config.RUN_TEST):
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

        # Run Greedy algorithm
        print("\n1. Greedy Algorithm:")
        greedy_solver = greedy_cvrp.GreedyCVRP(cvrp_problem)
        start_time = time.time()
        greedy_path = greedy_solver.find_path()
        greedy_time = time.time() - start_time
        greedy_movements, greedy_valid = cvrp_problem.is_valid_path(greedy_path)
        print(f"Valid: {greedy_valid}, Movements: {greedy_movements}, Time: {greedy_time:.4f}s")
        print(f"Path: {greedy_path}")

        test_results['methods']['greedy'] = {
            'valid': greedy_valid,
            'movements': greedy_movements,
            'time': greedy_time,
            'path': greedy_path
        }

        # Run Random algorithm
        print("\n2. Random Algorithm:")
        random_solver = random_cvrp.RandomCVRP(cvrp_problem)
        start_time = time.time()
        random_path = random_solver.random_path()
        random_time = time.time() - start_time
        random_movements, random_valid = cvrp_problem.is_valid_path(random_path)
        print(f"Valid: {random_valid}, Movements: {random_movements}, Time: {random_time:.4f}s")
        print(f"Path: {random_path}")

        test_results['methods']['random'] = {
            'valid': random_valid,
            'movements': random_movements,
            'time': random_time,
            'path': random_path
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

            ga_solver = genetic_cvrp.GeneticCVRP(
                cvrp_problem,
                population_size=ga_setting['GA_POPULATION_SIZE'],
                generations=ga_setting['GA_GENERATIONS'],
                crossover_rate=ga_setting['GA_CROSSOVER_RATE'],
                mutation_rate=ga_setting['GA_MUTATION_RATE'],
                tournament_size=ga_setting['GA_TOURNAMENT_SIZE'],
                elitism_count=ga_setting['GA_ELITISM_SIZE']
            )

            start_time = time.time()
            ga_path = ga_solver.solve()
            ga_time = time.time() - start_time

            ga_result = {
                'config_id': ga_idx + 1,
                'settings': ga_setting,
                'time': ga_time
            }

            if ga_path is not None:
                ga_movements, ga_valid = cvrp_problem.is_valid_path(ga_path)
                print(f"Valid: {ga_valid}, Movements: {ga_movements}, Time: {ga_time:.4f}s")
                print(f"Path: {ga_path}")

                ga_result.update({
                    'valid': ga_valid,
                    'movements': ga_movements,
                    'path': ga_path
                })
            else:
                print(f"No valid solution found, Time: {ga_time:.4f}s")
                ga_result['valid'] = False

            test_results['methods']['genetic'].append(ga_result)

        results.append(test_results)

        # Print summary for this test
        print(f"\n{'-'*40} SUMMARY FOR TEST {test_idx+1} {'-'*40}")
        print(f"CVRP: Cities={test_config['CVRP_NUMBER_OF_CITIES']}, Distance={test_config['CVRP_DISTANCE_BETWEEN_CITIES']}, Capacity={test_config['CVRP_CAPACITY']}")
        print(f"Greedy: Valid={greedy_valid}, Movements={greedy_movements}, Time={greedy_time:.4f}s")
        print(f"Random: Valid={random_valid}, Movements={random_movements}, Time={random_time:.4f}s")

        for ga_idx, ga_result in enumerate(test_results['methods']['genetic']):
            if ga_result['valid']:
                ga_movements = ga_result['movements']
                ga_time = ga_result['time']
                ga_settings = ga_result['settings']
                print(f"GA (Config {ga_idx+1}): Valid=True, Movements={ga_movements}, Time={ga_time:.4f}s")
                print(f"  Parameters: Pop={ga_settings['GA_POPULATION_SIZE']}, Gen={ga_settings['GA_GENERATIONS']}, " +
                      f"Cx={ga_settings['GA_CROSSOVER_RATE']}, Mut={ga_settings['GA_MUTATION_RATE']}, " +
                      f"Tour={ga_settings['GA_TOURNAMENT_SIZE']}, Elite={ga_settings['GA_ELITISM_SIZE']}")
            else:
                print(f"GA (Config {ga_idx+1}): No valid solution, Time={ga_result['time']:.4f}s")

    # Return results for further analysis if needed
    return results

def create_comparison_table(results):
    """Create a comparison table for the report"""
    for test_result in results:
        test_id = test_result['test_id']
        cvrp_config = test_result['cvrp_config']

        print(f"\nComparison Table for Test {test_id}")
        print(f"CVRP: Cities={cvrp_config['CVRP_NUMBER_OF_CITIES']}, Distance={cvrp_config['CVRP_DISTANCE_BETWEEN_CITIES']}, Capacity={cvrp_config['CVRP_CAPACITY']}")

        print(f"\n{'Method':<20} {'Valid':<10} {'Movements':<15} {'Time (s)':<15}")
        print('-' * 60)

        # Greedy
        greedy = test_result['methods']['greedy']
        print(f"{'Greedy':<20} {str(greedy['valid']):<10} {greedy['movements']:<15} {greedy['time']:<15.4f}")

        # Random
        random = test_result['methods']['random']
        print(f"{'Random':<20} {str(random['valid']):<10} {random['movements']:<15} {random['time']:<15.4f}")

        # Genetic algorithm configurations
        for ga_result in test_result['methods']['genetic']:
            config_id = ga_result['config_id']
            method_name = f"GA (Config {config_id})"

            if ga_result['valid']:
                print(f"{method_name:<20} {str(ga_result['valid']):<10} {ga_result['movements']:<15} {ga_result['time']:<15.4f}")
            else:
                print(f"{method_name:<20} {str(ga_result['valid']):<10} {'N/A':<15} {ga_result['time']:<15.4f}")

if __name__ == "__main__":
    results = run_test()
    create_comparison_table(results)