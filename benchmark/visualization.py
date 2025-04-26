import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class Visualization:
    def __init__(self, base_dir='results'):
        """
        Initialize the visualization module.

        Args:
            base_dir: Base directory for results
        """
        self.base_dir = base_dir

    def plot_cvrp_solution(self, problem, routes, title=None, save_path=None):
        """
        Plot a CVRP solution.

        Args:
            problem: CVRP problem instance
            routes: List of routes
            title: Plot title
            save_path: Path to save the plot

        Returns:
            plt.Figure: The matplotlib figure
        """
        cities = problem.cities
        depot = problem.depot

        fig, ax = plt.subplots(figsize=(12, 10))

        # Plot depot
        ax.scatter(cities[depot][0], cities[depot][1], c='red', s=200, marker='*',
                   label='Depot', edgecolors='black', zorder=3)

        # Plot cities
        for i, (x, y) in enumerate(cities):
            if i != depot:
                ax.scatter(x, y, c='blue', s=100, edgecolors='black', zorder=2)
                ax.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points')

        # Plot routes with different colors
        colors = plt.cm.rainbow(np.linspace(0, 1, len(routes)))

        for i, route in enumerate(routes):
            route_x = [cities[depot][0]]
            route_y = [cities[depot][1]]

            for city in route:
                route_x.append(cities[city][0])
                route_y.append(cities[city][1])

            # Return to depot
            route_x.append(cities[depot][0])
            route_y.append(cities[depot][1])

            ax.plot(route_x, route_y, 'o-', color=colors[i], linewidth=2,
                    label=f'Route {i+1}', zorder=1)

        if title:
            ax.set_title(title, fontsize=14)
        else:
            ax.set_title('CVRP Solution', fontsize=14)

        ax.set_xlabel('X Coordinate', fontsize=12)
        ax.set_ylabel('Y Coordinate', fontsize=12)
        ax.grid(True, alpha=0.3)

        # Add legend with custom placement
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_convergence(self, stats, title=None, save_path=None):
        """
        Plot convergence of an algorithm.

        Args:
            stats: List of (generation/iteration, fitness) tuples
            title: Plot title
            save_path: Path to save the plot

        Returns:
            plt.Figure: The matplotlib figure
        """
        x, y = zip(*stats)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x, y, 'b-')

        # Find the best solution
        best_idx = y.index(min(y))
        best_x, best_y = x[best_idx], y[best_idx]

        # Mark the best solution
        ax.plot(best_x, best_y, 'ro', markersize=8,
                label=f'Best: {best_y:.2f} at {best_x}')

        ax.set_xlabel('Generation/Iteration', fontsize=12)
        ax.set_ylabel('Fitness (Total Distance)', fontsize=12)

        if title:
            ax.set_title(title, fontsize=14)
        else:
            ax.set_title('Convergence Plot', fontsize=14)

        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)

        return fig

    def plot_ga_progress(self, stats, title=None, save_path=None):
        """
        Plot the progress of a Genetic Algorithm.

        Args:
            stats: List of (generation, best, avg, worst) tuples
            title: Plot title
            save_path: Path to save the plot

        Returns:
            plt.Figure: The matplotlib figure
        """
        generations, best, avg, worst = zip(*stats)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(generations, best, 'g-', label='Best')
        ax.plot(generations, avg, 'b-', label='Average')
        ax.plot(generations, worst, 'r-', label='Worst')

        ax.set_xlabel('Generation', fontsize=12)
        ax.set_ylabel('Fitness (Total Distance)', fontsize=12)

        if title:
            ax.set_title(title, fontsize=14)
        else:
            ax.set_title('Genetic Algorithm Progress', fontsize=14)

        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)

        return fig

    def plot_parameter_effect(self, results, parameter, metric='distance', title=None, save_path=None):
        """
        Plot the effect of a parameter on a metric.

        Args:
            results: List of dictionaries with algorithm results
            parameter: Parameter name
            metric: Metric to plot (e.g., 'distance', 'time')
            title: Plot title
            save_path: Path to save the plot

        Returns:
            plt.Figure: The matplotlib figure
        """
        # Extract parameter values and corresponding metrics
        param_values = [r['value'] for r in results if r['parameter'] == parameter]
        metric_values = [r[metric] for r in results if r['parameter'] == parameter]

        # Sort by parameter value
        sorted_indices = sorted(range(len(param_values)), key=lambda i: param_values[i])
        param_values = [param_values[i] for i in sorted_indices]
        metric_values = [metric_values[i] for i in sorted_indices]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(param_values, metric_values, 'bo-', markersize=8)

        # Find the best value
        best_idx = metric_values.index(min(metric_values))
        best_param, best_metric = param_values[best_idx], metric_values[best_idx]

        # Mark the best value
        ax.plot(best_param, best_metric, 'ro', markersize=10,
                label=f'Best: {best_metric:.2f} at {parameter}={best_param}')

        ax.set_xlabel(parameter, fontsize=12)
        ax.set_ylabel(metric.capitalize(), fontsize=12)

        if title:
            ax.set_title(title, fontsize=14)
        else:
            ax.set_title(f'Effect of {parameter} on {metric.capitalize()}', fontsize=14)

        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)

        return fig

    def create_comparison_table(self, results, save_path=None):
        """
        Create a comparison table of algorithm results.

        Args:
            results: List of dictionaries with algorithm results
            save_path: Path to save the table

        Returns:
            pd.DataFrame: The comparison table
        """
        table_data = []

        for result in results:
            row = {
                'Algorithm': result['algorithm'],
                'Distance': result['distance'],
                'Valid': result['valid']
            }

            if 'time' in result:
                row['Time (s)'] = result['time']

            if 'parameters' in result:
                for param, value in result['parameters'].items():
                    row[param] = value

            table_data.append(row)

        table = pd.DataFrame(table_data)

        if save_path:
            table.to_csv(save_path, index=False)

        return table

    def plot_all_algorithms_comparison(self, results, title=None, save_path=None):
        """
        Plot a comparison of all algorithms.

        Args:
            results: Dictionary with algorithm results
            title: Plot title
            save_path: Path to save the plot

        Returns:
            plt.Figure: The matplotlib figure
        """
        algorithms = []
        distances = []

        # Collect algorithm names and distances
        for algo_type, algo_results in results.items():
            if isinstance(algo_results, dict):
                algorithms.append(algo_type)
                distances.append(algo_results['distance'])
            elif isinstance(algo_results, list) and algo_results:
                # Take the best result if multiple runs
                best_result = min(algo_results, key=lambda r: r['distance'])
                algorithms.append(algo_type)
                distances.append(best_result['distance'])

        # Sort by distance (better to worse)
        sorted_indices = sorted(range(len(distances)), key=lambda i: distances[i])
        algorithms = [algorithms[i] for i in sorted_indices]
        distances = [distances[i] for i in sorted_indices]

        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.bar(algorithms, distances)

        # Customize colors
        colors = ['green', 'blue', 'red', 'orange', 'purple']
        for i, bar in enumerate(bars):
            bar.set_color(colors[i % len(colors)])

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{height:.1f}', ha='center', va='bottom')

        ax.set_ylabel('Total Distance (lower is better)', fontsize=12)

        if title:
            ax.set_title(title, fontsize=14)
        else:
            ax.set_title('Algorithm Performance Comparison', fontsize=14)

        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)

        return fig