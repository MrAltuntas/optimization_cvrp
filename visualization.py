import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from matplotlib.gridspec import GridSpec

class VisualizationTools:
    def __init__(self, csv_file_path):
        """
        Initialize with path to the CSV file containing algorithm comparison data.

        Args:
            csv_file_path: Path to the CSV file exported by TestRunner
        """
        self.csv_file_path = csv_file_path
        self.data = None
        self.output_dir = os.path.dirname(csv_file_path)

    def load_data(self):
        """Load the CSV data and perform basic preprocessing."""
        self.data = pd.read_csv(self.csv_file_path)

        # Convert string values to numeric where possible
        numeric_cols = ['best_movements', 'worst_movements', 'avg_movements',
                        'std_movements', 'time', 'best_gen_count',
                        'worst_gen_count', 'avg_gen_count', 'std_gen_count']

        for col in numeric_cols:
            if col in self.data.columns:
                # Replace 'N/A' with np.nan
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')

        # Convert success_rate from string percentage to float
        if 'success_rate' in self.data.columns:
            self.data['success_rate'] = self.data['success_rate'].str.rstrip('%').astype(float) / 100

        return self.data

    def plot_fitness_trends(self, output_dir=None):
        """
        Plot the fitness trends over generations for GA configurations with simplified visualization.
        Only shows average fitness values with intuitive scaling for better readability.

        Args:
            output_dir: Directory to save the plot
        """
        if output_dir is None:
            output_dir = os.path.join(self.output_dir, 'plots')

        os.makedirs(output_dir, exist_ok=True)

        # Get all fitness data files from the results directory
        fitness_files = []
        for file in os.listdir(self.output_dir):
            if file.startswith('fitness_data_test') and file.endswith('.csv'):
                fitness_files.append(file)

        if not fitness_files:
            print("No fitness data files found to plot fitness trends")
            return

        # Group files by test
        test_files = {}
        for file in fitness_files:
            # Extract test ID from filename
            test_id = file.split('_')[2].replace('test', '')
            if test_id not in test_files:
                test_files[test_id] = []
            test_files[test_id].append(file)

        # Create a plot for each test
        for test_id, files in test_files.items():
            # Find test name from our data
            if self.data is None:
                self.load_data()

            test_data = self.data[self.data['test_id'] == int(test_id)]
            if test_data.empty:
                test_name = f"Test {test_id}"
            else:
                test_name = test_data['test_name'].iloc[0]

            # Create a single figure with nice styling
            plt.figure(figsize=(12, 8))

            # Collect data for all configurations to determine appropriate scales
            all_fitness_data = []
            config_names = []

            for file in files:
                # Extract config ID from filename
                config_id = file.split('_')[-1].replace('.csv', '')

                # Load fitness data
                fitness_data = pd.read_csv(os.path.join(self.output_dir, file))

                # Find the configuration details
                config_row = test_data[test_data['algorithm'].str.startswith(f'GA (Config {config_id})')]

                if not config_row.empty:
                    config_name = config_row['algorithm'].iloc[0]
                    params = config_row['parameters'].iloc[0]
                    if 'Mut=' in params:
                        mutations = params.split('Mut=')[1].split(',')[0]
                        label = f"{config_name} (Mut={mutations})"
                    else:
                        label = config_name
                else:
                    label = f"GA Config {config_id}"

                all_fitness_data.append(fitness_data)
                config_names.append(label)

            # Process data to make visualization more readable
            # We'll use a piecewise approach:
            # 1. First transform values: subtract minimum value and add small offset to avoid zeros
            # 2. Then use logarithmic scale for the very high values

            # Find overall minimum fitness value across all configs (excluding infinity)
            min_fitness = float('inf')
            for data in all_fitness_data:
                curr_min = data['avg_fitness'].replace([np.inf, -np.inf], np.nan).min()
                if not np.isnan(curr_min) and curr_min < min_fitness:
                    min_fitness = curr_min

            # Apply grid styling
            plt.grid(True, alpha=0.3, linestyle='--')

            # Plot each configuration with transformed values
            for i, (fitness_data, label) in enumerate(zip(all_fitness_data, config_names)):
                # Create a copy of data for transformation
                transformed_fitness = fitness_data['avg_fitness'].copy()

                # Find where fitness drops to valid solution range (below 100)
                valid_solution_idx = None
                for idx, val in enumerate(transformed_fitness):
                    if val < 100:
                        valid_solution_idx = idx
                        break

                # Plot with nice coloring
                plt.plot(fitness_data['generation'], transformed_fitness,
                         linewidth=2.5, label=label)

                # Add annotation where fitness becomes a valid solution
                if valid_solution_idx is not None:
                    gen = fitness_data['generation'].iloc[valid_solution_idx]
                    val = transformed_fitness.iloc[valid_solution_idx]
                    plt.annotate(f"Valid: {int(val)}",
                                 xy=(gen, val),
                                 xytext=(10, 10),
                                 textcoords="offset points",
                                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

                # Add annotation for final value
                if len(fitness_data) > 0:
                    final_idx = len(fitness_data) - 1
                    final_gen = fitness_data['generation'].iloc[final_idx]
                    final_val = transformed_fitness.iloc[final_idx]
                    if final_val < 1000:  # Only annotate if it's a reasonable value
                        plt.annotate(f"Final: {int(final_val)}",
                                     xy=(final_gen, final_val),
                                     xytext=(-40, -20),
                                     textcoords="offset points",
                                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

            # Set y-axis to log scale for better visualization of the full range
            plt.yscale('log')

            # Set titles and labels
            plt.title(f'Average Fitness Progress - {test_name}', fontsize=14)
            plt.xlabel('Generation', fontsize=12)
            plt.ylabel('Average Fitness (log scale, lower is better)', fontsize=12)

            # Add legend with better positioning
            plt.legend(loc='upper right', fontsize=10)

            # Ensure y-axis has a reasonable minimum value
            plt.ylim(bottom=max(1, min_fitness * 0.9))

            plt.tight_layout()

            # Save the plot
            plot_filename = f"fitness_trends_{test_name.replace(' ', '_')}.png"
            plt.savefig(os.path.join(output_dir, plot_filename), dpi=300, bbox_inches='tight')
            plt.close()

        return output_dir

    def generate_all_charts(self):
        """Generate all charts and save them to the output directory."""
        if self.data is None:
            self.load_data()

        # Create output directory for plots if it doesn't exist
        plots_dir = os.path.join(self.output_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)

        # Generate individual charts
        self.plot_algorithm_performance_comparison(output_dir=plots_dir)
        self.plot_success_rates(output_dir=plots_dir)
        self.plot_execution_times(output_dir=plots_dir)

        # Try to generate GA and TS specific charts
        try:
            self.plot_ga_generation_performance(output_dir=plots_dir)
            self.plot_ga_parameter_effects(output_dir=plots_dir)
            self.plot_ts_parameter_effects(output_dir=plots_dir)
            self.plot_fitness_trends(output_dir=plots_dir)  # Add this line
        except Exception as e:
            print(f"Warning: Could not generate some algorithm-specific charts: {e}")

        # Create the dashboard
        try:
            self.create_dashboard(output_dir=plots_dir)
        except Exception as e:
            print(f"Warning: Could not create dashboard: {e}")

        print(f"All visualizations saved to: {plots_dir}")
        return plots_dir

    def plot_algorithm_performance_comparison(self, output_dir=None):
        """
        Plot comparison of best/avg/worst movements across algorithms.

        Args:
            output_dir: Directory to save the plot
        """
        if self.data is None:
            self.load_data()

        # Filter out rows with non-valid solutions
        valid_data = self.data[self.data['best_movements'].notna()]

        if valid_data.empty:
            print("No valid data to plot algorithm performance comparison")
            return

        # Group by test_name and algorithm
        grouped = valid_data.groupby(['test_name', 'algorithm']).agg({
            'best_movements': 'min',
            'avg_movements': 'mean',
            'worst_movements': 'max'
        }).reset_index()

        # Create figure
        plt.figure(figsize=(14, 8))

        # Iterate through test instances
        for i, test_name in enumerate(grouped['test_name'].unique()):
            plt.subplot(len(grouped['test_name'].unique()), 1, i+1)

            test_data = grouped[grouped['test_name'] == test_name]

            # Sort by average movements for better comparison
            test_data = test_data.sort_values('avg_movements')

            x = np.arange(len(test_data))
            width = 0.25

            # Plot bars
            plt.bar(x - width, test_data['best_movements'], width, label='Best', color='green', alpha=0.7)
            plt.bar(x, test_data['avg_movements'], width, label='Average', color='blue', alpha=0.7)
            plt.bar(x + width, test_data['worst_movements'], width, label='Worst', color='red', alpha=0.7)

            plt.xlabel('Algorithm')
            plt.ylabel('Movements')
            plt.title(f'Performance Comparison for {test_name}')
            plt.xticks(x, test_data['algorithm'], rotation=45, ha='right')
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()

        plt.tight_layout()

        if output_dir:
            plt.savefig(os.path.join(output_dir, 'algorithm_performance_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_success_rates(self, output_dir=None):
        """
        Plot success rates across algorithms.

        Args:
            output_dir: Directory to save the plot
        """
        if self.data is None:
            self.load_data()

        # Group by test_name and algorithm
        grouped = self.data.groupby(['test_name', 'algorithm'])['success_rate'].mean().reset_index()

        if grouped.empty:
            print("No data to plot success rates")
            return

        # Create figure
        plt.figure(figsize=(14, 8))

        # Iterate through test instances
        for i, test_name in enumerate(grouped['test_name'].unique()):
            plt.subplot(len(grouped['test_name'].unique()), 1, i+1)

            test_data = grouped[grouped['test_name'] == test_name]

            # Sort by success rate for better visualization
            test_data = test_data.sort_values('success_rate', ascending=False)

            # Create horizontal bar chart
            bars = plt.barh(test_data['algorithm'], test_data['success_rate'], color=plt.cm.viridis(test_data['success_rate']))

            # Add percentage labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                plt.text(max(0.01, width - 0.1), bar.get_y() + bar.get_height()/2,
                         f'{width:.1%}', va='center', color='white' if width > 0.3 else 'black', fontweight='bold')

            plt.xlabel('Success Rate')
            plt.title(f'Success Rates for {test_name}')
            plt.xlim(0, 1.05)
            plt.grid(axis='x', linestyle='--', alpha=0.7)

        plt.tight_layout()

        if output_dir:
            plt.savefig(os.path.join(output_dir, 'success_rates.png'), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_execution_times(self, output_dir=None):
        """
        Plot execution times comparison.

        Args:
            output_dir: Directory to save the plot
        """
        if self.data is None:
            self.load_data()

        # Group by test_name and algorithm
        grouped = self.data.groupby(['test_name', 'algorithm'])['time'].mean().reset_index()

        if grouped.empty:
            print("No data to plot execution times")
            return

        # Create figure
        plt.figure(figsize=(14, 8))

        # Iterate through test instances
        for i, test_name in enumerate(grouped['test_name'].unique()):
            plt.subplot(len(grouped['test_name'].unique()), 1, i+1)

            test_data = grouped[grouped['test_name'] == test_name]

            # Sort by execution time for better visualization
            test_data = test_data.sort_values('time')

            # Create bar chart
            bars = plt.barh(test_data['algorithm'], test_data['time'], color=plt.cm.coolwarm_r(test_data['time'] / test_data['time'].max()))

            # Add time labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                plt.text(width + 0.1, bar.get_y() + bar.get_height()/2,
                         f'{width:.2f}s', va='center')

            plt.xlabel('Average Execution Time (seconds)')
            plt.title(f'Execution Times for {test_name}')
            plt.grid(axis='x', linestyle='--', alpha=0.7)

        plt.tight_layout()

        if output_dir:
            plt.savefig(os.path.join(output_dir, 'execution_times.png'), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_ga_generation_performance(self, output_dir=None):
        """
        Plot GA generation counts vs. performance.

        Args:
            output_dir: Directory to save the plot
        """
        if self.data is None:
            self.load_data()

        # Filter GA data with valid solutions
        ga_data = self.data[self.data['algorithm'].str.contains('GA')]
        valid_ga_data = ga_data[ga_data['best_movements'].notna()]

        if valid_ga_data.empty:
            print("No valid GA data to plot generation performance")
            return

        # Check if generation count data is available
        if valid_ga_data['avg_gen_count'].isna().all():
            print("No generation count data available for GA")
            return

        # Create figure
        plt.figure(figsize=(14, 8))

        # Create scatter plot
        plt.scatter(valid_ga_data['avg_gen_count'], valid_ga_data['best_movements'],
                    c=valid_ga_data['time'], cmap='viridis', alpha=0.7, s=100)

        # Add colorbar for execution time
        cbar = plt.colorbar()
        cbar.set_label('Execution Time (seconds)')

        # Add labels for each point
        for i, row in valid_ga_data.iterrows():
            plt.annotate(row['algorithm'],
                         (row['avg_gen_count'], row['best_movements']),
                         xytext=(5, 5), textcoords='offset points',
                         fontsize=8)

        plt.xlabel('Average Generation Count')
        plt.ylabel('Best Movements')
        plt.title('GA Performance vs. Generation Count')
        plt.grid(linestyle='--', alpha=0.7)

        if output_dir:
            plt.savefig(os.path.join(output_dir, 'ga_generation_performance.png'), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def extract_ga_parameters(self):
        """Extract GA parameters from the parameters column."""
        if self.data is None:
            self.load_data()

        # Filter GA data
        ga_data = self.data[self.data['algorithm'].str.contains('GA')].copy()

        if ga_data.empty:
            return pd.DataFrame()

        # Extract parameters - using safer methods
        try:
            # Extract numeric parameters
            ga_data['population_size'] = ga_data['parameters'].str.extract(r'Pop=(\d+)').astype(float)
            ga_data['generations'] = ga_data['parameters'].str.extract(r'Gen=(\d+)').astype(float)
            ga_data['crossover_rate'] = ga_data['parameters'].str.extract(r'Cx=(\d+\.\d+)').astype(float)
            ga_data['mutation_rate'] = ga_data['parameters'].str.extract(r'Mut=(\d+\.\d+)').astype(float)
            ga_data['tournament_size'] = ga_data['parameters'].str.extract(r'Tour=(\d+)').astype(float)
            ga_data['elitism_size'] = ga_data['parameters'].str.extract(r'Elite=(\d+)').astype(float)

            # Fix the improved_evaluation extraction
            # Instead of using map(), which may cause issues, we'll use a different approach
            eval_imp = ga_data['parameters'].str.extract(r'Eval_Imp=(\w+)')
            ga_data['improved_evaluation'] = eval_imp.applymap(lambda x: x == 'True' if pd.notna(x) else np.nan)
        except Exception as e:
            print(f"Warning: Could not extract some GA parameters: {e}")

        return ga_data

    def extract_ts_parameters(self):
        """Extract TS parameters from the parameters column."""
        if self.data is None:
            self.load_data()

        # Filter TS data
        ts_data = self.data[self.data['algorithm'].str.contains('TS')].copy()

        if ts_data.empty:
            return pd.DataFrame()

        # Extract parameters - using safer methods
        try:
            ts_data['iterations'] = ts_data['parameters'].str.extract(r'Iter=(\d+)').astype(float)
            ts_data['tabu_size'] = ts_data['parameters'].str.extract(r'TabuSize=(\d+)').astype(float)
            ts_data['neighborhood_size'] = ts_data['parameters'].str.extract(r'NeighSize=(\d+)').astype(float)
        except Exception as e:
            print(f"Warning: Could not extract some TS parameters: {e}")

        return ts_data

    def plot_ga_parameter_effects(self, output_dir=None):
        """
        Plot effects of different parameters on GA performance.

        Args:
            output_dir: Directory to save the plot
        """
        ga_data = self.extract_ga_parameters()

        if ga_data.empty:
            print("No GA data to plot parameter effects")
            return

        # Filter valid solutions
        valid_ga_data = ga_data[ga_data['best_movements'].notna()]

        if valid_ga_data.empty:
            print("No valid GA solutions to plot parameter effects")
            return

        # Check if we have at least one parameter to plot
        if (valid_ga_data[['mutation_rate', 'crossover_rate', 'tournament_size',
                           'elitism_size', 'improved_evaluation', 'population_size']].notna().sum() <= 0).all():
            print("No GA parameters available to plot effects")
            return

        # Create figure
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(2, 3, figure=fig)

        parameter_plots = [
            ('mutation_rate', 'Mutation Rate', gs[0, 0]),
            ('crossover_rate', 'Crossover Rate', gs[0, 1]),
            ('tournament_size', 'Tournament Size', gs[0, 2]),
            ('elitism_size', 'Elitism Size', gs[1, 0]),
            ('improved_evaluation', 'Improved Evaluation', gs[1, 1]),
            ('population_size', 'Population Size', gs[1, 2])
        ]

        for param_name, param_title, grid_pos in parameter_plots:
            # Skip if no data for this parameter
            if valid_ga_data[param_name].isna().all():
                continue

            ax = fig.add_subplot(grid_pos)
            try:
                sns.boxplot(x=param_name, y='best_movements', data=valid_ga_data, ax=ax)
                ax.set_title(f'Effect of {param_title} on Performance')
                ax.set_xlabel(param_title)
                ax.set_ylabel('Best Movements')
                ax.grid(axis='y', linestyle='--', alpha=0.7)
            except Exception as e:
                print(f"Warning: Could not plot {param_title} effects: {e}")

        plt.tight_layout()

        if output_dir:
            plt.savefig(os.path.join(output_dir, 'ga_parameter_effects.png'), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_ts_parameter_effects(self, output_dir=None):
        """
        Plot effects of different parameters on TS performance.

        Args:
            output_dir: Directory to save the plot
        """
        ts_data = self.extract_ts_parameters()

        if ts_data.empty:
            print("No TS data to plot parameter effects")
            return

        # Filter valid solutions
        valid_ts_data = ts_data[ts_data['best_movements'].notna()]

        if valid_ts_data.empty:
            print("No valid TS solutions to plot parameter effects")
            return

        # Check if we have at least one parameter to plot
        if (valid_ts_data[['iterations', 'tabu_size', 'neighborhood_size']].notna().sum() <= 0).all():
            print("No TS parameters available to plot effects")
            return

        # Create figure
        fig = plt.figure(figsize=(15, 8))
        gs = GridSpec(1, 3, figure=fig)

        parameter_plots = [
            ('iterations', 'Iterations', gs[0, 0]),
            ('tabu_size', 'Tabu Size', gs[0, 1]),
            ('neighborhood_size', 'Neighborhood Size', gs[0, 2])
        ]

        for param_name, param_title, grid_pos in parameter_plots:
            # Skip if no data for this parameter
            if valid_ts_data[param_name].isna().all():
                continue

            ax = fig.add_subplot(grid_pos)
            try:
                sns.boxplot(x=param_name, y='best_movements', data=valid_ts_data, ax=ax)
                ax.set_title(f'Effect of {param_title} on Performance')
                ax.set_xlabel(param_title)
                ax.set_ylabel('Best Movements')
                ax.grid(axis='y', linestyle='--', alpha=0.7)
            except Exception as e:
                print(f"Warning: Could not plot {param_title} effects: {e}")

        plt.tight_layout()

        if output_dir:
            plt.savefig(os.path.join(output_dir, 'ts_parameter_effects.png'), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def create_dashboard(self, output_dir=None):
        """
        Create a dashboard with key performance metrics.

        Args:
            output_dir: Directory to save the plot
        """
        if self.data is None:
            self.load_data()

        # Filter out rows with non-valid solutions
        valid_data = self.data[self.data['best_movements'].notna()]

        if valid_data.empty:
            print("No valid solutions to create dashboard")
            return

        # Create figure
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(2, 2, figure=fig)

        # 1. Algorithm Performance Comparison
        ax1 = fig.add_subplot(gs[0, 0])

        # Group by algorithm
        performance_by_alg = valid_data.groupby('algorithm').agg({
            'best_movements': 'min',
            'avg_movements': 'mean',
            'worst_movements': 'max',
            'time': 'mean',
            'success_rate': 'mean'
        }).reset_index()

        # Sort by average performance
        performance_by_alg = performance_by_alg.sort_values('avg_movements')

        # Plot performance
        performance_by_alg.plot(x='algorithm', y=['best_movements', 'avg_movements', 'worst_movements'],
                                kind='bar', ax=ax1, color=['green', 'blue', 'red'], alpha=0.7)
        ax1.set_title('Algorithm Performance Comparison')
        ax1.set_xlabel('Algorithm')
        ax1.set_ylabel('Movements')
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        ax1.legend(['Best', 'Average', 'Worst'])

        # 2. Success Rates
        ax2 = fig.add_subplot(gs[0, 1])

        # Plot success rates
        performance_by_alg.plot(x='algorithm', y='success_rate', kind='bar', ax=ax2, color='purple', alpha=0.7)

        # Add percentage labels
        for i, value in enumerate(performance_by_alg['success_rate']):
            ax2.text(i, value + 0.02, f'{value:.1%}', ha='center', va='bottom')

        ax2.set_title('Algorithm Success Rates')
        ax2.set_xlabel('Algorithm')
        ax2.set_ylabel('Success Rate')
        ax2.set_ylim(0, 1.1)
        ax2.grid(axis='y', linestyle='--', alpha=0.7)

        # 3. Time vs. Performance
        ax3 = fig.add_subplot(gs[1, 0])

        # Scatter plot of time vs. best movements
        scatter = ax3.scatter(valid_data['time'], valid_data['best_movements'],
                              c=valid_data['success_rate'], cmap='viridis',
                              alpha=0.7, s=80, edgecolors='k')

        # Add algorithm labels
        for i, row in valid_data.iterrows():
            ax3.annotate(row['algorithm'],
                         (row['time'], row['best_movements']),
                         xytext=(5, 5), textcoords='offset points',
                         fontsize=8)

        ax3.set_title('Time vs. Performance')
        ax3.set_xlabel('Execution Time (seconds)')
        ax3.set_ylabel('Best Movements')
        ax3.grid(linestyle='--', alpha=0.7)

        # Add colorbar for success rate
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label('Success Rate')

        # 4. Performance by Problem Size
        ax4 = fig.add_subplot(gs[1, 1])

        # Check if we have multiple problem sizes
        if len(valid_data['cities'].unique()) > 1:
            try:
                # Group by problem size (cities) and algorithm
                size_perf = valid_data.groupby(['cities', 'algorithm']).agg({
                    'best_movements': 'min'
                }).reset_index()

                # Pivot for easier plotting
                pivot_data = size_perf.pivot(index='cities', columns='algorithm', values='best_movements')
                pivot_data.plot(kind='bar', ax=ax4, alpha=0.7)

                ax4.set_title('Best Performance by Problem Size')
                ax4.set_xlabel('Number of Cities')
                ax4.set_ylabel('Best Movements')
                ax4.grid(axis='y', linestyle='--', alpha=0.7)
                ax4.legend(title='Algorithm', loc='upper left', bbox_to_anchor=(1, 1))
            except Exception as e:
                print(f"Warning: Could not create problem size comparison plot: {e}")
                ax4.text(0.5, 0.5, "Performance by Problem Size\n(Not enough data with different sizes)",
                         ha='center', va='center', fontsize=12)
                ax4.set_axis_off()
        else:
            ax4.text(0.5, 0.5, "Performance by Problem Size\n(Only one problem size available)",
                     ha='center', va='center', fontsize=12)
            ax4.set_axis_off()

        plt.tight_layout()

        if output_dir:
            plt.savefig(os.path.join(output_dir, 'performance_dashboard.png'), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()