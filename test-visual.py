from visualization import VisualizationTools
import os
import glob
import sys

print(f"\n{'-'*20} GENERATING VISUALIZATIONS {'-'*20}")

# Find the most recent CSV file if not specified
if len(sys.argv) > 1:
    csv_file_path = sys.argv[1]
else:
    # Find the most recent algorithm comparison CSV file
    csv_files = glob.glob("results/algorithm_comparison_*.csv")
    if not csv_files:
        print("No algorithm comparison CSV files found. Please specify a file path.")
        sys.exit(1)

    # Sort by modification time (most recent first)
    csv_file_path = max(csv_files, key=os.path.getmtime)

print(f"Reading data from: {csv_file_path}")

# Initialize visualization tool with the CSV file
viz_tool = VisualizationTools(csv_file_path)

# Generate all charts, including the new fitness trends
plots_dir = viz_tool.generate_all_charts()

# Generate only fitness trends if needed
# viz_tool.plot_fitness_trends()

print(f"\n{'-'*20} VISUALIZATION COMPLETE {'-'*20}")
print(f"All visualizations saved to: {plots_dir}")