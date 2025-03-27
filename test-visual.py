from visualization import VisualizationTools

print(f"\n{'-'*20} GENERATING VISUALIZATIONS {'-'*20}")

# Initialize visualization tool with the generated CSV file
csv_file_path = "results/algorithm_comparison_20250326_220606.csv"
print(f"Reading data from: {csv_file_path}")

# Initialize visualization tool with the CSV file
viz_tool = VisualizationTools(csv_file_path)

# Generate all charts
plots_dir = viz_tool.generate_all_charts()

print(f"\n{'-'*20} VISUALIZATION COMPLETE {'-'*20}")