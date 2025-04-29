import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # enables 3D plotting
import numpy as np

def plot_3d(qid: int = 2172253923) -> None:
    """
    Plots a 3D surface of the alpha sweep results for a given query ID.

    Args:
        qid: The ID of the query to visualize.
    """
    # 1) Load and filter
    df = pd.read_csv('entire-database-spark-experiments/alpha_results.csv')
    sub = df[df['query_id'] == qid]

    # 2) Pivot into matrix, fill diagonal & missing with 0
    mat = sub.pivot(index='alpha1', columns='alpha2', values='difference')
    alphas = sorted(sub['alpha1'].unique())
    mat = mat.reindex(index=alphas, columns=alphas, fill_value=0.0)

    # 3) Prepare meshgrid
    X, Y = np.meshgrid(alphas, alphas)
    Z = mat.values

    # 4) Plot 3D surface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, linewidth=0, antialiased=True)
    ax.set_xlabel('α₁')
    ax.set_ylabel('α₂')
    ax.set_zlabel('difference')
    ax.set_title(f'α-Sweep Surface for Query {qid}')
    plt.tight_layout()
    plt.show()

def draw_heatmap(
    csv_path='entire-database-spark-experiments/alpha_results.csv',
    output_path='entire-database-spark-experiments/alpha_diff_heatmap.png'
) -> None:
    """
    Reads the given CSV file, computes the average difference for each α₁–α₂ combination,
    and saves a heatmap plot of these averages to the specified output path.
    """
    # 1. Load the data from CSV
    df = pd.read_csv(csv_path)
    
    # 2. Compute average 'difference' for each (alpha1, alpha2) combination
    avg_diff = df.groupby(['alpha1', 'alpha2'], as_index=False)['difference'].mean()
    
    # 3. Pivot data to create a matrix of alpha1 vs alpha2 with average differences
    heatmap_data = avg_diff.pivot(index='alpha1', columns='alpha2', values='difference')
    # Ensure the index and columns are sorted for proper axis ordering
    heatmap_data = heatmap_data.sort_index()           # sort by alpha1
    heatmap_data = heatmap_data.sort_index(axis=1)     # sort by alpha2
    
    # 4. Plot the heatmap using matplotlib
    fig, ax = plt.subplots()
    # Display the matrix as an image with origin at lower-left (so smallest α₁, α₂ is at bottom-left)
    cax = ax.imshow(heatmap_data.values, origin='lower', aspect='auto')
    
    # Set axis tick positions and labels to match α₂ and α₁ values
    ax.set_xticks(np.arange(len(heatmap_data.columns)))
    ax.set_xticklabels(heatmap_data.columns)
    ax.set_yticks(np.arange(len(heatmap_data.index)))
    ax.set_yticklabels(heatmap_data.index)
    
    # Label the axes with α₂ (x-axis) and α₁ (y-axis)
    ax.set_xlabel('α₂')
    ax.set_ylabel('α₁')
    
    # Add a color bar to show the difference scale
    fig.colorbar(cax, ax=ax, label='Difference')
    
    # 5. Save the figure to a file and close the plot to free memory
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)