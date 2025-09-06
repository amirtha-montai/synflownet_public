import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# Load correlation data with proper indexing
correlation_df = pd.read_csv("/home/ubuntu/synflownet_public/src/interpretability/results_qed_run/sparse_autoencoder_dim_128/sparse_autoencoder_results_correlations.csv", index_col=0)
save_dir = "/home/ubuntu/synflownet_public/src/interpretability/results_qed_run/sparse_autoencoder_dim_128"

print("Correlation DataFrame shape:", correlation_df.shape)
print("DataFrame columns:", list(correlation_df.columns))
print("DataFrame index:", list(correlation_df.index))
print("Data types:", correlation_df.dtypes)
print("\nFirst few rows:")
print(correlation_df.head())

def plot_correlation_heatmap(corr_df, output_dir):
    # Ensure we have numeric data only
    numeric_df = corr_df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        print("No numeric data found! Converting all data to numeric...")
        # Try to convert all data to numeric, replacing errors with NaN
        numeric_df = corr_df.apply(pd.to_numeric, errors='coerce')
    
    print(f"Plotting correlation matrix with shape: {numeric_df.shape}")
    
    # Calculate appropriate figure size based on data dimensions
    n_factors = numeric_df.shape[0]
    n_rewards = numeric_df.shape[1]
    
    # Make figure reasonably sized with space for y-axis labels
    fig_width = max(7, min(16, n_rewards * 0.5))
    fig_height = max(10, min(12, n_factors * 0.2))
    
    plt.figure(figsize=(fig_width, fig_height))
    
    # Create heatmap with larger font sizes
    ax = sns.heatmap(
        numeric_df,
        annot=False,
        cmap='RdBu_r',
        center=0,
        fmt='.3f',
        cbar_kws={'label': 'Correlation'},
        xticklabels=True,
        yticklabels=True
    )
    
    # Increase font sizes for better readability
    plt.title('Latent Factor - Reward Correlations', fontsize=16, pad=20)
    plt.xlabel('Reward Signals', fontsize=10)
    plt.ylabel('Latent Factors', fontsize=10)
    
    # Make y-axis labels (factor names) larger and more readable
    ax.tick_params(axis='y', labelsize=7, rotation=0)
    ax.tick_params(axis='x', labelsize=8, rotation=0)
    
    # Adjust layout to prevent clipping of tick-labels
    plt.tight_layout()
    plt.savefig(f"{output_dir}/factor_reward_correlations_updated.png", dpi=300, bbox_inches='tight')
    plt.show()

plot_correlation_heatmap(correlation_df, save_dir)