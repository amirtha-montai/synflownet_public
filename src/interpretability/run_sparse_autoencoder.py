"""
Configuration and Quick Runner for Sparse Autoencoder Analysis

This script provides easy configuration and execution of the sparse autoencoder
for discovering reward-specific chemical factors.
"""

import json
import sys
import os
from pathlib import Path

# Add the current directory to path
sys.path.append(str(Path(__file__).parent))

from interpretability.sparse_autoencoder import SparseAutoencoderTrainer

result_dir = "/home/ubuntu/synflownet_public/src/interpretability/results_qed_run/sparse_autoencoder_dim_128" #TODO: change this

def create_config():
    """Create default configuration for sparse autoencoder."""
    config = {
        "data": {
            "embedding_file": "/home/ubuntu/synflownet_public/src/interpretability/results_qed_run/embeddings.csv", #TODO: change this
            "test_size": 0.1
        },
        "autoencoder": {
            "hidden_dim": 128, #64
            "epochs": 200,
            "batch_size": 128,
            "learning_rate": 0.001,
            "sparsity_weight": 0.01,
            "sparsity_target": 0.05,
            "sparsity_method": "l1",  # 'l1' or 'kl'
            "dropout": 0.1
        },
        "reward_predictor": {
            "epochs": 100,
            "learning_rate": 0.001,
            "dropout": 0.2
        },
        "output": {
            "results_dir": result_dir,
            "save_models": True,
            "save_plots": True
        }
    }
    return config

def run_analysis_with_config(config_path: str = None):
    """Run sparse autoencoder analysis with configuration."""
    
    # Load or create config
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Loaded configuration from {config_path}")
    else:
        config = create_config()
        if config_path:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"Created default configuration at {config_path}")
    
    # Print configuration
    print("\nConfiguration:")
    print(json.dumps(config, indent=2))
    
    # Create results directory
    os.makedirs(config["output"]["results_dir"], exist_ok=True)
    
    # Initialize trainer
    trainer = SparseAutoencoderTrainer(config["data"]["embedding_file"])
    
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)
    
    # Load and prepare data
    trainer.load_data()
    trainer.prepare_data(test_size=config["data"]["test_size"])
    
    print("\n" + "="*60)
    print("TRAINING SPARSE AUTOENCODER")
    print("="*60)
    
    # Train sparse autoencoder
    train_losses, recon_losses, sparse_losses = trainer.train_autoencoder(
        hidden_dim=config["autoencoder"]["hidden_dim"],
        epochs=config["autoencoder"]["epochs"],
        batch_size=config["autoencoder"]["batch_size"],
        lr=config["autoencoder"]["learning_rate"],
        sparsity_weight=config["autoencoder"]["sparsity_weight"],
        sparsity_method=config["autoencoder"]["sparsity_method"]
    )
    
    print("\n" + "="*60)
    print("EXTRACTING LATENT FACTORS")
    print("="*60)
    
    # Extract latent factors
    all_factors, train_factors, test_factors = trainer.extract_latent_factors()
    
    print("\n" + "="*60)
    print("TRAINING REWARD PREDICTORS")
    print("="*60)
    
    # Train reward predictors
    trainer.train_reward_predictors(all_factors, train_factors, test_factors)
    
    print("\n" + "="*60)
    print("ANALYZING FACTOR SPECIFICITY")
    print("="*60)
    
    # Analyze factor specificity
    correlation_df, factor_specificity = trainer.analyze_factor_specificity(all_factors)
    
    if config["output"]["save_plots"]:
        print("\n" + "="*60)
        print("GENERATING PLOTS")
        print("="*60)
        
        # Plot results
        trainer.plot_results(train_losses, all_factors, correlation_df, config["output"]["results_dir"])
    
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    # Save results
    results_path = os.path.join(config["output"]["results_dir"], "sparse_autoencoder_results.pkl")
    trainer.save_results(all_factors, correlation_df, factor_specificity, results_path)
    
    # Generate comprehensive report
    generate_report(trainer, all_factors, correlation_df, factor_specificity, config)
    
    return trainer, all_factors, correlation_df, factor_specificity

def generate_report(trainer, all_factors, correlation_df, factor_specificity, config):
    """Generate a comprehensive analysis report."""
    
    report_path = os.path.join(config["output"]["results_dir"], "analysis_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("SPARSE AUTOENCODER CHEMICAL FACTOR ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Dataset summary
        f.write("DATASET SUMMARY:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total molecules analyzed: {len(trainer.df)}\n")
        f.write(f"Embedding dimension: {trainer.embeddings.shape[1]}\n")
        f.write(f"Number of latent factors discovered: {all_factors.shape[1]}\n")
        f.write(f"Number of reward signals: {len(trainer.rewards.columns)}\n")
        f.write(f"Train/test split: {len(trainer.train_indices)}/{len(trainer.test_indices)}\n\n")
        
        # Model performance
        f.write("REWARD PREDICTION PERFORMANCE:\n")
        f.write("-" * 40 + "\n")
        for reward_name, perf in trainer.reward_predictors.items():
            f.write(f"{reward_name:20s}: Train R² = {perf['train_r2']:.3f}, Test R² = {perf['test_r2']:.3f}\n")
        f.write("\n")
        
        # Factor analysis
        f.write("LATENT FACTOR ANALYSIS:\n")
        f.write("-" * 40 + "\n")
        
        # Sparsity statistics
        sparsity = (all_factors > 0.1).mean(axis=0)
        f.write(f"Sparsity Statistics (fraction of molecules with factor > 0.1):\n")
        f.write(f"  Mean: {sparsity.mean():.3f}\n")
        f.write(f"  Std:  {sparsity.std():.3f}\n")
        f.write(f"  Min:  {sparsity.min():.3f}\n")
        f.write(f"  Max:  {sparsity.max():.3f}\n\n")
        
        # Top factor-reward associations
        f.write("TOP FACTOR-REWARD ASSOCIATIONS:\n")
        f.write("-" * 40 + "\n")
        
        # Sort factors by strongest correlation magnitude
        factor_strength = {}
        for factor_name, specificity in factor_specificity.items():
            factor_strength[factor_name] = abs(specificity['correlation'])
        
        sorted_factors = sorted(factor_strength.items(), key=lambda x: x[1], reverse=True)
        
        for factor_name, strength in sorted_factors[:10]:  # Top 10 factors
            specificity = factor_specificity[factor_name]
            reward = specificity['strongest_reward']
            corr = specificity['correlation']
            f.write(f"{factor_name:12s} → {reward:20s} (r = {corr:6.3f})\n")
        f.write("\n")
        
        # Reward-specific factors
        f.write("REWARD-SPECIFIC FACTOR SUMMARY:\n")
        f.write("-" * 40 + "\n")
        for reward in trainer.rewards.columns:
            # Find factors most correlated with this reward
            reward_corrs = correlation_df[reward].abs().sort_values(ascending=False)
            top_factors = reward_corrs.head(3)
            
            f.write(f"{reward:20s}: ")
            factor_list = []
            for factor_name, corr in top_factors.items():
                factor_list.append(f"{factor_name}({corr:.3f})")
            f.write(", ".join(factor_list) + "\n")
        f.write("\n")
        
        # Configuration used
        f.write("CONFIGURATION USED:\n")
        f.write("-" * 40 + "\n")
        f.write(json.dumps(config, indent=2))
    
    print(f"Comprehensive report saved to: {report_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Sparse Autoencoder Analysis for Chemical Factors")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--quick", action="store_true", help="Run quick analysis with default settings")
    parser.add_argument("--hidden-dim", type=int, default=32, help="Number of latent factors")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--embedding-file", type=str, help="Path to embedding file")
    
    args = parser.parse_args()
    
    run_analysis_with_config(args.config)
