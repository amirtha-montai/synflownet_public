"""
Sparse Autoencoder for Chemical Reward Factor Discovery

This module trains sparse autoencoders on graph embeddings to discover
reward-specific latent factors that activate on similar chemistries.
The sparse autoencoder learns interpretable features that can reveal
chemical patterns associated with specific rewards or properties.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import ast
import pickle
import logging
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SparseAutoencoder(nn.Module):
    """Sparse autoencoder for discovering interpretable chemical factors."""
    
    def __init__(self, input_dim: int, hidden_dim: int, sparsity_weight: float = 0.01, 
                 sparsity_target: float = 0.05, dropout: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sparsity_weight = sparsity_weight
        self.sparsity_target = sparsity_target
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, input_dim)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """Forward pass through autoencoder."""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded
    
    def kl_divergence_sparsity(self, hidden_activations):
        """Compute KL divergence for sparsity regularization with numerical stability."""
        # Apply sigmoid to ensure activations are in (0,1)
        activations_sigmoid = torch.sigmoid(hidden_activations)
        
        # Compute average activation for each neuron
        rho_hat = torch.mean(activations_sigmoid, dim=0)
        rho = self.sparsity_target
        
        # Clamp rho_hat to avoid log(0) or log(1)
        rho_hat = torch.clamp(rho_hat, min=1e-7, max=1-1e-7)
        
        # KL divergence: KL(rho || rho_hat) with numerical stability
        kl = rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))
        
        # Check for NaN and return zero if found
        kl_sum = torch.sum(kl)
        if torch.isnan(kl_sum) or torch.isinf(kl_sum):
            return torch.tensor(0.0, device=hidden_activations.device, requires_grad=True)
        
        return kl_sum
    
    def l1_sparsity(self, hidden_activations):
        """Alternative L1 sparsity regularization (more stable)."""
        return torch.mean(torch.abs(hidden_activations))

class RewardPredictor(nn.Module):
    """Neural network to predict rewards from latent factors."""
    
    def __init__(self, latent_dim: int, output_dim: int = 1, dropout: float = 0.2):
        super().__init__()
        
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim // 2, latent_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim // 4, output_dim)
        )
    
    def forward(self, x):
        return self.predictor(x)

class SparseAutoencoderTrainer:
    """Train sparse autoencoders on chemical embeddings."""
    
    def __init__(self, embedding_file: str, device: str = 'auto'):
        self.embedding_file = embedding_file
        self.device = self._get_device(device)
        self.scaler = StandardScaler()
        
        # Data containers
        self.df = None
        self.embeddings = None
        self.rewards = None
        self.molecular_properties = None
        
        # Models
        self.autoencoder = None
        self.reward_predictors = {}
        
    def _get_device(self, device: str) -> torch.device:
        """Get the appropriate device."""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)
    
    def load_data(self):
        """Load embeddings and compute molecular properties as rewards."""
        logger.info(f"Loading data from {self.embedding_file}")
        
        # Load CSV data
        self.df = pd.read_csv(self.embedding_file)
        logger.info(f"Loaded {len(self.df)} molecules")
        
        # Parse embeddings
        embeddings_list = []
        valid_indices = []
        
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Parsing embeddings"):
            try:
                if pd.notna(row['graph_embeddings']):
                    emb_str = row['graph_embeddings']
                    if emb_str.startswith('tensor('):
                        emb_str = emb_str[7:-1]  # Remove 'tensor(' and ')'
                    
                    # Parse the tensor values
                    emb_values = ast.literal_eval(emb_str) if isinstance(emb_str, str) else emb_str
                    emb_tensor = torch.tensor(emb_values).float()
                    
                    # Flatten if multi-dimensional
                    if len(emb_tensor.shape) > 1:
                        emb_tensor = emb_tensor.flatten()
                    
                    embeddings_list.append(emb_tensor.numpy())
                    valid_indices.append(idx)
            except Exception as e:
                logger.warning(f"Could not parse embedding for row {idx}: {e}")
                continue
        
        if not embeddings_list:
            raise ValueError("No valid embeddings found in the data")
        
        # Store embeddings
        self.embeddings = np.array(embeddings_list)
        self.df = self.df.iloc[valid_indices].reset_index(drop=True)
        
        logger.info(f"Successfully parsed {len(embeddings_list)} embeddings with dimension {self.embeddings.shape[1]}")
        
        # Compute molecular properties as reward signals
        self._compute_molecular_rewards()
        
    def _compute_molecular_rewards(self):
        """Compute molecular properties as reward signals."""
        logger.info("Computing molecular properties as reward signals...")
        
        rewards = []
        properties = []
        
        for smiles in tqdm(self.df['smiles'], desc="Computing properties"):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                # Use default values for invalid molecules
                mol_props = {
                    'logp': 0.0,
                    'mw': 0.0,
                    'tpsa': 0.0,
                    'qed': 0.0,
                    'sa_score': 0.0,
                    'num_rings': 0,
                    'num_rotatable_bonds': 0,
                    'num_hbd': 0,
                    'num_hba': 0
                }
            else:
                try:
                    mol_props = {
                        'logp': Descriptors.MolLogP(mol),
                        'mw': Descriptors.MolWt(mol),
                        'tpsa': Descriptors.TPSA(mol),
                        'qed': Descriptors.qed(mol),
                        'sa_score': self._compute_sa_score(mol),
                        'num_rings': rdMolDescriptors.CalcNumRings(mol),
                        'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                        'num_hbd': Descriptors.NumHDonors(mol),
                        'num_hba': Descriptors.NumHAcceptors(mol)
                    }
                    
                    # Replace any NaN values with defaults
                    for key, value in mol_props.items():
                        if pd.isna(value) or np.isnan(value) or np.isinf(value):
                            mol_props[key] = 0.0
                            
                except Exception as e:
                    logger.warning(f"Error computing properties for SMILES {smiles}: {e}")
                    # Use default values for problematic molecules
                    mol_props = {
                        'logp': 0.0,
                        'mw': 0.0,
                        'tpsa': 0.0,
                        'qed': 0.0,
                        'sa_score': 0.0,
                        'num_rings': 0,
                        'num_rotatable_bonds': 0,
                        'num_hbd': 0,
                        'num_hba': 0
                    }
            
            properties.append(mol_props)
            
            # Create composite reward signals with safety checks
            reward_dict = {
                'drug_likeness': max(0.0, min(1.0, mol_props['qed'])),  # Clamp QED score to [0,1]
                'complexity': max(0.0, min(1.0, 1.0 - mol_props['sa_score'])),  # Clamp complexity
                'lipophilicity': max(-10.0, min(10.0, mol_props['logp'])),  # Reasonable LogP range
                'size': max(0.0, min(2.0, mol_props['mw'] / 500.0)),  # Normalized MW, max 1000 Da
                'polarity': max(0.0, min(2.0, mol_props['tpsa'] / 200.0)),  # Normalized TPSA, max 400
                'flexibility': max(0.0, min(2.0, mol_props['num_rotatable_bonds'] / 10.0))  # Max 20 rotatable bonds
            }
            
            # Final safety check: replace any remaining problematic values
            for key, value in reward_dict.items():
                if pd.isna(value) or np.isnan(value) or np.isinf(value):
                    reward_dict[key] = 0.0
                    
            rewards.append(reward_dict)
        
        self.molecular_properties = pd.DataFrame(properties)
        self.rewards = pd.DataFrame(rewards)
        
        logger.info(f"Computed {len(self.rewards.columns)} reward signals")
        logger.info(f"Reward signals: {list(self.rewards.columns)}")
        
    def _compute_sa_score(self, mol):
        """Compute synthetic accessibility score (simplified version)."""
        try:
            # This is a simplified version - in practice, you might use rdkit-contrib
            # or other libraries for a more accurate SA score
            return min(1.0, Descriptors.NumAliphaticRings(mol) * 0.1 + 
                      Descriptors.NumAromaticRings(mol) * 0.05 + 0.5)
        except:
            return 0.5
    
    def prepare_data(self, test_size: float = 0.2):
        """Prepare train/test splits."""
        # Normalize embeddings
        train_indices, test_indices = train_test_split(
            np.arange(len(self.embeddings)), 
            test_size=test_size, 
            random_state=42
        )
        
        # Fit scaler on training data only
        train_embeddings = self.embeddings[train_indices]
        self.scaler.fit(train_embeddings)
        
        # Transform all data
        embeddings_scaled = self.scaler.transform(self.embeddings)
        
        # Create splits
        self.train_embeddings = torch.tensor(embeddings_scaled[train_indices], dtype=torch.float32)
        self.test_embeddings = torch.tensor(embeddings_scaled[test_indices], dtype=torch.float32)
        
        self.train_rewards = self.rewards.iloc[train_indices].values
        self.test_rewards = self.rewards.iloc[test_indices].values
        
        self.train_indices = train_indices
        self.test_indices = test_indices
        
        logger.info(f"Data split: Train {len(train_indices)}, Test {len(test_indices)}")
    
    def train_autoencoder(self, hidden_dim: int = 64, epochs: int = 200, 
                         batch_size: int = 32, lr: float = 0.001,
                         sparsity_weight: float = 0.01, sparsity_method: str = 'l1'):
        """Train the sparse autoencoder."""
        logger.info(f"Training sparse autoencoder with {hidden_dim} latent factors using {sparsity_method} sparsity...")
        
        # Create autoencoder
        input_dim = self.train_embeddings.shape[1]
        self.autoencoder = SparseAutoencoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            sparsity_weight=sparsity_weight
        ).to(self.device)
        
        # Create data loader
        train_dataset = TensorDataset(self.train_embeddings)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Optimizer and scheduler
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Training loop
        train_losses = []
        reconstruction_losses = []
        sparsity_losses = []
        
        self.autoencoder.train()
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_recon_loss = 0
            epoch_sparse_loss = 0
            
            for batch_idx, (batch_embeddings,) in enumerate(train_loader):
                batch_embeddings = batch_embeddings.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                reconstructed, encoded = self.autoencoder(batch_embeddings)
                
                # Reconstruction loss
                recon_loss = F.mse_loss(reconstructed, batch_embeddings)
                
                # Sparsity loss with method selection
                if sparsity_method == 'kl':
                    sparse_loss = self.autoencoder.kl_divergence_sparsity(encoded)
                elif sparsity_method == 'l1':
                    sparse_loss = self.autoencoder.l1_sparsity(encoded)
                else:
                    # Default to L1 if unknown method
                    sparse_loss = self.autoencoder.l1_sparsity(encoded)
                
                # Total loss with NaN protection
                total_loss = recon_loss + sparsity_weight * sparse_loss
                
                # Check for NaN/Inf in losses
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    logger.warning(f"NaN/Inf detected in total_loss at epoch {epoch}, batch {batch_idx}. Skipping batch.")
                    continue
                
                # Backward pass
                total_loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.autoencoder.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Use safe values for logging
                epoch_loss += total_loss.item() if not torch.isnan(total_loss) else 0.0
                epoch_recon_loss += recon_loss.item() if not torch.isnan(recon_loss) else 0.0
                epoch_sparse_loss += sparse_loss.item() if not torch.isnan(sparse_loss) else 0.0
            
            # Average losses
            epoch_loss /= len(train_loader)
            epoch_recon_loss /= len(train_loader)
            epoch_sparse_loss /= len(train_loader)
            
            train_losses.append(epoch_loss)
            reconstruction_losses.append(epoch_recon_loss)
            sparsity_losses.append(epoch_sparse_loss)
            
            scheduler.step(epoch_loss)
            
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}: Total Loss {epoch_loss:.6f}, "
                          f"Recon Loss {epoch_recon_loss:.6f}, Sparse Loss {epoch_sparse_loss:.6f}")
        
        logger.info("Autoencoder training completed!")
        return train_losses, reconstruction_losses, sparsity_losses
    
    def extract_latent_factors(self):
        """Extract latent factors from trained autoencoder."""
        if self.autoencoder is None:
            raise ValueError("Autoencoder must be trained first!")
        
        self.autoencoder.eval()
        
        # Extract factors for all data
        all_embeddings = torch.tensor(self.scaler.transform(self.embeddings), dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            _, latent_factors = self.autoencoder(all_embeddings)
            latent_factors = latent_factors.cpu().numpy()
        
        # Split into train/test
        train_factors = latent_factors[self.train_indices]
        test_factors = latent_factors[self.test_indices]
        
        logger.info(f"Extracted {latent_factors.shape[1]} latent factors")
        return latent_factors, train_factors, test_factors
    
    def train_reward_predictors(self, latent_factors, train_factors, test_factors):
        """Train reward predictors from latent factors."""
        logger.info("Training reward predictors from latent factors...")
        
        latent_dim = train_factors.shape[1]
        
        for reward_name in self.rewards.columns:
            logger.info(f"Training predictor for {reward_name}...")
            
            # Create reward predictor
            predictor = RewardPredictor(latent_dim=latent_dim).to(self.device)
            
            # Prepare data
            train_targets = torch.tensor(self.train_rewards[:, self.rewards.columns.get_loc(reward_name)], 
                                       dtype=torch.float32).to(self.device)
            test_targets = torch.tensor(self.test_rewards[:, self.rewards.columns.get_loc(reward_name)], 
                                      dtype=torch.float32).to(self.device)
            
            train_factors_tensor = torch.tensor(train_factors, dtype=torch.float32).to(self.device)
            test_factors_tensor = torch.tensor(test_factors, dtype=torch.float32).to(self.device)
            
            # Training setup
            optimizer = torch.optim.Adam(predictor.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            # Train predictor
            predictor.train()
            for epoch in range(100):
                optimizer.zero_grad()
                predictions = predictor(train_factors_tensor).squeeze()
                loss = criterion(predictions, train_targets)
                loss.backward()
                optimizer.step()
            
            # Evaluate predictor
            predictor.eval()
            with torch.no_grad():
                train_preds = predictor(train_factors_tensor).squeeze().cpu().numpy()
                test_preds = predictor(test_factors_tensor).squeeze().cpu().numpy()
                
                train_r2 = r2_score(self.train_rewards[:, self.rewards.columns.get_loc(reward_name)], train_preds)
                test_r2 = r2_score(self.test_rewards[:, self.rewards.columns.get_loc(reward_name)], test_preds)
                
                logger.info(f"{reward_name} - Train R²: {train_r2:.3f}, Test R²: {test_r2:.3f}")
            
            self.reward_predictors[reward_name] = {
                'model': predictor,
                'train_r2': train_r2,
                'test_r2': test_r2
            }
    
    def analyze_factor_specificity(self, latent_factors):
        """Analyze which factors are specific to which rewards."""
        logger.info("Analyzing factor-reward relationships...")
        
        # Clean data: replace NaN/inf values
        clean_factors = np.nan_to_num(latent_factors, nan=0.0, posinf=1.0, neginf=-1.0)
        clean_rewards = np.nan_to_num(self.rewards.values, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Check for constant columns (which would cause correlation issues)
        factor_stds = np.std(clean_factors, axis=0)
        reward_stds = np.std(clean_rewards, axis=0)
        
        # Add small noise to constant columns to avoid correlation issues
        constant_factor_mask = factor_stds < 1e-10
        constant_reward_mask = reward_stds < 1e-10
        
        if np.any(constant_factor_mask):
            logger.warning(f"Found {np.sum(constant_factor_mask)} constant factors, adding small noise")
            clean_factors[:, constant_factor_mask] += np.random.normal(0, 1e-6, 
                                                                      (clean_factors.shape[0], np.sum(constant_factor_mask)))
        
        if np.any(constant_reward_mask):
            logger.warning(f"Found {np.sum(constant_reward_mask)} constant rewards, adding small noise")
            clean_rewards[:, constant_reward_mask] += np.random.normal(0, 1e-6, 
                                                                      (clean_rewards.shape[0], np.sum(constant_reward_mask)))
        
        # Compute correlations between factors and rewards
        try:
            correlations = np.corrcoef(clean_factors.T, clean_rewards.T)
            factor_reward_corrs = correlations[:latent_factors.shape[1], latent_factors.shape[1]:]
        except Exception as e:
            logger.error(f"Correlation computation failed: {e}")
            # Fallback: create zero correlation matrix
            factor_reward_corrs = np.zeros((latent_factors.shape[1], clean_rewards.shape[1]))
        
        # Replace any remaining NaN values with 0
        factor_reward_corrs = np.nan_to_num(factor_reward_corrs, nan=0.0)
        
        # Create correlation DataFrame
        factor_names = [f"Factor_{i}" for i in range(latent_factors.shape[1])]
        correlation_df = pd.DataFrame(
            factor_reward_corrs,
            index=factor_names,
            columns=self.rewards.columns
        )
        
        # Find strongest correlations for each factor
        factor_specificity = {}
        for i, factor_name in enumerate(factor_names):
            factor_corrs = correlation_df.iloc[i]
            
            # Handle NaN values in correlations
            abs_corrs = factor_corrs.abs()
            
            # Check if all correlations are NaN
            if abs_corrs.isna().all():
                strongest_reward = self.rewards.columns[0]  # Default to first reward
                strongest_corr = 0.0
                logger.warning(f"All correlations are NaN for {factor_name}, using default values")
            else:
                # Drop NaN values before finding max
                valid_corrs = abs_corrs.dropna()
                if len(valid_corrs) == 0:
                    strongest_reward = self.rewards.columns[0]
                    strongest_corr = 0.0
                else:
                    strongest_reward = valid_corrs.idxmax()
                    strongest_corr = factor_corrs[strongest_reward]
            
            factor_specificity[factor_name] = {
                'strongest_reward': strongest_reward,
                'correlation': strongest_corr,
                'all_correlations': factor_corrs.fillna(0.0).to_dict()  # Fill NaN with 0
            }
        
        return correlation_df, factor_specificity
    
    def plot_results(self, train_losses, latent_factors, correlation_df, save_dir: str):
        """Plot analysis results."""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Training curves
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses)
        plt.title('Autoencoder Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        # 2. Factor activation distribution
        plt.subplot(1, 2, 2)
        factor_activations = np.mean(latent_factors, axis=0)
        plt.bar(range(len(factor_activations)), factor_activations)
        plt.title('Average Factor Activations')
        plt.xlabel('Factor Index')
        plt.ylabel('Average Activation')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/training_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Factor-reward correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            correlation_df,
            annot=True,
            cmap='RdBu_r',
            center=0,
            fmt='.3f',
            cbar_kws={'label': 'Correlation'}
        )
        plt.title('Latent Factor - Reward Correlations')
        plt.xlabel('Reward Signals')
        plt.ylabel('Latent Factors')
        plt.tight_layout()
        plt.savefig(f"{save_dir}/factor_reward_correlations.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # 4. Factor sparsity analysis
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        sparsity_scores = np.mean(latent_factors > 0.1, axis=0)  # Fraction of molecules where factor > 0.1
        plt.bar(range(len(sparsity_scores)), sparsity_scores)
        plt.title('Factor Sparsity (Fraction of Active Molecules)')
        plt.xlabel('Factor Index')
        plt.ylabel('Activation Frequency')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.hist(sparsity_scores, bins=20, alpha=0.7, edgecolor='black')
        plt.title('Distribution of Factor Sparsity')
        plt.xlabel('Activation Frequency')
        plt.ylabel('Number of Factors')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/sparsity_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self, latent_factors, correlation_df, factor_specificity, save_path: str):
        """Save analysis results."""
        results = {
            'latent_factors': latent_factors,
            'correlation_matrix': correlation_df,
            'factor_specificity': factor_specificity,
            'reward_predictors_performance': {name: {
                'train_r2': pred['train_r2'],
                'test_r2': pred['test_r2']
            } for name, pred in self.reward_predictors.items()},
            'molecular_data': {
                'smiles': self.df['smiles'].tolist(),
                'rewards': self.rewards,
                'properties': self.molecular_properties
            }
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(results, f)
        
        # Save correlation matrix as CSV
        correlation_df.to_csv(save_path.replace('.pkl', '_correlations.csv'))
        
        logger.info(f"Results saved to {save_path}")
