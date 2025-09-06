"""
Chemical Motif Probe Training

This module trains probes to predict chemical motifs using graph embeddings
extracted from the SynFlowNet model. It uses AUROC as the primary metric
for evaluating motif classification performance.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import ast
import pickle
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
MOTIFS = {
    # --- Functional groups ---
    "alcohol":            "[OX2H][#6]",                     # includes phenols
    "phenol":             "[OX2H]c",                        # phenols only
    "carboxylic_acid":    "[CX3](=O)[OX2H1]",
    "ester":              "[CX3](=O)[OX2][#6]",
    "ether":              "[OX2;!$([OX2][CX3]=O)][#6]",     # exclude esters
    "ketone":             "[#6][CX3](=O)[#6]",
    "aldehyde":           "[#6][CX3H](=O)",
    "amine_primary":      "[NX3;H2][#6]",
    "amine_secondary":    "[NX3;H1]([#6])[#6]",
    "amine_tertiary":     "[NX3;H0]([#6])([#6])[#6]",
    "amide":              "[NX3][CX3](=O)[#6]",
    "nitro":              "[N+](=O)[O-]",
    "nitrile":            "[CX2]#N",
    "halogen_F":          "[F]",
    "halogen_Cl":         "[Cl]",
    "halogen_Br":         "[Br]",
    "halogen_I":          "[I]",
    "sulfone":            "[SX4](=O)(=O)[#6]",
    "sulfoxide":          "[SX3](=O)[#6]",
    "thiol":              "[SX2H][#6]",
    "phosphate":          "[PX4](=O)([OX1,OX2H,OX2-])([OX1,OX2H,OX2-])[OX1,OX2H,OX2-]",

    # --- Ring systems ---
    "benzene":            "c1ccccc1",
    "pyridine":           "n1ccccc1",
    "pyrrole":            "[nH]1cccc1",
    "furan":              "o1cccc1",
    "thiophene":          "s1cccc1",
    "imidazole":          "c1ncc[nH]1",
    "indole":             "c1ccc2[nH]ccc2c1",
    "quinoline":          "c1ccc2ncccc2c1",
    "naphthalene":        "c1ccc2ccccc2c1",
    "cyclohexane":        "C1CCCCC1",
    "cyclopentane":       "C1CCCC1",
    "cyclopropane":       "C1CC1",

    # --- Structural motifs ---
    "double_bond":        "[#6]=[#6]",
    "triple_bond":        "[#6]#[#6]",
    "aromatic_N":         "[n]",
    "aromatic_O":         "[o]",
    "aromatic_S":         "[s]",
    "quaternary_carbon":  "[CX4](C)(C)(C)C",
    "terminal_alkyne":    "[#6]#[#6H]",
    "allyl":              "[#6]=[#6]-[CH2]",
    "vinyl":              "[#6]=[#6H1]",
    "phenyl":             "c1ccccc1",
    "methyl":             "[CH3]",
    "methylene":          "[CH2]",
    "methine":            "[CH]",

    # --- Pharmacophore-like motifs ---
    "hbond_donor":        "[$([OX2H]),$([NX3;H1,H2,H3;+0]),$([SX2H])]",
    "hbond_acceptor":     "[$([O;H0;v2;-]),$([O;H0;v2]),$([N;H0;v3;!+]),$([S;H0;v2])]",
    "positive_ionizable": "[$([N+]),$([NX3;H3+]),$([NX3;H2+]),$([NX3;H1+])]",
    "negative_ionizable": "[$([O-]),$(C(=O)[O-]),$(S(=O)(=O)[O-]),$(P(=O)(O)(O)[O-])]",
    "aromatic_ring":      "[a]"
}

class ChemicalMotifExtractor:
    """Extract chemical motifs from SMILES strings using RDKit."""
    
    def __init__(self):
        # Define common chemical motifs as SMARTS patterns
        self.motifs = MOTIFS
        
        # Compile SMARTS patterns
        self.compiled_patterns = {}
        for name, pattern in self.motifs.items():
            try:
                self.compiled_patterns[name] = Chem.MolFromSmarts(pattern)
            except Exception as e:
                logger.warning(f"Could not compile pattern {name}: {pattern} - {e}")
    
    def extract_motifs(self, smiles: str) -> Dict[str, int]:
        """Extract motif counts from a SMILES string."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {name: 0 for name in self.compiled_patterns.keys()}
        
        motif_counts = {}
        for name, pattern in self.compiled_patterns.items():
            if pattern is not None:
                matches = mol.GetSubstructMatches(pattern)
                motif_counts[name] = len(matches)
            else:
                motif_counts[name] = 0
        
        return motif_counts
    
    def extract_motifs_batch(self, smiles_list: List[str]) -> pd.DataFrame:
        """Extract motifs for a batch of SMILES strings."""
        results = []
        for smiles in tqdm(smiles_list, desc="Extracting motifs"):
            motif_counts = self.extract_motifs(smiles)
            motif_counts['smiles'] = smiles
            results.append(motif_counts)
        
        return pd.DataFrame(results)

class MotifDataset(Dataset):
    """Dataset for motif classification."""
    
    def __init__(self, embeddings: torch.Tensor, labels: torch.Tensor):
        self.embeddings = embeddings
        self.labels = labels
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

class MotifProbe(nn.Module):
    """Neural network probe for motif classification."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_motifs: int = 1, dropout: float = 0.2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_motifs)
            # No sigmoid here - BCEWithLogitsLoss will handle it
        )
    
    def forward(self, x):
        return self.classifier(x)

class MotifProbeTrainer:
    """Train and evaluate motif classification probes."""
    
    def __init__(self, embedding_file: str, device: str = 'auto'):
        self.embedding_file = embedding_file
        self.device = self._get_device(device)
        self.motif_extractor = ChemicalMotifExtractor()
        self.scaler = StandardScaler()
        
        # Load data
        self.df = None
        self.embeddings = None
        self.motif_labels = None
        self.motif_names = None
        
    def _get_device(self, device: str) -> torch.device:
        """Get the appropriate device."""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)
    
    def load_data(self):
        """Load embeddings and extract motifs."""
        logger.info(f"Loading data from {self.embedding_file}")
        
        # Load CSV data
        self.df = pd.read_csv(self.embedding_file)
        logger.info(f"Loaded {len(self.df)} molecules")
        
        # Parse embeddings
        embeddings_list = []
        valid_indices = []
        
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Parsing embeddings"):
            try:
                # Parse graph embeddings from string representation
                if pd.notna(row['graph_embeddings']):
                    emb_str = row['graph_embeddings']
                    # Remove 'tensor(' and ')' and convert to numpy
                    if emb_str.startswith('tensor('):
                        emb_str = emb_str[7:-1]  # Remove 'tensor(' and ')'
                    
                    # Parse the tensor values
                    emb_values = ast.literal_eval(emb_str) if isinstance(emb_str, str) else emb_str
                    emb_tensor = torch.tensor(emb_values).float()
                    
                    embeddings_list.append(emb_tensor)
                    valid_indices.append(idx)
            except Exception as e:
                logger.warning(f"Could not parse embedding for row {idx}: {e}")
                continue
        
        if not embeddings_list:
            raise ValueError("No valid embeddings found in the data")
        
        # Stack embeddings
        self.embeddings = torch.stack(embeddings_list)
        self.df = self.df.iloc[valid_indices].reset_index(drop=True)
        
        logger.info(f"Successfully parsed {len(self.embeddings)} embeddings")
        logger.info(f"Embedding dimension: {self.embeddings.shape[1]}")
        
        # Extract motifs
        logger.info("Extracting chemical motifs...")
        motif_df = self.motif_extractor.extract_motifs_batch(self.df['smiles'].tolist())
        
        # Prepare motif labels
        self.motif_names = [col for col in motif_df.columns if col != 'smiles']
        motif_matrix = motif_df[self.motif_names].values
        
        # Convert to binary labels (presence/absence)
        self.motif_labels = torch.tensor((motif_matrix > 0).astype(int), dtype=torch.float)
        
        logger.info(f"Extracted {len(self.motif_names)} motif types")
        logger.info(f"Motif prevalence:")
        for i, name in enumerate(self.motif_names):
            prevalence = self.motif_labels[:, i].mean().item()
            logger.info(f"  {name}: {prevalence:.3f}")
    
    def prepare_data(self, test_size: float = 0.2, val_size: float = 0.1):
        """Prepare train/val/test splits."""
        # Handle multi-dimensional embeddings by flattening if needed
        embeddings_np = self.embeddings.numpy()
        original_shape = embeddings_np.shape
        if len(original_shape) > 2:
            # Flatten to 2D for StandardScaler
            embeddings_np = embeddings_np.reshape(original_shape[0], -1)
            logger.info(f"Reshaped embeddings from {original_shape} to {embeddings_np.shape}")
        
        # Split data FIRST to avoid data leakage
        indices = np.arange(len(embeddings_np))
        train_indices, test_indices = train_test_split(
            indices, test_size=test_size, random_state=42, stratify=None
        )
        train_indices, val_indices = train_test_split(
            train_indices, test_size=val_size/(1-test_size), random_state=42, stratify=None
        )
        
        # Fit scaler ONLY on training data to avoid data leakage
        train_embeddings_np = embeddings_np[train_indices]
        self.scaler.fit(train_embeddings_np)
        
        # Apply scaling to all splits using fitted scaler
        embeddings_scaled = self.scaler.transform(embeddings_np)
        embeddings_tensor = torch.tensor(embeddings_scaled, dtype=torch.float)
        
        # Create splits
        self.train_embeddings = embeddings_tensor[train_indices]
        self.train_labels = self.motif_labels[train_indices]
        
        self.val_embeddings = embeddings_tensor[val_indices]
        self.val_labels = self.motif_labels[val_indices]
        
        self.test_embeddings = embeddings_tensor[test_indices]
        self.test_labels = self.motif_labels[test_indices]
        
        logger.info(f"Data split: Train {len(train_indices)}, Val {len(val_indices)}, Test {len(test_indices)}")
    
    def train_probe(self, motif_idx: int, epochs: int = 100, batch_size: int = 64, lr: float = 0.001):
        """Train a probe for a specific motif."""
        motif_name = self.motif_names[motif_idx]
        logger.info(f"Training probe for motif: {motif_name}")
        
        # Create datasets
        train_dataset = MotifDataset(self.train_embeddings, self.train_labels[:, motif_idx])
        val_dataset = MotifDataset(self.val_embeddings, self.val_labels[:, motif_idx])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Create model
        model = MotifProbe(
            input_dim=self.train_embeddings.shape[1],
            hidden_dim=256,
            num_motifs=1
        ).to(self.device)
        
        # Loss and optimizer - use BCEWithLogitsLoss for better numerical stability
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        # NOTE ReduceLROnPlateau defaults to mode='min', but you’re tracking AUC (higher is better).
        # REDO ANALYSIS WITH THIS FIXED
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-6)

        # Training loop
        best_val_auc = 0
        best_model_state = model.state_dict().copy()  # Initialize with current state
        train_losses = []
        val_aucs = []
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            for embeddings, labels in train_loader:
                embeddings, labels = embeddings.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(embeddings).squeeze()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation
            model.eval()
            val_preds = []
            val_targets = []
            
            with torch.no_grad():
                for embeddings, labels in val_loader:
                    embeddings, labels = embeddings.to(self.device), labels.to(self.device)
                    logits = model(embeddings).squeeze()
                    # Apply sigmoid for probability calculation
                    probabilities = torch.sigmoid(logits)
                    val_preds.extend(probabilities.cpu().numpy())
                    val_targets.extend(labels.cpu().numpy())
            
            val_auc = roc_auc_score(val_targets, val_preds) if len(set(val_targets)) > 1 else 0.5
            val_aucs.append(val_auc)
            
            scheduler.step(val_auc)
            
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_state = model.state_dict().copy()
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss {train_loss:.4f}, Val AUC {val_auc:.4f}")
        
        # Load best model
        model.load_state_dict(best_model_state)
        
        return model, train_losses, val_aucs, best_val_auc
    
    def evaluate_probe(self, model: nn.Module, motif_idx: int):
        """Evaluate a trained probe."""
        motif_name = self.motif_names[motif_idx]
        
        # Test dataset
        test_dataset = MotifDataset(self.test_embeddings, self.test_labels[:, motif_idx])
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        model.eval()
        test_preds = []
        test_targets = []
        
        with torch.no_grad():
            for embeddings, labels in test_loader:
                embeddings, labels = embeddings.to(self.device), labels.to(self.device)
                logits = model(embeddings).squeeze()
                # Apply sigmoid for probability calculation
                probabilities = torch.sigmoid(logits)
                test_preds.extend(probabilities.cpu().numpy())
                test_targets.extend(labels.cpu().numpy())
        
        # Calculate metrics
        test_auc = roc_auc_score(test_targets, test_preds) if len(set(test_targets)) > 1 else 0.5
        test_ap = average_precision_score(test_targets, test_preds) if len(set(test_targets)) > 1 else 0.5
        
        # Binary predictions for classification report
        binary_preds = (np.array(test_preds) > 0.5).astype(int)
        
        return {
            'motif': motif_name,
            'auc': test_auc,
            'average_precision': test_ap,
            'predictions': test_preds,
            'targets': test_targets,
            'binary_predictions': binary_preds
        }
    
    def train_all_probes(self, epochs: int = 100, min_prevalence: float = 0.01):
        """Train probes for all motifs with sufficient prevalence."""
        results = {}
        
        # Filter motifs by prevalence
        valid_motifs = []
        for i, name in enumerate(self.motif_names):
            prevalence = self.motif_labels[:, i].mean().item()
            if prevalence >= min_prevalence and prevalence <= (1 - min_prevalence):
                valid_motifs.append((i, name, prevalence))
        
        logger.info(f"Training probes for {len(valid_motifs)} motifs (prevalence >= {min_prevalence})")
        
        for motif_idx, motif_name, prevalence in tqdm(valid_motifs, desc="Training probes"):
            try:
                model, train_losses, val_aucs, best_val_auc = self.train_probe(
                    motif_idx, epochs=epochs
                )
                eval_results = self.evaluate_probe(model, motif_idx)
                
                results[motif_name] = {
                    'model': model,
                    'train_losses': train_losses,
                    'val_aucs': val_aucs,
                    'best_val_auc': best_val_auc,
                    'test_results': eval_results,
                    'prevalence': prevalence
                }
                
                logger.info(f"{motif_name}: Val AUC {best_val_auc:.4f}, Test AUC {eval_results['auc']:.4f}")
                
            except Exception as e:
                logger.error(f"Failed to train probe for {motif_name}: {e}")
                continue
        
        return results
    
    def plot_results(self, results: Dict, save_path: str = None):
        """Plot training results and performance metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. AUC scores distribution
        aucs = [r['test_results']['auc'] for r in results.values()]
        axes[0, 0].hist(aucs, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('Test AUC')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Distribution of Test AUC Scores')
        axes[0, 0].axvline(np.mean(aucs), color='red', linestyle='--', label=f'Mean: {np.mean(aucs):.3f}')
        axes[0, 0].legend()
        
        # 2. AUC vs Prevalence
        prevalences = [r['prevalence'] for r in results.values()]
        axes[0, 1].scatter(prevalences, aucs, alpha=0.7)
        axes[0, 1].set_xlabel('Motif Prevalence')
        axes[0, 1].set_ylabel('Test AUC')
        axes[0, 1].set_title('Test AUC vs Motif Prevalence')
        
        # 3. Top performing motifs
        sorted_results = sorted(results.items(), key=lambda x: x[1]['test_results']['auc'], reverse=True)
        top_motifs = sorted_results[:15]
        
        motif_names = [name for name, _ in top_motifs]
        motif_aucs = [result['test_results']['auc'] for _, result in top_motifs]
        
        axes[1, 0].barh(range(len(motif_names)), motif_aucs)
        axes[1, 0].set_yticks(range(len(motif_names)))
        axes[1, 0].set_yticklabels(motif_names)
        axes[1, 0].set_xlabel('Test AUC')
        axes[1, 0].set_title('Top 15 Motifs by AUC')
        
        # 4. Learning curves for best motif
        best_motif_name = top_motifs[0][0]
        best_result = results[best_motif_name]
        
        epochs_range = range(len(best_result['train_losses']))
        axes[1, 1].plot(epochs_range, best_result['train_losses'], label='Train Loss')
        ax2 = axes[1, 1].twinx()
        ax2.plot(epochs_range, best_result['val_aucs'], 'r-', label='Val AUC')
        
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss', color='b')
        ax2.set_ylabel('AUC', color='r')
        axes[1, 1].set_title(f'Learning Curves: {best_motif_name}')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plots saved to {save_path}")
        
        plt.show()
    
    def save_results(self, results: Dict, save_path: str):
        """Save training results."""
        # Prepare summary data
        summary_data = []
        for motif_name, result in results.items():
            summary_data.append({
                'motif': motif_name,
                'prevalence': result['prevalence'],
                'best_val_auc': result['best_val_auc'],
                'test_auc': result['test_results']['auc'],
                'test_average_precision': result['test_results']['average_precision']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('test_auc', ascending=False)
        
        # Save summary
        summary_path = save_path.replace('.pkl', '_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Summary saved to {summary_path}")
        
        # Save full results
        with open(save_path, 'wb') as f:
            pickle.dump(results, f)
        logger.info(f"Full results saved to {save_path}")
        
        return summary_df

def main():
    """Main training pipeline."""
    # Configuration
    embedding_file = "/home/ubuntu/synflownet_public/src/neurips_poster/results_qed_run/embeddings.csv"
    results_dir = "/home/ubuntu/synflownet_public/src/neurips_poster/results_qed_run"
    # actually 30676 molecules in results_7669
    # actually 32054 molecules in results_qed_run
    
    # Initialize trainer
    trainer = MotifProbeTrainer(embedding_file)
    
    # Load and prepare data
    trainer.load_data()
    trainer.prepare_data()
    
    # Train all probes
    results = trainer.train_all_probes(epochs=100, min_prevalence=0.02)
    
    # Plot and save results
    trainer.plot_results(results, save_path=f"{results_dir}/motif_probe_results.png")
    summary_df = trainer.save_results(results, f"{results_dir}/motif_probe_results.pkl")
    
    # Print summary
    print("\n" + "="*80)
    print("MOTIF PROBE TRAINING SUMMARY")
    print("="*80)
    print(f"Total motifs trained: {len(results)}")
    print(f"Mean test AUC: {summary_df['test_auc'].mean():.4f}")
    print(f"Best performing motif: {summary_df.iloc[0]['motif']} (AUC: {summary_df.iloc[0]['test_auc']:.4f})")
    print("\nTop 10 motifs by AUC:")
    print(summary_df.head(10)[['motif', 'prevalence', 'test_auc']].to_string(index=False))

if __name__ == "__main__":
    main()
