"""
Molecular Saliency Analysis using Integrated Gradients

This module provides tools for analyzing molecular property predictions using 
Integrated Gradients (IG) to understand which atoms contribute most to the model's 
decisions. It includes functionality for:
- Computing attribution scores for molecular stop actions
- Identifying important molecular motifs
- Generating counterfactual edits to improve QED scores
- Visualizing attribution heatmaps

Author: SynFlowNet Team
"""

import os
import io
from typing import Dict, List, Tuple, Optional, Set, Union, Any
import torch
import numpy as np
from torch.nn.functional import log_softmax
from rdkit import Chem
from rdkit.Chem import QED, Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.rdmolops import GetSymmSSSR

# Project imports
from embedding_analysis import init_model


class MolecularSaliencyAnalyzer:
    """Main class for molecular saliency analysis using Integrated Gradients."""
    
    def __init__(self, model, ctx, device: torch.device):
        """
        Initialize the saliency analyzer.
        
        Args:
            model: The trained molecular generation model
            ctx: Model context containing graph conversion utilities
            device: PyTorch device for computations
        """
        self.model = model
        self.ctx = ctx
        self.device = device
        
    def _prep_cond_synflownet(self, n_graphs: int) -> torch.Tensor:
        """Prepare conditional tensor for SynFlowNet model."""
        if hasattr(self.ctx, 'num_cond_dim'):
            cond_dim = self.ctx.num_cond_dim
        elif hasattr(self.model, 'transf') and hasattr(self.model.transf, 'c2h'):
            cond_dim = self.model.transf.c2h[0].in_features
        else:
            cond_dim = 1
        return torch.zeros((n_graphs, cond_dim), device=self.device)

    def _graph_out_from_model(self, batch, cond: torch.Tensor) -> torch.Tensor:
        """Extract output logits from model forward pass."""
        out = self.model(batch, cond)
        if isinstance(out, (tuple, list)):
            return out[-1]
        return out

    def _ensure_batch_attributes(self, data) -> Any:
        """Ensure PyTorch Geometric data has required batch attributes."""
        if not hasattr(data, "batch") or data.batch is None:
            data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=self.device)
        if not hasattr(data, "ptr") or data.ptr is None:
            data.ptr = torch.tensor([0, data.x.size(0)], dtype=torch.long, device=self.device)
        if not hasattr(data, "num_graphs"):
            data.num_graphs = 1
        return data

    def _infer_stop_action_index(self, smiles: str) -> int:
        """
        Infer the stop action index from model/context.
        
        Args:
            smiles: SMILES string to process
            
        Returns:
            Index of the stop action in the model's action space
        """
        # Check context attributes for stop index
        for attr_name in ("stop_action_idx", "stop_idx", "STOP_IDX", "stop_action", "stop_token_id"):
            if hasattr(self.ctx, attr_name) and isinstance(getattr(self.ctx, attr_name), int):
                return int(getattr(self.ctx, attr_name))
        
        # Check action vocabulary
        if hasattr(self.ctx, "action_vocab"):
            for key, value in self.ctx.action_vocab.items():
                if str(key).lower() in {"stop", "<stop>", "[stop]"}:
                    return int(value)
        
        # Fallback: infer from forward pass (assume stop is last action)
        mol = Chem.MolFromSmiles(smiles)
        data = self._ensure_batch_attributes(
            self.ctx.graph_to_Data(self.ctx.obj_to_graph(mol), traj_len=0).to(self.device)
        )
        cond = self._prep_cond_synflownet(1)
        
        with torch.no_grad():
            logits = self._graph_out_from_model(data, cond)
        
        return logits.shape[-1] - 1 if logits.ndim > 0 else 0

    def compute_integrated_gradients(
        self, 
        smiles: str, 
        steps: int = 64, 
        baseline: str = "zeros"
    ) -> Tuple[Chem.Mol, torch.Tensor, torch.Tensor, int]:
        """
        Compute Integrated Gradients for stop action prediction.
        
        Args:
            smiles: Input molecule SMILES string
            steps: Number of integration steps
            baseline: Baseline type ("zeros" or "mean")
            
        Returns:
            Tuple of (molecule, atom_scores, attributions, stop_action_index)
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        
        # Prepare molecular graph data
        data = self._ensure_batch_attributes(
            self.ctx.graph_to_Data(self.ctx.obj_to_graph(mol), traj_len=0).to(self.device)
        )
        
        if not torch.is_floating_point(data.x):
            data.x = data.x.float()
        
        x = data.x
        
        # Set up baseline
        if baseline == "zeros":
            x_baseline = torch.zeros_like(x)
        elif baseline == "mean":
            x_baseline = x.mean(0, keepdim=True).expand_as(x)
        else:
            raise ValueError("baseline must be 'zeros' or 'mean'")

        stop_idx = self._infer_stop_action_index(smiles)
        cond = self._prep_cond_synflownet(1)
        
        self.model.eval()
        if hasattr(self.model, "enable_interpretability_mode"):
            self.model.enable_interpretability_mode(True)

        delta = x - x_baseline
        total_gradients = torch.zeros_like(x)

        # Integrate gradients over path from baseline to input
        for step in range(1, steps + 1):
            alpha = step / steps
            x_interpolated = (x_baseline + alpha * delta).detach().clone().requires_grad_(True)
            
            # Clone data and update features
            data_step = self._ensure_batch_attributes(data.clone())
            data_step.x = x_interpolated

            # Forward pass and compute gradients
            logits = self._graph_out_from_model(data_step, cond)
            if logits.ndim == 2:
                logits = logits[0]
            
            log_prob_stop = log_softmax(logits, dim=-1)[stop_idx]
            gradients = torch.autograd.grad(
                log_prob_stop, x_interpolated, 
                retain_graph=False, create_graph=False
            )[0]
            total_gradients += gradients

        if hasattr(self.model, "enable_interpretability_mode"):
            self.model.enable_interpretability_mode(False)

        # Compute final attributions
        avg_gradients = total_gradients / steps
        attributions = delta * avg_gradients
        atom_scores = attributions.abs().sum(1).cpu()  # Per-atom importance
        
        return mol, atom_scores, attributions.cpu(), stop_idx

    def identify_important_motifs(self, mol: Chem.Mol, atom_scores: torch.Tensor, k: int = 3) -> List[Set[int]]:
        """
        Identify the top-k most important molecular motifs based on attribution scores.
        
        Args:
            mol: RDKit molecule object
            atom_scores: Per-atom attribution scores
            k: Number of top motifs to return
            
        Returns:
            List of sets containing atom indices for each motif
        """
        n_atoms = mol.GetNumAtoms()
        scores = atom_scores.numpy().astype(float)
        
        # Find atoms above 75th percentile of importance
        threshold = np.percentile(scores, 75)
        important_atoms = set(np.nonzero(scores >= threshold)[0])
        
        # Build adjacency list
        adjacency = [[] for _ in range(n_atoms)]
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            adjacency[i].append(j)
            adjacency[j].append(i)
        
        # Find connected components of important atoms
        connected_components = []
        visited = set()
        
        for atom_idx in important_atoms:
            if atom_idx in visited:
                continue
                
            # DFS to find connected component
            stack = [atom_idx]
            component = set()
            
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                    
                visited.add(current)
                component.add(current)
                
                # Add neighboring important atoms
                for neighbor in adjacency[current]:
                    if neighbor in important_atoms and neighbor not in visited:
                        stack.append(neighbor)
            
            connected_components.append(component)
        
        # Add ring systems as potential motifs (chemically meaningful)
        ring_systems = [set(ring) for ring in GetSymmSSSR(mol)]
        
        # Score candidates (components + rings)
        candidates = []
        for component in connected_components:
            score = float(atom_scores[list(component)].sum())
            candidates.append((score, component))
        
        for ring in ring_systems:
            score = 1.2 * float(atom_scores[list(ring)].sum())  # Boost ring importance
            candidates.append((score, ring))
        
        # Sort by importance score
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Select non-overlapping motifs
        motifs = []
        for score, motif_atoms in candidates:
            # Check for significant overlap with existing motifs
            is_novel = all(
                len(motif_atoms & existing) / max(1, len(motif_atoms | existing)) < 0.6
                for existing in motifs
            )
            
            if is_novel:
                motifs.append(motif_atoms)
            
            if len(motifs) >= k:
                break
        
        return motifs

class CounterfactualAnalyzer:
    """Analyzes counterfactual molecular edits to improve QED scores."""
    
    # Chemical transformation rules for counterfactual generation
    TRANSFORMATION_RULES = [
        (Chem.MolFromSmarts("[OD2]-[CX4]"), Chem.MolFromSmiles("S-*")),        # Ether → thioether
        (Chem.MolFromSmarts("[CH3]"),        Chem.MolFromSmiles("F")),         # Methyl → fluorine
        (Chem.MolFromSmarts("C(=O)O[CX4]"),  Chem.MolFromSmiles("C(=O)O")),    # Dealkylate ester
        (Chem.MolFromSmarts("C(=O)N"),        Chem.MolFromSmiles("C(=O)O")),   # Amide → ester
        (Chem.MolFromSmarts("[OX2H]"),        Chem.MolFromSmiles("C=O")),     # Alcohol → ketone
        (Chem.MolFromSmarts("[Cl]"),           Chem.MolFromSmiles("Br")),     # Chloro → bromo
        (Chem.MolFromSmarts("[Br]"),           Chem.MolFromSmiles("I")),      # Bromo → iodo
        (Chem.MolFromSmarts("C(=O)O"),         Chem.MolFromSmiles("CO")),     # Acid → alcohol
        (Chem.MolFromSmarts("[nH]"),           Chem.MolFromSmiles("nC")),     # N-H → N-methyl
    ]
    
    def find_counterfactual_edits(self, mol: Chem.Mol, motif_atoms: Set[int]) -> List[Tuple[int, Chem.Mol]]:
        """
        Find valid counterfactual edits for a given motif.
        
        Args:
            mol: Original molecule
            motif_atoms: Set of atom indices in the motif
            
        Returns:
            List of (rule_index, modified_molecule) tuples
        """
        valid_edits = []
        
        for rule_idx, (pattern, replacement) in enumerate(self.TRANSFORMATION_RULES):
            for match in mol.GetSubstructMatches(pattern):
                # Check if transformation overlaps with motif
                if not set(match) & motif_atoms:
                    continue
                
                try:
                    # Apply transformation
                    modified_mols = Chem.ReplaceSubstructs(mol, pattern, replacement, replaceAll=False)
                    if modified_mols:
                        modified_mol = modified_mols[0]
                        Chem.SanitizeMol(modified_mol)
                        valid_edits.append((rule_idx, modified_mol))
                        
                except Exception:
                    # Skip invalid transformations
                    continue
        
        return valid_edits
    
    def find_best_qed_improvement(self, mol: Chem.Mol, motif_atoms: Set[int]) -> Optional[Dict[str, Any]]:
        """
        Find the counterfactual edit that provides the best QED improvement.
        
        Args:
            mol: Original molecule
            motif_atoms: Set of atom indices in the motif
            
        Returns:
            Dictionary with best counterfactual information or None
        """
        base_qed = QED.qed(mol)
        best_result = None
        
        valid_edits = self.find_counterfactual_edits(mol, motif_atoms)
        
        for rule_idx, modified_mol in valid_edits:
            try:
                modified_qed = QED.qed(modified_mol)
                delta_qed = modified_qed - base_qed
                
                if best_result is None or delta_qed > best_result["delta_qed"]:
                    best_result = {
                        "rule": rule_idx,
                        "delta_qed": delta_qed,
                        "smiles_cf": Chem.MolToSmiles(modified_mol)
                    }
                    
            except Exception:
                continue
        
        return best_result
    
    def analyze_stop_action_and_qed(
        self, 
        saliency_analyzer: MolecularSaliencyAnalyzer,
        smiles: str, 
        steps: int = 64, 
        k_motifs: int = 3
    ) -> Dict[str, Any]:
        """
        Complete analysis pipeline: IG → motifs → counterfactual QED improvements.
        
        Args:
            saliency_analyzer: Initialized saliency analyzer
            smiles: Input molecule SMILES
            steps: Integration steps for IG
            k_motifs: Number of top motifs to analyze
            
        Returns:
            Complete analysis results dictionary
        """
        # Compute integrated gradients
        mol, atom_scores, _, stop_idx = saliency_analyzer.compute_integrated_gradients(
            smiles, steps=steps
        )
        
        # Identify important motifs
        motifs = saliency_analyzer.identify_important_motifs(mol, atom_scores, k=k_motifs)
        
        # Analyze counterfactual improvements for each motif
        results = []
        for motif_atoms in motifs:
            best_cf = self.find_best_qed_improvement(mol, motif_atoms)
            results.append({
                "motif_atoms": sorted(list(motif_atoms)),
                "best_cf": best_cf if best_cf else {"note": "no valid counterfactual edit found"}
            })
        
        # Get top atoms by attribution score
        top_k = min(10, mol.GetNumAtoms())
        top_atom_indices = atom_scores.topk(top_k)[1].numpy().tolist()
        
        return {
            "smiles": smiles,
            "stop_idx": stop_idx,
            "qed_base": QED.qed(mol),
            "top_atoms_by_score": [int(idx) for idx in top_atom_indices],
            "results": results
        }

class MolecularVisualization:
    """Handles visualization of molecular saliency analysis results."""
    
    @staticmethod
    def _viridis_colormap(x: float) -> Tuple[float, float, float]:
        """
        Apply viridis-like colormap for continuous attribution visualization.
        
        Args:
            x: Normalized value between 0 and 1
            
        Returns:
            RGB color tuple
        """
        x = float(np.clip(x, 0.0, 1.0))
        
        # Viridis color anchors: dark purple → teal → yellow
        color_anchors = np.array([
            [0.27, 0.00, 0.33],  # Dark purple
            [0.25, 0.27, 0.53],  # Purple
            [0.16, 0.47, 0.56],  # Teal
            [0.99, 0.91, 0.14]   # Yellow
        ])
        
        # Interpolate between anchors
        positions = np.linspace(0, 1, len(color_anchors))
        segment_idx = int(np.clip(np.searchsorted(positions, x) - 1, 0, len(color_anchors) - 2))
        
        # Linear interpolation within segment
        t = (x - positions[segment_idx]) / (positions[segment_idx + 1] - positions[segment_idx] + 1e-12)
        color = color_anchors[segment_idx] * (1 - t) + color_anchors[segment_idx + 1] * t
        
        return tuple(color.tolist())
    
    def create_attribution_heatmap(
        self,
        mol: Chem.Mol,
        atom_scores: np.ndarray,
        motifs: List[Set[int]],
        output_path: str,
        legend: Optional[str] = None,
        image_size: Tuple[int, int] = (900, 600)
    ) -> None:
        """
        Create and save molecular attribution heatmap.
        
        Args:
            mol: RDKit molecule object
            atom_scores: Per-atom attribution scores
            motifs: List of important motifs (currently unused for simplicity)
            output_path: Path to save the image
            legend: Optional legend text
            image_size: Image dimensions (width, height)
        """
        # Normalize attribution scores
        scores = atom_scores.astype(float)
        if scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min())
        else:
            scores = np.zeros_like(scores)
        
        n_atoms = mol.GetNumAtoms()
        
        # Prepare molecule for drawing
        mol_copy = Chem.Mol(mol)
        rdMolDraw2D.PrepareMolForDrawing(mol_copy)
        
        # Create drawer
        drawer = rdMolDraw2D.MolDraw2DCairo(*image_size)
        draw_options = drawer.drawOptions()
        draw_options.legendFontSize = 20
        draw_options.bondLineWidth = 2
        
        # Enable continuous highlighting if available
        if hasattr(draw_options, "continuousHighlight"):
            draw_options.continuousHighlight = True
        
        # Create color and radius mappings
        atom_colors = {i: self._viridis_colormap(scores[i]) for i in range(n_atoms)}
        atom_radii = {i: 0.15 + 0.35 * scores[i] for i in range(n_atoms)}  # Variable radius
        
        # Draw molecule with attribution highlighting
        drawer.DrawMolecule(
            mol_copy,
            highlightAtoms=list(range(n_atoms)),
            highlightAtomColors=atom_colors,
            highlightAtomRadii=atom_radii,
            legend=legend or ""
        )
        
        drawer.FinishDrawing()
        
        # Save image
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(drawer.GetDrawingText())
    
    def create_counterfactual_grid(
        self,
        base_smiles: str,
        counterfactual_results: List[Dict[str, Any]],
        output_path: str,
        n_cols: int = 3
    ) -> None:
        """
        Create grid visualization of counterfactual molecules.
        
        Args:
            base_smiles: Original molecule SMILES
            counterfactual_results: List of counterfactual analysis results
            output_path: Path to save the grid image
            n_cols: Number of columns in the grid
        """
        molecules = []
        legends = []
        
        # Add base molecule
        base_mol = Chem.MolFromSmiles(base_smiles)
        if base_mol is not None:
            molecules.append(base_mol)
            base_qed = QED.qed(base_mol)
            legends.append(f"Base (QED={base_qed:.3f})")
        
        # Add counterfactual molecules
        for cf_result in counterfactual_results:
            if not isinstance(cf_result, dict) or "smiles_cf" not in cf_result:
                continue
                
            cf_mol = Chem.MolFromSmiles(cf_result["smiles_cf"])
            if cf_mol is None:
                continue
                
            molecules.append(cf_mol)
            rule_idx = cf_result.get("rule", "-")
            delta_qed = cf_result.get("delta_qed", 0.0)
            legends.append(f"CF r{rule_idx}: ΔQED={delta_qed:+.3f}")
        
        if not molecules:
            print("Warning: No valid molecules to visualize")
            return
        
        # Create grid image
        try:
            grid_image = Draw.MolsToGridImage(
                molecules,
                molsPerRow=n_cols,
                subImgSize=(300, 260),
                legends=legends,
                useSVG=False
            )
            
            # Save image
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            grid_image.save(output_path)
            print(f"Saved counterfactual grid: {output_path}")
            
        except Exception as e:
            print(f"Error creating counterfactual grid: {e}")


def main():
    """Main execution function demonstrating the saliency analysis pipeline."""
    
    # Configuration
    CHECKPOINT_PATH = "/home/ubuntu/synflownet/training_logs/logs/backup_debug_run_reactions_task_2025-09-04_21-38-54/model_state.pt"
    SMILES = "Cc1nc(N)c(C(=O)N2CC(C(=O)O)C(C3CCCCC3)C2)cc1F"
    OUTPUT_DIR = "./ig_cf_renders"
    
    # Initialize model and analyzers
    trainer, device = init_model(CHECKPOINT_PATH)
    print(f"Loaded model from: {CHECKPOINT_PATH}")
    print(f"Using device: {device}")
    print(f"Model type: {type(trainer.model).__name__}")
    
    saliency_analyzer = MolecularSaliencyAnalyzer(trainer.model, trainer.ctx, device)
    cf_analyzer = CounterfactualAnalyzer()
    visualizer = MolecularVisualization()
    
    # Run complete analysis
    analysis_results = cf_analyzer.analyze_stop_action_and_qed(
        saliency_analyzer, SMILES, steps=64, k_motifs=3
    )
    
    # Print results
    print(f"STOP action index: {analysis_results['stop_idx']}")
    print(f"Base QED: {analysis_results['qed_base']:.4f}")
    print(f"Top atoms by attribution: {analysis_results['top_atoms_by_score']}")
    
    for i, result in enumerate(analysis_results['results']):
        print(f"Motif {i+1}: atoms {result['motif_atoms']}")
        print(f"  Best counterfactual: {result['best_cf']}")
    
    # Generate visualizations
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Compute IG for visualization
    mol, atom_scores, _, stop_idx = saliency_analyzer.compute_integrated_gradients(SMILES, steps=64)
    motifs = saliency_analyzer.identify_important_motifs(mol, atom_scores, k=3)
    
    # Create attribution heatmap
    heatmap_path = os.path.join(OUTPUT_DIR, "attribution_heatmap.png")
    visualizer.create_attribution_heatmap(
        mol, atom_scores.numpy(), motifs, heatmap_path,
        legend=f"IG Attribution Heatmap (Stop Action {stop_idx})"
    )
    print(f"Saved attribution heatmap: {heatmap_path}")
    
    # Create counterfactual grid
    valid_cfs = [
        result["best_cf"] for result in analysis_results["results"]
        if isinstance(result.get("best_cf"), dict) and "smiles_cf" in result["best_cf"]
    ]
    
    if valid_cfs:
        cf_grid_path = os.path.join(OUTPUT_DIR, "counterfactual_grid.png")
        visualizer.create_counterfactual_grid(SMILES, valid_cfs, cf_grid_path)


if __name__ == "__main__":
    main()
