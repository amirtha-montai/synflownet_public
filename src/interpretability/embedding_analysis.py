import os
import torch
import json
import pickle
import pandas as pd
from synflownet.tasks.reactions_task import ReactionTrainer
from synflownet.config import Config
from rdkit import Chem


# Safe device detection
def get_safe_device():
    """Safely detect and return available device."""
    try:
        if torch.cuda.is_available():
            torch.cuda.init()  # Test CUDA initialization
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    except RuntimeError as e:
        device = torch.device("cpu")
    return device


def load_checkpoint_and_model_state(checkpoint_path, device=NotImplemented):
    """Load model checkpoint safely onto specified device."""
    if device is NotImplemented:
        device = get_safe_device()

    if checkpoint_path is not None:
        try:
            state = torch.load(checkpoint_path, map_location=device)
            print(f"✅ Checkpoint loaded on {device}")
        except Exception:
            print(f"⚠️  Failed to load on {device}, trying CPU")
            state = torch.load(checkpoint_path, map_location='cpu')
            device = torch.device('cpu')

        config = Config(**state["cfg"])
        
        # Override config device to match our safe device
        config.device = str(device)
        trainer = ReactionTrainer(config)
        
        # Ensure trainer model is on correct device
        if hasattr(trainer, 'model') and trainer.model is not None:
            trainer.model.to(device)
        
        print(f"  - Model type: {type(trainer.model).__name__}")
        print(f"  - Model device: {next(trainer.model.parameters()).device if hasattr(trainer.model, 'parameters') else 'N/A'}")
        print(f"  - Context type: {type(trainer.ctx).__name__}")
        
    
def extract_graph_embeddings_batch(smiles_list, model, ctx, device):
    """
    Extract graph embeddings for multiple molecules efficiently in a single batch.
    
    Args:
        smiles_list (list): List of SMILES strings
        model: The trained GraphTransformerSynGFN model
        ctx: The environment context (ReactionTemplateEnvContext)
        device: The device to run computations on
        
    Returns:
        dict: Contains embeddings and metadata for all molecules
    """
    
    print(f"🧪 Processing batch of {len(smiles_list)} molecules...")
    
    # Step 1: Convert all SMILES to RDKit molecules
    molecules = []
    valid_indices = []
    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            molecules.append(mol)
            valid_indices.append(i)
        else:
            print(f"❌ Invalid SMILES: {smiles}")
    
    if not molecules:
        print("❌ No valid molecules to process")
        return None
    
    
    # Step 2: Convert all molecules to graphs
    graphs = []
    graph_valid_indices = []
    for i, mol in enumerate(molecules):
        try:
            graph = ctx.obj_to_graph(mol)
            graphs.append(graph)
            graph_valid_indices.append(valid_indices[i])
        except Exception as e:
            print(f"❌ Failed to convert molecule {valid_indices[i]} to graph: {e}")
    
    if not graphs:
        print("❌ No valid graphs created")
        return None
    
    print(f"✅ {len(graphs)} graphs created successfully")
    
    # Step 3: Convert all graphs to PyTorch Geometric format and create batch
    try:
        data_list = []
        for graph in graphs:
            # Use consistent trajectory length for all molecules
            data = ctx.graph_to_Data(graph, traj_len=0)
            data_list.append(data)
        
        # Create single batch containing all molecules
        batch = ctx.collate(data_list)
        batch = batch.to(device)
        
        print(f"✅ Batch created:")
        print(f"   - Number of graphs: {batch.num_graphs}")
        print(f"   - Total nodes: {batch.x.shape[0]}")
        print(f"   - Node features shape: {batch.x.shape}")
        print(f"   - Edge features shape: {batch.edge_attr.shape}")
        
    except Exception as e:
        print(f"❌ Failed to create batch: {e}")
        return None
    
    # Step 4: Extract embeddings for entire batch
    try:
        model.eval()
        with torch.no_grad():
            # Enable interpretability mode
            model.enable_interpretability_mode(True)
            
            # Create conditioning tensor for the batch
            num_graphs = batch.num_graphs
            if hasattr(ctx, 'num_cond_dim'):
                cond_dim = ctx.num_cond_dim
            elif hasattr(model, 'transf') and hasattr(model.transf, 'c2h'):
                cond_dim = model.transf.c2h[0].in_features
            else:
                cond_dim = 1
            
            cond = torch.zeros((num_graphs, cond_dim), device=device)
            
            print(f"   - Conditioning shape: {cond.shape}")
            
            # Single forward pass for all molecules
            node_embeddings, graph_embeddings = model.transf(batch, cond)
            
            # Get full model output
            if model.do_bck:
                fwd_cat, bck_cat, graph_out = model(batch, cond)
            else:
                fwd_cat, graph_out = model(batch, cond)
            
            model.enable_interpretability_mode(False)
            
        print(f"✅ Batch embeddings extracted successfully!")
        print(f"   - Node embeddings shape: {node_embeddings.shape}")
        print(f"   - Graph embeddings shape: {graph_embeddings.shape}")
        
        # Step 5: Split embeddings back to individual molecules
        results = {}
        node_start_idx = 0
        
        for i, (graph_idx, orig_idx) in enumerate(zip(range(len(graphs)), graph_valid_indices)):
            # Get number of nodes for this molecule
            num_nodes = graphs[graph_idx].number_of_nodes()
            
            # Extract node embeddings for this molecule
            mol_node_embeddings = node_embeddings[node_start_idx:node_start_idx + num_nodes]
            
            # Extract graph embedding for this molecule
            mol_graph_embedding = graph_embeddings[graph_idx:graph_idx + 1]
            
            results[f'molecule_{orig_idx}'] = {
                'smiles': smiles_list[orig_idx],
                'molecule': molecules[graph_valid_indices.index(orig_idx)],
                'node_embeddings': mol_node_embeddings.cpu(),
                'graph_embeddings': mol_graph_embedding.cpu(),
                'graph_output': graph_out[graph_idx:graph_idx + 1].cpu() if 'graph_out' in locals() else None,
                'batch_info': {
                    'num_atoms': num_nodes,
                    'embedding_dim': mol_node_embeddings.shape[1],
                    'original_index': orig_idx
                }
            }
            
            node_start_idx += num_nodes
        
        # Add batch-level information
        batch_info = {
            'total_molecules': len(graphs),
            'valid_molecules': len(results) - 1,  # -1 for batch_info itself
            'total_nodes': node_embeddings.shape[0],
            'embedding_dim': node_embeddings.shape[1],
            'graph_embedding_dim': graph_embeddings.shape[1],
            'processed_indices': graph_valid_indices
        }
        
        return results, batch_info
        
    except Exception as e:
        print(f"❌ Failed to extract embeddings: {e}")
        return None




def extract_embeddings_large_dataset(smiles_list, model, ctx, device, batch_size=64, save_path=None, save_dir=None):
    """
    Process a large dataset of molecules in efficient batches.
    
    Args:
        smiles_list (list): Large list of SMILES strings
        model, ctx, device: As above
        batch_size (int): Number of molecules to process per batch
        save_path (str): Optional path to save results
        
    Returns:
        dict: All embeddings and statistics
    """
    print(f"🚀 Processing {len(smiles_list)} molecules in batches of {batch_size}")
    
    all_results = {}
    failed_indices = []
    
    # Process in batches
    for i in range(0, len(smiles_list), batch_size):
        batch_smiles = smiles_list[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(smiles_list) + batch_size - 1) // batch_size
        
        print(f"\n📦 Processing batch {batch_num}/{total_batches} ({len(batch_smiles)} molecules)")
        
        try:
            batch_results, batch_info = extract_graph_embeddings_batch(batch_smiles, model, ctx, device)

            if batch_results:
                # Merge results, adjusting indices
                for key, result in batch_results.items():
                    if key.startswith('molecule_'):
                        original_idx = int(key.split('_')[1])
                        global_idx = i + original_idx
                        all_results[f'molecule_{global_idx}'] = result
                        
                        # Update the original index in the result
                        if 'batch_info' in result:
                            result['batch_info']['global_index'] = global_idx
                
                print(f"✅ Batch {batch_num} completed successfully")
            else:
                print(f"❌ Batch {batch_num} failed")
                failed_indices.extend(range(i, i + len(batch_smiles)))
                
        except Exception as e:
            print(f"❌ Batch {batch_num} failed with error: {e}")
            failed_indices.extend(range(i, i + len(batch_smiles)))
    
    # Compute overall statistics
    valid_embeddings = []
    for key, result in all_results.items():
        if key.startswith('molecule_'):
            valid_embeddings.append(result['graph_embeddings'][0])
    
    if valid_embeddings:
        embeddings_matrix = torch.stack(valid_embeddings)


        print(f"\n📊 Dataset Processing Complete!")
        print(f"   - Total molecules: {len(smiles_list)}")
        print(f"   - Successful: {len(valid_embeddings)}")
        print(f"   - Failed: {len(failed_indices)}")
        print(f"   - Success rate: {len(valid_embeddings)/len(smiles_list):.1%}")
        
        # Save results if requested
        if save_path:
            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, os.path.basename(save_path))
            with open(save_path, 'wb') as f:
                # Save without the large embedding tensors for now
                save_dict = {k: v for k, v in all_results.items() if k != 'dataset_stats'}
                pickle.dump(save_dict, f)
            df = pd.DataFrame(all_results).T
            df.to_csv(os.path.join(save_dir, 'embeddings.csv'))
            print(f"💾 Results saved to {save_path}")
    
    return all_results



def init_model(checkpoint_path=None):

    DEVICE = get_safe_device()
    
    # Load checkpoint and create trainer
    if checkpoint_path is not None:
        print(f"Loading checkpoint from: {checkpoint_path}")

        # Load checkpoint with safe device mapping
        try:
            state = torch.load(checkpoint_path, map_location=DEVICE)
            print(f"✅ Checkpoint loaded on {DEVICE}")
        except Exception:
            print(f"⚠️  Failed to load on {DEVICE}, trying CPU")
            state = torch.load(checkpoint_path, map_location='cpu')
            DEVICE = torch.device('cpu')
        
        config = Config(**state["cfg"])
        
        # Override config device to match our safe device
        config.device = str(DEVICE)
        print("Config loaded from checkpoint.", config)
        trainer = ReactionTrainer(config)
        
        # Ensure trainer model is on correct device
        if hasattr(trainer, 'model') and trainer.model is not None:
            trainer.model.to(DEVICE)
        
        print(f"  - Model type: {type(trainer.model).__name__}")
        print(f"  - Model device: {next(trainer.model.parameters()).device if hasattr(trainer.model, 'parameters') else 'N/A'}")
        print(f"  - Context type: {type(trainer.ctx).__name__}")
        return trainer, DEVICE
    


def main():
    # Initialize device and load model
    # accept config_path from user 
    #config_path = "config_interpretability.json"
    config_path = "/home/ubuntu/synflownet_public/src/interpretability/results_qed_run/config_interpretability_qed.json"
    with open(config_path, 'r') as f:
        user_config = json.load(f)
    trainer, DEVICE = init_model(user_config.get('CHECKPOINT_PATH', None))
    smiles_list = pd.read_csv(user_config['SMILES_PATH'])['smi'].tolist()
    extract_embeddings_large_dataset(smiles_list, trainer.model, trainer.ctx, DEVICE, batch_size=user_config.get('BATCH_SIZE', 4), save_path=user_config.get('SAVE_PATH', 'embeddings.pkl'), save_dir=user_config.get('SAVE_DIR', './results'))

if __name__ == "__main__":
    # Initialize device and load model  
    main()
