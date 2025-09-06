"""
Extract the smiles from generated database file with proper error handling and organization
"""

import sqlite3
import pandas as pd
import os
import json
from datetime import datetime
from collections import Counter

# Configuration
DB_PATH =  "/home/ubuntu/synflownet_public/training_logs/logs/debug_run_reactions_task_2025-09-04_21-38-54/final/generated_objs_0.db"
OUTPUT_DIR = "results_qed_run"


def setup_output_directory():
    """Create output directory if it doesn't exist"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")
    return OUTPUT_DIR


def validate_database(db_path):
    """Validate that the database exists and is accessible"""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found at {db_path}")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        if not any('results' in table[0] for table in tables):
            raise ValueError("No 'results' table found in database")
        
        conn.close()
        print(f"✓ Database validation passed: {db_path}")
        return True
        
    except sqlite3.Error as e:
        raise sqlite3.Error(f"Database validation failed: {e}")


def get_database_info(db_path):
    """Get basic information about the database structure"""
    conn = sqlite3.connect(db_path)
    
    try:
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(results)")
        columns_info = cursor.fetchall()
        column_names = [col[1] for col in columns_info]
        
        cursor.execute("SELECT COUNT(*) FROM results")
        total_records = cursor.fetchone()[0]
        
        return {
            'columns': column_names,
            'total_records': total_records
        }
        
    except sqlite3.Error as e:
        raise sqlite3.Error(f"Error getting database info: {e}")
    finally:
        conn.close()


def load_and_clean_data(db_path):
    """Load data from database and perform cleaning operations"""
    conn = sqlite3.connect(db_path)
    
    try:
        print("Loading data from database...")
        query = "SELECT * FROM results ORDER BY r DESC"
        df = pd.read_sql_query(query, conn)
        
        print(f"Initial records loaded: {len(df)}")
        
        # Data cleaning steps
        initial_count = len(df)
        
        # Step 1: Remove rows with null SMILES
        if 'smi' in df.columns:
            df = df.dropna(subset=['smi'])
            print(f"After removing null SMILES: {len(df)} records ({initial_count - len(df)} removed)")
            
            # Step 2: Remove rows with empty SMILES
            df = df[df['smi'].str.strip() != '']
            print(f"After removing empty SMILES: {len(df)} records")
        else:
            print("WARNING: 'smi' column not found!")
        
        # Step 3: Remove rows with null trajectories (if traj column exists)
        if 'traj' in df.columns:
            pre_traj_count = len(df)
            df = df.dropna(subset=['traj'])
            print(f"After removing null trajectories: {len(df)} records ({pre_traj_count - len(df)} removed)")
        else:
            print("INFO: 'traj' column not found, skipping trajectory cleaning")
        
        # Step 4: Sort by reward
        if 'r' in df.columns:
            df = df.sort_values(by='r', ascending=False)
            print(f"Data sorted by reward (r) in descending order")
        else:
            print("WARNING: 'r' (reward) column not found!")
        
        return df
        
    except Exception as e:
        raise Exception(f"Error loading and cleaning data: {e}")
    finally:
        conn.close()


def analyze_data_quality(df):
    """Analyze the quality and characteristics of the data"""
    print("\n" + "="*60)
    print("DATA QUALITY ANALYSIS")
    print("="*60)
    
    analysis = {}
    
    # Basic statistics
    analysis['total_records'] = len(df)
    
    if 'smi' in df.columns:
        analysis['unique_smiles'] = df['smi'].nunique()
        analysis['duplicate_smiles'] = len(df) - analysis['unique_smiles']
        analysis['diversity_ratio'] = analysis['unique_smiles'] / analysis['total_records']
        
        print(f"Total records: {analysis['total_records']}")
        print(f"Unique SMILES: {analysis['unique_smiles']}")
        print(f"Duplicate SMILES: {analysis['duplicate_smiles']}")
        print(f"Diversity ratio: {analysis['diversity_ratio']:.3f}")
    
    if 'traj' in df.columns:
        analysis['unique_trajectories'] = df['traj'].nunique()
        analysis['duplicate_trajectories'] = len(df) - analysis['unique_trajectories']
        print(f"Unique trajectories: {analysis['unique_trajectories']}")
        print(f"Duplicate trajectories: {analysis['duplicate_trajectories']}")
    
    if 'r' in df.columns:
        analysis['reward_stats'] = {
            'min': df['r'].min(),
            'max': df['r'].max(),
            'mean': df['r'].mean(),
            'std': df['r'].std()
        }
        print(f"Reward range: {analysis['reward_stats']['min']:.4f} - {analysis['reward_stats']['max']:.4f}")
        print(f"Reward mean: {analysis['reward_stats']['mean']:.4f} ± {analysis['reward_stats']['std']:.4f}")
    
    return analysis


def create_derivative_datasets(df):
    """Create derivative datasets (unique SMILES, unique trajectories)"""
    datasets = {}
    
    if 'smi' in df.columns:
        # Unique SMILES (keep highest reward for each)
        df_unique_smiles = df.drop_duplicates(subset=['smi'], keep='first')
        datasets['unique_smiles'] = df_unique_smiles
        print(f"Created unique SMILES dataset: {len(df_unique_smiles)} records")
    
    if 'traj' in df.columns:
        # Unique trajectories (keep highest reward for each)
        df_unique_trajs = df.drop_duplicates(subset=['traj'], keep='first')
        datasets['unique_trajectories'] = df_unique_trajs
        print(f"Created unique trajectories dataset: {len(df_unique_trajs)} records")
    
    return datasets


def save_datasets(df, datasets, output_dir):
    """Save all datasets to CSV files"""
    print("\n" + "="*60)
    print("SAVING DATASETS")
    print("="*60)
    
    try:
        # Save main dataset
        main_path = os.path.join(output_dir, "all_results.csv")
        df.to_csv(main_path, index=False)
        print(f"✓ Saved main dataset: {main_path}")
        
        # Save derivative datasets
        for name, dataset in datasets.items():
            file_path = os.path.join(output_dir, f"{name}.csv")
            dataset.to_csv(file_path, index=False)
            print(f"✓ Saved {name}: {file_path}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error saving datasets: {e}")
        return False


def analyze_top_molecules(df, top_counts=[10, 30]):
    """Analyze and save information about the most frequently occurring molecules"""
    if 'smi' not in df.columns:
        print("Cannot analyze top molecules: 'smi' column not found")
        return
    
    print("\n" + "="*60)
    print("TOP MOLECULES ANALYSIS")
    print("="*60)
    
    for top in top_counts:
        try:
            save_info_on_top_smiles(df, top)
            print(f"✓ Saved top {top} molecules analysis")
        except Exception as e:
            print(f"✗ Error analyzing top {top} molecules: {e}")
    return
    
    # Also save complete analysis
    try:
        save_info_on_top_smiles(df, len(df))
        print(f"✓ Saved complete molecules analysis")
    except Exception as e:
        print(f"✗ Error saving complete analysis: {e}")


def save_info_on_top_smiles(df, top):
    """Save detailed information about top occurring SMILES"""
    df = df.copy()
    
    # Get top SMILES by count
    top_smiles = df['smi'].value_counts().head(top)
    top_smiles_df = top_smiles.reset_index()
    top_smiles_df.columns = ['smi', 'count']
    
    # Add additional statistics for each SMILES
    def safe_apply(func, default_value):
        """Safely apply function with error handling"""
        def wrapper(x):
            try:
                subset = df[df['smi'] == x]
                if subset.empty:
                    return default_value
                return func(subset)
            except Exception:
                return default_value
        return wrapper
    
    # Highest reward for each SMILES
    top_smiles_df['highest_reward'] = top_smiles_df['smi'].apply(
        safe_apply(lambda subset: subset['r'].max() if 'r' in subset.columns else 0, 0)
    )
    
    # Average reward for each SMILES
    top_smiles_df['average_reward'] = top_smiles_df['smi'].apply(
        safe_apply(lambda subset: subset['r'].mean() if 'r' in subset.columns else 0, 0)
    )
    
    # First occurrence trajectory (if available)
    if 'traj' in df.columns:
        top_smiles_df['first_occurrence_trajectory'] = top_smiles_df['smi'].apply(
            safe_apply(lambda subset: subset['traj'].iloc[0] if 'traj' in subset.columns else '', '')
        )
        
        # Number of unique trajectories for each SMILES
        top_smiles_df['num_unique_trajectories'] = top_smiles_df['smi'].apply(
            safe_apply(lambda subset: subset['traj'].nunique() if 'traj' in subset.columns else 0, 0)
        )
    
    # Number of unique rewards for each SMILES
    if 'r' in df.columns:
        top_smiles_df['num_unique_rewards'] = top_smiles_df['smi'].apply(
            safe_apply(lambda subset: subset['r'].nunique() if 'r' in subset.columns else 0, 0)
        )
    
    # Save to file
    output_path = os.path.join(OUTPUT_DIR, f"top_smiles_{top}.csv")
    top_smiles_df.to_csv(output_path, index=False)
    
    return top_smiles_df


def print_summary_statistics(df, analysis):
    """Print a summary of the extraction results"""
    print("\n" + "="*60)
    print("EXTRACTION SUMMARY")
    print("="*60)
    
    print(f"Database processed successfully!")
    print(f"Total records processed: {len(df)}")
    
    if 'unique_smiles' in analysis:
        print(f"Unique molecules: {analysis['unique_smiles']}")
        print(f"Diversity ratio: {analysis['diversity_ratio']:.3f}")
        
        if analysis['diversity_ratio'] > 0.7:
            print("✅ Excellent diversity - Low mode collapse")
        elif analysis['diversity_ratio'] > 0.5:
            print("🟡 Good diversity - Moderate mode collapse")
        elif analysis['diversity_ratio'] > 0.3:
            print("⚠️ Poor diversity - Significant mode collapse")
        else:
            print("🔴 Very poor diversity - Severe mode collapse")
    
    if 'reward_stats' in analysis:
        rs = analysis['reward_stats']
        print(f"Reward statistics: {rs['mean']:.4f} ± {rs['std']:.4f}")
        print(f"Reward range: [{rs['min']:.4f}, {rs['max']:.4f}]")
    
    print(f"Output directory: {OUTPUT_DIR}")

def analyze_top_scoring_mols(df, top=50):
    """Analyze and save information about the top scoring molecules"""
    if 'smi' not in df.columns or 'r' not in df.columns:
        print("Cannot analyze top scoring molecules: 'smi' or 'r' column not found")
        return
    
    print("\n" + "="*60)
    print(f"TOP {top} SCORING MOLECULES ANALYSIS")
    print("="*60)
    df = df.sort_values(by='r', ascending=False)
    # drop duplicate smi, keep one with highest r value 
    df = df.drop_duplicates(subset=['smi'], keep='first')
    # Get top scoring molecules
    top_scoring = df.nlargest(top, 'r')
    
    # Save to CSV
    output_path = os.path.join(OUTPUT_DIR, f"top_scoring_molecules_{top}.csv")
    top_scoring.to_csv(output_path, index=False)
    
    print(f"✓ Saved top {top} scoring molecules to {output_path}")

def main():
    """Main execution function with comprehensive error handling"""
    try:
        print("SynFlowNet SMILES Extraction Tool")
        print("="*60)
        
        # Step 1: Setup
        output_dir = setup_output_directory()
        
        # Step 2: Validate database
        validate_database(DB_PATH)
        
        # Step 3: Get database info
        db_info = get_database_info(DB_PATH)
        print(f"Database columns: {db_info['columns']}")
        print(f"Total records in database: {db_info['total_records']}")
        
        # Step 4: Load and clean data
        df = load_and_clean_data(DB_PATH)
        
        if df.empty:
            print("❌ No valid data found after cleaning!")
            return
        
        # Step 5: Analyze data quality
        analysis = analyze_data_quality(df)
        
        # Step 6: Create derivative datasets
        datasets = create_derivative_datasets(df)
        
        # Step 7: Save all datasets
        if save_datasets(df, datasets, output_dir):
            print("✅ All datasets saved successfully")
        else:
            print("❌ Error saving some datasets")
        
        # Step 8: Analyze top molecules
        analyze_top_molecules(df)
        analyze_top_scoring_mols(df, top=50)

        
        # Step 9: Print summary
        print_summary_statistics(df, analysis)
        
        print("\n✅ Extraction completed successfully!")
        
    except FileNotFoundError as e:
        print(f"❌ File error: {e}")
    except sqlite3.Error as e:
        print(f"❌ Database error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()