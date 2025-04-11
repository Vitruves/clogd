#!/usr/bin/env python3

import os
import sys
import argparse
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import AllChem
from collections import defaultdict
import random
import time

def print_message(message):
    sys.stdout.write(f"-- {message}\n")
    sys.stdout.flush()

def safe_mol_from_smiles(smiles):
    if not smiles or pd.isna(smiles) or not isinstance(smiles, str):
        return None
    
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        try:
            mol = Chem.MolFromSmiles(smiles.strip(), sanitize=False)
            if mol is not None:
                Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL^Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        except:
            pass
    
    return mol

def get_scaffold(mol, include_chirality=False):
    if mol is None:
        return None
    
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if include_chirality:
            return Chem.MolToSmiles(scaffold, isomericSmiles=True)
        else:
            return Chem.MolToSmiles(scaffold, isomericSmiles=False)
    except:
        return None

def get_generic_framework(mol):
    if mol is None:
        return None
    
    try:
        # Get scaffold first
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        # Make it even more generic by removing atom types
        generic = MurckoScaffold.MakeScaffoldGeneric(scaffold)
        return Chem.MolToSmiles(generic, isomericSmiles=False)
    except:
        return None

def murcko_scaffold_split(df, smiles_col, output_dir, ratio=(0.8, 0.1, 0.1), 
                          chirality=False, balanced=True, seed=42):
    print_message("Performing Murcko scaffold split")
    random.seed(seed)
    np.random.seed(seed)
    
    # Create dictionary of scaffolds
    scaffolds = {}
    scaffold_counts = defaultdict(int)
    
    print_message("Extracting scaffolds from molecules")
    for idx, row in df.iterrows():
        smiles = row[smiles_col]
        mol = safe_mol_from_smiles(smiles)
        if mol is None:
            print_message(f"[  0%] Warning: Failed to parse SMILES: {smiles}")
            continue
        
        scaffold = get_scaffold(mol, include_chirality=chirality)
        if scaffold is None:
            print_message(f"[  0%] Warning: Failed to generate scaffold for: {smiles}")
            continue
        
        if scaffold not in scaffolds:
            scaffolds[scaffold] = []
        scaffolds[scaffold].append(idx)
        scaffold_counts[scaffold] += 1
    
    # Sort scaffolds by frequency
    scaffold_sets = sorted(list(scaffolds.items()), key=lambda x: (scaffold_counts[x[0]], x[0]), reverse=True)
    
    train_indices = []
    valid_indices = []
    test_indices = []
    
    if balanced:
        print_message("Using balanced scaffold split method")
        # Balanced split - distribute scaffolds evenly
        train_cutoff = ratio[0]
        valid_cutoff = ratio[0] + ratio[1]
        
        train_weight = 0
        valid_weight = 0
        test_weight = 0
        
        # Initialize with empty scaffold sets
        for scaffold, indices in scaffold_sets:
            if train_weight / len(df) < train_cutoff:
                train_indices.extend(indices)
                train_weight += len(indices)
            elif valid_weight / len(df) < ratio[1]:
                valid_indices.extend(indices)
                valid_weight += len(indices)
            else:
                test_indices.extend(indices)
                test_weight += len(indices)
    else:
        print_message("Using random scaffold split method")
        # Random split - shuffle scaffolds first
        random.shuffle(scaffold_sets)
        
        # Calculate cumulative scaffold counts
        total_mols = sum(len(indices) for _, indices in scaffold_sets)
        
        train_cutoff = int(ratio[0] * total_mols)
        valid_cutoff = int((ratio[0] + ratio[1]) * total_mols)
        
        count = 0
        for scaffold, indices in scaffold_sets:
            if count < train_cutoff:
                train_indices.extend(indices)
            elif count < valid_cutoff:
                valid_indices.extend(indices)
            else:
                test_indices.extend(indices)
            count += len(indices)
    
    # Create output DataFrames
    train_df = df.iloc[train_indices].copy()
    valid_df = df.iloc[valid_indices].copy()
    test_df = df.iloc[test_indices].copy()
    
    # Print statistics
    print_message(f"[100%] Split complete")
    print_message(f"Train set: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print_message(f"Validation set: {len(valid_df)} samples ({len(valid_df)/len(df)*100:.1f}%)")
    print_message(f"Test set: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
    
    # Count unique scaffolds
    train_scaffolds = set()
    valid_scaffolds = set()
    test_scaffolds = set()
    
    for idx in train_indices:
        smiles = df.iloc[idx][smiles_col]
        mol = safe_mol_from_smiles(smiles)
        if mol:
            scaffold = get_scaffold(mol, include_chirality=chirality)
            if scaffold:
                train_scaffolds.add(scaffold)
    
    for idx in valid_indices:
        smiles = df.iloc[idx][smiles_col]
        mol = safe_mol_from_smiles(smiles)
        if mol:
            scaffold = get_scaffold(mol, include_chirality=chirality)
            if scaffold:
                valid_scaffolds.add(scaffold)
    
    for idx in test_indices:
        smiles = df.iloc[idx][smiles_col]
        mol = safe_mol_from_smiles(smiles)
        if mol:
            scaffold = get_scaffold(mol, include_chirality=chirality)
            if scaffold:
                test_scaffolds.add(scaffold)
    
    print_message(f"Unique scaffolds in train: {len(train_scaffolds)}")
    print_message(f"Unique scaffolds in valid: {len(valid_scaffolds)}")
    print_message(f"Unique scaffolds in test: {len(test_scaffolds)}")
    
    # Calculate overlap
    train_valid_overlap = len(train_scaffolds.intersection(valid_scaffolds))
    train_test_overlap = len(train_scaffolds.intersection(test_scaffolds))
    valid_test_overlap = len(valid_scaffolds.intersection(test_scaffolds))
    
    print_message(f"Scaffold overlap between train-valid: {train_valid_overlap}")
    print_message(f"Scaffold overlap between train-test: {train_test_overlap}")
    print_message(f"Scaffold overlap between valid-test: {valid_test_overlap}")
    
    # Save output files
    os.makedirs(output_dir, exist_ok=True)
    
    combined_df = pd.concat([train_df, valid_df, test_df])
    combined_df.to_csv(os.path.join(output_dir, "combined.csv"), index=False)
    
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    valid_df.to_csv(os.path.join(output_dir, "valid.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    
    print_message(f"Output files saved to {output_dir}")
    return train_df, valid_df, test_df

def generic_framework_split(df, smiles_col, output_dir, ratio=(0.8, 0.1, 0.1), seed=42):
    print_message("Performing generic framework scaffold split")
    random.seed(seed)
    np.random.seed(seed)
    
    frameworks = {}
    framework_counts = defaultdict(int)
    
    print_message("Extracting generic frameworks from molecules")
    for idx, row in df.iterrows():
        smiles = row[smiles_col]
        mol = safe_mol_from_smiles(smiles)
        if mol is None:
            print_message(f"[  0%] Warning: Failed to parse SMILES: {smiles}")
            continue
        
        framework = get_generic_framework(mol)
        if framework is None:
            print_message(f"[  0%] Warning: Failed to generate framework for: {smiles}")
            continue
        
        if framework not in frameworks:
            frameworks[framework] = []
        frameworks[framework].append(idx)
        framework_counts[framework] += 1
    
    # Sort frameworks by frequency
    framework_sets = sorted(list(frameworks.items()), key=lambda x: (framework_counts[x[0]], x[0]), reverse=True)
    
    train_indices = []
    valid_indices = []
    test_indices = []
    
    print_message("Using balanced framework split method")
    train_cutoff = ratio[0]
    valid_cutoff = ratio[0] + ratio[1]
    
    train_weight = 0
    valid_weight = 0
    test_weight = 0
    
    # Initialize with empty framework sets
    for framework, indices in framework_sets:
        if train_weight / len(df) < train_cutoff:
            train_indices.extend(indices)
            train_weight += len(indices)
        elif valid_weight / len(df) < ratio[1]:
            valid_indices.extend(indices)
            valid_weight += len(indices)
        else:
            test_indices.extend(indices)
            test_weight += len(indices)
    
    # Create output DataFrames
    train_df = df.iloc[train_indices].copy()
    valid_df = df.iloc[valid_indices].copy()
    test_df = df.iloc[test_indices].copy()
    
    # Print statistics
    print_message(f"[100%] Split complete")
    print_message(f"Train set: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print_message(f"Validation set: {len(valid_df)} samples ({len(valid_df)/len(df)*100:.1f}%)")
    print_message(f"Test set: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
    
    # Count unique frameworks
    train_frameworks = set()
    valid_frameworks = set()
    test_frameworks = set()
    
    for idx in train_indices:
        smiles = df.iloc[idx][smiles_col]
        mol = safe_mol_from_smiles(smiles)
        if mol:
            framework = get_generic_framework(mol)
            if framework:
                train_frameworks.add(framework)
    
    for idx in valid_indices:
        smiles = df.iloc[idx][smiles_col]
        mol = safe_mol_from_smiles(smiles)
        if mol:
            framework = get_generic_framework(mol)
            if framework:
                valid_frameworks.add(framework)
    
    for idx in test_indices:
        smiles = df.iloc[idx][smiles_col]
        mol = safe_mol_from_smiles(smiles)
        if mol:
            framework = get_generic_framework(mol)
            if framework:
                test_frameworks.add(framework)
    
    print_message(f"Unique frameworks in train: {len(train_frameworks)}")
    print_message(f"Unique frameworks in valid: {len(valid_frameworks)}")
    print_message(f"Unique frameworks in test: {len(test_frameworks)}")
    
    # Save output files
    os.makedirs(output_dir, exist_ok=True)
    
    combined_df = pd.concat([train_df, valid_df, test_df])
    combined_df.to_csv(os.path.join(output_dir, "combined.csv"), index=False)
    
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    valid_df.to_csv(os.path.join(output_dir, "valid.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    
    print_message(f"Output files saved to {output_dir}")
    return train_df, valid_df, test_df

def parse_args():
    parser = argparse.ArgumentParser(description="Split chemical datasets by molecular scaffolds")
    
    parser.add_argument("--input", "-i", required=True, help="Input CSV file with SMILES column")
    parser.add_argument("--output-dir", "-o", required=True, help="Output directory for split files")
    parser.add_argument("--smiles-col", "-s", default="SMILES", help="Name of SMILES column")
    parser.add_argument("--method", choices=["murcko", "generic"], default="murcko",
                       help="Scaffold splitting method (murcko or generic framework)")
    parser.add_argument("--chirality", action="store_true", 
                       help="Consider stereochemistry in scaffold generation")
    parser.add_argument("--ratio", default="0.8,0.1,0.1", 
                       help="Split ratio as train,valid,test (comma-separated)")
    parser.add_argument("--balanced", action="store_true", 
                       help="Use balanced scaffold distribution (vs random)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    try:
        ratio = [float(x) for x in args.ratio.split(",")]
        if len(ratio) != 3 or sum(ratio) != 1.0:
            print_message("Error: Ratio must be three comma-separated values that sum to 1.0")
            return 1
    except:
        print_message("Error: Invalid ratio format. Must be three comma-separated values (e.g., 0.8,0.1,0.1)")
        return 1
    
    print_message(f"Loading input data from {args.input}")
    df = pd.read_csv(args.input)
    print_message(f"Loaded {len(df)} rows")
    
    if args.smiles_col not in df.columns:
        print_message(f"Error: SMILES column '{args.smiles_col}' not found")
        print_message(f"Available columns: {', '.join(df.columns)}")
        return 1
    
    start_time = time.time()
    
    if args.method == "murcko":
        print_message("Using Murcko scaffold splitting method")
        train_df, valid_df, test_df = murcko_scaffold_split(
            df, args.smiles_col, args.output_dir, ratio=ratio, 
            chirality=args.chirality, balanced=args.balanced, seed=args.seed
        )
    else:
        print_message("Using generic framework splitting method")
        train_df, valid_df, test_df = generic_framework_split(
            df, args.smiles_col, args.output_dir, ratio=ratio, seed=args.seed
        )
    
    runtime = time.time() - start_time
    print_message(f"Splitting completed in {runtime:.2f} seconds")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 