#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Lipinski
import argparse
from tqdm import tqdm

# Use RDKit's built-in functionality to identify acidic/basic groups
def calculate_acid_base_properties(smiles):
    """Calculate acidic and basic properties for a molecule using RDKit."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    
    # RDKit doesn't have direct NumAcidAtoms/NumBasicAtoms functions
    # Let's implement our own counting based on Lipinski rules
    
    # Count basic nitrogen atoms (NH, NH2, NH3+, guanidine, amidine)
    basic_n = sum(1 for atom in mol.GetAtoms() if 
                 atom.GetSymbol() == 'N' and 
                 atom.GetFormalCharge() >= 0 and
                 atom.GetTotalDegree() < 4 and
                 atom.GetTotalNumHs() > 0)
    
    # Count acidic groups (COOH, tetrazole, sulfonamides, etc.)
    acidic_count = 0
    
    # Check for carboxylic acids
    pattern_cooh = Chem.MolFromSmarts('C(=O)[OH]')
    if pattern_cooh:
        acidic_count += len(mol.GetSubstructMatches(pattern_cooh))
    
    # Check for tetrazoles (often acidic)
    pattern_tetrazole = Chem.MolFromSmarts('c1nnn[nH]1')
    if pattern_tetrazole:
        acidic_count += len(mol.GetSubstructMatches(pattern_tetrazole))
    
    # Check for sulfonamides
    pattern_sulfonamide = Chem.MolFromSmarts('S(=O)(=O)[NH]')
    if pattern_sulfonamide:
        acidic_count += len(mol.GetSubstructMatches(pattern_sulfonamide))
    
    # Check for phenols
    pattern_phenol = Chem.MolFromSmarts('c[OH]')
    if pattern_phenol:
        acidic_count += len(mol.GetSubstructMatches(pattern_phenol))
    
    return acidic_count, basic_n

def split_dataset_by_acid_base(input_file, output_dir, smiles_col='SMILES'):
    """Split a dataset into two groups based on acid/base properties."""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read input file
    print(f"-- Reading dataset from {input_file}")
    df = pd.read_csv(input_file)
    
    if smiles_col not in df.columns:
        potential_cols = [col for col in df.columns if 'smiles' in col.lower()]
        if potential_cols:
            smiles_col = potential_cols[0]
            print(f"-- SMILES column not found. Using {smiles_col} instead")
        else:
            raise ValueError(f"SMILES column not found in dataset")
    
    total_molecules = len(df)
    print(f"-- Processing {total_molecules} molecules")
    
    # Calculate acid/base properties for each molecule
    results = []
    for smiles in tqdm(df[smiles_col], desc="Calculating acid/base properties"):
        n_acid, n_base = calculate_acid_base_properties(smiles)
        results.append((n_acid, n_base))
    
    # Add results to dataframe
    df['nAcid'] = [r[0] for r in results]
    df['nBase'] = [r[1] for r in results]
    
    # Create masks for the two groups
    group1_mask = (df['nAcid'] > 0) | (df['nBase'] > 0)
    group2_mask = (df['nAcid'] == 0) & (df['nBase'] == 0)
    
    # Split dataframe into groups
    group1_df = df[group1_mask]
    group2_df = df[group2_mask]
    
    # Count invalid molecules (where descriptors calculation failed)
    invalid_mask = df['nAcid'].isna() | df['nBase'].isna()
    invalid_count = invalid_mask.sum()
    
    # Get base filename without extension
    base_filename = os.path.splitext(os.path.basename(input_file))[0]
    
    # Save the groups to separate files
    group1_output = os.path.join(output_dir, f"{base_filename}_acidic_or_basic.csv")
    group2_output = os.path.join(output_dir, f"{base_filename}_neutral.csv")
    
    group1_df.to_csv(group1_output, index=False)
    group2_df.to_csv(group2_output, index=False)
    
    # Print statistics
    print(f"-- [100%] Processing complete")
    print(f"-- Summary:")
    print(f"   Total molecules processed: {total_molecules}")
    print(f"   Group 1 (acidic/basic): {len(group1_df)} molecules ({len(group1_df)/total_molecules*100:.1f}%)")
    print(f"   Group 2 (neutral): {len(group2_df)} molecules ({len(group2_df)/total_molecules*100:.1f}%)")
    if invalid_count > 0:
        print(f"   Invalid molecules: {invalid_count} ({invalid_count/total_molecules*100:.1f}%)")
    print(f"-- Results saved to:")
    print(f"   Group 1: {group1_output}")
    print(f"   Group 2: {group2_output}")

def main():
    parser = argparse.ArgumentParser(description='Split molecules by acid/base properties')
    parser.add_argument('--input', '-i', type=str, required=True, 
                        help='Input CSV file with SMILES column')
    parser.add_argument('--output-dir', '-o', type=str, default='acid_base_split',
                        help='Output directory for split files')
    parser.add_argument('--smiles-col', '-s', type=str, default='SMILES',
                        help='Name of the SMILES column')
    
    args = parser.parse_args()
    
    split_dataset_by_acid_base(args.input, args.output_dir, args.smiles_col)

if __name__ == "__main__":
    main()