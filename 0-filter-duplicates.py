#!/usr/bin/env python3

import os
import sys
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit import RDLogger
import argparse
from tqdm import tqdm

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

def canonicalize_smiles(smiles):
    """Convert SMILES to canonical form"""
    if not isinstance(smiles, str) or not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    except:
        return None

def log_message(msg):
    """Log message in CMake-style format"""
    print(f"-- {msg}")
    sys.stdout.flush()

def log_success(msg):
    """Log success message in CMake-style format with green color"""
    print(f"-- {msg} - \033[92msuccess\033[0m")
    sys.stdout.flush()

def log_warning(msg):
    """Log warning message in CMake-style format with orange color"""
    print(f"-- {msg} - \033[93mwarning\033[0m")
    sys.stdout.flush()

def log_error(msg):
    """Log error message in CMake-style format with red color"""
    print(f"-- {msg} - \033[91merror\033[0m", file=sys.stderr)
    sys.stdout.flush()

def compare_smiles_datasets(file1, file2, smiles_col1, smiles_col2, output=None, cleaned_output=None):
    """Compare two datasets and calculate SMILES similarity metrics"""
    # Load datasets
    log_message(f"Loading first dataset from {file1}")
    try:
        df1 = pd.read_csv(file1)
        if smiles_col1 not in df1.columns:
            log_error(f"SMILES column '{smiles_col1}' not found in {file1}")
            return False
    except Exception as e:
        log_error(f"Failed to load {file1}: {str(e)}")
        return False
    
    log_message(f"Loading second dataset from {file2}")
    try:
        df2 = pd.read_csv(file2)
        if smiles_col2 not in df2.columns:
            log_error(f"SMILES column '{smiles_col2}' not found in {file2}")
            return False
    except Exception as e:
        log_error(f"Failed to load {file2}: {str(e)}")
        return False
    
    # Get SMILES columns
    smiles1 = df1[smiles_col1].astype(str)
    smiles2 = df2[smiles_col2].astype(str)
    
    # Canonicalize SMILES
    log_message(f"Canonicalizing {len(smiles1)} SMILES from first dataset")
    canonical_smiles1 = []
    for s in tqdm(smiles1, disable=None):
        canonical_smiles1.append(canonicalize_smiles(s))
    
    log_message(f"Canonicalizing {len(smiles2)} SMILES from second dataset")
    canonical_smiles2 = []
    for s in tqdm(smiles2, disable=None):
        canonical_smiles2.append(canonicalize_smiles(s))
    
    # Remove None values
    canonical_smiles1 = [s for s in canonical_smiles1 if s is not None]
    canonical_smiles2 = [s for s in canonical_smiles2 if s is not None]
    
    # Create sets for comparison
    set1 = set(canonical_smiles1)
    set2 = set(canonical_smiles2)
    
    # Calculate metrics
    intersection = set1.intersection(set2)
    unique_to_set1 = set1 - set2
    unique_to_set2 = set2 - set1
    
    # Calculate percentages
    pct_1_in_2 = (len(intersection) / len(set1)) * 100 if len(set1) > 0 else 0
    pct_2_in_1 = (len(intersection) / len(set2)) * 100 if len(set2) > 0 else 0
    jaccard_index = len(intersection) / len(set1.union(set2)) if len(set1.union(set2)) > 0 else 0
    
    # Display results
    log_message(f"Dataset 1 valid SMILES: {len(canonical_smiles1)}")
    log_message(f"Dataset 2 valid SMILES: {len(canonical_smiles2)}")
    log_message(f"Unique SMILES in dataset 1: {len(set1)}")
    log_message(f"Unique SMILES in dataset 2: {len(set2)}")
    log_message(f"SMILES common to both datasets: {len(intersection)}")
    log_message(f"SMILES unique to dataset 1: {len(unique_to_set1)}")
    log_message(f"SMILES unique to dataset 2: {len(unique_to_set2)}")
    log_success(f"Percentage of dataset 1 SMILES found in dataset 2: {pct_1_in_2:.2f}%")
    log_success(f"Percentage of dataset 2 SMILES found in dataset 1: {pct_2_in_1:.2f}%")
    log_success(f"Jaccard similarity index: {jaccard_index:.4f}")
    
    # Save results to file if requested
    if output:
        try:
            log_message(f"Saving results to {output}")
            results = {
                "dataset1_total": len(smiles1),
                "dataset1_valid": len(canonical_smiles1),
                "dataset1_unique": len(set1),
                "dataset2_total": len(smiles2),
                "dataset2_valid": len(canonical_smiles2),
                "dataset2_unique": len(set2),
                "common_smiles": len(intersection),
                "unique_to_dataset1": len(unique_to_set1),
                "unique_to_dataset2": len(unique_to_set2),
                "pct_dataset1_in_dataset2": pct_1_in_2,
                "pct_dataset2_in_dataset1": pct_2_in_1,
                "jaccard_index": jaccard_index
            }
            
            # Save as CSV
            results_df = pd.DataFrame([results])
            results_df.to_csv(output, index=False)
            
            # Optional: save the intersection and differences
            base, ext = os.path.splitext(output)
            intersection_file = f"{base}_common{ext}"
            unique1_file = f"{base}_unique_to_1{ext}"
            unique2_file = f"{base}_unique_to_2{ext}"
            
            pd.DataFrame(list(intersection), columns=["SMILES"]).to_csv(intersection_file, index=False)
            pd.DataFrame(list(unique_to_set1), columns=["SMILES"]).to_csv(unique1_file, index=False)
            pd.DataFrame(list(unique_to_set2), columns=["SMILES"]).to_csv(unique2_file, index=False)
            
            log_success(f"Saved common SMILES to {intersection_file}")
            log_success(f"Saved SMILES unique to dataset 1 to {unique1_file}")
            log_success(f"Saved SMILES unique to dataset 2 to {unique2_file}")
            
        except Exception as e:
            log_error(f"Failed to save results: {str(e)}")
    
    # Save largest dataset cleaned from SMILES of smallest dataset
    if cleaned_output:
        try:
            log_message(f"Creating cleaned dataset from largest input")
            
            # Determine which dataset is larger
            is_set1_larger = len(set1) > len(set2)
            
            if is_set1_larger:
                # First dataset is larger
                largest_df = df1.copy()
                largest_col = smiles_col1
                unique_smiles = unique_to_set1
                log_message(f"Dataset 1 is larger ({len(set1)} vs {len(set2)} unique SMILES)")
            else:
                # Second dataset is larger
                largest_df = df2.copy()
                largest_col = smiles_col2
                unique_smiles = unique_to_set2
                log_message(f"Dataset 2 is larger ({len(set2)} vs {len(set1)} unique SMILES)")
            
            # Create a mask for rows to keep
            unique_smiles_set = set(unique_smiles)
            mask = largest_df[largest_col].apply(
                lambda x: canonicalize_smiles(x) in unique_smiles_set if isinstance(x, str) else False
            )
            
            # Filter the dataframe
            cleaned_df = largest_df[mask]
            
            # Save to file
            cleaned_df.to_csv(cleaned_output, index=False)
            log_success(f"Saved cleaned dataset with {len(cleaned_df)} rows to {cleaned_output}")
            log_message(f"Removed {len(largest_df) - len(cleaned_df)} entries that were in the smaller dataset")
            
        except Exception as e:
            log_error(f"Failed to save cleaned dataset: {str(e)}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Compare SMILES between two datasets after canonicalization")
    parser.add_argument("--file1", required=True, help="First CSV file with SMILES")
    parser.add_argument("--file2", required=True, help="Second CSV file with SMILES")
    parser.add_argument("--smiles-col1", default="SMILES", help="SMILES column name in first file")
    parser.add_argument("--smiles-col2", default="SMILES", help="SMILES column name in second file")
    parser.add_argument("--output", help="Output file to save comparison results (CSV)")
    parser.add_argument("--cleaned-output", help="Output file to save the largest dataset cleaned from SMILES in the smallest dataset (CSV)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    if len(sys.argv) == 1:
        parser.print_help()
        return 0
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file1):
        log_error(f"File not found: {args.file1}")
        return 1
    
    if not os.path.exists(args.file2):
        log_error(f"File not found: {args.file2}")
        return 1
    
    success = compare_smiles_datasets(
        args.file1, 
        args.file2, 
        args.smiles_col1, 
        args.smiles_col2, 
        args.output,
        args.cleaned_output
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 