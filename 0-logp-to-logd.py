#!/usr/bin/env python3

import sys
import argparse
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import time
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
import signal
import atexit
import tempfile
import traceback

TEMP_FILES = []

def exit_handler():
    clean_temp_files()
    print("-- Script terminated")

def clean_temp_files():
    global TEMP_FILES
    for file in TEMP_FILES:
        try:
            if os.path.exists(file):
                os.remove(file)
        except Exception as e:
            print(f"-- Warning: Could not remove temporary file {file}: {e}")

def signal_handler(sig, frame):
    print("-- Received interrupt signal. Cleaning up...")
    clean_temp_files()
    sys.exit(1)

def print_message(message):
    print(f"-- {message}")

def update_progress(progress, description="Processing"):
    bar_length = 30
    filled_length = int(round(bar_length * progress))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    sys.stdout.write(f"\r-- {description}: [{bar}] {progress*100:.1f}%")
    sys.stdout.flush()

def is_ionizable(mol):
    if mol is None:
        return False
    
    # Define common ionizable groups
    acid_patterns = [
        "[C,S,P](=[O,S])[O;H1,-1]",  # Carboxylic, sulfonic, phosphonic acids
        "[c,n]1[n,c][c,n][c,n][c,n]1[O;H1,-1]",  # Tetrazoles
        "[S]([O;H1,-1])(=O)(=O)-[c,C]",  # Sulfonamides
        "[N;H1][S;D4](=O)(=O)[C,c]",  # Sulfonamides
        "[c,n]1[n,c][c,n][c,n][c,n]1[S;D4](=O)(=O)[O;H1,-1]"  # Tetrazole sulfonic acids
    ]
    
    base_patterns = [
        "[N;H0;$(NC);!$(N=*)]",  # Tertiary amines
        "[N;H2;$(NC);!$(N=*)]",  # Primary amines
        "[N;H1;$(NC);!$(N=*)]",  # Secondary amines
        "c1ncnc[nH0]1",  # Pyrimidines
        "c1ncccc1",  # Pyridines
        "c1nsnc1",  # Thiazoles
        "c1ncoc1",  # Oxazoles
        "c1n[nH0]c[nH0]1",  # Triazoles
        "c1n[nH0]nn1",  # Tetrazoles
        "C(=N)N",  # Amidines
        "C(=N)NN",  # Guanidines
        "[n;H0;+;!$([n][!c])]"  # Quaternary nitrogen
    ]
    
    # Check for ionizable groups
    for pattern in acid_patterns + base_patterns:
        if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
            return True
    
    return False

def process_molecule(smiles, options):
    result = {
        'SMILES': smiles,
        'is_valid': False,
        'is_ionizable': False,
        'LogP': None,
        'errors': []
    }
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            result['errors'].append("Invalid SMILES")
            return result
        
        result['is_valid'] = True
        
        # Calculate basic properties
        result['LogP'] = Descriptors.MolLogP(mol)
        
        # Check if molecule is ionizable
        result['is_ionizable'] = is_ionizable(mol)
        
    except Exception as e:
        result['errors'].append(f"Error processing molecule: {str(e)}")
    
    return result

def process_smiles_wrapper(args):
    smiles, options = args
    return process_molecule(smiles, options)

def prepare_output_dir(output_path):
    output_dir = Path(output_path).parent
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def create_argparser():
    parser = argparse.ArgumentParser(description="Filter compounds based on ionizability (LogD = LogP)")
    
    # Input/output options
    parser.add_argument("--input", required=True, help="Input file with SMILES (CSV, TSV, SDF)")
    parser.add_argument("--output", help="Output file (default: input_filtered.csv)")
    parser.add_argument("--smiles-col", default="SMILES", help="Column name containing SMILES strings")
    parser.add_argument("--keep-original-cols", action="store_true", help="Keep all original columns in output")
    parser.add_argument("--target", default=None, help="Target column to be optimized")
    
    # Processing options
    parser.add_argument("--only-non-ionizable", action="store_true", help="Keep only non-ionizable compounds (LogD = LogP)")
    parser.add_argument("--only-ionizable", action="store_true", help="Keep only ionizable compounds (LogD ≠ LogP)")
    parser.add_argument("--parallel", action="store_true", help="Use parallel processing")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Number of parallel jobs (-1 for all cores)")
    
    # Output format options
    parser.add_argument("--output-format", choices=["csv", "tsv", "sdf", "auto"], default="auto",
                      help="Output file format (default: auto-detect from extension)")
    
    # General options
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    parser.add_argument("--save-config", help="Save configuration to specified JSON file")
    parser.add_argument("--load-config", help="Load configuration from JSON file")
    
    return parser

def get_file_extension(file_path):
    return os.path.splitext(file_path)[1].lower()

def load_input_file(file_path, format_type='auto'):
    print_message(f"Loading input file: {file_path}")
    
    if format_type == 'auto':
        ext = get_file_extension(file_path)
        if ext == '.csv':
            format_type = 'csv'
        elif ext == '.tsv':
            format_type = 'tsv'
        elif ext == '.sdf':
            format_type = 'sdf'
        else:
            format_type = 'csv'  # Default to CSV
    
    try:
        if format_type == 'csv':
            df = pd.read_csv(file_path)
        elif format_type == 'tsv':
            df = pd.read_csv(file_path, sep='\t')
        elif format_type == 'sdf':
            from rdkit.Chem import PandasTools
            df = PandasTools.LoadSDF(file_path)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        print_message(f"Loaded {len(df)} rows from {file_path}")
        return df
    
    except Exception as e:
        print_message(f"Error loading file: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

def save_output_file(df, file_path, format_type='auto'):
    print_message(f"Saving {len(df)} rows to output file: {file_path}")
    
    if format_type == 'auto':
        ext = get_file_extension(file_path)
        if ext == '.csv':
            format_type = 'csv'
        elif ext == '.tsv':
            format_type = 'tsv'
        elif ext == '.sdf':
            format_type = 'sdf'
        else:
            format_type = 'csv'  # Default to CSV
    
    try:
        if format_type == 'csv':
            df.to_csv(file_path, index=False)
        elif format_type == 'tsv':
            df.to_csv(file_path, sep='\t', index=False)
        elif format_type == 'sdf':
            from rdkit.Chem import PandasTools
            PandasTools.WriteSDF(df, file_path, molColName='ROMol', properties=list(df.columns))
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        print_message(f"Successfully saved to {file_path}")
    
    except Exception as e:
        print_message(f"Error saving file: {str(e)}")
        traceback.print_exc()

def save_processing_info(df_original, df_filtered, output_path, options, runtime):
    info_file = os.path.splitext(output_path)[0] + "_info.json"
    
    # Calculate statistics
    total_compounds = len(df_original)
    valid_compounds = df_filtered['is_valid'].sum() if 'is_valid' in df_filtered.columns else len(df_filtered)
    ionizable_compounds = df_filtered['is_ionizable'].sum() if 'is_ionizable' in df_filtered.columns else 'unknown'
    non_ionizable_compounds = len(df_filtered[~df_filtered['is_ionizable']]) if 'is_ionizable' in df_filtered.columns else 'unknown'
    
    # Prepare info dict
    info = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'runtime_seconds': runtime,
        'input_file': options.input,
        'output_file': output_path,
        'statistics': {
            'total_compounds': total_compounds,
            'valid_molecules': valid_compounds,
            'ionizable_compounds': ionizable_compounds,
            'non_ionizable_compounds': non_ionizable_compounds,
            'filtered_compounds': total_compounds - len(df_filtered)
        },
        'options': vars(options)
    }
    
    # Save to file
    try:
        import json
        with open(info_file, 'w') as f:
            json.dump(info, f, indent=2)
        print_message(f"Saved processing info to {info_file}")
    except Exception as e:
        print_message(f"Warning: Could not save processing info: {str(e)}")

def main():
    # Register cleanup handlers
    atexit.register(exit_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Parse arguments
    parser = create_argparser()
    args = parser.parse_args()
    
    # Set up output file if not specified
    if not args.output:
        input_base = os.path.splitext(args.input)[0]
        args.output = f"{input_base}_filtered.csv"
    
    print_message("Ionizability Filter for Chemical Compounds")
    print_message(f"Input file: {args.input}")
    print_message(f"Output file: {args.output}")
    
    start_time = time.time()
    
    # Load input file
    df = load_input_file(args.input)
    
    # Check if required columns exist
    if args.smiles_col not in df.columns:
        print_message(f"Error: SMILES column '{args.smiles_col}' not found in input file")
        sys.exit(1)
    
    if args.target and args.target not in df.columns:
        print_message(f"Warning: Target column '{args.target}' not found in input file")
    
    # Process molecules
    print_message("Processing molecules to determine ionizability...")
    
    if args.parallel:
        try:
            from multiprocessing import Pool, cpu_count
            n_jobs = args.n_jobs if args.n_jobs > 0 else cpu_count()
            print_message(f"Using parallel processing with {n_jobs} cores")
            
            # Prepare arguments for parallel processing
            smiles_list = df[args.smiles_col].tolist()
            args_list = [(smiles, args) for smiles in smiles_list]
            
            with Pool(n_jobs) as pool:
                results = list(pool.map(process_smiles_wrapper, args_list))
        except Exception as e:
            print_message(f"Error in parallel processing: {str(e)}")
            print_message("Falling back to sequential processing")
            results = [process_molecule(smiles, args) for smiles in df[args.smiles_col]]
    else:
        results = []
        total = len(df)
        for i, smiles in enumerate(df[args.smiles_col]):
            results.append(process_molecule(smiles, args))
            if i % max(1, total // 100) == 0:
                update_progress(i / total)
        update_progress(1.0)
        print()
    
    # Convert results to dataframe
    results_df = pd.DataFrame(results)
    
    # Merge with original dataframe if keeping original columns
    if args.keep_original_cols:
        results_df = pd.concat([df.reset_index(drop=True), results_df[['is_valid', 'is_ionizable', 'LogP', 'errors']]], axis=1)
    
    # Filter based on ionizability
    if args.only_non_ionizable and args.only_ionizable:
        print_message("Warning: Both --only-non-ionizable and --only-ionizable specified. No filtering will be applied.")
        filtered_df = results_df
    elif args.only_non_ionizable:
        print_message("Filtering to keep only non-ionizable compounds (LogD = LogP)")
        filtered_df = results_df[~results_df['is_ionizable']]
    elif args.only_ionizable:
        print_message("Filtering to keep only ionizable compounds (LogD ≠ LogP)")
        filtered_df = results_df[results_df['is_ionizable']]
    else:
        print_message("No ionizability filtering applied - keeping all compounds")
        filtered_df = results_df
    
    # Save output
    prepare_output_dir(args.output)
    save_output_file(filtered_df, args.output, args.output_format)
    
    # Calculate and display stats
    runtime = time.time() - start_time
    print_message(f"Processing completed in {runtime:.2f} seconds")
    print_message(f"Total compounds: {len(df)}")
    print_message(f"Valid molecules: {results_df['is_valid'].sum()}")
    print_message(f"Ionizable compounds: {results_df['is_ionizable'].sum()}")
    print_message(f"Non-ionizable compounds: {len(results_df[~results_df['is_ionizable']])}")
    print_message(f"Filtered output: {len(filtered_df)} compounds")
    
    # Save processing information
    save_processing_info(df, filtered_df, args.output, args, runtime)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())