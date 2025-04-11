#!/usr/bin/env python3

import os
import sys
import argparse
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator, GetRDKitFPGenerator, GetAtomPairGenerator, GetTopologicalTorsionGenerator
from pathlib import Path
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import psutil
import signal
import tempfile
import atexit
import shutil
import gc
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_selection import VarianceThreshold, mutual_info_regression

TEMP_DIR = None
TEMP_FILES = []
INTERRUPTED = False
EXECUTOR = None

def exit_handler():
    global EXECUTOR
    if EXECUTOR:
        EXECUTOR.shutdown(wait=False, cancel_futures=True)
    cleanup_temp_files()

def signal_handler(sig, frame):
    global INTERRUPTED, EXECUTOR
    INTERRUPTED = True
    print_message("\nInterrupted by user. Cleaning up...")
    if EXECUTOR:
        EXECUTOR.shutdown(wait=False, cancel_futures=True)
    cleanup_temp_files()
    sys.exit(1)

def cleanup_temp_files():
    global TEMP_DIR, TEMP_FILES
    for file in TEMP_FILES:
        try:
            if os.path.exists(file):
                os.remove(file)
        except:
            pass
    
    if TEMP_DIR and os.path.exists(TEMP_DIR):
        try:
            shutil.rmtree(TEMP_DIR)
        except:
            pass

def print_message(message):
    sys.stdout.write(f"-- {message}\n")
    sys.stdout.flush()

def update_progress(progress, description="Computing"):
    sys.stdout.write(f"\r-- [ {progress:6.2f}% ] {description}")
    sys.stdout.flush()

class ProgressTracker:
    def __init__(self, total_steps, description="Computing"):
        self.total_steps = total_steps
        self.completed_steps = multiprocessing.Value('i', 0)
        self.description = description
        update_progress(0.0, self.description)
    
    def increment(self, steps=1):
        with self.completed_steps.get_lock():
            self.completed_steps.value += steps
            completed = self.completed_steps.value
            if self.total_steps > 0:
                current_progress = min(100, (completed / self.total_steps * 100))
                update_progress(current_progress, self.description)
    
    def finalize(self):
        update_progress(100.0, self.description)
        sys.stdout.write("\n")
        sys.stdout.flush()

def check_memory_usage(threshold=0.7):
    try:
        vm = psutil.virtual_memory()
        return vm.percent >= (threshold * 100)
    except:
        return False

def save_to_disk(data, file_path):
    import pickle
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(data, f, protocol=4)
        return True
    except:
        return False

def load_from_disk(file_path):
    import pickle
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except:
        return None

def compute_fingerprints(mol_data):
    smiles, fp_types, fp_size, prefix, idx, concat_fp = mol_data
    
    result_dict = {}
    fp_bits = {}
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return idx, {}, {}, smiles
        
        if 'morgan' in fp_types:
            morgan_gen = GetMorganGenerator(radius=2, fpSize=fp_size)
            morgan_fp = morgan_gen.GetFingerprint(mol)
            on_bits = list(morgan_fp.GetOnBits())
            if concat_fp:
                bit_string = ['0'] * fp_size
                for bit in on_bits:
                    if bit < fp_size:
                        bit_string[bit] = '1'
                fp_bits[f"{prefix}_morgan_concat"] = ''.join(bit_string)
            else:
                for bit in on_bits:
                    result_dict[f"{prefix}_morgan_fp_{bit}"] = 1
        
        if 'rdkit' in fp_types:
            rdkit_gen = GetRDKitFPGenerator(fpSize=fp_size)
            rdkit_fp = rdkit_gen.GetFingerprint(mol)
            on_bits = list(rdkit_fp.GetOnBits())
            if concat_fp:
                bit_string = ['0'] * fp_size
                for bit in on_bits:
                    if bit < fp_size:
                        bit_string[bit] = '1'
                fp_bits[f"{prefix}_rdkit_concat"] = ''.join(bit_string)
            else:
                for bit in on_bits:
                    result_dict[f"{prefix}_rdkit_fp_{bit}"] = 1
        
        if 'maccs' in fp_types:
            maccs = MACCSkeys.GenMACCSKeys(mol)
            on_bits = list(maccs.GetOnBits())
            if concat_fp:
                bit_string = ['0'] * 167  # MACCS keys are 167 bits
                for bit in on_bits:
                    if bit < 167:
                        bit_string[bit] = '1'
                fp_bits[f"{prefix}_maccs_concat"] = ''.join(bit_string)
            else:
                for bit in on_bits:
                    result_dict[f"{prefix}_maccs_fp_{bit}"] = 1
                
        if 'atompair' in fp_types:
            ap_gen = GetAtomPairGenerator()
            ap_fp = ap_gen.GetSparseCountFingerprint(mol)
            elements = ap_fp.GetNonzeroElements()
            if concat_fp:
                if elements:
                    max_idx = max(elements.keys())
                    bit_string = ['0'] * (max_idx + 1)
                    for idx, count in elements.items():
                        bit_string[idx] = '1'
                    fp_bits[f"{prefix}_atompair_concat"] = ''.join(bit_string)
            else:
                for idx, count in elements.items():
                    result_dict[f"{prefix}_atompair_fp_{idx}"] = count
                
        if 'torsion' in fp_types:
            tt_gen = GetTopologicalTorsionGenerator()
            tt_fp = tt_gen.GetSparseCountFingerprint(mol)
            elements = tt_fp.GetNonzeroElements()
            if concat_fp:
                if elements:
                    max_idx = max(elements.keys())
                    bit_string = ['0'] * (max_idx + 1)
                    for idx, count in elements.items():
                        bit_string[idx] = '1'
                    fp_bits[f"{prefix}_torsion_concat"] = ''.join(bit_string)
            else:
                for idx, count in elements.items():
                    result_dict[f"{prefix}_torsion_fp_{idx}"] = count
                
        if 'morgan3' in fp_types:
            morgan_gen = GetMorganGenerator(radius=3, fpSize=fp_size)
            morgan_fp = morgan_gen.GetFingerprint(mol)
            on_bits = list(morgan_fp.GetOnBits())
            if concat_fp:
                bit_string = ['0'] * fp_size
                for bit in on_bits:
                    if bit < fp_size:
                        bit_string[bit] = '1'
                fp_bits[f"{prefix}_morgan3_concat"] = ''.join(bit_string)
            else:
                for bit in on_bits:
                    result_dict[f"{prefix}_morgan3_fp_{bit}"] = 1
        
        return idx, result_dict, fp_bits, smiles
    except:
        return idx, {}, {}, smiles

def reduce_dimensionality(df, fp_prefix, method='pca', n_components=100, target_col=None):
    print_message(f"Reducing dimensionality using {method} method")
    
    fp_columns = [col for col in df.columns if col.startswith(f"{fp_prefix}_") and "_fp_" in col]
    if not fp_columns:
        print_message(f"No fingerprint columns found with prefix {fp_prefix}_")
        return df
    
    print_message(f"Found {len(fp_columns)} fingerprint columns")
    X = df[fp_columns].fillna(0).values
    
    if method == 'pca':
        if n_components >= min(X.shape[0], X.shape[1]):
            n_components = min(X.shape[0], X.shape[1]) - 1
            print_message(f"Adjusted n_components to {n_components}")
            
        model = PCA(n_components=n_components, random_state=42)
        transformed = model.fit_transform(X)
        
        variance_ratio = model.explained_variance_ratio_
        cumulative_variance = np.cumsum(variance_ratio)
        print_message(f"Explained variance with {n_components} components: {cumulative_variance[-1]:.4f}")
        
        for i in range(transformed.shape[1]):
            df[f"{fp_prefix}_pca_{i+1}"] = transformed[:, i]
            
        return df
    
    elif method == 'svd':
        if n_components >= min(X.shape[0], X.shape[1]):
            n_components = min(X.shape[0], X.shape[1]) - 1
            print_message(f"Adjusted n_components to {n_components}")
            
        model = TruncatedSVD(n_components=n_components, random_state=42)
        transformed = model.fit_transform(X)
        
        variance_ratio = model.explained_variance_ratio_
        cumulative_variance = np.cumsum(variance_ratio)
        print_message(f"Explained variance with {n_components} components: {cumulative_variance[-1]:.4f}")
        
        for i in range(transformed.shape[1]):
            df[f"{fp_prefix}_svd_{i+1}"] = transformed[:, i]
            
        return df
    
    elif method == 'variance':
        selector = VarianceThreshold(threshold=0.01)
        selected = selector.fit_transform(X)
        selected_indices = selector.get_support(indices=True)
        
        print_message(f"Selected {len(selected_indices)} features based on variance")
        
        selected_columns = [fp_columns[i] for i in selected_indices]
        retained_df = df[selected_columns].copy()
        
        for col in retained_df.columns:
            new_col = col.replace("_fp_", "_var_")
            df[new_col] = retained_df[col]
            
        return df
    
    elif method == 'mutual_info' and target_col is not None and target_col in df.columns:
        y = df[target_col].values
        
        mi_scores = mutual_info_regression(X, y, random_state=42)
        top_indices = np.argsort(mi_scores)[-n_components:]
        
        print_message(f"Selected {len(top_indices)} features based on mutual information")
        
        selected_columns = [fp_columns[i] for i in top_indices]
        retained_df = df[selected_columns].copy()
        
        for col in retained_df.columns:
            new_col = col.replace("_fp_", "_mi_")
            df[new_col] = retained_df[col]
            
        return df
    
    elif method == 'folding':
        n_bits = len(fp_columns)
        folded_bits = n_components
        folding_factor = max(1, n_bits // folded_bits)
        
        folded_fps = np.zeros((X.shape[0], folded_bits))
        
        for i in range(n_bits):
            folded_idx = i % folded_bits
            folded_fps[:, folded_idx] |= X[:, i]
        
        print_message(f"Folded {n_bits} bits into {folded_bits} bits")
        
        for i in range(folded_bits):
            df[f"{fp_prefix}_folded_{i+1}"] = folded_fps[:, i]
            
        return df
    
    else:
        print_message(f"Unknown dimensionality reduction method: {method}")
        return df

def get_optimal_worker_count():
    try:
        cpus = multiprocessing.cpu_count()
        mem = psutil.virtual_memory()
        mem_gb = mem.available / (1024**3)
        mem_workers = max(1, int(mem_gb))
        return min(max(1, cpus - 1), mem_workers)
    except:
        return max(1, multiprocessing.cpu_count() - 1)

def main():
    global TEMP_DIR, TEMP_FILES, EXECUTOR, INTERRUPTED
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    atexit.register(exit_handler)
    
    parser = argparse.ArgumentParser(description='Compute molecular fingerprints with one bit per column')
    
    # Input/Output options
    io_group = parser.add_argument_group('Input/Output')
    io_group.add_argument('--input', '-i', required=True, help='Input file path (CSV, TSV, SDF)')
    io_group.add_argument('--output', '-o', required=True, help='Output file path')
    io_group.add_argument('--smiles-col', '-s', default='SMILES', help='Name of column containing SMILES (default: SMILES)')
    io_group.add_argument('--append', '-a', action='store_true', help='Append fingerprints to input file')
    io_group.add_argument('--prefix', '-p', default='rdkit', help='Prefix for fingerprint column names')
    
    # Fingerprint options
    fp_group = parser.add_argument_group('Fingerprint Generation')
    fp_group.add_argument('--fp-types', '-t', nargs='+', default=['morgan'], 
                        choices=['morgan', 'rdkit', 'maccs', 'atompair', 'torsion', 'morgan3', 'all'],
                        help='Fingerprint types to compute (default: morgan). Use "all" for all types.')
    fp_group.add_argument('--fp-size', type=int, default=1024, 
                        help='Fingerprint size for fixed-length fingerprints (default: 1024)')
    fp_group.add_argument('--morgan-radius', type=int, default=2,
                        help='Radius for Morgan fingerprints (default: 2)')
    
    # Concatenation options
    concat_group = parser.add_mutually_exclusive_group()
    concat_group.add_argument('--concat-per-fp', action='store_true', 
                             help='Concatenate bits as a single string per fingerprint type (e.g., 01001...)')
    concat_group.add_argument('--concat-all', action='store_true',
                             help='Concatenate all fingerprint bits into one combined string')
    # For backward compatibility
    concat_group.add_argument('--concat', action='store_true', help=argparse.SUPPRESS)
    
    # Dimensionality reduction options
    dim_group = parser.add_argument_group('Dimensionality Reduction')
    dim_group.add_argument('--reduce', '-r', action='store_true', help='Enable dimension reduction')
    dim_group.add_argument('--reduction-method', '-m', default='pca', 
                        choices=['pca', 'svd', 'variance', 'mutual_info', 'folding'],
                        help='Dimension reduction method (default: pca)')
    dim_group.add_argument('--n-components', '-c', type=int, default=100, 
                        help='Number of components after reduction (default: 100)')
    dim_group.add_argument('--target-col', 
                        help='Target column for mutual information based selection')
    
    # Performance options
    perf_group = parser.add_argument_group('Performance')
    perf_group.add_argument('--n-jobs', '-j', type=int, default=None, 
                          help='Number of parallel jobs (default: auto)')
    perf_group.add_argument('--ram-offloading', action='store_true', 
                          help='Enable RAM offloading to disk when needed')
    perf_group.add_argument('--batch-size', type=int, default=1000,
                          help='Batch size for processing (default: 1000)')
    perf_group.add_argument('--temp-dir', type=str, default=None,
                          help='Directory for temporary files (default: auto)')
    
    # Output control
    output_group = parser.add_argument_group('Output Control')
    output_group.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    output_group.add_argument('--quiet', '-q', action='store_true', help='Minimal output')
    output_group.add_argument('--no-progress', action='store_true', help='Disable progress bar')
    output_group.add_argument('--save-format', choices=['csv', 'tsv', 'parquet', 'pickle'], 
                            help='Force output format regardless of file extension')
    
    args = parser.parse_args()
    
    print_message("Molecular Fingerprint Generator")
    
    # Process fingerprint types
    all_fp_types = ['morgan', 'rdkit', 'maccs', 'atompair', 'torsion', 'morgan3']
    if 'all' in args.fp_types:
        fp_types = all_fp_types
        print_message("Computing all fingerprint types")
    else:
        fp_types = args.fp_types
    
    # Handle concatenation options
    concat_mode = None
    if args.concat_per_fp:
        concat_mode = 'per_fp'
        print_message("Concatenating bits per fingerprint type")
    elif args.concat_all:
        concat_mode = 'all'
        print_message("Concatenating all fingerprint bits into one string")
    elif args.concat:  # Backward compatibility
        concat_mode = 'per_fp'
        print_message("Using legacy --concat option (equivalent to --concat-per-fp)")
    
    # Setup temporary directory for RAM offloading
    if args.ram_offloading:
        print_message("RAM offloading enabled")
        try:
            if args.temp_dir and os.path.exists(args.temp_dir):
                TEMP_DIR = os.path.join(args.temp_dir, f"fp_gen_{os.getpid()}")
                os.makedirs(TEMP_DIR, exist_ok=True)
            else:
                TEMP_DIR = tempfile.mkdtemp(prefix="fp_gen_")
        except:
            TEMP_DIR = os.path.join(os.path.dirname(args.output), "temp_fp_gen")
            os.makedirs(TEMP_DIR, exist_ok=True)
    
    input_path = args.input
    output_path = args.output
    smiles_col = args.smiles_col
    fp_size = args.fp_size
    n_jobs = args.n_jobs if args.n_jobs else get_optimal_worker_count()
    
    print_message(f"Processing input file: {input_path}")
    
    file_ext = os.path.splitext(input_path)[1].lower()
    
    try:
        if file_ext == '.sdf':
            from rdkit.Chem import PandasTools
            df = PandasTools.LoadSDF(input_path)
            if 'ROMol' in df.columns:
                df[smiles_col] = [Chem.MolToSmiles(mol) if mol is not None else None for mol in df['ROMol']]
            print_message(f"Loaded SDF with {len(df)} molecules")
        elif file_ext == '.csv':
            df = pd.read_csv(input_path)
            print_message(f"Loaded CSV with {len(df)} rows")
        elif file_ext == '.tsv':
            df = pd.read_csv(input_path, sep='\t')
            print_message(f"Loaded TSV with {len(df)} rows")
        elif file_ext == '.parquet':
            df = pd.read_parquet(input_path)
            print_message(f"Loaded Parquet file with {len(df)} rows")
        elif file_ext == '.pkl' or file_ext == '.pickle':
            df = pd.read_pickle(input_path)
            print_message(f"Loaded Pickle file with {len(df)} rows")
        else:
            try:
                df = pd.read_csv(input_path)
                print_message(f"Loaded file as CSV with {len(df)} rows")
            except:
                df = pd.read_csv(input_path, sep='\t')
                print_message(f"Loaded file as TSV with {len(df)} rows")
    except Exception as e:
        print_message(f"Error loading input file: {str(e)}")
        sys.exit(1)
    
    if smiles_col not in df.columns:
        print_message(f"SMILES column '{smiles_col}' not found in input file")
        available_columns = ', '.join(df.columns)
        print_message(f"Available columns: {available_columns}")
        sys.exit(1)
    
    print_message(f"Computing fingerprints for {len(df)} molecules")
    print_message(f"Fingerprint types: {', '.join(fp_types)}")
    print_message(f"Fingerprint size: {fp_size}")
    if concat_mode:
        if concat_mode == 'per_fp':
            print_message("Concatenating fingerprint bits into single strings per type")
        elif concat_mode == 'all':
            print_message("Concatenating all fingerprint bits into one combined string")
    print_message(f"Using {n_jobs} parallel processes")
    
    # Batch processing setup
    batch_size = min(args.batch_size, len(df))
    if len(df) > batch_size and not args.quiet:
        print_message(f"Processing in batches of {batch_size} molecules")
    
    smiles_list = df[smiles_col].tolist()
    mol_data = [(smiles, fp_types, fp_size, args.prefix, i, bool(concat_mode)) for i, smiles in enumerate(smiles_list)]
    
    fp_columns = {}
    concat_columns = {}
    memory_check_interval = max(1, min(100, len(smiles_list) // 20))
    offload_count = 0
    
    if not args.no_progress and not args.quiet:
        progress_tracker = ProgressTracker(len(smiles_list), "Computing fingerprints")
    
    try:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            EXECUTOR = executor
            futures = [executor.submit(compute_fingerprints, data) for data in mol_data]
            
            results_buffer = {}
            
            for future_idx, future in enumerate(as_completed(futures)):
                if INTERRUPTED:
                    break
                
                try:
                    idx, result_dict, concat_dict, smiles = future.result()
                    
                    if result_dict:
                        results_buffer[idx] = result_dict
                        
                        for col_name in result_dict:
                            if col_name not in fp_columns:
                                fp_columns[col_name] = np.zeros(len(df))
                            
                            fp_columns[col_name][idx] = result_dict[col_name]
                    
                    if concat_dict:
                        for col_name, bit_string in concat_dict.items():
                            if col_name not in concat_columns:
                                concat_columns[col_name] = [""] * len(df)
                            concat_columns[col_name][idx] = bit_string
                    
                    if args.ram_offloading and future_idx % memory_check_interval == 0:
                        if check_memory_usage(0.75):
                            temp_file = os.path.join(TEMP_DIR, f"fp_{offload_count}.pkl")
                            data_to_save = {'fp_columns': fp_columns, 'concat_columns': concat_columns}
                            save_to_disk(data_to_save, temp_file)
                            TEMP_FILES.append(temp_file)
                            
                            offload_count += 1
                            fp_columns = {}
                            concat_columns = {}
                            gc.collect()
                    
                    if not args.no_progress and not args.quiet:
                        progress_tracker.increment()
                    
                except Exception as e:
                    if args.verbose:
                        print_message(f"Error processing molecule {idx}: {str(e)}")
                    if not args.no_progress and not args.quiet:
                        progress_tracker.increment()
                
                if INTERRUPTED:
                    break
            
            EXECUTOR = None
        
        if not INTERRUPTED:
            if not args.no_progress and not args.quiet:
                progress_tracker.finalize()
            
            print_message("Creating result DataFrame")
            
            # Combine all fingerprints into one string if concat_all is specified
            if concat_mode == 'all' and concat_columns:
                all_fp_combined = [""] * len(df)
                for col_name, values in concat_columns.items():
                    for idx, val in enumerate(values):
                        all_fp_combined[idx] += val
                
                # Replace individual concat columns with a single combined one
                concat_columns = {f"{args.prefix}_combined_concat": all_fp_combined}
            
            if args.ram_offloading and TEMP_FILES:
                if fp_columns or concat_columns:
                    temp_file = os.path.join(TEMP_DIR, f"fp_{offload_count}.pkl")
                    data_to_save = {'fp_columns': fp_columns, 'concat_columns': concat_columns}
                    save_to_disk(data_to_save, temp_file)
                    TEMP_FILES.append(temp_file)
                    fp_columns = {}
                    concat_columns = {}
                
                all_fp_columns = {}
                all_concat_columns = {}
                
                for temp_file in TEMP_FILES:
                    if INTERRUPTED:
                        break
                        
                    temp_data = load_from_disk(temp_file)
                    if temp_data:
                        if 'fp_columns' in temp_data:
                            for col_name, values in temp_data['fp_columns'].items():
                                if col_name not in all_fp_columns:
                                    all_fp_columns[col_name] = values
                                else:
                                    all_fp_columns[col_name] = np.maximum(all_fp_columns[col_name], values)
                        
                        if 'concat_columns' in temp_data:
                            for col_name, values in temp_data['concat_columns'].items():
                                if col_name not in all_concat_columns:
                                    all_concat_columns[col_name] = values
                                else:
                                    for i, val in enumerate(values):
                                        if val and not all_concat_columns[col_name][i]:
                                            all_concat_columns[col_name][i] = val
                    
                    try:
                        os.remove(temp_file)
                        TEMP_FILES.remove(temp_file)
                    except:
                        pass
                
                result_data = {}
                
                if args.append:
                    if all_fp_columns:
                        new_df = pd.DataFrame(all_fp_columns)
                        output_df = pd.concat([df, new_df], axis=1)
                    else:
                        output_df = df.copy()
                        
                    if all_concat_columns:
                        for col_name, values in all_concat_columns.items():
                            output_df[col_name] = values
                            
                    print_message(f"Appended fingerprint features to input data")
                else:
                    result_data = {smiles_col: smiles_list}
                    if all_fp_columns:
                        result_data.update(all_fp_columns)
                    
                    output_df = pd.DataFrame(result_data)
                    
                    if all_concat_columns:
                        for col_name, values in all_concat_columns.items():
                            output_df[col_name] = values
                            
                    print_message(f"Created new dataframe with fingerprint features")
            else:
                # Combine all fingerprints into one string if concat_all is specified
                if concat_mode == 'all' and concat_columns:
                    all_fp_combined = [""] * len(df)
                    for col_name, values in concat_columns.items():
                        for idx, val in enumerate(values):
                            all_fp_combined[idx] += val
                    
                    # Replace individual concat columns with a single combined one
                    concat_columns = {f"{args.prefix}_combined_concat": all_fp_combined}
                    
                if args.append:
                    if fp_columns:
                        new_df = pd.DataFrame(fp_columns)
                        output_df = pd.concat([df, new_df], axis=1)
                    else:
                        output_df = df.copy()
                        
                    if concat_columns:
                        for col_name, values in concat_columns.items():
                            output_df[col_name] = values
                            
                    print_message(f"Appended fingerprint features to input data")
                else:
                    result_data = {smiles_col: smiles_list}
                    if fp_columns:
                        result_data.update(fp_columns)
                    
                    output_df = pd.DataFrame(result_data)
                    
                    if concat_columns:
                        for col_name, values in concat_columns.items():
                            output_df[col_name] = values
                            
                    print_message(f"Created new dataframe with fingerprint features")
            
            if args.ram_offloading:
                cleanup_temp_files()
            
            if INTERRUPTED:
                print_message("Operation interrupted by user before saving")
                sys.exit(1)
                
            if args.reduce and not concat_mode:
                print_message(f"Reducing dimensionality using {args.reduction_method}")
                output_df = reduce_dimensionality(
                    output_df, 
                    args.prefix,
                    method=args.reduction_method,
                    n_components=args.n_components,
                    target_col=args.target_col
                )
            elif args.reduce and concat_mode:
                print_message("Skipping dimensionality reduction as concatenation option is enabled")
            
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            print_message(f"Saving results to {output_path}")
            
            try:
                # Determine output format
                output_format = args.save_format if args.save_format else os.path.splitext(output_path)[1].lower().lstrip('.')
                
                if output_format == 'csv' or output_format == '.csv':
                    output_df.to_csv(output_path, index=False)
                elif output_format == 'tsv' or output_format == '.tsv':
                    output_df.to_csv(output_path, sep='\t', index=False)
                elif output_format == 'parquet' or output_format == '.parquet':
                    output_df.to_parquet(output_path, index=False)
                elif output_format in ['pickle', 'pkl', '.pickle', '.pkl']:
                    output_df.to_pickle(output_path)
                else:
                    # Default to CSV
                    output_df.to_csv(output_path, index=False)
                    
                print_message(f"Results saved to {output_path}")
                
                if concat_mode:
                    concat_count = sum(1 for col in output_df.columns if (col.startswith(f"{args.prefix}_") and "_concat" in col))
                    print_message(f"Generated {concat_count} concatenated fingerprint string{'s' if concat_count > 1 else ''} for {len(df)} molecules")
                else:
                    fp_count = sum(1 for col in output_df.columns if col.startswith(f"{args.prefix}_"))
                    print_message(f"Generated {fp_count} fingerprint-related features for {len(df)} molecules")
            except Exception as e:
                print_message(f"Error saving file: {str(e)}")
    
    except Exception as e:
        print_message(f"Error: {str(e)}")
    finally:
        if args.ram_offloading:
            cleanup_temp_files()

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()