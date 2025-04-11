#!/usr/bin/env python3

import os
import sys
import argparse
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from pathlib import Path
import multiprocessing
import time
import gc
import psutil
import tempfile
import signal
import shutil
import atexit
from concurrent.futures import ProcessPoolExecutor, as_completed

TEMP_DIR = None
TEMP_FILES = []
INTERRUPTED = False
EXECUTOR = None

np.seterr(all='ignore')

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

def prepare_3d_molecule(mol):
    if mol is None:
        return None
    
    try:
        if mol.GetNumConformers() > 0:
            conf = mol.GetConformer()
            if conf.Is3D():
                return mol
        
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.UFFOptimizeMolecule(mol)
        return mol
    except:
        return None

def compute_mordred_descriptors(mol_data):
    smiles, compute_3d, prefix, idx = mol_data
    
    try:
        from mordred import Calculator, descriptors
        
        calc = Calculator(descriptors, ignore_3D=not compute_3d)
        
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            if compute_3d:
                mol = prepare_3d_molecule(mol)
                if mol is None:
                    return idx, {}, smiles
            
            desc_dict = calc(mol)
            result_dict = {}
            
            for desc_name, value in desc_dict.items():
                if value is not None:
                    try:
                        if hasattr(value, "as_dict"):
                            for k, v in value.as_dict().items():
                                column_name = f"{prefix}_{desc_name}_{k}"
                                result_dict[column_name] = float(v) if v is not None else np.nan
                        else:
                            column_name = f"{prefix}_{desc_name}"
                            result_dict[column_name] = float(value) if value is not None else np.nan
                    except:
                        pass
            
            return idx, result_dict, smiles
        else:
            return idx, {}, smiles
    except:
        return idx, {}, smiles

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
    
    parser = argparse.ArgumentParser(description='Compute Mordred descriptors for molecules')
    parser.add_argument('--input', '-i', required=True, help='Input file path (CSV, TSV, SDF)')
    parser.add_argument('--output', '-o', required=True, help='Output file path')
    parser.add_argument('--smiles-col', '-s', default='SMILES', help='Name of column containing SMILES (default: SMILES)')
    parser.add_argument('--compute-3d', '-3d', action='store_true', help='Compute 3D descriptors')
    parser.add_argument('--append', '-a', action='store_true', help='Append descriptors to input file')
    parser.add_argument('--prefix', '-p', default='mordred', help='Prefix for descriptor column names')
    parser.add_argument('--n-jobs', '-j', type=int, default=None, help='Number of parallel jobs')
    parser.add_argument('--ram-offloading', action='store_true', help='Enable RAM offloading to disk when needed')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    print_message("Mordred Descriptor Calculator")
    
    try:
        import mordred
        print_message("Mordred package found")
    except ImportError:
        print_message("Mordred package not installed. To install, run: pip install mordred")
        sys.exit(1)
    
    if args.ram_offloading:
        print_message("RAM offloading enabled")
        try:
            TEMP_DIR = tempfile.mkdtemp(prefix="mordred_")
        except:
            TEMP_DIR = os.path.join(os.path.dirname(args.output), "temp_mordred")
            os.makedirs(TEMP_DIR, exist_ok=True)
    
    input_path = args.input
    output_path = args.output
    smiles_col = args.smiles_col
    compute_3d = args.compute_3d
    n_jobs = args.n_jobs if args.n_jobs else get_optimal_worker_count()
    ram_offloading = args.ram_offloading
    
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
        elif file_ext in ['.xlsx', '.xls']:
            df = pd.read_excel(input_path)
            print_message(f"Loaded Excel file with {len(df)} rows")
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
    
    print_message(f"Computing Mordred descriptors for {len(df)} molecules")
    print_message(f"3D descriptors: {'enabled' if compute_3d else 'disabled'}")
    print_message(f"Using {n_jobs} parallel processes")
    
    smiles_list = df[smiles_col].tolist()
    mol_data = [(smiles, compute_3d, args.prefix, i) for i, smiles in enumerate(smiles_list)]
    
    descriptor_columns = {}
    memory_check_interval = max(1, min(100, len(smiles_list) // 20))
    offload_count = 0
    
    progress_tracker = ProgressTracker(len(smiles_list), "Computing descriptors")
    
    try:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            EXECUTOR = executor
            futures = [executor.submit(compute_mordred_descriptors, data) for data in mol_data]
            
            results_buffer = {}
            
            for future_idx, future in enumerate(as_completed(futures)):
                if INTERRUPTED:
                    break
                
                try:
                    idx, result_dict, smiles = future.result()
                    
                    if result_dict:
                        results_buffer[idx] = result_dict
                        
                        for col_name in result_dict:
                            if col_name not in descriptor_columns:
                                descriptor_columns[col_name] = np.full(len(df), np.nan)
                            
                            descriptor_columns[col_name][idx] = result_dict[col_name]
                    
                    if ram_offloading and future_idx % memory_check_interval == 0:
                        if check_memory_usage(0.75):
                            temp_file = os.path.join(TEMP_DIR, f"mordred_{offload_count}.pkl")
                            save_to_disk(descriptor_columns, temp_file)
                            TEMP_FILES.append(temp_file)
                            
                            offload_count += 1
                            descriptor_columns = {}
                            gc.collect()
                    
                    progress_tracker.increment()
                    
                except Exception:
                    progress_tracker.increment()
                
                if INTERRUPTED:
                    break
            
            EXECUTOR = None
        
        if not INTERRUPTED:
            progress_tracker.finalize()
            
            print_message("Creating result DataFrame")
            
            if ram_offloading and TEMP_FILES:
                if descriptor_columns:
                    temp_file = os.path.join(TEMP_DIR, f"mordred_{offload_count}.pkl")
                    save_to_disk(descriptor_columns, temp_file)
                    TEMP_FILES.append(temp_file)
                    descriptor_columns = {}
                
                all_columns = {}
                
                for temp_file in TEMP_FILES:
                    if INTERRUPTED:
                        break
                        
                    temp_data = load_from_disk(temp_file)
                    if temp_data:
                        for col_name, values in temp_data.items():
                            if col_name not in all_columns:
                                all_columns[col_name] = values
                            else:
                                mask = ~np.isnan(values)
                                all_columns[col_name][mask] = values[mask]
                    
                    try:
                        os.remove(temp_file)
                        TEMP_FILES.remove(temp_file)
                    except:
                        pass
                
                if args.append:
                    new_df = pd.DataFrame(all_columns)
                    output_df = pd.concat([df, new_df], axis=1)
                    print_message(f"Appended {len(all_columns)} descriptors to input data")
                else:
                    result_data = {smiles_col: smiles_list}
                    result_data.update(all_columns)
                    output_df = pd.DataFrame(result_data)
                    print_message(f"Created new dataframe with {len(all_columns)} descriptors")
            else:
                if args.append:
                    new_df = pd.DataFrame(descriptor_columns)
                    output_df = pd.concat([df, new_df], axis=1)
                    print_message(f"Appended {len(descriptor_columns)} descriptors to input data")
                else:
                    result_data = {smiles_col: smiles_list}
                    result_data.update(descriptor_columns)
                    output_df = pd.DataFrame(result_data)
                    print_message(f"Created new dataframe with {len(descriptor_columns)} descriptors")
            
            if ram_offloading:
                cleanup_temp_files()
            
            if INTERRUPTED:
                print_message("Operation interrupted by user before saving")
                sys.exit(1)
            
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            print_message(f"Saving results to {output_path}")
            
            try:
                file_ext = os.path.splitext(output_path)[1].lower()
                
                if file_ext == '.csv':
                    # Use efficient chunked saving for large files
                    if len(output_df) > 10000:
                        print_message("Using chunked CSV writing for large dataset")
                        chunk_size = min(10000, max(1000, len(output_df) // 10))
                        
                        # Write header first
                        output_df.iloc[:0].to_csv(output_path, index=False)
                        
                        # Append chunks without headers
                        for i in range(0, len(output_df), chunk_size):
                            end_idx = min(i + chunk_size, len(output_df))
                            if args.verbose:
                                print_message(f"Writing rows {i+1}-{end_idx} of {len(output_df)}")
                            chunk = output_df.iloc[i:end_idx]
                            chunk.to_csv(output_path, mode='a', header=False, index=False)
                    else:
                        output_df.to_csv(output_path, index=False)
                
                elif file_ext == '.tsv':
                    # Use efficient chunked saving for large files
                    if len(output_df) > 10000:
                        print_message("Using chunked TSV writing for large dataset")
                        chunk_size = min(10000, max(1000, len(output_df) // 10))
                        
                        # Write header first
                        output_df.iloc[:0].to_csv(output_path, sep='\t', index=False)
                        
                        # Append chunks without headers
                        for i in range(0, len(output_df), chunk_size):
                            end_idx = min(i + chunk_size, len(output_df))
                            if args.verbose:
                                print_message(f"Writing rows {i+1}-{end_idx} of {len(output_df)}")
                            chunk = output_df.iloc[i:end_idx]
                            chunk.to_csv(output_path, sep='\t', mode='a', header=False, index=False)
                    else:
                        output_df.to_csv(output_path, sep='\t', index=False)
                
                elif file_ext in ['.xlsx', '.xls']:
                    if len(output_df) > 100000:
                        print_message("Warning: Large dataset for Excel format. This may be slow.")
                    
                    # Try to use a more efficient Excel writer
                    try:
                        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                            output_df.to_excel(writer, index=False, sheet_name='Descriptors')
                    except:
                        print_message("Falling back to default Excel writer")
                        output_df.to_excel(output_path, index=False)
                
                else:
                    # Default to chunked CSV for unknown formats
                    print_message(f"Unknown format '{file_ext}', saving as CSV")
                    if len(output_df) > 10000:
                        print_message("Using chunked CSV writing for large dataset")
                        chunk_size = min(10000, max(1000, len(output_df) // 10))
                        
                        # Write header first
                        output_df.iloc[:0].to_csv(output_path, index=False)
                        
                        # Append chunks without headers
                        for i in range(0, len(output_df), chunk_size):
                            end_idx = min(i + chunk_size, len(output_df))
                            if args.verbose:
                                print_message(f"Writing rows {i+1}-{end_idx} of {len(output_df)}")
                            chunk = output_df.iloc[i:end_idx]
                            chunk.to_csv(output_path, mode='a', header=False, index=False)
                    else:
                        output_df.to_csv(output_path, index=False)
                
                print_message(f"Results saved to {output_path}")
                descriptor_count = len(output_df.columns) - (1 if not args.append else len(df.columns))
                print_message(f"Computed {descriptor_count} Mordred descriptors for {len(df)} molecules")
            
            except Exception as e:
                print_message(f"Error saving file: {str(e)}")
                
                # Fallback to save in CSV format with different name if the original save fails
                try:
                    fallback_path = os.path.splitext(output_path)[0] + "_fallback.csv"
                    print_message(f"Attempting to save as CSV to {fallback_path}")
                    
                    # Use chunked writing for the fallback too
                    chunk_size = min(5000, max(1000, len(output_df) // 20))
                    
                    # Write header first
                    output_df.iloc[:0].to_csv(fallback_path, index=False)
                    
                    # Append chunks without headers
                    for i in range(0, len(output_df), chunk_size):
                        end_idx = min(i + chunk_size, len(output_df))
                        chunk = output_df.iloc[i:end_idx]
                        chunk.to_csv(fallback_path, mode='a', header=False, index=False)
                    
                    print_message(f"Successfully saved fallback file to {fallback_path}")
                except Exception as fallback_error:
                    print_message(f"Fallback save also failed: {str(fallback_error)}")
    
    except Exception as e:
        print_message(f"Error: {str(e)}")
    finally:
        if ram_offloading:
            cleanup_temp_files()

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()