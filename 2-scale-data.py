#!/usr/bin/env python3
import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
from joblib import dump, load, Parallel, delayed
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
import multiprocessing

def parse_arguments():
    parser = argparse.ArgumentParser(description="Data Scaling Tool for Large Datasets")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV/Parquet file")
    parser.add_argument("--output", type=str, required=True, help="Path to output CSV/Parquet file")
    parser.add_argument("--scaler", type=str, default="standard", choices=["standard", "minmax", "robust", "maxabs", "quantile", "power"],
                        help="Type of scaler to use")
    parser.add_argument("--batch_size", type=int, default=100000, help="Batch size for processing")
    parser.add_argument("--save_scaler", type=str, default=None, help="Path to save fitted scaler (optional)")
    parser.add_argument("--load_scaler", type=str, default=None, help="Path to load pre-fitted scaler (optional)")
    parser.add_argument("--id_cols", type=str, default=None, help="Comma-separated list of ID columns to preserve")
    parser.add_argument("--exclude_cols", type=str, default=None, help="Comma-separated list of columns to exclude from scaling")
    parser.add_argument("--n_jobs", type=int, default=-1, help="Number of jobs for parallel processing (-1 for all cores)")
    parser.add_argument("--float_precision", type=str, default="float32", choices=["float32", "float64"], 
                        help="Float precision for optimized memory usage")
    parser.add_argument("--memory_limit", type=int, default=None, help="Memory limit in GB (None for no limit)")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    parser.add_argument("--quantile_options", type=str, default=None, 
                        help="Options for QuantileTransformer as JSON string: {\"n_quantiles\": 1000, \"output_distribution\": \"normal\"}")
    parser.add_argument("--power_options", type=str, default=None,
                        help="Options for PowerTransformer as JSON string: {\"method\": \"yeo-johnson\", \"standardize\": true}")
    return parser.parse_args()

def log_message(msg):
    print(f"-- {msg}")
    sys.stdout.flush()

def get_file_size(file_path):
    return os.path.getsize(file_path) / (1024 * 1024 * 1024)  # Size in GB

def create_scaler(scaler_type, **kwargs):
    scalers = {
        "standard": StandardScaler,
        "minmax": MinMaxScaler,
        "robust": RobustScaler,
        "maxabs": MaxAbsScaler,
        "quantile": QuantileTransformer,
        "power": PowerTransformer
    }
    
    if scaler_type not in scalers:
        log_message(f"WARNING: Unknown scaler type '{scaler_type}', defaulting to standard")
        return StandardScaler()
    
    return scalers[scaler_type](**kwargs)

def get_dtypes(file_path):
    extension = os.path.splitext(file_path)[1].lower()
    if extension == '.csv':
        return pd.read_csv(file_path, nrows=10).dtypes
    elif extension == '.parquet':
        return pd.read_parquet(file_path, engine='pyarrow').dtypes
    else:
        raise ValueError(f"Unsupported file extension: {extension}")

def determine_chunksize(file_path, batch_size, memory_limit=None):
    file_size = get_file_size(file_path)
    log_message(f"File size: {file_size:.2f} GB")
    
    if memory_limit is not None:
        available_memory = memory_limit
    else:
        available_memory = multiprocessing.cpu_count() * 2  # Estimate 2GB per core as default
    
    max_chunk_size = int(batch_size * (available_memory / file_size))
    recommended_chunk = min(batch_size, max_chunk_size)
    return max(10000, recommended_chunk)  # Ensure minimum reasonable chunk size

def load_data_efficient(file_path, batch_size, dtypes=None):
    extension = os.path.splitext(file_path)[1].lower()
    
    if extension == '.csv':
        for chunk in pd.read_csv(file_path, chunksize=batch_size, dtype=dtypes):
            yield chunk
    elif extension == '.parquet':
        # Read parquet in batches
        table = pd.read_parquet(file_path, engine='pyarrow')
        total_rows = len(table)
        for start in range(0, total_rows, batch_size):
            end = min(start + batch_size, total_rows)
            yield table.iloc[start:end]
    else:
        raise ValueError(f"Unsupported file extension: {extension}")

def fit_scaler_on_batches(file_path, scaler, batch_size, numeric_cols, exclude_cols, float_precision, verbose=False):
    log_message("Analyzing dataset structure")
    total_rows = 0
    data_reader = load_data_efficient(file_path, batch_size)
    
    log_message("Starting first pass to fit scaler")
    start_time = time.time()
    batch_counter = 0
    
    for batch in data_reader:
        batch_counter += 1
        total_rows += len(batch)
        if verbose and batch_counter % 10 == 0:
            log_message(f"Processing batch {batch_counter} ({total_rows:,} rows)")
        
        # Extract only numeric columns that are not excluded
        batch_numeric = batch[numeric_cols].astype(float_precision)
        
        # Partial fit the scaler
        scaler.partial_fit(batch_numeric)
    
    fit_time = time.time() - start_time
    log_message(f"Fitted scaler on {total_rows:,} rows in {fit_time:.2f} seconds")
    
    return scaler, total_rows

def transform_batch(batch, scaler, numeric_cols, id_cols, exclude_cols, float_precision):
    # Create a copy of the batch to preserve structure
    transformed_batch = batch.copy()
    
    # Extract numeric columns for transformation
    numeric_data = batch[numeric_cols].astype(float_precision)
    
    # Apply transformation to numeric data
    transformed_numeric = scaler.transform(numeric_data)
    
    # Replace original columns with transformed values
    for i, col in enumerate(numeric_cols):
        transformed_batch[col] = transformed_numeric[:, i]
    
    return transformed_batch

def transform_dataset(file_path, output_path, scaler, batch_size, numeric_cols, id_cols, exclude_cols, float_precision, n_jobs, verbose=False):
    log_message(f"Starting second pass to transform data")
    start_time = time.time()
    
    # Determine output file extension
    output_ext = os.path.splitext(output_path)[1].lower()
    
    data_reader = load_data_efficient(file_path, batch_size)
    
    if output_ext == '.csv':
        # For CSV, create empty file with header only for first batch
        first_batch = True
        for batch in data_reader:
            start_batch_time = time.time()
            transformed_batch = transform_batch(batch, scaler, numeric_cols, id_cols, exclude_cols, float_precision)
            
            # Write to output file
            if first_batch:
                transformed_batch.to_csv(output_path, index=False)
                first_batch = False
            else:
                transformed_batch.to_csv(output_path, mode='a', header=False, index=False)
            
            if verbose:
                batch_time = time.time() - start_batch_time
                log_message(f"Transformed batch of {len(batch):,} rows in {batch_time:.2f} seconds")
    
    elif output_ext == '.parquet':
        # For parquet, collect all batches and write at once (more efficient for parquet)
        transformed_data = []
        batch_counter = 0
        
        for batch in data_reader:
            batch_counter += 1
            start_batch_time = time.time()
            
            transformed_batch = transform_batch(batch, scaler, numeric_cols, id_cols, exclude_cols, float_precision)
            transformed_data.append(transformed_batch)
            
            if verbose and batch_counter % 10 == 0:
                batch_time = time.time() - start_batch_time
                log_message(f"Transformed batch {batch_counter} ({len(batch):,} rows) in {batch_time:.2f} seconds")
        
        # Combine all batches and write to parquet
        log_message("Combining batches and writing to parquet file")
        pd.concat(transformed_data).to_parquet(output_path, engine='pyarrow', index=False)
    
    transform_time = time.time() - start_time
    log_message(f"Transformed data in {transform_time:.2f} seconds")

def parallel_transform_dataset(file_path, output_path, scaler, batch_size, numeric_cols, id_cols, exclude_cols, float_precision, n_jobs, verbose=False):
    # This function uses joblib to parallelize the transformation
    log_message(f"Starting parallel transformation with {n_jobs} workers")
    start_time = time.time()
    
    # Determine output file extension
    output_ext = os.path.splitext(output_path)[1].lower()
    
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
    
    # Read data in chunks
    data_reader = load_data_efficient(file_path, batch_size)
    
    # Process batches in parallel
    if output_ext == '.csv':
        # For CSV output
        first_batch = True
        batch_number = 0
        
        for batch in data_reader:
            batch_number += 1
            if verbose:
                log_message(f"Processing batch {batch_number}")
            
            transformed_batch = transform_batch(batch, scaler, numeric_cols, id_cols, exclude_cols, float_precision)
            
            # Write to output file
            if first_batch:
                transformed_batch.to_csv(output_path, index=False)
                first_batch = False
            else:
                transformed_batch.to_csv(output_path, mode='a', header=False, index=False)
    
    elif output_ext == '.parquet':
        # For parquet, we'll collect batches and process them in parallel
        batches = []
        for batch in data_reader:
            batches.append(batch)
        
        if verbose:
            log_message(f"Collected {len(batches)} batches, starting parallel processing")
        
        # Process in parallel
        transformed_batches = Parallel(n_jobs=n_jobs)(
            delayed(transform_batch)(batch, scaler, numeric_cols, id_cols, exclude_cols, float_precision)
            for batch in batches
        )
        
        # Combine and write
        log_message("Combining batches and writing to parquet file")
        pd.concat(transformed_batches).to_parquet(output_path, engine='pyarrow', index=False)
    
    transform_time = time.time() - start_time
    log_message(f"Parallel transformation completed in {transform_time:.2f} seconds")

def main():
    args = parse_arguments()
    start_time = time.time()
    
    # Validate input file
    if not os.path.exists(args.input):
        log_message(f"ERROR: Input file {args.input} does not exist")
        return 1
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Parse JSON options for specialized scalers
    quantile_options = {}
    power_options = {}
    
    if args.quantile_options and args.scaler == "quantile":
        try:
            quantile_options = json.loads(args.quantile_options)
        except json.JSONDecodeError:
            log_message(f"WARNING: Could not parse quantile_options JSON, using defaults")
    
    if args.power_options and args.scaler == "power":
        try:
            power_options = json.loads(args.power_options)
        except json.JSONDecodeError:
            log_message(f"WARNING: Could not parse power_options JSON, using defaults")
    
    # Determine appropriate options based on scaler type
    scaler_options = {}
    if args.scaler == "quantile":
        scaler_options = quantile_options
    elif args.scaler == "power":
        scaler_options = power_options
    
    # Create or load scaler
    if args.load_scaler and os.path.exists(args.load_scaler):
        log_message(f"Loading pre-fitted scaler from {args.load_scaler}")
        scaler = load(args.load_scaler)
    else:
        log_message(f"Creating new {args.scaler} scaler")
        scaler = create_scaler(args.scaler, **scaler_options)
    
    # Parse column specifications
    id_cols = []
    if args.id_cols:
        id_cols = [col.strip() for col in args.id_cols.split(',')]
    
    exclude_cols = []
    if args.exclude_cols:
        exclude_cols = [col.strip() for col in args.exclude_cols.split(',')]
    
    # Combine ID and exclude columns
    all_excluded = list(set(id_cols + exclude_cols))
    
    # Determine batch size based on file size and memory constraints
    batch_size = determine_chunksize(args.input, args.batch_size, args.memory_limit)
    log_message(f"Using batch size of {batch_size:,} rows")
    
    # Read a sample to determine numeric columns
    sample_df = next(load_data_efficient(args.input, 1000))
    all_cols = sample_df.columns.tolist()
    
    # Identify numeric columns (excluding ID and explicitly excluded columns)
    numeric_cols = sample_df.select_dtypes(include=np.number).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in all_excluded]
    
    if not numeric_cols:
        log_message("ERROR: No numeric columns found for scaling after exclusions")
        return 1
    
    log_message(f"Found {len(numeric_cols)} numeric columns to scale")
    
    # If not loading a pre-fitted scaler, fit it on the data
    if not args.load_scaler or not os.path.exists(args.load_scaler):
        scaler, total_rows = fit_scaler_on_batches(
            args.input, scaler, batch_size, numeric_cols, 
            all_excluded, args.float_precision, args.verbose
        )
        
        # Save the fitted scaler if requested
        if args.save_scaler:
            scaler_dir = os.path.dirname(args.save_scaler)
            if scaler_dir and not os.path.exists(scaler_dir):
                os.makedirs(scaler_dir)
            dump(scaler, args.save_scaler)
            log_message(f"Saved fitted scaler to {args.save_scaler}")
    
    # Transform the dataset using the fitted/loaded scaler
    if args.n_jobs == 1:
        # Single process transformation
        transform_dataset(
            args.input, args.output, scaler, batch_size, numeric_cols,
            id_cols, all_excluded, args.float_precision, args.n_jobs, args.verbose
        )
    else:
        # Parallel transformation
        parallel_transform_dataset(
            args.input, args.output, scaler, batch_size, numeric_cols,
            id_cols, all_excluded, args.float_precision, args.n_jobs, args.verbose
        )
    
    total_time = time.time() - start_time
    log_message(f"Complete! Processed dataset in {total_time:.2f} seconds")
    
    # Print scaled data statistics (if verbose)
    if args.verbose:
        log_message("Scaled data statistics:")
        try:
            result_ext = os.path.splitext(args.output)[1].lower()
            if result_ext == '.csv':
                sample = pd.read_csv(args.output, nrows=1000)
            else:
                sample = pd.read_parquet(args.output, engine='pyarrow')
                
            for col in numeric_cols:
                if col in sample.columns:
                    stats = sample[col].describe()
                    log_message(f"  {col}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, min={stats['min']:.4f}, max={stats['max']:.4f}")
        except Exception as e:
            log_message(f"Could not read statistics from result file: {str(e)}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 