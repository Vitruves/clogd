#!/usr/bin/env python3
import pandas as pd
import argparse
import numpy as np
import sys

def scale_values(df, column, scale_type):
    """Apply scaling to a column."""
    if scale_type == 'log':
        # Handle zeros and negative values for log scaling
        min_val = df[column].min()
        if min_val <= 0:
            offset = abs(min_val) + 1
            df[column] = np.log(df[column] + offset)
        else:
            df[column] = np.log(df[column])
    elif scale_type == 'minmax':
        df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
    elif scale_type == 'zscore':
        df[column] = (df[column] - df[column].mean()) / df[column].std()
    return df

def main():
    parser = argparse.ArgumentParser(description='Filter CSV data by column values and apply scaling')
    parser.add_argument('input_file', help='Input CSV file path')
    parser.add_argument('--output', '-o', help='Output CSV file path (default: stdout)', default=None)
    parser.add_argument('--target', '-t', help='Target column name to filter and/or scale')
    parser.add_argument('--min', type=float, help='Minimum value filter')
    parser.add_argument('--max', type=float, help='Maximum value filter')
    parser.add_argument('--scale', choices=['log', 'minmax', 'zscore'], help='Apply scaling to the target column')
    parser.add_argument('--show-cols', action='store_true', help='Show available columns and exit')
    
    args = parser.parse_args()
    
    try:
        # Read the CSV file
        df = pd.read_csv(args.input_file)
        
        # Show columns and exit if requested
        if args.show_cols:
            print("Available columns:")
            for col in df.columns:
                print(f"  {col}")
            return
        
        # Check if target column exists
        if args.target and args.target not in df.columns:
            print(f"Error: Column '{args.target}' not found in the CSV file", file=sys.stderr)
            print(f"Available columns: {', '.join(df.columns)}", file=sys.stderr)
            return
        
        # Apply filters if specified
        if args.target:
            if args.min is not None:
                df = df[df[args.target] >= args.min]
            
            if args.max is not None:
                df = df[df[args.target] <= args.max]
            
            # Apply scaling if specified
            if args.scale:
                df = scale_values(df, args.target, args.scale)
        
        # Output the filtered data
        if args.output:
            df.to_csv(args.output, index=False)
            print(f"Filtered data saved to {args.output}")
        else:
            df.to_csv(sys.stdout, index=False)
    
    except FileNotFoundError:
        print(f"Error: File '{args.input_file}' not found", file=sys.stderr)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()