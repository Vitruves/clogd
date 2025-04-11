#!/usr/bin/env python3

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, MACCSkeys, AllChem, DataStructs
from rdkit.ML.Descriptors import MoleculeDescriptors # type: ignore
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from scipy import stats
import argparse
from pathlib import Path
import json
import joblib
from datetime import datetime
from rdkit import DataStructs

def update_progress(progress, stage="Analyzing"):
    """Update progress on a single line"""
    sys.stdout.write(f"\r[ {progress:6.2f}% ] -- {stage}")
    sys.stdout.flush()
    if progress >= 100:
        sys.stdout.write("\n")

def calculate_basic_stats(df, true_col, pred_col):
    """Calculate basic statistical metrics between true and predicted values"""
    # Filter out any NaN values
    valid_data = df.dropna(subset=[true_col, pred_col])
    
    if len(valid_data) == 0:
        return {
            "count": 0,
            "error": "No valid data points found"
        }
    
    true_values = valid_data[true_col].values
    pred_values = valid_data[pred_col].values
    errors = pred_values - true_values
    abs_errors = np.abs(errors)
    
    # Calculate metrics
    stats = {
        "count": len(valid_data),
        "mae": mean_absolute_error(true_values, pred_values),
        "rmse": np.sqrt(mean_squared_error(true_values, pred_values)),
        "r2": r2_score(true_values, pred_values),
        "bias": np.mean(errors),
        "max_error": np.max(abs_errors),
        "std_error": np.std(errors),
        "median_error": np.median(abs_errors),
        "q1_error": np.percentile(abs_errors, 25),
        "q3_error": np.percentile(abs_errors, 75),
        "true_min": np.min(true_values),
        "true_max": np.max(true_values),
        "true_mean": np.mean(true_values),
        "pred_min": np.min(pred_values),
        "pred_max": np.max(pred_values),
        "pred_mean": np.mean(pred_values)
    }
    
    # Add percentile errors
    for percentile in [90, 95, 99]:
        stats[f"p{percentile}_error"] = np.percentile(abs_errors, percentile)
        
    return stats

def calculate_error_distribution(df, true_col, pred_col, bins=20):
    """Calculate error distribution for histogram"""
    valid_data = df.dropna(subset=[true_col, pred_col])
    errors = valid_data[pred_col] - valid_data[true_col]
    hist, bin_edges = np.histogram(errors, bins=bins)
    return {
        "counts": hist.tolist(),
        "bin_edges": bin_edges.tolist(),
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "std": float(np.std(errors))
    }

def identify_outliers(df, true_col, pred_col, z_threshold=3.0):
    """Identify outliers based on z-score of prediction error"""
    valid_data = df.dropna(subset=[true_col, pred_col]).copy()
    errors = valid_data[pred_col] - valid_data[true_col]
    abs_errors = np.abs(errors)
    
    # Calculate z-scores of errors
    z_scores = np.abs(stats.zscore(errors))
    valid_data['z_score'] = z_scores
    valid_data['error'] = errors
    valid_data['abs_error'] = abs_errors
    
    # Filter outliers
    outliers = valid_data[z_scores > z_threshold].sort_values('abs_error', ascending=False)
    return outliers

def calculate_chemical_descriptors(smiles):
    """Calculate key chemical descriptors for error analysis"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    
    descriptors = {}
    
    # Molecular weight
    descriptors['MW'] = Descriptors.MolWt(mol)
    
    # LogP and related
    descriptors['cLogP'] = Descriptors.MolLogP(mol)
    
    # Topological descriptors
    descriptors['TPSA'] = Descriptors.TPSA(mol)
    descriptors['NumRotatableBonds'] = Descriptors.NumRotatableBonds(mol)
    descriptors['NumHDonors'] = Lipinski.NumHDonors(mol)
    descriptors['NumHAcceptors'] = Lipinski.NumHAcceptors(mol)
    
    # Structural features
    descriptors['NumAromaticRings'] = Descriptors.NumAromaticRings(mol)
    descriptors['NumAliphaticRings'] = Descriptors.NumAliphaticRings(mol)
    descriptors['FractionCSP3'] = Descriptors.FractionCSP3(mol)
    
    # Charge-related
    descriptors['NumHeteroatoms'] = Descriptors.NumHeteroatoms(mol)
    
    # Complexity
    descriptors['BertzCT'] = Descriptors.BertzCT(mol)
    
    return descriptors

def analyze_descriptor_correlation(df, true_col, pred_col, smiles_col):
    """Analyze correlation between prediction errors and molecular descriptors"""
    valid_data = df.dropna(subset=[true_col, pred_col, smiles_col]).copy()
    
    # Calculate errors
    valid_data['error'] = valid_data[pred_col] - valid_data[true_col]
    valid_data['abs_error'] = np.abs(valid_data['error'])
    
    # Calculate descriptors for each molecule
    descriptors_list = []
    for idx, row in valid_data.iterrows():
        try:
            desc = calculate_chemical_descriptors(row[smiles_col])
            desc['index'] = idx
            descriptors_list.append(desc)
        except:
            pass
    
    if not descriptors_list:
        return {
            "error": "Could not calculate descriptors for any molecules"
        }
    
    # Create descriptor dataframe
    desc_df = pd.DataFrame(descriptors_list)
    desc_df = desc_df.set_index('index')
    
    # Merge with original dataframe
    analysis_df = pd.merge(valid_data, desc_df, left_index=True, right_index=True)
    
    # Calculate correlations between descriptors and error metrics
    error_corr = []
    abs_error_corr = []
    desc_cols = [col for col in desc_df.columns if col != 'index']
    
    for col in desc_cols:
        ec = analysis_df[['error', col]].corr().iloc[0,1]
        aec = analysis_df[['abs_error', col]].corr().iloc[0,1]
        error_corr.append(ec)
        abs_error_corr.append(aec)
    
    return {
        "descriptors": desc_cols,
        "error_correlation": error_corr,
        "abs_error_correlation": abs_error_corr,
        "dataframe": analysis_df
    }

def analyze_error_vs_true_value(df, true_col, pred_col, num_bins=10):
    """Analyze how prediction error varies across the range of true values"""
    valid_data = df.dropna(subset=[true_col, pred_col]).copy()
    
    # Calculate errors
    valid_data['error'] = valid_data[pred_col] - valid_data[true_col]
    valid_data['abs_error'] = np.abs(valid_data['error'])
    
    # Create bins based on true values
    min_val = valid_data[true_col].min()
    max_val = valid_data[true_col].max()
    bin_edges = np.linspace(min_val, max_val, num_bins + 1)
    
    # Assign each value to a bin
    valid_data['bin'] = pd.cut(valid_data[true_col], bins=bin_edges, labels=False)
    
    # Compute statistics per bin
    bin_stats = valid_data.groupby('bin').agg({
        true_col: ['mean', 'count'],
        'error': ['mean', 'std'],
        'abs_error': ['mean', 'median', 'max']
    }).reset_index()
    
    # Flatten the column names
    bin_stats.columns = ['_'.join(col).strip() for col in bin_stats.columns.values]
    
    return {
        "bin_edges": bin_edges.tolist(),
        "bin_stats": bin_stats
    }

def train_error_predictor(df, true_col, pred_col, smiles_col):
    """Train a model to predict the error based on molecular features"""
    valid_data = df.dropna(subset=[true_col, pred_col, smiles_col]).copy()
    
    # Calculate errors
    valid_data['error'] = valid_data[pred_col] - valid_data[true_col]
    valid_data['abs_error'] = np.abs(valid_data['error'])
    
    # Generate fingerprints for each molecule
    fingerprints = []
    valid_indices = []
    
    update_progress(0, "Generating fingerprints")
    
    # Import DataStructs for converting fingerprints to numpy arrays
    from rdkit import DataStructs
    # Use MorganGenerator to avoid deprecation warning
    from rdkit.Chem import rdFingerprintGenerator
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    
    for i, (idx, row) in enumerate(valid_data.iterrows()):
        try:
            mol = Chem.MolFromSmiles(row[smiles_col])
            if mol:
                # Get fingerprint as bit vector
                fp = fpgen.GetFingerprint(mol)
                # Convert to numpy array
                arr = np.zeros((2048,), dtype=np.int32)
                DataStructs.ConvertToNumpyArray(fp, arr)
                fingerprints.append(arr)
                valid_indices.append(idx)
        except Exception as e:
            pass
        
        if i % 100 == 0:
            update_progress(min(95, i/len(valid_data)*100), "Generating fingerprints")
    
    update_progress(95, "Training error predictor")
    
    if not fingerprints:
        return {"error": "Failed to generate valid fingerprints"}
    
    # Convert to numpy array
    X = np.vstack(fingerprints)
    y = valid_data.loc[valid_indices, 'abs_error'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train a simple model to predict the error
    model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    # Feature importance
    importance = model.feature_importances_
    
    update_progress(100, "Completed error analysis")
    
    return {
        "train_r2": train_score,
        "test_r2": test_score,
        "feature_importance": importance.tolist(),
        "model": model
    }

def create_visualizations(df, true_col, pred_col, output_dir):
    """Create visualizations for error analysis"""
    valid_data = df.dropna(subset=[true_col, pred_col]).copy()
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Calculate errors
    valid_data['error'] = valid_data[pred_col] - valid_data[true_col]
    valid_data['abs_error'] = np.abs(valid_data['error'])
    
    # 1. Scatter plot: True vs Predicted
    plt.figure(figsize=(10, 8))
    max_val = max(valid_data[true_col].max(), valid_data[pred_col].max())
    min_val = min(valid_data[true_col].min(), valid_data[pred_col].min())
    
    # Create a density scatter plot
    density_scatter = plt.hexbin(
        valid_data[true_col], valid_data[pred_col], 
        gridsize=50, cmap='viridis', 
        mincnt=1, bins='log'
    )
    
    # Add diagonal line (perfect prediction)
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    
    # Add colorbar
    cbar = plt.colorbar(density_scatter)
    cbar.set_label('log10(count)')
    
    plt.title(f'True vs Predicted LogD Values\nR² = {r2_score(valid_data[true_col], valid_data[pred_col]):.3f}')
    plt.xlabel(f'True {true_col}')
    plt.ylabel(f'Predicted {pred_col}')
    plt.axis('equal')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'true_vs_predicted.png', dpi=300)
    plt.close()
    
    # 2. Error histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(valid_data['error'], kde=True, bins=50)
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.7)
    plt.title(f'Error Distribution\nMean = {valid_data["error"].mean():.3f}, StdDev = {valid_data["error"].std():.3f}')
    plt.xlabel('Prediction Error (Predicted - True)')
    plt.ylabel('Count')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'error_histogram.png', dpi=300)
    plt.close()
    
    # 3. Absolute error vs true value
    plt.figure(figsize=(10, 6))
    plt.scatter(valid_data[true_col], valid_data['abs_error'], alpha=0.5, s=5)
    
    # Add trend line
    z = np.polyfit(valid_data[true_col], valid_data['abs_error'], 1)
    p = np.poly1d(z)
    plt.plot(sorted(valid_data[true_col]), p(sorted(valid_data[true_col])), "r--", linewidth=2)
    
    plt.title('Absolute Error vs True Value')
    plt.xlabel(f'True {true_col}')
    plt.ylabel('Absolute Error')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'error_vs_true.png', dpi=300)
    plt.close()
    
    # 4. Bias analysis: Error vs True Value
    plt.figure(figsize=(10, 6))
    plt.scatter(valid_data[true_col], valid_data['error'], alpha=0.5, s=5)
    
    # Add trend line
    z = np.polyfit(valid_data[true_col], valid_data['error'], 1)
    p = np.poly1d(z)
    plt.plot(sorted(valid_data[true_col]), p(sorted(valid_data[true_col])), "r--", linewidth=2)
    
    # Add zero line
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    plt.title('Error vs True Value (Bias Analysis)')
    plt.xlabel(f'True {true_col}')
    plt.ylabel('Error (Predicted - True)')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'bias_analysis.png', dpi=300)
    plt.close()
    
    # 5. Error distribution by value range
    bin_results = analyze_error_vs_true_value(valid_data, true_col, pred_col, num_bins=10)
    bin_stats = bin_results["bin_stats"]
    
    plt.figure(figsize=(12, 6))
    
    # Bar chart for mean absolute error per bin
    x = np.arange(len(bin_stats))
    width = 0.35
    
    plt.bar(x, bin_stats['abs_error_mean'], width, alpha=0.7, label='Mean Abs Error')
    plt.bar(x + width, bin_stats['error_std'], width, alpha=0.7, label='Error StdDev')
    
    # Fix the mismatch between tick locations and labels
    # Make sure we have the same number of labels as we have bin indices
    bin_labels = []
    for i in range(len(bin_stats)):
        bin_idx = int(bin_stats.iloc[i]['bin_'])
        if bin_idx < len(bin_results['bin_edges']) - 1:
            bin_labels.append(f"{bin_results['bin_edges'][bin_idx]:.1f}-{bin_results['bin_edges'][bin_idx+1]:.1f}")
        else:
            # Fallback for any out-of-range indices
            bin_labels.append(f"Bin {bin_idx}")
    
    # Now set the tick positions and labels
    plt.xticks(x + width/2, bin_labels, rotation=45)
    
    plt.xlabel(f'Range of True {true_col}')
    plt.ylabel('Error Metric')
    plt.title('Error Distribution by Value Range')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'error_by_range.png', dpi=300)
    plt.close()
    
    # Return paths to generated visualizations
    return {
        "true_vs_predicted": str(output_dir / 'true_vs_predicted.png'),
        "error_histogram": str(output_dir / 'error_histogram.png'),
        "error_vs_true": str(output_dir / 'error_vs_true.png'),
        "bias_analysis": str(output_dir / 'bias_analysis.png'),
        "error_by_range": str(output_dir / 'error_by_range.png')
    }

def main():
    parser = argparse.ArgumentParser(description="LogD Model Analysis Tool")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input CSV file with true and predicted LogD values")
    parser.add_argument("--true-col", "-t", type=str, default="LogD", help="Column name for true LogD values")
    parser.add_argument("--pred-col", "-p", type=str, default="cLogD", help="Column name for predicted LogD values")
    parser.add_argument("--smiles-col", "-s", type=str, default="SMILES", help="Column name for SMILES strings")
    parser.add_argument("--output-dir", "-o", type=str, default="logd_analysis", help="Output directory for analysis results")
    parser.add_argument("--full-analysis", "-f", action="store_true", help="Perform full analysis including error predictor training")
    
    args = parser.parse_args()
    
    # Load the data
    try:
        df = pd.read_csv(args.input)
        print(f"Loaded {len(df)} records from {args.input}")
    except Exception as e:
        print(f"Error loading input file: {str(e)}")
        sys.exit(1)
    
    # Verify required columns exist
    required_cols = [args.true_col, args.pred_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {', '.join(missing_cols)}")
        print(f"Available columns: {', '.join(df.columns)}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    results = {}
    
    # Calculate basic statistics
    update_progress(10, "Calculating basic statistics")
    results["basic_stats"] = calculate_basic_stats(df, args.true_col, args.pred_col)
    
    # Calculate error distribution
    update_progress(20, "Analyzing error distribution")
    results["error_distribution"] = calculate_error_distribution(df, args.true_col, args.pred_col)
    
    # Identify outliers
    update_progress(30, "Identifying outliers")
    outliers = identify_outliers(df, args.true_col, args.pred_col)
    results["outliers_count"] = len(outliers)
    
    # Export top outliers
    if len(outliers) > 0:
        top_n = min(50, len(outliers))
        outliers.head(top_n).to_csv(output_dir / "top_outliers.csv", index=True)
        results["outliers_file"] = str(output_dir / "top_outliers.csv")
    
    # Analyze error vs true value relationship
    update_progress(40, "Analyzing error patterns")
    results["error_by_value"] = analyze_error_vs_true_value(df, args.true_col, args.pred_col)
    
    # If SMILES column is available, perform chemical analysis
    if args.smiles_col in df.columns:
        # Analyze correlation with chemical descriptors
        update_progress(50, "Analyzing chemical descriptor correlations")
        descriptor_correlations = analyze_descriptor_correlation(df, args.true_col, args.pred_col, args.smiles_col)
        
        if "error" not in descriptor_correlations:
            results["descriptor_correlations"] = {
                "descriptors": descriptor_correlations["descriptors"],
                "error_correlation": descriptor_correlations["error_correlation"],
                "abs_error_correlation": descriptor_correlations["abs_error_correlation"]
            }
            
            # Save correlation results
            corr_df = pd.DataFrame({
                'Descriptor': descriptor_correlations["descriptors"],
                'Error_Correlation': descriptor_correlations["error_correlation"],
                'AbsError_Correlation': descriptor_correlations["abs_error_correlation"]
            })
            corr_df = corr_df.sort_values('AbsError_Correlation', ascending=False)
            corr_df.to_csv(output_dir / "descriptor_correlations.csv", index=False)
            results["descriptor_correlations_file"] = str(output_dir / "descriptor_correlations.csv")
        
        # If full analysis is requested, train error predictor
        if args.full_analysis:
            update_progress(60, "Training error predictor")
            error_predictor = train_error_predictor(df, args.true_col, args.pred_col, args.smiles_col)
            
            if "error" not in error_predictor:
                results["error_predictor"] = {
                    "train_r2": error_predictor["train_r2"],
                    "test_r2": error_predictor["test_r2"]
                }
                
                # Save model if it shows any predictive power
                if error_predictor["test_r2"] > 0.1:
                    joblib.dump(error_predictor["model"], output_dir / "error_predictor.pkl")
                    results["error_predictor_file"] = str(output_dir / "error_predictor.pkl")
    
    # Create visualizations
    update_progress(80, "Creating visualizations")
    viz_paths = create_visualizations(df, args.true_col, args.pred_col, output_dir)
    results["visualizations"] = viz_paths
    
    # Save full results
    with open(output_dir / "analysis_results.json", "w") as f:
        # Convert numpy values to Python native types
        def convert_numpy(obj):
            if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')  # Convert DataFrame to list of dictionaries
            elif isinstance(obj, pd.Series):
                return obj.to_dict()  # Convert Series to dictionary
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy(x) for x in obj]
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            else:
                return obj
                
        json.dump(convert_numpy(results), f, indent=2)
    
    # Generate HTML report
    update_progress(90, "Generating report")
    
    # Basic HTML template
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>LogD Model Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 15px; margin-bottom: 20px; }}
            .metric {{ display: inline-block; margin-right: 20px; min-width: 150px; }}
            .metric-value {{ font-size: 1.2em; font-weight: bold; color: #3498db; }}
            .metric-label {{ font-size: 0.9em; color: #7f8c8d; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .viz-container {{ display: flex; flex-wrap: wrap; }}
            .viz {{ margin: 10px; max-width: 45%; }}
            img {{ max-width: 100%; border: 1px solid #ddd; }}
        </style>
    </head>
    <body>
        <h1>LogD Model Analysis Report</h1>
        <p>Analysis of {args.pred_col} predictions compared to {args.true_col} true values.</p>
        
        <div class="card">
            <h2>Summary Statistics</h2>
            <div class="metrics-container">
                <div class="metric">
                    <div class="metric-value">{results['basic_stats']['count']}</div>
                    <div class="metric-label">Total Compounds</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{results['basic_stats']['r2']:.3f}</div>
                    <div class="metric-label">R² Score</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{results['basic_stats']['rmse']:.3f}</div>
                    <div class="metric-label">RMSE</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{results['basic_stats']['mae']:.3f}</div>
                    <div class="metric-label">MAE</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{results['basic_stats']['bias']:.3f}</div>
                    <div class="metric-label">Bias</div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>Error Distribution</h2>
            <div class="metrics-container">
                <div class="metric">
                    <div class="metric-value">{results['error_distribution']['mean']:.3f}</div>
                    <div class="metric-label">Mean Error</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{results['error_distribution']['std']:.3f}</div>
                    <div class="metric-label">Error StdDev</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{results['basic_stats']['median_error']:.3f}</div>
                    <div class="metric-label">Median Abs Error</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{results['basic_stats']['p95_error']:.3f}</div>
                    <div class="metric-label">95% Error</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{results['basic_stats']['max_error']:.3f}</div>
                    <div class="metric-label">Max Error</div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>Value Ranges</h2>
            <div class="metrics-container">
                <div class="metric">
                    <div class="metric-value">{results['basic_stats']['true_min']:.2f} - {results['basic_stats']['true_max']:.2f}</div>
                    <div class="metric-label">True Range</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{results['basic_stats']['pred_min']:.2f} - {results['basic_stats']['pred_max']:.2f}</div>
                    <div class="metric-label">Predicted Range</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{results['basic_stats']['true_mean']:.2f}</div>
                    <div class="metric-label">True Mean</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{results['basic_stats']['pred_mean']:.2f}</div>
                    <div class="metric-label">Predicted Mean</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{results['outliers_count']}</div>
                    <div class="metric-label">Outliers</div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>Visualizations</h2>
            <div class="viz-container">
                <div class="viz">
                    <h3>True vs Predicted</h3>
                    <img src="{os.path.basename(viz_paths['true_vs_predicted'])}" alt="True vs Predicted">
                </div>
                <div class="viz">
                    <h3>Error vs True Value</h3>
                    <img src="{os.path.basename(viz_paths['error_vs_true'])}" alt="Error vs True Value">
                </div>
                <div class="viz">
                    <h3>Bias Analysis</h3>
                    <img src="{os.path.basename(viz_paths['bias_analysis'])}" alt="Bias Analysis">
                </div>
                <div class="viz">
                    <h3>Error Distribution by Range</h3>
                    <img src="{os.path.basename(viz_paths['error_by_range'])}" alt="Error Distribution by Range">
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>Error by Value Range</h2>
            <table>
                <tr>
                    <th>Range</th>
                    <th>Count</th>
                    <th>Mean Abs Error</th>
                    <th>Error StdDev</th>
                    <th>Max Error</th>
                    <th>Mean Error (Bias)</th>
                </tr>
    """
    
    # Add bin rows
    bin_stats = results["error_by_value"]["bin_stats"]
    bin_edges = results["error_by_value"]["bin_edges"]
    
    for i in range(len(bin_stats)):
        row = bin_stats.iloc[i]
        range_text = f"{bin_edges[i]:.2f} - {bin_edges[i+1]:.2f}"
        html_content += f"""
                <tr>
                    <td>{range_text}</td>
                    <td>{int(row[f'{args.true_col}_count'])}</td>
                    <td>{row['abs_error_mean']:.3f}</td>
                    <td>{row['error_std']:.3f}</td>
                    <td>{row['abs_error_max']:.3f}</td>
                    <td>{row['error_mean']:.3f}</td>
                </tr>
        """
    
    html_content += """
            </table>
        </div>
    """
    
    # Add chemical descriptor section if available
    if "descriptor_correlations" in results:
        corr_df = pd.read_csv(output_dir / "descriptor_correlations.csv")
        top_corr = corr_df.head(10)
        
        html_content += """
        <div class="card">
            <h2>Chemical Descriptor Correlations</h2>
            <p>Top 10 descriptors correlated with prediction error:</p>
            <table>
                <tr>
                    <th>Descriptor</th>
                    <th>Correlation with Error</th>
                    <th>Correlation with Abs Error</th>
                </tr>
        """
        
        for i, row in top_corr.iterrows():
            html_content += f"""
                <tr>
                    <td>{row['Descriptor']}</td>
                    <td>{row['Error_Correlation']:.3f}</td>
                    <td>{row['AbsError_Correlation']:.3f}</td>
                </tr>
            """
        
        html_content += """
            </table>
        </div>
        """
    
    # Add outliers section if outliers were found
    if "outliers_file" in results:
        outliers_df = pd.read_csv(output_dir / "top_outliers.csv", index_col=0)
        top_outliers = outliers_df.head(10)
        
        html_content += """
        <div class="card">
            <h2>Top Outliers</h2>
            <p>Compounds with largest prediction errors:</p>
            <table>
                <tr>
                    <th>Index</th>
        """
        
        if args.smiles_col in top_outliers.columns:
            html_content += f"""
                    <th>{args.smiles_col}</th>
            """
        
        html_content += f"""
                    <th>{args.true_col}</th>
                    <th>{args.pred_col}</th>
                    <th>Error</th>
                    <th>Abs Error</th>
                </tr>
        """
        
        for idx, row in top_outliers.iterrows():
            html_content += f"""
                <tr>
                    <td>{idx}</td>
            """
            
            if args.smiles_col in row:
                html_content += f"""
                    <td>{row[args.smiles_col]}</td>
                """
            
            html_content += f"""
                    <td>{row[args.true_col]:.3f}</td>
                    <td>{row[args.pred_col]:.3f}</td>
                    <td>{row['error']:.3f}</td>
                    <td>{row['abs_error']:.3f}</td>
                </tr>
            """
        
        html_content += """
            </table>
        </div>
        """
    
    # Add error predictor section if trained
    if "error_predictor" in results:
        html_content += f"""
        <div class="card">
            <h2>Error Predictor Analysis</h2>
            <p>Can we predict which molecules will have high prediction errors?</p>
            <div class="metrics-container">
                <div class="metric">
                    <div class="metric-value">{results['error_predictor']['train_r2']:.3f}</div>
                    <div class="metric-label">Training R²</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{results['error_predictor']['test_r2']:.3f}</div>
                    <div class="metric-label">Testing R²</div>
                </div>
            </div>
            <p>
                {'The error predictor model shows some ability to anticipate which molecules will have higher prediction errors. This suggests systematic biases in the model.' if results['error_predictor']['test_r2'] > 0.2 else 'The error predictor has low predictive power, suggesting errors may be more random than systematic.'}
            </p>
        </div>
        """
    
    # Add improvement recommendations
    html_content += """
        <div class="card">
            <h2>Model Improvement Recommendations</h2>
            <ul>
    """
    
    # Add recommendations based on analysis results
    if "basic_stats" in results:
        if results["basic_stats"]["bias"] > 0.3:
            html_content += """
                <li><strong>Systematic Bias:</strong> The model shows significant positive bias. Consider recalibration or adjusting the training objective.</li>
            """
        elif results["basic_stats"]["bias"] < -0.3:
            html_content += """
                <li><strong>Systematic Bias:</strong> The model shows significant negative bias. Consider recalibration or adjusting the training objective.</li>
            """
        
        if results["basic_stats"]["r2"] < 0.7:
            html_content += """
                <li><strong>Overall Accuracy:</strong> The model's R² score indicates room for improvement. Consider additional descriptors or more complex modeling approaches.</li>
            """
    
    if "error_by_value" in results:
        bin_stats = results["error_by_value"]["bin_stats"]
        max_error_bin = bin_stats.loc[bin_stats['abs_error_mean'].idxmax()]
        min_error_bin = bin_stats.loc[bin_stats['abs_error_mean'].idxmin()]
        
        # Get the bin edges for the max error bin
        max_bin_idx = int(max_error_bin['bin_'])
        max_range_start = results["error_by_value"]["bin_edges"][max_bin_idx]
        max_range_end = results["error_by_value"]["bin_edges"][max_bin_idx+1]
        
        html_content += f"""
            <li><strong>Range-Specific Errors:</strong> The model performs worst in the range [{max_range_start:.2f}, {max_range_end:.2f}]. Consider adding more training data in this range or creating a specialized model for this segment.</li>
        """
    
    if "descriptor_correlations" in results:
        corr_df = pd.read_csv(output_dir / "descriptor_correlations.csv")
        top_corr = corr_df.iloc[0]
        
        if abs(top_corr['AbsError_Correlation']) > 0.3:
            html_content += f"""
                <li><strong>Descriptor Sensitivity:</strong> The model shows sensitivity to the '{top_corr['Descriptor']}' descriptor (correlation: {top_corr['AbsError_Correlation']:.3f}). Consider how this property is represented in your model.</li>
            """
    
    if "outliers_count" in results and results["outliers_count"] > 0:
        html_content += f"""
            <li><strong>Outlier Analysis:</strong> The model has {results["outliers_count"]} significant outliers. Review the outliers file to identify potential chemical patterns or data issues.</li>
        """
    
    html_content += """
                <li><strong>Data Enhancement:</strong> Consider enriching the training dataset with compounds similar to those with high errors.</li>
                <li><strong>Feature Engineering:</strong> Explore additional descriptors or transformations that might better capture the physicochemical properties relevant to LogD.</li>
                <li><strong>Model Architecture:</strong> If using a linear model or simple ensemble, consider more complex architectures like deep neural networks that can capture non-linear relationships.</li>
            </ul>
        </div>
        
        <div class="card">
            <h2>Conclusion</h2>
            <p>
                This analysis provides insights into the strengths and weaknesses of the current LogD prediction model.
                By addressing the specific patterns and biases identified here, the next model iteration can achieve improved accuracy and reliability.
            </p>
            <p>
                Generated on: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """
            </p>
        </div>
    </body>
    </html>
    """
    
    # Write HTML report
    with open(output_dir / "analysis_report.html", "w") as f:
        f.write(html_content)
    
    update_progress(100, "Analysis complete")
    
    print(f"\nAnalysis complete. Results saved to {output_dir}")
    print(f"Open {output_dir}/analysis_report.html to view the full report.")

if __name__ == "__main__":
    main()