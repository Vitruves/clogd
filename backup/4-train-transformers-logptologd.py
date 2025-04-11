#!/usr/bin/env python3

import argparse
import os
import numpy as np
import pandas as pd
import joblib
import gc
from sklearn.metrics import mean_squared_error, r2_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parse_args():
    parser = argparse.ArgumentParser(description="Extract LogD predictions using pretrained LogP transformer")
    parser.add_argument("--input", required=True, help="Input training data CSV file")
    parser.add_argument("--external-test", required=True, help="External test set for evaluation")
    parser.add_argument("--output-dir", required=True, help="Directory to save results")
    parser.add_argument("--transformer-model", required=True, help="Path to pretrained transformer model")
    parser.add_argument("--target-column", default="LOGD", help="Target column name")
    parser.add_argument("--smiles-col", default="SMILES", help="Column name containing SMILES strings")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for transformer inference")
    parser.add_argument("--use-mps", action="store_true", help="Use MPS (Metal) acceleration on Apple Silicon")
    parser.add_argument("--scale-factor", type=float, default=1.0, help="Scale factor for transformer predictions")
    parser.add_argument("--offset", type=float, default=0.0, help="Offset for transformer predictions")
    return parser.parse_args()

def load_data(input_file, external_test_file, target_column, smiles_col):
    print(f"-- Loading training data from {input_file}")
    df_train = pd.read_csv(input_file)
    
    print(f"-- Loading external test data from {external_test_file}")
    df_test = pd.read_csv(external_test_file)
    
    if target_column not in df_train.columns:
        raise ValueError(f"Target column '{target_column}' not found in training data")
    
    if target_column not in df_test.columns:
        raise ValueError(f"Target column '{target_column}' not found in test data")
    
    if smiles_col not in df_train.columns or smiles_col not in df_test.columns:
        raise ValueError(f"SMILES column '{smiles_col}' not found in data")
    
    # Extract targets
    y_train = df_train[target_column].values
    y_test = df_test[target_column].values
    
    # Prepare SMILES strings
    smiles_train = df_train[smiles_col].tolist()
    smiles_test = df_test[smiles_col].tolist()
    
    print(f"-- Training set: {len(df_train)} samples, Test set: {len(df_test)} samples")
    
    return y_train, y_test, smiles_train, smiles_test, df_train, df_test

def get_transformer_predictions(model_path, smiles_list, batch_size=16, use_mps=False):
    print(f"-- Loading transformer model from {model_path}")
    
    # Determine device
    device = "cpu"
    if use_mps and torch.backends.mps.is_available():
        print("-- Using MPS acceleration for transformer inference")
        device = torch.device("mps")
    elif torch.cuda.is_available():
        print("-- Using CUDA acceleration for transformer inference")
        device = torch.device("cuda")
    else:
        print("-- Using CPU for transformer inference")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    # Function to get predictions in batches with memory management
    def get_predictions(smiles_list):
        all_predictions = []
        
        # Process in smaller chunks to avoid memory issues
        chunk_size = 1000
        num_chunks = (len(smiles_list) + chunk_size - 1) // chunk_size
        
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min((chunk_idx + 1) * chunk_size, len(smiles_list))
            chunk = smiles_list[chunk_start:chunk_end]
            
            chunk_predictions = []
            
            # Process each batch within the chunk
            for i in tqdm(range(0, len(chunk), batch_size), 
                        desc=f"-- Processing chunk {chunk_idx+1}/{num_chunks}"):
                batch = chunk[i:i+batch_size]
                
                try:
                    inputs = tokenizer(batch, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        # Get LogP predictions
                        outputs = model(**inputs)
                        batch_preds = outputs.logits.cpu().numpy().flatten()
                        chunk_predictions.extend(batch_preds)
                        
                        # Clear GPU memory
                        if device != "cpu":
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            elif hasattr(torch.mps, 'empty_cache'):
                                torch.mps.empty_cache()
                
                except Exception as e:
                    print(f"-- Error processing batch: {str(e)}")
                    # Use zeros as fallback for failed batches
                    chunk_predictions.extend([0.0] * len(batch))
            
            # Add chunk predictions
            all_predictions.extend(chunk_predictions)
            
            # Force garbage collection
            gc.collect()
        
        return np.array(all_predictions)
    
    print(f"-- Generating predictions for {len(smiles_list)} compounds")
    predictions = get_predictions(smiles_list)
    
    # Clear memory
    del model, tokenizer
    if device != "cpu":
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
    gc.collect()
    
    return predictions

def optimize_linear_transform(logp_preds, logd_true):
    """Find optimal scaling and offset to convert LogP predictions to LogD"""
    from scipy.optimize import minimize
    
    def rmse_loss(params):
        scale, offset = params
        logd_pred = scale * logp_preds + offset
        return np.sqrt(mean_squared_error(logd_true, logd_pred))
    
    # Start with scale=1, offset=0
    initial_guess = [1.0, 0.0]
    result = minimize(rmse_loss, initial_guess, method='Nelder-Mead')
    
    optimal_scale, optimal_offset = result.x
    print(f"-- Optimized transformation: LogD = {optimal_scale:.4f} × LogP + {optimal_offset:.4f}")
    
    return optimal_scale, optimal_offset

def evaluate_predictions(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    print(f"-- RMSE: {rmse:.4f}")
    print(f"-- R²: {r2:.4f}")
    
    return rmse, r2

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    y_train, y_test, smiles_train, smiles_test, df_train, df_test = load_data(
        args.input, args.external_test, args.target_column, args.smiles_col
    )
    
    # Get transformer LogP predictions
    logp_train = get_transformer_predictions(
        args.transformer_model, smiles_train, 
        args.batch_size, args.use_mps
    )
    
    logp_test = get_transformer_predictions(
        args.transformer_model, smiles_test, 
        args.batch_size, args.use_mps
    )
    
    # Optimize linear transformation from LogP to LogD
    if args.scale_factor == 1.0 and args.offset == 0.0:
        print("-- Optimizing linear transformation from LogP to LogD")
        scale, offset = optimize_linear_transform(logp_train, y_train)
    else:
        print(f"-- Using provided transformation: LogD = {args.scale_factor:.4f} × LogP + {args.offset:.4f}")
        scale, offset = args.scale_factor, args.offset
    
    # Apply transformation to get LogD predictions
    logd_train_pred = scale * logp_train + offset
    logd_test_pred = scale * logp_test + offset
    
    # Evaluate model
    print("-- Training set performance:")
    train_rmse, train_r2 = evaluate_predictions(y_train, logd_train_pred)
    
    print("-- Test set performance:")
    test_rmse, test_r2 = evaluate_predictions(y_test, logd_test_pred)
    
    # Save predictions
    train_predictions = pd.DataFrame({
        'SMILES': smiles_train,
        'LogP_predicted': logp_train,
        'LogD_observed': y_train,
        'LogD_predicted': logd_train_pred,
        'error': y_train - logd_train_pred
    })
    
    test_predictions = pd.DataFrame({
        'SMILES': smiles_test,
        'LogP_predicted': logp_test,
        'LogD_observed': y_test,
        'LogD_predicted': logd_test_pred,
        'error': y_test - logd_test_pred
    })
    
    train_pred_path = os.path.join(args.output_dir, "train_predictions.csv")
    test_pred_path = os.path.join(args.output_dir, "test_predictions.csv")
    
    train_predictions.to_csv(train_pred_path, index=False)
    test_predictions.to_csv(test_pred_path, index=False)
    
    print(f"-- Saved training predictions to {train_pred_path}")
    print(f"-- Saved test predictions to {test_pred_path}")
    
    # Save transformation parameters
    transform_params = {
        'scale_factor': float(scale),
        'offset': float(offset),
        'train_rmse': float(train_rmse),
        'train_r2': float(train_r2),
        'test_rmse': float(test_rmse),
        'test_r2': float(test_r2)
    }
    
    params_path = os.path.join(args.output_dir, "transform_params.json")
    with open(params_path, "w") as f:
        import json
        json.dump(transform_params, f, indent=2)
    
    print(f"-- Saved transformation parameters to {params_path}")
    
    # Create correlation plot
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 5))
        
        # Training set
        plt.subplot(1, 2, 1)
        plt.scatter(y_train, logd_train_pred, alpha=0.5)
        plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], 'r--')
        plt.xlabel("Observed LogD")
        plt.ylabel("Predicted LogD")
        plt.title(f"Training Set (R² = {train_r2:.3f}, RMSE = {train_rmse:.3f})")
        
        # Test set
        plt.subplot(1, 2, 2)
        plt.scatter(y_test, logd_test_pred, alpha=0.5)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
        plt.xlabel("Observed LogD")
        plt.ylabel("Predicted LogD")
        plt.title(f"Test Set (R² = {test_r2:.3f}, RMSE = {test_rmse:.3f})")
        
        plt.tight_layout()
        
        plot_path = os.path.join(args.output_dir, "prediction_correlation.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        
        print(f"-- Saved correlation plot to {plot_path}")
    except ImportError:
        print("-- Matplotlib not available, skipping correlation plot")
    
    print("-- Done")
    print(f"-- Model performance: Test RMSE = {test_rmse:.4f}, Test R² = {test_r2:.4f}")

if __name__ == "__main__":
    main()