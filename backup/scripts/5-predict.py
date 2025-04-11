#!/usr/bin/env python3

import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
import argparse
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def log_message(msg):
    print(f"-- {msg}")
    sys.stdout.flush()

def print_error(msg):
    print(f"-- {msg} - \033[91merror\033[0m", file=sys.stderr)
    sys.stdout.flush()
    
def print_success(msg):
    print(f"-- {msg} - \033[92msuccess\033[0m")
    sys.stdout.flush()

def load_model(model_dir):
    log_message(f"Loading model from {model_dir}")
    
    model_path = os.path.join(model_dir, "model.txt")
    if not os.path.exists(model_path):
        print_error(f"Model file not found at {model_path}")
        return None
    
    model = lgb.Booster(model_file=model_path)
    
    imputer_path = os.path.join(model_dir, "imputer.pkl")
    if not os.path.exists(imputer_path):
        print_error(f"Imputer not found at {imputer_path}")
        return None
    
    imputer = joblib.load(imputer_path)
    
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    if not os.path.exists(scaler_path):
        print_error(f"Scaler not found at {scaler_path}")
        return None
    
    scaler = joblib.load(scaler_path)
    
    feature_path = os.path.join(model_dir, "features.txt")
    if not os.path.exists(feature_path):
        print_error(f"Features list not found at {feature_path}")
        return None
    
    with open(feature_path, 'r') as f:
        feature_names = [line.strip() for line in f]
    
    return {
        "model": model,
        "imputer": imputer,
        "scaler": scaler,
        "feature_names": feature_names
    }

def prepare_features(df, feature_names, smiles_col):
    log_message(f"Preparing features using {len(feature_names)} known features")
    
    missing_features = [f for f in feature_names if f not in df.columns]
    if missing_features:
        log_message(f"Missing {len(missing_features)} features in input data")
        
        # Create a new DataFrame with missing features (avoids fragmentation)
        missing_df = pd.DataFrame({feature: np.zeros(len(df)) * np.nan for feature in missing_features})
        df = pd.concat([df, missing_df], axis=1)
    
    extra_features = [f for f in df.columns if f not in feature_names and f != smiles_col]
    if extra_features:
        log_message(f"Input data contains {len(extra_features)} extra features that will be ignored")
    
    # Check for features with all NaN values
    all_nan_features = [f for f in feature_names if df[f].isna().all()]
    if all_nan_features:
        log_message(f"Warning: {len(all_nan_features)} features have all NaN values")
        # Fill these features with 0 to avoid imputer warnings
        for f in all_nan_features:
            df[f] = 0.0
    
    X = df[feature_names].values.astype(np.float32)
    
    return X

def preprocess_features(X, imputer, scaler):
    log_message("Preprocessing features")
    
    # Handle non-finite values
    X = np.where(np.isfinite(X), X, np.nan)
    
    # Simply fill all NaN values with 0
    X_filled = np.nan_to_num(X, nan=0.0)
    
    # Handle the case where all values in a column are NaN
    # This avoids the imputer warning and error
    try:
        X_imputed = imputer.transform(X_filled)
        X_scaled = scaler.transform(X_imputed)
        return X_scaled
    except Exception as e:
        log_message(f"Warning: Error in preprocessing: {e}")
        log_message("Falling back to simpler preprocessing method")
        
        # Alternative approach: create a new imputer and scaler
        # This is a workaround for columns with all zeros
        simple_imputer = SimpleImputer(strategy='constant', fill_value=0)
        simple_scaler = StandardScaler()
        
        # Try to apply the preprocessing
        try:
            X_imputed = simple_imputer.fit_transform(X_filled)
            X_scaled = simple_scaler.fit_transform(X_imputed)
            return X_scaled
        except Exception as e2:
            log_message(f"Error in fallback preprocessing: {e2}")
            log_message("Using filled values without scaling")
            return X_filled

def predict(model, X):
    log_message(f"Making predictions for {len(X)} samples")
    return model.predict(X)

def main():
    parser = argparse.ArgumentParser(description="Predict using a trained LightGBM LogD model")
    parser.add_argument("--model-dir", required=True, help="Directory containing trained model files")
    parser.add_argument("--input", required=True, help="Input CSV file with SMILES and features")
    parser.add_argument("--output", required=True, help="Output CSV file for predictions")
    parser.add_argument("--smiles-col", default="SMILES", help="Column name containing SMILES strings")
    parser.add_argument("--output-col", default="PREDICTED_LOGD", help="Column name for predictions")
    parser.add_argument("--keep-original-cols", action="store_true", help="Keep all original columns in output")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_dir):
        print_error(f"Model directory {args.model_dir} not found")
        return 1
    
    if not os.path.exists(args.input):
        print_error(f"Input file {args.input} not found")
        return 1
    
    # Ensure output has correct extension
    output_path = args.output
    if not output_path.lower().endswith('.csv'):
        log_message(f"Warning: Output file '{output_path}' doesn't have .csv extension")
        if '.' not in os.path.basename(output_path):
            output_path = f"{output_path}.csv"
            log_message(f"Adding .csv extension: {output_path}")
    
    loaded = load_model(args.model_dir)
    if loaded is None:
        return 1
    
    model = loaded["model"]
    imputer = loaded["imputer"]
    scaler = loaded["scaler"]
    feature_names = loaded["feature_names"]
    
    log_message(f"Loading input data from {args.input}")
    try:
        df = pd.read_csv(args.input)
        log_message(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    except Exception as e:
        print_error(f"Failed to load input data: {e}")
        return 1
    
    if args.smiles_col not in df.columns:
        print_error(f"SMILES column '{args.smiles_col}' not found in input data")
        return 1
    
    X = prepare_features(df, feature_names, args.smiles_col)
    X = preprocess_features(X, imputer, scaler)
    predictions = predict(model, X)
    
    if args.keep_original_cols:
        df[args.output_col] = predictions
        output_df = df
        log_message("Keeping all original columns in output")
    else:
        output_df = pd.DataFrame({
            args.smiles_col: df[args.smiles_col],
            args.output_col: predictions
        })
        log_message("Output will contain only SMILES and prediction columns")
    
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    log_message(f"Saving predictions to {output_path}")
    output_df.to_csv(output_path, index=False)
    
    log_message(f"Statistics: min={predictions.min():.4f}, max={predictions.max():.4f}, mean={predictions.mean():.4f}, std={predictions.std():.4f}")
    print_success(f"Predictions complete for {len(df)} compounds")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 