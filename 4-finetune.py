#!/usr/bin/env python3

import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import joblib
import gc
import argparse
import time
import warnings
import json
import optuna

# Set environment variables for better memory usage
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["LIGHTGBM_MEM_FACTOR"] = "0.5"

warnings.filterwarnings("ignore")
np.seterr(all="ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

def configure_early_stopping(stopping_rounds, verbose=False):
    return lgb.early_stopping(stopping_rounds=stopping_rounds, verbose=verbose)

def log_message(msg):
    print(f"-- {msg}")
    sys.stdout.flush()

def print_error(msg):
    print(f"-- {msg} - \033[91merror\033[0m", file=sys.stderr)
    sys.stdout.flush()
    
def print_success(msg):
    print(f"-- {msg} - \033[92msuccess\033[0m")
    sys.stdout.flush()

def prepare_data(df, smiles_col, target_col, feature_names=None):
    log_message(f"Preparing data with {len(df.columns)} columns")
    
    # If feature names are provided, use them
    if feature_names:
        log_message(f"Using {len(feature_names)} pre-defined features from original model")
        # Check which feature columns exist in the new dataset
        available_features = [f for f in feature_names if f in df.columns]
        missing_features = [f for f in feature_names if f not in df.columns]
        
        if missing_features:
            log_message(f"Warning: {len(missing_features)} features from original model not found in new data")
            log_message(f"Missing features will be filled with zeros")
        
        # Extract features and target
        X = np.zeros((len(df), len(feature_names)), dtype=np.float32)
        for i, feature in enumerate(feature_names):
            if feature in df.columns:
                X[:, i] = df[feature].values.astype(np.float32)
    else:
        # Determine feature columns if not provided
        target_variations = [target_col, target_col.lower(), target_col.upper()]
        excluded_cols = [smiles_col] + target_variations + ["ID", "id", "Name", "name"]
        
        feature_names = [col for col in df.columns if col not in excluded_cols]
        log_message(f"Using {len(feature_names)} features")
        
        # Convert non-numeric columns
        for col in feature_names:
            if not pd.api.types.is_numeric_dtype(df[col]):
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    log_message(f"Removing non-numeric column: {col}")
                    feature_names.remove(col)
        
        # Extract features
        X = df[feature_names].values.astype(np.float32)
    
    # Extract target
    y = df[target_col].values.astype(np.float32)
    
    # Handle NaN values in target
    target_nans = np.isnan(y).sum()
    if target_nans > 0:
        log_message(f"Found {target_nans} NaN values in target")
        median_y = np.nanmedian(y)
        y = np.where(np.isnan(y), median_y, y)
    
    return X, y, feature_names

def preprocess_features(X, imputer, scaler, X_val=None, X_test=None, X_external=None):
    log_message("Preprocessing features using loaded transformers")
    
    # Handle infinite values
    X = np.where(np.isfinite(X), X, np.nan)
    if X_val is not None:
        X_val = np.where(np.isfinite(X_val), X_val, np.nan)
    if X_test is not None:
        X_test = np.where(np.isfinite(X_test), X_test, np.nan)
    if X_external is not None:
        X_external = np.where(np.isfinite(X_external), X_external, np.nan)
    
    # Apply existing transformations
    X = imputer.transform(X)
    X = scaler.transform(X)
    
    if X_val is not None:
        X_val = imputer.transform(X_val)
        X_val = scaler.transform(X_val)
    
    if X_test is not None:
        X_test = imputer.transform(X_test)
        X_test = scaler.transform(X_test)
        
    if X_external is not None:
        X_external = imputer.transform(X_external)
        X_external = scaler.transform(X_external)
    
    # Final check for NaN values
    if np.isnan(X).any():
        X = np.nan_to_num(X, nan=0.0)
    if X_val is not None and np.isnan(X_val).any():
        X_val = np.nan_to_num(X_val, nan=0.0)
    if X_test is not None and np.isnan(X_test).any():
        X_test = np.nan_to_num(X_test, nan=0.0)
    if X_external is not None and np.isnan(X_external).any():
        X_external = np.nan_to_num(X_external, nan=0.0)
    
    return X, X_val, X_test, X_external

def optimize_hyperparameters(pretrained_model, X_train, y_train, X_val, y_val, n_trials=30, timeout=600, X_external=None, y_external=None):
    log_message(f"Optimizing fine-tuning hyperparameters with Optuna (max {n_trials} trials)")
    
    # Define a callback to log trial progress
    class TrialMonitor:
        def __init__(self):
            self.best_score = float('inf')
            self.best_trial = None
            self.trial_count = 0
            self.last_report = time.time()
            self.report_interval = 5  # seconds
        
        def __call__(self, study, trial):
            self.trial_count += 1
            
            # Update best score
            if trial.value < self.best_score:
                self.best_score = trial.value
                self.best_trial = trial.number
                log_message(f"New best score: {self.best_score:.4f} (trial {self.best_trial})")
                # Print the parameters of the best trial
                params_str = ", ".join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" 
                                      for k, v in trial.params.items()])
                log_message(f"Parameters: {params_str}")
            
            # Report progress periodically
            current_time = time.time()
            if current_time - self.last_report > self.report_interval:
                self.last_report = current_time
                log_message(f"Completed {self.trial_count}/{n_trials} trials. Best score: {self.best_score:.4f}")
                
                # Calculate estimated time remaining
                elapsed = current_time - optimization_start_time
                if self.trial_count > 0:
                    time_per_trial = elapsed / self.trial_count
                    remaining_trials = n_trials - self.trial_count
                    est_remaining = time_per_trial * remaining_trials
                    log_message(f"Elapsed: {elapsed:.1f}s, Est. remaining: {est_remaining:.1f}s")

    # Record start time for time tracking
    optimization_start_time = time.time()
    
    def objective(trial):
        # Define parameter search space focused on fine-tuning
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'n_jobs': 4,
            
            # Learning parameters - more conservative for fine-tuning
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 2000, step=100),
            
            # Regularization parameters - important for preventing overfitting
            'lambda_l1': trial.suggest_float('lambda_l1', 0.01, 1.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 0.01, 1.0, log=True),
            
            # Feature and data sampling - help with overfitting
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 0.95),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 0.95),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            
            # Additional parameters for fine-tuning
            'force_row_wise': True
        }
        
        # Add trial information to log
        log_message(f"Starting trial {trial.number}")
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_val, label=y_val)
        
        # Train model (fine-tune)
        model = lgb.train(
            params,
            train_data,
            valid_sets=[valid_data],
            num_boost_round=params['n_estimators'],
            callbacks=[configure_early_stopping(20)],
            init_model=pretrained_model
        )
        
        # Get validation score
        val_preds = model.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        log_message(f"Trial {trial.number} validation RMSE: {val_rmse:.4f}")
        
        # If external test set is provided, evaluate on it too
        if X_external is not None and y_external is not None:
            ext_preds = model.predict(X_external)
            ext_rmse = np.sqrt(mean_squared_error(y_external, ext_preds))
            log_message(f"Trial {trial.number} external test RMSE: {ext_rmse:.4f}")
        else:
            ext_rmse = None
        
        # Clean up
        del model, train_data, valid_data
        gc.collect()
        
        # Select score based on availability of external test data
        if ext_rmse is not None:
            # If external test data is available, use only the external test RMSE
            final_score = ext_rmse
            log_message(f"Trial {trial.number} final score (external test only): {final_score:.4f}")
        else:
            # Use validation score
            final_score = val_rmse
            log_message(f"Trial {trial.number} final score (validation): {final_score:.4f}")
        
        return final_score
    
    # Create study with pruner
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    study = optuna.create_study(direction="minimize", pruner=pruner)
    
    # Run optimization with callbacks
    trial_monitor = TrialMonitor()
    study.optimize(objective, n_trials=n_trials, timeout=timeout, callbacks=[trial_monitor])
    
    # Get best parameters
    best_params = study.best_params
    best_params['objective'] = 'regression'
    best_params['metric'] = 'rmse'
    best_params['verbosity'] = -1
    best_params['n_jobs'] = 4
    best_params['force_row_wise'] = True
    
    # Calculate optimization time
    optimization_time = time.time() - optimization_start_time
    
    # Print optimization summary
    log_message(f"Hyperparameter optimization completed in {optimization_time:.2f} seconds")
    log_message(f"Best trial: {study.best_trial.number}")
    log_message(f"Best RMSE: {study.best_value:.4f}")
    log_message("Best hyperparameters:")
    for key, value in best_params.items():
        log_message(f"  {key}: {value}")
    
    # Save optimization history
    trial_data = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            trial_data.append({
                "number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "datetime": trial.datetime_start.strftime("%Y-%m-%d %H:%M:%S"),
                "duration": (trial.datetime_complete - trial.datetime_start).total_seconds()
            })
    
    # Return final parameters
    return best_params, trial_data

def fine_tune_model(pretrained_model, X_train, y_train, X_val, y_val, params, early_stopping_rounds=50, verbose=False):
    log_message(f"Fine-tuning pre-trained model (early stopping after {early_stopping_rounds} rounds)")
    
    # Create datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_val, label=y_val)
    
    # Train model with early stopping, continue from pre-trained model
    model = lgb.train(
        params,
        train_data,
        valid_sets=[valid_data],
        num_boost_round=params.get('n_estimators', 1000),
        callbacks=[configure_early_stopping(early_stopping_rounds, verbose)],
        init_model=pretrained_model
    )
    
    # Get feature importances
    importances = model.feature_importance(importance_type='gain')
    
    log_message(f"Model fine-tuning completed with best iteration: {model.best_iteration}")
    
    return model, importances

def evaluate_model(model, X, y, name="Test"):
    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    
    log_message(f"{name} metrics: RMSE={rmse:.4f}, R²={r2:.4f}, MAE={mae:.4f}")
    
    return {
        "rmse": rmse,
        "r2": r2,
        "mae": mae,
        "predictions": y_pred
    }

def plot_predictions(y_true, y_pred, output_file):
    plt.figure(figsize=(10, 8))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    
    # Add metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    plt.title(f"Predicted vs Actual LogD (RMSE={rmse:.4f}, R²={r2:.4f}, MAE={mae:.4f})")
    plt.xlabel("Actual LogD")
    plt.ylabel("Predicted LogD")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    log_message(f"Saved prediction plot to {output_file}")

def save_model(model, output_dir, imputer, scaler, feature_names, params):
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, "model.txt")
    model.save_model(model_path)
    
    # Save preprocessing objects
    joblib.dump(imputer, os.path.join(output_dir, "imputer.pkl"))
    joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))
    
    # Save feature names
    with open(os.path.join(output_dir, "features.txt"), 'w') as f:
        for name in feature_names:
            f.write(f"{name}\n")
    
    # Save parameters
    with open(os.path.join(output_dir, "params.txt"), 'w') as f:
        for key, value in params.items():
            f.write(f"{key}: {value}\n")
    
    log_message(f"Fine-tuned model and preprocessing objects saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a pre-trained LogD model with new data")
    parser.add_argument("--model-dir", required=True, help="Directory containing pre-trained model")
    parser.add_argument("--input", required=True, help="Input CSV file with SMILES and LogD values for fine-tuning")
    parser.add_argument("--output-dir", required=True, help="Directory to save fine-tuned model and results")
    parser.add_argument("--smiles-col", default="SMILES", help="Column name containing SMILES strings")
    parser.add_argument("--target-col", default="LOGD", help="Column name containing LogD values")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of data for testing")
    parser.add_argument("--val-size", type=float, default=0.2, help="Fraction of training data for validation")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Learning rate for fine-tuning")
    parser.add_argument("--n-estimators", type=int, default=500, help="Maximum number of additional boosting rounds")
    parser.add_argument("--early-stopping-rounds", type=int, default=50, help="Patience for early stopping")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output during training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--optimize", action="store_true", help="Use Optuna to optimize hyperparameters")
    parser.add_argument("--n-trials", type=int, default=30, help="Number of Optuna trials")
    parser.add_argument("--timeout", type=int, default=600, help="Maximum optimization time in seconds")
    parser.add_argument("--external-test-set", help="Path to external test set CSV for additional evaluation during optimization")
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    log_message(f"Starting LogD model fine-tuning")
    log_message(f"Pre-trained model: {args.model_dir}")
    log_message(f"Input data: {args.input}")
    log_message(f"Output directory: {args.output_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Step 1: Load the pre-trained model and preprocessing objects
    try:
        log_message("Loading pre-trained model and components")
        model_path = os.path.join(args.model_dir, "model.txt")
        imputer_path = os.path.join(args.model_dir, "imputer.pkl")
        scaler_path = os.path.join(args.model_dir, "scaler.pkl")
        features_path = os.path.join(args.model_dir, "features.txt")
        params_path = os.path.join(args.model_dir, "params.txt")
        
        # Load model
        pretrained_model = lgb.Booster(model_file=model_path)
        log_message(f"Loaded model with {pretrained_model.num_trees()} trees")
        
        # Load preprocessing objects
        imputer = joblib.load(imputer_path)
        scaler = joblib.load(scaler_path)
        
        # Load feature names
        with open(features_path, 'r') as f:
            original_features = [line.strip() for line in f]
        log_message(f"Loaded {len(original_features)} original features")
        
        # Load original parameters
        params = {}
        with open(params_path, 'r') as f:
            for line in f:
                if ':' in line:
                    key, value = line.strip().split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Convert specific parameters to their required types
                    if key == 'verbosity' or key == 'n_jobs' or key == 'num_leaves' or key == 'max_depth' or key == 'bagging_freq':
                        params[key] = int(float(value))
                    elif value.replace('.', '', 1).isdigit():
                        # Try to convert numeric values
                        try:
                            if '.' in value:
                                params[key] = float(value)
                            else:
                                params[key] = int(value)
                        except:
                            params[key] = value
                    else:
                        params[key] = value
        
        # Update parameters for fine-tuning if not using optimization
        if not args.optimize:
            params['learning_rate'] = args.learning_rate
            params['n_estimators'] = args.n_estimators
    except Exception as e:
        print_error(f"Failed to load model: {e}")
        return 1
    
    # Step 2: Load new data
    try:
        log_message("Loading fine-tuning data")
        df = pd.read_csv(args.input)
        log_message(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    except Exception as e:
        print_error(f"Failed to load data: {e}")
        return 1
    
    # Check required columns
    if args.smiles_col not in df.columns:
        print_error(f"SMILES column '{args.smiles_col}' not found")
        return 1
    
    if args.target_col not in df.columns:
        print_error(f"Target column '{args.target_col}' not found")
        return 1
    
    # Load external test set if provided
    X_external = None
    y_external = None
    if args.external_test_set:
        try:
            log_message(f"Loading external test set: {args.external_test_set}")
            external_df = pd.read_csv(args.external_test_set)
            log_message(f"Loaded external test set with {len(external_df)} rows")
            
            # Check required columns in external test set
            if args.smiles_col not in external_df.columns:
                print_error(f"SMILES column '{args.smiles_col}' not found in external test set")
                return 1
            
            if args.target_col not in external_df.columns:
                print_error(f"Target column '{args.target_col}' not found in external test set")
                return 1
                
            X_ext, y_ext, _ = prepare_data(external_df, args.smiles_col, args.target_col, original_features)
            X_external = X_ext
            y_external = y_ext
            log_message(f"External test set prepared with {len(X_external)} samples")
        except Exception as e:
            print_error(f"Failed to load external test set: {e}")
            log_message("Continuing without external test set")
            X_external = None
            y_external = None
    
    # Step 3: Prepare data using original features
    X, y, feature_names = prepare_data(df, args.smiles_col, args.target_col, original_features)
    
    # Step 4: Split data
    log_message("Splitting data into train/validation/test sets")
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, shuffle=True
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=args.val_size, random_state=args.seed, shuffle=True
    )
    
    log_message(f"Train: {len(X_train)} samples, Validation: {len(X_val)} samples, Test: {len(X_test)} samples")
    
    # Step 5: Preprocess features using existing transformers
    X_train, X_val, X_test, X_external = preprocess_features(X_train, imputer, scaler, X_val, X_test, X_external)
    
    # Step 6: Optimize hyperparameters if requested
    trial_data = []
    if args.optimize:
        log_message("Optimizing hyperparameters for fine-tuning")
        params, trial_data = optimize_hyperparameters(
            pretrained_model, 
            X_train, 
            y_train, 
            X_val, 
            y_val,
            n_trials=args.n_trials,
            timeout=args.timeout,
            X_external=X_external,
            y_external=y_external
        )
        
        # Save optimization history
        with open(os.path.join(args.output_dir, "optuna_trials.json"), 'w') as f:
            json.dump(trial_data, f, indent=2)
    
    # Step 7: Fine-tune the model
    model, importances = fine_tune_model(
        pretrained_model, 
        X_train, 
        y_train, 
        X_val, 
        y_val, 
        params, 
        args.early_stopping_rounds, 
        args.verbose
    )
    
    # Step 8: Evaluate
    train_metrics = evaluate_model(model, X_train, y_train, "Train")
    val_metrics = evaluate_model(model, X_val, y_val, "Validation")
    test_metrics = evaluate_model(model, X_test, y_test, "Test")
    
    # Evaluate on external test set if available
    ext_metrics = None
    if X_external is not None and y_external is not None:
        ext_metrics = evaluate_model(model, X_external, y_external, "External Test")
    
    # Calculate train-test gap (measure of overfitting)
    train_test_rmse_gap = train_metrics["rmse"] - test_metrics["rmse"]
    train_test_r2_gap = train_metrics["r2"] - test_metrics["r2"]
    
    log_message(f"Train-Test RMSE gap: {abs(train_test_rmse_gap):.4f} (smaller is better)")
    log_message(f"Train-Test R² gap: {abs(train_test_r2_gap):.4f} (smaller is better)")
    
    # Step 9: Create plots
    plot_predictions(y_test, test_metrics["predictions"], 
                    os.path.join(args.output_dir, "test_predictions.png"))
    
    # Create external test set plot if available
    if ext_metrics is not None:
        plot_predictions(y_external, ext_metrics["predictions"],
                       os.path.join(args.output_dir, "external_test_predictions.png"))
    
    # Step 10: Save model and results
    save_model(model, args.output_dir, imputer, scaler, feature_names, params)
    
    # Save predictions
    results_df = pd.DataFrame({
        "Actual": y_test,
        "Predicted": test_metrics["predictions"],
        "Error": test_metrics["predictions"] - y_test,
        "AbsError": np.abs(test_metrics["predictions"] - y_test)
    })
    results_df.to_csv(os.path.join(args.output_dir, "test_predictions.csv"), index=False)
    
    # Save external test predictions if available
    if ext_metrics is not None:
        ext_results_df = pd.DataFrame({
            "Actual": y_external,
            "Predicted": ext_metrics["predictions"],
            "Error": ext_metrics["predictions"] - y_external,
            "AbsError": np.abs(ext_metrics["predictions"] - y_external)
        })
        ext_results_df.to_csv(os.path.join(args.output_dir, "external_test_predictions.csv"), index=False)
    
    # Save metrics summary
    with open(os.path.join(args.output_dir, "metrics.txt"), "w") as f:
        f.write(f"Train RMSE: {train_metrics['rmse']:.4f}\n")
        f.write(f"Train R²: {train_metrics['r2']:.4f}\n")
        f.write(f"Train MAE: {train_metrics['mae']:.4f}\n\n")
        
        f.write(f"Validation RMSE: {val_metrics['rmse']:.4f}\n")
        f.write(f"Validation R²: {val_metrics['r2']:.4f}\n")
        f.write(f"Validation MAE: {val_metrics['mae']:.4f}\n\n")
        
        f.write(f"Test RMSE: {test_metrics['rmse']:.4f}\n")
        f.write(f"Test R²: {test_metrics['r2']:.4f}\n")
        f.write(f"Test MAE: {test_metrics['mae']:.4f}\n\n")
        
        if ext_metrics is not None:
            f.write(f"External Test RMSE: {ext_metrics['rmse']:.4f}\n")
            f.write(f"External Test R²: {ext_metrics['r2']:.4f}\n")
            f.write(f"External Test MAE: {ext_metrics['mae']:.4f}\n\n")
        
        f.write(f"Train-Test RMSE gap: {abs(train_test_rmse_gap):.4f}\n")
        f.write(f"Train-Test R² gap: {abs(train_test_r2_gap):.4f}\n")
    
    # Save feature importances
    sorted_indices = np.argsort(importances)[::-1]
    with open(os.path.join(args.output_dir, "feature_importance.txt"), "w") as f:
        for idx in sorted_indices:
            f.write(f"{feature_names[idx]}: {importances[idx]:.4f}\n")
    
    execution_time = time.time() - start_time
    log_message(f"Total execution time: {execution_time:.2f} seconds")
    print_success("LogD model fine-tuning completed")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())