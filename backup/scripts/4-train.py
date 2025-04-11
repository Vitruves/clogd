#!/usr/bin/env python3

import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import joblib
import gc
import argparse
import time
import warnings
import optuna
import json

# Set environment variables for better memory usage
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["LIGHTGBM_MEM_FACTOR"] = "0.5"

# Ignore warnings
warnings.filterwarnings("ignore")
np.seterr(all="ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

def configure_early_stopping(stopping_rounds, verbose=False):
    """Configure early stopping callback with specified parameters"""
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

def prepare_data(df, smiles_col, target_col, max_features=None):
    log_message(f"Preparing data with {len(df.columns)} columns")
    
    # Determine feature columns
    target_variations = [target_col, target_col.lower(), target_col.upper()]
    excluded_cols = [smiles_col] + target_variations + ["ID", "id", "Name", "name"]
    
    feature_cols = [col for col in df.columns if col not in excluded_cols]
    log_message(f"Using {len(feature_cols)} features")
    
    # Convert non-numeric columns
    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                log_message(f"Removing non-numeric column: {col}")
                feature_cols.remove(col)
    
    # Extract features and target
    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32)
    
    # Handle NaN values in target
    target_nans = np.isnan(y).sum()
    if target_nans > 0:
        log_message(f"Found {target_nans} NaN values in target")
        median_y = np.nanmedian(y)
        y = np.where(np.isnan(y), median_y, y)
    
    # Optionally reduce features based on variance
    if max_features is not None and len(feature_cols) > max_features:
        log_message(f"Selecting top {max_features} features by variance")
        variances = np.nanvar(X, axis=0)
        top_indices = np.argsort(variances)[-max_features:]
        X = X[:, top_indices]
        feature_cols = [feature_cols[i] for i in top_indices]
    
    return X, y, feature_cols

def preprocess_features(X, X_val=None, X_test=None):
    log_message("Preprocessing features")
    
    # Handle infinite values
    X = np.where(np.isfinite(X), X, np.nan)
    if X_val is not None:
        X_val = np.where(np.isfinite(X_val), X_val, np.nan)
    if X_test is not None:
        X_test = np.where(np.isfinite(X_test), X_test, np.nan)
    
    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)
    
    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Apply transformations to validation/test sets
    if X_val is not None:
        X_val = imputer.transform(X_val)
        X_val = scaler.transform(X_val)
    
    if X_test is not None:
        X_test = imputer.transform(X_test)
        X_test = scaler.transform(X_test)
    
    # Final check for NaN values
    if np.isnan(X).any():
        X = np.nan_to_num(X, nan=0.0)
    if X_val is not None and np.isnan(X_val).any():
        X_val = np.nan_to_num(X_val, nan=0.0)
    if X_test is not None and np.isnan(X_test).any():
        X_test = np.nan_to_num(X_test, nan=0.0)
    
    return {
        "X": X,
        "X_val": X_val if X_val is not None else None,
        "X_test": X_test if X_test is not None else None,
        "imputer": imputer,
        "scaler": scaler
    }

def optimize_hyperparameters(X_train, y_train, X_val, y_val, n_trials=50, timeout=600):
    log_message(f"Optimizing hyperparameters with Optuna (max {n_trials} trials)")
    
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
        # Define parameter search space focused on preventing overfitting
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'n_jobs': 4,
            
            # Learning parameters
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'n_estimators': 1000,  # Will use early stopping
            
            # Tree structure parameters - conservative to prevent overfitting
            'num_leaves': trial.suggest_int('num_leaves', 10, 50),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 50),
            
            # Regularization parameters - important for preventing overfitting
            'lambda_l1': trial.suggest_float('lambda_l1', 0.01, 1.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 0.01, 1.0, log=True),
            'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.01, 0.3),
            
            # Feature and data sampling - help with overfitting
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 0.9),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 0.9),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            
            # Additional parameters for memory efficiency
            'force_row_wise': True
        }
        
        # Add trial information to log
        log_message(f"Starting trial {trial.number}")
        
        # Use cross-validation for more robust evaluation
        k_fold = 3
        kf = KFold(n_splits=k_fold, shuffle=True, random_state=42)
        
        # Store cross-validation scores
        scores = []
        fold_scores = []
        
        # Track best_iteration to save later
        best_iteration = 0
        
        # Perform cross-validation
        for fold, (train_idx, valid_idx) in enumerate(kf.split(X_train)):
            X_fold_train, X_fold_valid = X_train[train_idx], X_train[valid_idx]
            y_fold_train, y_fold_valid = y_train[train_idx], y_train[valid_idx]
            
            # Create datasets
            train_data = lgb.Dataset(X_fold_train, label=y_fold_train)
            valid_data = lgb.Dataset(X_fold_valid, label=y_fold_valid)
            
            # Train model with early stopping
            gbm = lgb.train(
                params,
                train_data,
                valid_sets=[valid_data],
                num_boost_round=500,
                callbacks=[configure_early_stopping(20)]
            )
            
            # Store the best iteration
            best_iteration = max(best_iteration, gbm.best_iteration)
            
            # Predict and evaluate
            y_pred = gbm.predict(X_fold_valid)
            rmse = np.sqrt(mean_squared_error(y_fold_valid, y_pred))
            scores.append(rmse)
            fold_scores.append(rmse)
            
            # Clean up to save memory
            del gbm, train_data, valid_data
            gc.collect()
        
        # Return mean cross-validation score
        cv_score = np.mean(scores)
        log_message(f"Trial {trial.number} CV scores: " + 
                   " ".join([f"{score:.4f}" for score in fold_scores]) + 
                   f" (mean: {cv_score:.4f})")
        
        # Also evaluate on validation set
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_val, label=y_val)
        
        # For validation evaluation, create a separate model
        val_gbm = lgb.train(
            params,
            train_data,
            valid_sets=[valid_data],
            num_boost_round=100,  # Reduced for speed during optimization
            callbacks=[configure_early_stopping(10)]
        )
        
        # Update best iteration if validation model found better value
        best_iteration = max(best_iteration, val_gbm.best_iteration)
        
        val_preds = val_gbm.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        log_message(f"Trial {trial.number} validation RMSE: {val_rmse:.4f}")
        
        # Clean up
        del val_gbm, train_data, valid_data
        gc.collect()
        
        # Use a weighted combination of CV score and validation score
        combined_score = 0.7 * cv_score + 0.3 * val_rmse
        log_message(f"Trial {trial.number} combined score: {combined_score:.4f}")
        
        # Store best iteration in trial user attributes
        if hasattr(trial, 'user_attrs'):
            trial.set_user_attr('best_iteration', best_iteration)
        
        return combined_score
    
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
    best_params['n_estimators'] = 1000  # Will use early stopping
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

def train_model(X_train, y_train, X_val, y_val, params, early_stopping_rounds=50, verbose=False):
    log_message(f"Training final model with optimized parameters (early stopping after {early_stopping_rounds} rounds)")
    
    # Create datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_val, label=y_val)
    
    # Train model with early stopping
    model = lgb.train(
        params,
        train_data,
        valid_sets=[valid_data],
        num_boost_round=params.get('n_estimators', 1000),
        callbacks=[configure_early_stopping(early_stopping_rounds, verbose)]
    )
    
    # Get feature importances
    importances = model.feature_importance(importance_type='gain')
    
    log_message(f"Model training completed with best iteration: {model.best_iteration}")
    
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
    
    log_message(f"Model and preprocessing objects saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Train a LightGBM model for LogD prediction with anti-overfitting strategies")
    parser.add_argument("--input", required=True, help="Input CSV file with SMILES and LogD values")
    parser.add_argument("--output-dir", required=True, help="Directory to save model and results")
    parser.add_argument("--smiles-col", default="SMILES", help="Column name containing SMILES strings")
    parser.add_argument("--target-col", default="LOGD", help="Column name containing LogD values")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of data for testing")
    parser.add_argument("--val-size", type=float, default=0.2, help="Fraction of training data for validation")
    parser.add_argument("--max-features", type=int, default=500, help="Maximum number of features to use")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--timeout", type=int, default=600, help="Maximum optimization time in seconds")
    parser.add_argument("--no-optimize", action="store_true", help="Skip hyperparameter optimization")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--manual-params", action="store_true", 
                    help="Use manually specified parameters instead of Optuna optimization")
    parser.add_argument("--learning-rate", type=float, default=0.05,
                    help="Learning rate for manual tuning")
    parser.add_argument("--num-leaves", type=int, default=31,
                    help="Number of leaves for manual tuning")
    parser.add_argument("--max-depth", type=int, default=6,
                    help="Maximum tree depth for manual tuning")
    parser.add_argument("--min-data-in-leaf", type=int, default=20,
                    help="Minimum data in leaf for manual tuning")
    parser.add_argument("--lambda-l1", type=float, default=0.1,
                    help="L1 regularization for manual tuning")
    parser.add_argument("--lambda-l2", type=float, default=0.1,
                    help="L2 regularization for manual tuning")
    parser.add_argument("--feature-fraction", type=float, default=0.8,
                    help="Feature fraction for manual tuning")
    parser.add_argument("--bagging-fraction", type=float, default=0.8,
                    help="Bagging fraction for manual tuning")
    parser.add_argument("--bagging-freq", type=int, default=5,
                    help="Bagging frequency for manual tuning")
    parser.add_argument("--min-gain-to-split", type=float, default=0.1,
                    help="Minimum gain to split for manual tuning")
    parser.add_argument("--n-estimators", type=int, default=1000,
                    help="Number of estimators/boosting rounds")
    parser.add_argument("--early-stopping-rounds", type=int, default=50,
                    help="Number of rounds with no improvement before stopping")
    parser.add_argument("--verbose", action="store_true",
                    help="Enable verbose output including early stopping logs")
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    log_message(f"Starting optimized LogD model training with anti-overfitting strategies")
    log_message(f"Input file: {args.input}")
    log_message(f"Output directory: {args.output_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Step 1: Load data
    try:
        log_message("Loading data")
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
    
    # Step 2: Prepare data with feature selection
    X, y, feature_names = prepare_data(df, args.smiles_col, args.target_col, args.max_features)
    
    # Step 3: Split data
    log_message("Splitting data into train/validation/test sets")
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, shuffle=True
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=args.val_size, random_state=args.seed, shuffle=True
    )
    
    log_message(f"Train: {len(X_train)} samples, Validation: {len(X_val)} samples, Test: {len(X_test)} samples")
    
    # Step 4: Preprocess features
    processed = preprocess_features(X_train, X_val, X_test)
    X_train = processed["X"]
    X_val = processed["X_val"]
    X_test = processed["X_test"]
    imputer = processed["imputer"]
    scaler = processed["scaler"]
    
    # Step 5: Hyperparameter selection
    if args.no_optimize:
        log_message("Skipping hyperparameter optimization as requested")
        # Use conservative parameters to prevent overfitting
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': 5,
            'min_data_in_leaf': 20,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_gain_to_split': 0.1,
            'verbosity': -1,
            'n_estimators': 1000,
            'force_row_wise': True,
            'n_jobs': 4
        }
        trial_data = []
    elif args.manual_params:
        # Use manually specified parameters
        log_message("Using manually specified parameters")
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'learning_rate': args.learning_rate,
            'num_leaves': args.num_leaves,
            'max_depth': args.max_depth,
            'min_data_in_leaf': args.min_data_in_leaf,
            'lambda_l1': args.lambda_l1,
            'lambda_l2': args.lambda_l2,
            'feature_fraction': args.feature_fraction,
            'bagging_fraction': args.bagging_fraction,
            'bagging_freq': args.bagging_freq,
            'min_gain_to_split': args.min_gain_to_split,
            'verbosity': -1,
            'n_estimators': args.n_estimators,
            'force_row_wise': True,
            'n_jobs': 4
        }
        
        # Print the manual parameters being used
        log_message("Manual parameters:")
        for key, value in params.items():
            log_message(f"  {key}: {value}")
        
        trial_data = []
    else:
        # Optimize hyperparameters
        params, trial_data = optimize_hyperparameters(
            X_train, y_train, X_val, y_val, n_trials=args.n_trials, timeout=args.timeout
        )

    # Save optimization history if available
    if trial_data:
        with open(os.path.join(args.output_dir, "optuna_trials.json"), 'w') as f:
            json.dump(trial_data, f, indent=2)
    
    # Step 6: Train final model
    model, importances = train_model(X_train, y_train, X_val, y_val, params, 
                                  args.early_stopping_rounds, args.verbose)
    
    # Step 7: Evaluate
    train_metrics = evaluate_model(model, X_train, y_train, "Train")
    val_metrics = evaluate_model(model, X_val, y_val, "Validation")
    test_metrics = evaluate_model(model, X_test, y_test, "Test")
    
    # Calculate train-test gap (measure of overfitting)
    train_test_rmse_gap = train_metrics["rmse"] - test_metrics["rmse"]
    train_test_r2_gap = train_metrics["r2"] - test_metrics["r2"]
    
    log_message(f"Train-Test RMSE gap: {abs(train_test_rmse_gap):.4f} (smaller is better)")
    log_message(f"Train-Test R² gap: {abs(train_test_r2_gap):.4f} (smaller is better)")
    
    # Step 8: Create plots
    plot_predictions(y_test, test_metrics["predictions"], 
                   os.path.join(args.output_dir, "test_predictions.png"))
    
    # Step 9: Save model and results
    save_model(model, args.output_dir, imputer, scaler, feature_names, params)
    
    # Save predictions
    results_df = pd.DataFrame({
        "Actual": y_test,
        "Predicted": test_metrics["predictions"],
        "Error": test_metrics["predictions"] - y_test,
        "AbsError": np.abs(test_metrics["predictions"] - y_test)
    })
    results_df.to_csv(os.path.join(args.output_dir, "test_predictions.csv"), index=False)
    
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
        
        f.write(f"Train-Test RMSE gap: {abs(train_test_rmse_gap):.4f}\n")
        f.write(f"Train-Test R² gap: {abs(train_test_r2_gap):.4f}\n")
    
    # Save feature importances
    sorted_indices = np.argsort(importances)[::-1]
    with open(os.path.join(args.output_dir, "feature_importance.txt"), "w") as f:
        for idx in sorted_indices:
            f.write(f"{feature_names[idx]}: {importances[idx]:.4f}\n")
    
    execution_time = time.time() - start_time
    log_message(f"Total execution time: {execution_time:.2f} seconds")
    print_success("Optimized LogD model training completed")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())