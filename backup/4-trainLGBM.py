#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
import joblib
import os
from sklearn.metrics import roc_auc_score, mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
import warnings
from optuna._experimental import ExperimentalWarning

# Suppress Optuna experimental warnings
warnings.filterwarnings('ignore', category=ExperimentalWarning)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a LightGBM model with Optuna optimization")
    parser.add_argument("--input", required=True, help="Input training data file (CSV)")
    parser.add_argument("--output-dir", required=True, help="Directory to save model and results")
    parser.add_argument("--use-file-descriptors", action="store_true", help="Use molecular descriptors in input file")
    parser.add_argument("--external-test", required=True, help="External test set for optimization (CSV)")
    parser.add_argument("--target-column", required=True, help="Target column name")
    parser.add_argument("--smiles-col", default="SMILES", help="Column name containing SMILES strings to exclude")
    parser.add_argument("--task", choices=["binary", "regression", "multiclass"], default="binary", 
                        help="Task type: binary, regression, or multiclass")
    parser.add_argument("--n-trials", type=int, default=100, help="Number of Optuna trials")
    parser.add_argument("--time-budget", type=int, default=3600, help="Time budget in seconds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Show verbose output")
    parser.add_argument("--descriptors", choices=["mordred", "fingerprints", "both"], 
                        default="both", help="Type of molecular descriptors to use")
    parser.add_argument("--feature-selection", action="store_true", help="Enable feature selection optimization")
    return parser.parse_args()

def load_and_preprocess_data(input_file, external_test_file, target_column, use_descriptors, descriptor_type="both", smiles_col="SMILES"):
    print(f"-- Loading training data from {input_file}")
    df_train = pd.read_csv(input_file)
    
    print(f"-- Loading external test data from {external_test_file}")
    df_test = pd.read_csv(external_test_file)
    
    y_train = df_train[target_column].values
    y_test = df_test[target_column].values
    
    # Exclude target and SMILES columns
    exclude_cols = [target_column, smiles_col]
    feature_cols = [col for col in df_train.columns if col not in exclude_cols]
    
    if not use_descriptors:
        feature_cols = [col for col in feature_cols if not col.startswith("descriptor_")]
    
    # Select features based on type
    if descriptor_type == "mordred":
        feature_cols = [col for col in feature_cols if col.startswith("mordred_")]
    elif descriptor_type == "fingerprints":
        feature_cols = [col for col in feature_cols if col.startswith("fp_")]
    # For "both", use all features
    
    print(f"-- Using {len(feature_cols)} features")
    
    # Keep features as DataFrames to preserve column names
    X_train = df_train[feature_cols]
    
    print("-- Aligning test set features with training set")
    missing_cols = set(feature_cols) - set(df_test.columns)
    if missing_cols:
        print(f"-- Warning: Missing {len(missing_cols)} features in test set")
        for col in missing_cols:
            df_test[col] = 0
    
    X_test = df_test[feature_cols]
    
    print("-- Replacing NaN values with 0")
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    
    # Auto-detect task type
    detected_task = "regression"  # Default assumption
    num_unique_values = len(np.unique(y_train))
    if num_unique_values == 2:
        detected_task = "binary"
    elif 2 < num_unique_values <= 10:  # Arbitrary threshold for categorical
        detected_task = "multiclass"
    
    return X_train, y_train, X_test, y_test, feature_cols, detected_task

def get_evaluation_metric(task):
    if task == "binary":
        return lambda y_true, y_pred: roc_auc_score(y_true, y_pred)
    elif task == "regression":
        # Use RMSE without the squared parameter
        return lambda y_true, y_pred: -np.sqrt(mean_squared_error(y_true, y_pred))
    else:  # multiclass
        return lambda y_true, y_pred: accuracy_score(y_true, np.argmax(y_pred, axis=1))

def select_features(trial, X_train, y_train, X_valid, X_test, task, use_feature_selection=True):
    if not use_feature_selection:
        return X_train, X_valid, X_test, list(X_train.columns)
    
    # First, remove low variance features
    variance_threshold = trial.suggest_float("variance_threshold", 0.0, 0.2)
    selector = VarianceThreshold(threshold=variance_threshold)
    X_train_var = selector.fit_transform(X_train)
    X_valid_var = selector.transform(X_valid)
    X_test_var = selector.transform(X_test)
    
    # Get the selected feature names after variance thresholding
    selected_feature_mask = selector.get_support()
    selected_features = X_train.columns[selected_feature_mask].tolist()
    
    # Create DataFrame with selected features
    X_train_selected = pd.DataFrame(X_train_var, columns=selected_features)
    X_valid_selected = pd.DataFrame(X_valid_var, columns=selected_features)
    X_test_selected = pd.DataFrame(X_test_var, columns=selected_features)
    
    # Optionally do a second selection with a simple model
    do_model_selection = trial.suggest_categorical("do_model_selection", [True, False])
    
    if do_model_selection and len(selected_features) > 10:
        # Determine selection method
        selection_method = trial.suggest_categorical("selection_method", ["model_importance", "l1"])
        
        if selection_method == "model_importance":
            # Use a simple LightGBM to select features based on importance
            importance_percentile = trial.suggest_float("importance_percentile", 0.5, 0.95)
            model_params = {
                "objective": "binary" if task == "binary" else "regression" if task == "regression" else "multiclass",
                "verbosity": -1,
                "boosting_type": "gbdt",
                "max_depth": 3,
                "learning_rate": 0.1,
                "n_estimators": 50
            }
            
            if task == "multiclass":
                model_params["num_class"] = len(np.unique(y_train))
                
            selector_model = lgb.LGBMClassifier(**model_params) if task in ["binary", "multiclass"] else lgb.LGBMRegressor(**model_params)
            selector_model.fit(X_train_selected, y_train)
            
            # Select features based on importance percentile
            feature_importances = selector_model.feature_importances_
            importance_threshold = np.percentile(feature_importances, (1.0 - importance_percentile) * 100)
            importance_mask = feature_importances >= importance_threshold
            
            selected_features = [selected_features[i] for i in range(len(selected_features)) if importance_mask[i]]
            
        elif selection_method == "l1":
            # Use L1 regularization for feature selection
            l1_threshold = trial.suggest_float("l1_threshold", 1e-5, 1e-3, log=True)
            l1_model_params = {
                "objective": "binary" if task == "binary" else "regression" if task == "regression" else "multiclass",
                "verbosity": -1,
                "boosting_type": "gbdt",
                "max_depth": 3,
                "reg_alpha": l1_threshold,
                "reg_lambda": 0.0,
                "learning_rate": 0.1,
                "n_estimators": 50
            }
            
            if task == "multiclass":
                l1_model_params["num_class"] = len(np.unique(y_train))
                
            l1_model = lgb.LGBMClassifier(**l1_model_params) if task in ["binary", "multiclass"] else lgb.LGBMRegressor(**l1_model_params)
            selector = SelectFromModel(l1_model, threshold="mean")
            selector.fit(X_train_selected, y_train)
            
            # Get selected features
            model_mask = selector.get_support()
            selected_features = [selected_features[i] for i in range(len(selected_features)) if model_mask[i]]
        
        # Create final DataFrames with selected features
        X_train_selected = X_train_selected[selected_features]
        X_valid_selected = X_valid_selected[selected_features]
        X_test_selected = X_test_selected[selected_features]
    
    print(f"-- Selected {len(selected_features)} features after feature selection")
    return X_train_selected, X_valid_selected, X_test_selected, selected_features

def objective(trial, X_train, y_train, X_valid, y_valid, X_test, y_test, task, eval_metric, use_feature_selection):
    # Feature selection
    X_train_selected, X_valid_selected, X_test_selected, selected_features = select_features(
        trial, X_train, y_train, X_valid, X_test, task, use_feature_selection
    )
    
    # Model hyperparameters with expanded search space
    boosting_type = trial.suggest_categorical("boosting_type", ["gbdt", "dart", "goss"])
    
    # Wider parameter ranges and more granular exploration
    params = {
        "objective": "binary" if task == "binary" else "regression" if task == "regression" else "multiclass",
        "verbosity": -1,
        "boosting_type": boosting_type,
        "num_leaves": trial.suggest_int("num_leaves", 7, 4095, log=True),  # Expanded range with log scale
        "max_depth": trial.suggest_int("max_depth", 2, 15),  # Wider range
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.5, log=True),  # Expanded range
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000, log=True),  # More options with log scale
        "min_child_samples": trial.suggest_int("min_child_samples", 1, 300, log=True),  # Wider range
        "subsample": trial.suggest_float("subsample", 0.3, 1.0) if boosting_type != "goss" else 1.0,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-10, 100.0, log=True),  # Much wider range
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-10, 100.0, log=True),  # Much wider range
    }
    
    # Add more hyperparameters to search
    if boosting_type == "dart":
        params["drop_rate"] = trial.suggest_float("drop_rate", 0.05, 0.5)
        params["skip_drop"] = trial.suggest_float("skip_drop", 0.05, 0.5)
        
    if boosting_type == "goss":
        # Ensure top_rate + other_rate <= 1.0
        top_rate = trial.suggest_float("top_rate", 0.1, 0.9)
        other_rate = trial.suggest_float("other_rate", 0.05, min(0.5, 1.0 - top_rate))
        params["top_rate"] = top_rate
        params["other_rate"] = other_rate
    
    # Advanced parameters for all boosting types
    params["min_child_weight"] = trial.suggest_float("min_child_weight", 1e-5, 100.0, log=True)
    params["min_split_gain"] = trial.suggest_float("min_split_gain", 0.0, 5.0)
    params["bagging_freq"] = trial.suggest_int("bagging_freq", 0, 10)
    
    if task == "multiclass":
        params["num_class"] = len(np.unique(y_train))
    
    model = lgb.LGBMClassifier(**params) if task in ["binary", "multiclass"] else lgb.LGBMRegressor(**params)
    
    # Apply early stopping only for gbdt and goss boosting types
    if boosting_type == "dart":
        model.fit(X_train_selected, y_train, eval_set=[(X_valid_selected, y_valid)])
    else:
        model.fit(
            X_train_selected, y_train, 
            eval_set=[(X_valid_selected, y_valid)],
            callbacks=[lgb.early_stopping(stopping_rounds=50)]
        )
    
    if task == "binary":
        y_pred = model.predict_proba(X_test_selected)[:, 1]
    elif task == "multiclass":
        y_pred = model.predict_proba(X_test_selected)
    else:
        y_pred = model.predict(X_test_selected)
    
    external_score = eval_metric(y_test, y_pred)
    
    # Save selected features for retrieval after optimization
    trial.set_user_attr("selected_features", selected_features)
    
    return external_score

def main():
    args = parse_args()
    np.random.seed(args.seed)
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"-- Created output directory: {args.output_dir}")
    
    print("-- LightGBM model training with Optuna optimization")
    
    X_train, y_train, X_test, y_test, feature_cols, detected_task = load_and_preprocess_data(
        args.input, args.external_test, args.target_column, args.use_file_descriptors, 
        args.descriptors, args.smiles_col
    )
    
    # Override task with detected task if there's a mismatch
    if args.task != detected_task:
        print(f"-- Warning: Specified task '{args.task}' doesn't match detected task '{detected_task}' based on target data")
        print(f"-- Using detected task: {detected_task}")
        args.task = detected_task
    else:
        print(f"-- Task: {args.task}")
    
    print(f"-- Training data shape: {X_train.shape}")
    print(f"-- External test data shape: {X_test.shape}")
    
    if args.feature_selection:
        print("-- Feature selection optimization enabled")
    
    X_train_opt, X_valid, y_train_opt, y_valid = train_test_split(
        X_train, y_train, test_size=0.2, random_state=args.seed
    )
    
    eval_metric = get_evaluation_metric(args.task)
    
    print(f"-- Starting Optuna optimization with {args.n_trials} trials")
    print(f"-- Optimizing directly on external test set performance")
    
    # Track best score for reporting improvement
    best_score_so_far = float('inf')
    
    def log_trial(study, trial):
        nonlocal best_score_so_far
        # For regression, convert negative value back to positive RMSE
        current_score = -trial.value if args.task == "regression" else trial.value
        if args.task == "regression" and current_score < best_score_so_far:
            best_score_so_far = current_score
            n_features = len(trial.user_attrs["selected_features"]) if "selected_features" in trial.user_attrs else X_train.shape[1]
            print(f"-- [Improvement] Trial {trial.number}: External RMSE = {current_score:.4f} with {n_features} features")
    
    # Create sampler with better exploration
    sampler = optuna.samplers.TPESampler(
        multivariate=True,  # Consider parameter correlations
        seed=args.seed,
        n_startup_trials=10,  # More random exploration at start
        constant_liar=True,  # Improve parallelization if multiple workers
        warn_independent_sampling=False  # Suppress independent sampling warnings
    )
    
    study = optuna.create_study(direction="maximize", sampler=sampler)
    
    # Add pruning to eliminate poor trials early
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=30,
        interval_steps=10
    )
    
    study.optimize(
        lambda trial: objective(
            trial, X_train_opt, y_train_opt, X_valid, y_valid, X_test, y_test, args.task, eval_metric, args.feature_selection
        ),
        n_trials=args.n_trials,
        timeout=args.time_budget,
        show_progress_bar=args.verbose,
        callbacks=[log_trial]
    )
    
    # Show actual metric score rather than optimization score
    if args.task == "regression":
        actual_best_score = -study.best_value
        print(f"-- Best trial: {study.best_trial.number}")
        print(f"-- Best external RMSE: {actual_best_score:.4f}")
    else:
        print(f"-- Best trial: {study.best_trial.number}")
        print(f"-- Best score: {study.best_value:.4f}")
    
    print("-- Best parameters:")
    for key, value in study.best_params.items():
        print(f"--   {key}: {value}")
    
    # Get the best selected features
    best_selected_features = study.best_trial.user_attrs.get("selected_features", feature_cols)
    print(f"-- Final model will use {len(best_selected_features)} features")
    
    # Train final model with best parameters and selected features
    best_params = study.best_params
    if args.task == "multiclass":
        best_params["num_class"] = len(np.unique(y_train))
    
    best_params["objective"] = "binary" if args.task == "binary" else "regression" if args.task == "regression" else "multiclass"
    
    print("-- Training final model with best parameters")
    final_model = lgb.LGBMClassifier(**best_params) if args.task in ["binary", "multiclass"] else lgb.LGBMRegressor(**best_params)
    
    # Use selected features for final model
    X_train_final = X_train[best_selected_features]
    X_test_final = X_test[best_selected_features]
    
    final_model.fit(X_train_final, y_train)
    
    model_path = os.path.join(args.output_dir, "best_model.pkl")
    joblib.dump(final_model, model_path)
    print(f"-- Saved model to {model_path}")
    
    params_path = os.path.join(args.output_dir, "best_params.json")
    with open(params_path, "w") as f:
        import json
        json.dump(study.best_params, f, indent=2)
    print(f"-- Saved best parameters to {params_path}")
    
    # Save selected features
    features_path = os.path.join(args.output_dir, "selected_features.txt")
    with open(features_path, "w") as f:
        for feature in best_selected_features:
            f.write(f"{feature}\n")
    print(f"-- Saved {len(best_selected_features)} selected features to {features_path}")
    
    feature_importance = pd.DataFrame({
        'feature': best_selected_features,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    fi_path = os.path.join(args.output_dir, "feature_importance.csv")
    feature_importance.to_csv(fi_path, index=False)
    print(f"-- Saved feature importance to {fi_path}")
    
    if args.task == "binary":
        y_pred = final_model.predict_proba(X_test_final)[:, 1]
        test_score = roc_auc_score(y_test, y_pred)
        print(f"-- Final external test AUC: {test_score:.4f}")
    elif args.task == "regression":
        y_pred = final_model.predict(X_test_final)
        test_score = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"-- Final external test RMSE: {test_score:.4f}")
    else:  # multiclass
        y_pred = final_model.predict(X_test_final)
        test_score = accuracy_score(y_test, y_pred)
        print(f"-- Final external test accuracy: {test_score:.4f}")
    
    print("-- Done")

if __name__ == "__main__":
    main()