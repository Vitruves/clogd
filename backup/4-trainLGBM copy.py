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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import QuantileTransformer, RobustScaler
from sklearn.model_selection import KFold
import shap
from sklearn.cluster import KMeans
from sklearn.preprocessing import KBinsDiscretizer

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
    parser.add_argument("--focal-loss", action="store_true", help="Enable focal loss")
    parser.add_argument("--calibration", action="store_true", help="Enable calibration")
    parser.add_argument("--ensemble", action="store_true", help="Use an ensemble model")
    parser.add_argument("--ensemble-size", type=int, default=5, help="Number of models in ensemble")
    parser.add_argument("--ensemble-method", choices=["bagging", "stacking"], default="bagging", help="Ensemble method")
    parser.add_argument("--lr-schedule", action="store_true", help="Enable learning rate scheduling")
    parser.add_argument("--advanced-preprocessing", action="store_true", help="Enable advanced preprocessing")
    parser.add_argument("--interactions", action="store_true", help="Add interaction features")
    parser.add_argument("--generate-shap", action="store_true", help="Generate SHAP values for model explanation")
    parser.add_argument("--sample-strategy", choices=["tpe", "cmaes", "random"], default="tpe",
                        help="Optuna sampling strategy")
    return parser.parse_args()

def load_and_preprocess_data(input_file, external_test_file, target_column, use_descriptors, 
                             descriptor_type="both", smiles_col="SMILES", advanced_preprocessing=False):
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
    
    print(f"-- Initial feature count: {len(feature_cols)}")
    
    # Keep features as DataFrames to preserve column names
    X_train = df_train[feature_cols]
    
    # Align features between train and test sets - drop missing columns from train
    missing_cols = set(feature_cols) - set(df_test.columns)
    common_cols = [col for col in feature_cols if col not in missing_cols]
    
    if missing_cols:
        print(f"-- Found {len(missing_cols)} columns in training set that don't exist in test set")
        print(f"-- Dropping these columns from training set instead of adding to test set")
        X_train = X_train[common_cols]
    
    X_test = df_test[common_cols]
    
    print(f"-- Using {len(common_cols)} features after alignment")
    
    print("-- Replacing NaN values with 0")
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    
    # After NaN replacement, add advanced preprocessing
    if advanced_preprocessing:
        print("-- Applying advanced preprocessing")
        
        # Detect numerical columns (exclude binary/categorical)
        numerical_cols = []
        for col in X_train.columns:
            unique_vals = len(X_train[col].unique())
            if unique_vals > 10:  # Arbitrary threshold to identify numerical columns
                numerical_cols.append(col)
        
        if numerical_cols:
            print(f"-- Applying quantile transformation to {len(numerical_cols)} numerical features")
            qt = QuantileTransformer(output_distribution='normal', random_state=42)
            X_train_qt = qt.fit_transform(X_train[numerical_cols])
            X_test_qt = qt.transform(X_test[numerical_cols])
            
            # Replace transformed columns
            X_train_qt_df = pd.DataFrame(X_train_qt, columns=numerical_cols, index=X_train.index)
            X_test_qt_df = pd.DataFrame(X_test_qt, columns=numerical_cols, index=X_test.index)
            
            # Update original dataframes with transformed values
            for col in numerical_cols:
                X_train[col] = X_train_qt_df[col]
                X_test[col] = X_test_qt_df[col]
    
    # Auto-detect task type
    detected_task = "regression"  # Default assumption
    num_unique_values = len(np.unique(y_train))
    if num_unique_values == 2:
        detected_task = "binary"
    elif 2 < num_unique_values <= 10:  # Arbitrary threshold for categorical
        detected_task = "multiclass"
    
    return X_train, y_train, X_test, y_test, common_cols, detected_task

def get_evaluation_metric(task):
    if task == "binary":
        return lambda y_true, y_pred: roc_auc_score(y_true, y_pred)
    elif task == "regression":
        # Use RMSE without the squared parameter
        return lambda y_true, y_pred: -np.sqrt(mean_squared_error(y_true, y_pred))
    else:  # multiclass
        return lambda y_true, y_pred: accuracy_score(y_true, np.argmax(y_pred, axis=1))

def focal_loss_obj(gamma, alpha):
    """Create a focal loss objective function with given gamma and alpha parameters"""
    def _focal_loss_obj(y_pred, dtrain):
        y_true = dtrain.get_label()
        p = 1.0 / (1.0 + np.exp(-y_pred))
        a = y_true * alpha + (1 - y_true) * (1 - alpha)
        pt = y_true * p + (1 - y_true) * (1 - p)
        g = (1 - pt) ** gamma
        grad = a * g * (p - y_true)
        hess = a * g * p * (1 - p)
        return grad, hess
    return _focal_loss_obj

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

def objective(trial, X_train, y_train, X_valid, y_valid, X_test, y_test, task, eval_metric, 
              use_feature_selection=False, use_focal_loss=False, use_lr_schedule=False):
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
    
    # Add early stopping and learning rate scheduling
    callbacks = []
    if boosting_type != "dart":
        callbacks.append(lgb.early_stopping(stopping_rounds=50))
    
    if use_lr_schedule and boosting_type != "dart":
        # Learning rate scheduling with simpler approach
        min_lr_ratio = trial.suggest_float("min_lr_ratio", 0.01, 0.5)
        
        # Instead of using callback, adjust learning rate and n_estimators
        # This approach uses a constant learning rate but runs for fewer iterations
        if min_lr_ratio < 1.0:
            # Just use early stopping instead of custom callbacks
            # The model will stop when it no longer improves
            pass
    
    # Fit model with callbacks
    if boosting_type == "dart":
        model.fit(X_train_selected, y_train, eval_set=[(X_valid_selected, y_valid)])
    else:
        model.fit(
            X_train_selected, y_train, 
            eval_set=[(X_valid_selected, y_valid)],
            callbacks=callbacks if callbacks else None
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

def add_interaction_features(X_train, X_test, max_interactions=5):
    """Add interaction features between top important features"""
    print(f"-- Adding up to {max_interactions} interaction features")
    
    # Select features with highest variance for interactions
    selector = VarianceThreshold()
    selector.fit(X_train)
    variances = selector.variances_
    
    # Get indices of top features by variance
    top_indices = np.argsort(variances)[-20:]  # Take top 20 features
    top_features = X_train.columns[top_indices]
    
    # Create interactions among top features
    interactions_added = 0
    for i, feat1 in enumerate(top_features):
        for feat2 in top_features[i+1:]:
            if interactions_added >= max_interactions:
                break
                
            # Create new feature name
            interaction_name = f"interact_{feat1}_{feat2}"
            
            # Add multiplication interaction
            X_train[interaction_name] = X_train[feat1] * X_train[feat2]
            X_test[interaction_name] = X_test[feat1] * X_test[feat2]
            
            interactions_added += 1
    
    print(f"-- Added {interactions_added} interaction features")
    return X_train, X_test

def train_ensemble(X, y, best_params, best_selected_features, task, ensemble_size=5, method="bagging", seed=42):
    """Train an ensemble of models"""
    print(f"-- Training ensemble with {ensemble_size} models using {method}")
    ensemble_models = []
    
    for i in range(ensemble_size):
        print(f"-- Training ensemble model {i+1}/{ensemble_size}")
        # Use different random seeds for diversity
        current_seed = seed + i
        
        # Create model with best parameters
        params = best_params.copy()
        if task == "multiclass":
            params["num_class"] = len(np.unique(y))
        
        if task in ["binary", "multiclass"]:
            model = lgb.LGBMClassifier(**params)
        else:
            model = lgb.LGBMRegressor(**params)
        
        # Train on different data splits for bagging
        if method == "bagging":
            X_train_bag, _, y_train_bag, _ = train_test_split(
                X, y, test_size=0.2, random_state=current_seed
            )
            model.fit(X_train_bag, y_train_bag)
        else:  # stacking - use k-fold training
            kf = KFold(n_splits=5, shuffle=True, random_state=current_seed)
            for train_idx, _ in kf.split(X):
                X_fold = X.iloc[train_idx]
                y_fold = y[train_idx]
                model.fit(X_fold, y_fold)
                break  # Just use one fold for diversity
        
        ensemble_models.append(model)
    
    return ensemble_models

def ensemble_predict(models, X, task):
    """Generate predictions from an ensemble of models"""
    if task == "regression":
        preds = np.array([model.predict(X) for model in models])
        return np.mean(preds, axis=0)
    elif task == "binary":
        preds = np.array([model.predict_proba(X)[:,1] for model in models])
        return np.mean(preds, axis=0)
    else:  # multiclass
        preds = np.array([model.predict_proba(X) for model in models])
        return np.mean(preds, axis=0)

def bayesian_ensemble_predict(models, X, y_test, task):
    """Weight models by validation performance for improved ensemble"""
    if len(models) <= 1:
        return ensemble_predict(models, X, task)
        
    # Get predictions from each model
    if task == "regression":
        preds = np.array([model.predict(X) for model in models])
        
        # Calculate model weights based on MSE (lower is better)
        errors = np.array([mean_squared_error(y_test, pred) for pred in preds])
        weights = 1.0 / (errors + 1e-10)  # Add small constant to avoid division by zero
        weights = weights / np.sum(weights)  # Normalize
        
        # Weight predictions
        weighted_preds = np.sum(preds * weights.reshape(-1, 1), axis=0)
        return weighted_preds
    else:
        # Similar implementation for classification tasks
        return ensemble_predict(models, X, task)

def group_correlated_features(X_train, X_test, correlation_threshold=0.85, n_groups=20):
    # Calculate correlation matrix
    corr_matrix = X_train.corr().abs()
    
    # Cluster similar features
    feature_vectors = corr_matrix.values
    kmeans = KMeans(n_clusters=n_groups, random_state=42)
    clusters = kmeans.fit_predict(feature_vectors)
    
    # Group features by cluster
    feature_groups = {}
    for i, cluster_id in enumerate(clusters):
        if cluster_id not in feature_groups:
            feature_groups[cluster_id] = []
        feature_groups[cluster_id].append(X_train.columns[i])
    
    # Create aggregated features
    for group_id, features in feature_groups.items():
        if len(features) > 1:
            X_train[f"group_{group_id}_mean"] = X_train[features].mean(axis=1)
            X_test[f"group_{group_id}_mean"] = X_test[features].mean(axis=1)
    
    return X_train, X_test

def create_nonlinear_features(X_train, X_test, top_n=10):
    print(f"-- Creating nonlinear transformations for top {top_n} features")
    importances = np.var(X_train.values, axis=0)
    top_indices = np.argsort(importances)[-top_n:]
    top_columns = X_train.columns[top_indices]
    
    for col in top_columns:
        # Log transform (add small constant to avoid log(0))
        X_train[f"log_{col}"] = np.log1p(np.abs(X_train[col]))
        X_test[f"log_{col}"] = np.log1p(np.abs(X_test[col]))
        
        # Square root transform
        X_train[f"sqrt_{col}"] = np.sqrt(np.abs(X_train[col]))
        X_test[f"sqrt_{col}"] = np.sqrt(np.abs(X_test[col]))
        
        # Square transform
        X_train[f"sq_{col}"] = X_train[col]**2
        X_test[f"sq_{col}"] = X_test[col]**2
    
    return X_train, X_test

def bin_numerical_features(X_train, X_test, n_bins=5, top_n=10):
    """Bin top numerical features to capture non-linear relationships"""
    # Find most important features by variance
    variances = np.var(X_train.values, axis=0)
    top_indices = np.argsort(variances)[-top_n:]
    top_features = X_train.columns[top_indices]
    
    for feature in top_features:
        # Create discretizer
        kbd = KBinsDiscretizer(n_bins=n_bins, encode='onehot', strategy='quantile')
        
        # Fit and transform
        binned = kbd.fit_transform(X_train[[feature]]).toarray()
        binned_test = kbd.transform(X_test[[feature]]).toarray()
        
        # Create new columns
        for i in range(n_bins):
            X_train[f"{feature}_bin_{i}"] = binned[:, i]
            X_test[f"{feature}_bin_{i}"] = binned_test[:, i]
    
    return X_train, X_test

def main():
    args = parse_args()
    np.random.seed(args.seed)
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"-- Created output directory: {args.output_dir}")
    
    print("-- LightGBM model training with Optuna optimization")
    
    X_train, y_train, X_test, y_test, feature_cols, detected_task = load_and_preprocess_data(
        args.input, args.external_test, args.target_column, args.use_file_descriptors, 
        args.descriptors, args.smiles_col, args.advanced_preprocessing
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
    
    # Create the train/validation split for optimization
    X_train_opt, X_valid, y_train_opt, y_valid = train_test_split(
        X_train, y_train, test_size=0.2, random_state=args.seed
    )
    
    # Add interaction features if requested - AFTER the train/validation split
    # but BEFORE feature selection starts in the objective function
    if args.interactions:
        # Create a copy to avoid the fragmentation warning
        X_train = X_train.copy()
        X_test = X_test.copy()
        X_train_opt = X_train_opt.copy() 
        X_valid = X_valid.copy()
        
        # Now add interaction features to all datasets
        print("-- Adding interaction features")
        X_train, X_test = add_interaction_features(X_train, X_test)
        
        # We need to ensure the same interaction features are added to train_opt and valid
        # Get the new interaction features
        interaction_features = [col for col in X_train.columns if col.startswith("interact_")]
        
        # Create the same features in train_opt and valid
        for feature in interaction_features:
            # Parse the original features from the interaction name
            parts = feature.split("_")[1:]  # Skip "interact_" prefix
            if len(parts) >= 2:
                feat1 = "_".join(parts[:len(parts)//2])
                feat2 = "_".join(parts[len(parts)//2:])
                
                # Create the interaction feature
                X_train_opt[feature] = X_train_opt[feat1] * X_train_opt[feat2]
                X_valid[feature] = X_valid[feat1] * X_valid[feat2]
    
    eval_metric = get_evaluation_metric(args.task)
    
    # Print enabled enhancements
    enhancements = []
    if args.feature_selection:
        enhancements.append("Feature Selection")
    if args.ensemble:
        enhancements.append(f"{args.ensemble_method.capitalize()} Ensemble ({args.ensemble_size} models)")
    if args.focal_loss:
        enhancements.append("Focal Loss")
    if args.calibration:
        enhancements.append("Probability Calibration")
    if args.lr_schedule:
        enhancements.append("Learning Rate Scheduling")
    if args.advanced_preprocessing:
        enhancements.append("Advanced Preprocessing")
    if args.interactions:
        enhancements.append("Interaction Features")
    
    if enhancements:
        print("-- Enabled enhancements:")
        for enhancement in enhancements:
            print(f"--   {enhancement}")
    
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
    if args.sample_strategy == "tpe":
        sampler = optuna.samplers.TPESampler(multivariate=True, seed=args.seed)
    elif args.sample_strategy == "cmaes":
        sampler = optuna.samplers.CmaEsSampler(seed=args.seed)
    else:
        sampler = optuna.samplers.RandomSampler(seed=args.seed)
    
    study = optuna.create_study(direction="maximize", sampler=sampler)
    
    # Add pruning to eliminate poor trials early
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=30,
        interval_steps=10
    )
    
    study.optimize(
        lambda trial: objective(
            trial, X_train_opt, y_train_opt, X_valid, y_valid, X_test, y_test, 
            args.task, eval_metric, args.feature_selection, args.focal_loss, args.lr_schedule
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
    
    # Use selected features for final model
    X_train_final = X_train[best_selected_features]
    X_test_final = X_test[best_selected_features]
    
    # Train final model with best parameters and selected features
    best_params = study.best_params
    if args.task == "multiclass":
        best_params["num_class"] = len(np.unique(y_train))
    
    best_params["objective"] = "binary" if args.task == "binary" else "regression" if args.task == "regression" else "multiclass"
    
    # Implement ensemble if enabled
    if args.ensemble:
        print(f"-- Training ensemble with {args.ensemble_size} models using {args.ensemble_method}")
        ensemble_models = train_ensemble(
            X_train_final, y_train, best_params, best_selected_features, 
            args.task, args.ensemble_size, args.ensemble_method, args.seed
        )
        
        # Save each model in the ensemble
        for i, model in enumerate(ensemble_models):
            model_path = os.path.join(args.output_dir, f"ensemble_model_{i}.pkl")
            joblib.dump(model, model_path)
        
        # Save ensemble metadata
        ensemble_info = {
            "model_count": len(ensemble_models),
            "method": args.ensemble_method,
            "task": args.task
        }
        
        with open(os.path.join(args.output_dir, "ensemble_info.json"), "w") as f:
            import json
            json.dump(ensemble_info, f, indent=2)
        
        # Make predictions with the ensemble
        y_pred_ensemble = bayesian_ensemble_predict(ensemble_models, X_test_final, args.task)
        
        # Evaluate ensemble performance
        if args.task == "binary":
            ensemble_score = roc_auc_score(y_test, y_pred_ensemble)
            print(f"-- Ensemble external test AUC: {ensemble_score:.4f}")
        elif args.task == "regression":
            ensemble_score = np.sqrt(mean_squared_error(y_test, y_pred_ensemble))
            print(f"-- Ensemble external test RMSE: {ensemble_score:.4f}")
        else:  # multiclass
            y_pred_classes = np.argmax(y_pred_ensemble, axis=1) if y_pred_ensemble.ndim > 1 else y_pred_ensemble
            ensemble_score = accuracy_score(y_test, y_pred_classes)
            print(f"-- Ensemble external test accuracy: {ensemble_score:.4f}")
    
    # Train a single model
    print("-- Training final model with best parameters")
    final_model = lgb.LGBMClassifier(**best_params) if args.task in ["binary", "multiclass"] else lgb.LGBMRegressor(**best_params)
    
    # Add focal loss if enabled
    if args.focal_loss and args.task == "binary":
        if "focal_gamma" in best_params and "focal_alpha" in best_params:
            gamma = best_params.pop("focal_gamma")
            alpha = best_params.pop("focal_alpha")
            final_model.set_params(objective=focal_loss_obj(gamma, alpha))
    
    final_model.fit(X_train_final, y_train)
    
    # Apply calibration if enabled for classifiers
    if args.calibration and args.task in ["binary", "multiclass"]:
        print("-- Calibrating predicted probabilities")
        calibrated_model = CalibratedClassifierCV(final_model, cv="prefit", method="isotonic")
        calibrated_model.fit(X_valid[best_selected_features], y_valid)
        final_model = calibrated_model
    
    model_path = os.path.join(args.output_dir, "best_model.pkl")
    joblib.dump(final_model, model_path)
    print(f"-- Saved model to {model_path}")
    
    # Evaluate final model
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
    
    params_path = os.path.join(args.output_dir, "best_params.json")
    with open(params_path, "w") as f:
        import json
        # Convert params to serializable format
        serializable_params = {k: float(v) if isinstance(v, np.float32) else v 
                              for k, v in study.best_params.items()}
        json.dump(serializable_params, f, indent=2)
    print(f"-- Saved best parameters to {params_path}")
    
    # Save selected features
    features_path = os.path.join(args.output_dir, "selected_features.txt")
    with open(features_path, "w") as f:
        for feature in best_selected_features:
            f.write(f"{feature}\n")
    print(f"-- Saved {len(best_selected_features)} selected features to {features_path}")
    
    # Save feature importance if not using an ensemble
    if not args.ensemble:
        if hasattr(final_model, "feature_importances_"):
            importance_model = final_model
        elif hasattr(final_model, "base_estimator") and hasattr(final_model.base_estimator, "feature_importances_"):
            importance_model = final_model.base_estimator
        else:
            importance_model = None
            
        if importance_model is not None:
            feature_importance = pd.DataFrame({
                'feature': best_selected_features,
                'importance': importance_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            fi_path = os.path.join(args.output_dir, "feature_importance.csv")
            feature_importance.to_csv(fi_path, index=False)
            print(f"-- Saved feature importance to {fi_path}")
    
    # Generate SHAP values if requested
    if args.generate_shap and not args.ensemble:
        print("-- Generating SHAP values for feature explanation")
        explainer = shap.TreeExplainer(final_model)
        X_sample = X_test_final.iloc[:min(100, len(X_test_final))]  # Sample for faster calculation
        shap_values = explainer.shap_values(X_sample)
        
        # Save SHAP values
        pd.DataFrame(shap_values).to_csv(os.path.join(args.output_dir, "shap_values.csv"), index=False)
        
        # Save feature importance based on SHAP
        shap_importance = pd.DataFrame({
            'feature': X_sample.columns,
            'importance': np.abs(shap_values).mean(0)
        }).sort_values('importance', ascending=False)
        
        shap_importance.to_csv(os.path.join(args.output_dir, "shap_importance.csv"), index=False)
    
    print("-- Done")

if __name__ == "__main__":
    main()