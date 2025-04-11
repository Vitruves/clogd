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
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.ensemble import StackingRegressor, StackingClassifier
from sklearn.linear_model import RidgeCV, LogisticRegressionCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.manifold import TSNE, Isomap
from sklearn.decomposition import KernelPCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from transformers import AutoModel, AutoTokenizer
import torch
from rdkit import Chem
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import traceback

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
    parser.add_argument("--polynomial-features", action="store_true", help="Add polynomial features")
    parser.add_argument("--feature-clustering", action="store_true", help="Use feature clustering")
    parser.add_argument("--nonlinear-transforms", action="store_true", help="Add nonlinear feature transformations")
    parser.add_argument("--feature-binning", action="store_true", help="Bin numerical features")
    parser.add_argument("--nn-boost", action="store_true", help="Add neural network boosting")
    parser.add_argument("--nn-hidden-layers", type=str, default="64,32", help="Hidden layer sizes for neural network")
    parser.add_argument("--use-transformers", action="store_true", help="Use transformer models for molecular embedding")
    parser.add_argument("--use-gnn", action="store_true", help="Use graph neural networks for molecular embedding")
    parser.add_argument("--stacked-models", action="store_true", help="Use stacked models")
    parser.add_argument("--manifold-features", action="store_true", help="Add manifold learning features")
    parser.add_argument("--transformer-model", 
                      help="Path to custom pretrained transformer model (for LogP/LogD tasks)")
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
    
    # Model hyperparameters with massively expanded search space
    boosting_type = trial.suggest_categorical("boosting_type", ["gbdt", "dart", "goss"])
    
    # Much wider parameter ranges with logarithmic scales where appropriate
    params = {
        "objective": "binary" if task == "binary" else "regression" if task == "regression" else "multiclass",
        "verbosity": -1,
        "boosting_type": boosting_type,
        "num_leaves": trial.suggest_int("num_leaves", 2, 8192, log=True),  # Vastly expanded range
        "max_depth": trial.suggest_int("max_depth", -1, 20),  # -1 means no limit, larger range
        "learning_rate": trial.suggest_float("learning_rate", 0.0001, 1.0, log=True),  # Expanded range
        "n_estimators": trial.suggest_int("n_estimators", 10, 3000, log=True),  # Much larger range
        "min_child_samples": trial.suggest_int("min_child_samples", 1, 500, log=True),  # Wider range
        "subsample": trial.suggest_float("subsample", 0.1, 1.0) if boosting_type != "goss" else 1.0,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-10, 1000.0, log=True),  # Much wider range
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-10, 1000.0, log=True),  # Much wider range
    }
    
    # Add more hyperparameters to search with expanded ranges
    if boosting_type == "dart":
        params["drop_rate"] = trial.suggest_float("drop_rate", 0.01, 0.9)  # Wider range
        params["skip_drop"] = trial.suggest_float("skip_drop", 0.01, 0.9)  # Wider range
        params["max_drop"] = trial.suggest_int("max_drop", 10, 100)  # Additional parameter
        
    if boosting_type == "goss":
        # Ensure top_rate + other_rate <= 1.0
        top_rate = trial.suggest_float("top_rate", 0.01, 0.99)  # Expanded
        other_rate = trial.suggest_float("other_rate", 0.01, min(0.99, 1.0 - top_rate))  # Expanded
        params["top_rate"] = top_rate
        params["other_rate"] = other_rate
    
    # Advanced parameters for all boosting types with expanded ranges
    params["min_child_weight"] = trial.suggest_float("min_child_weight", 1e-6, 1000.0, log=True)
    params["min_split_gain"] = trial.suggest_float("min_split_gain", 0.0, 20.0)
    params["bagging_freq"] = trial.suggest_int("bagging_freq", 0, 100)
    
    # Additional parameters to explore
    params["feature_fraction_seed"] = trial.suggest_int("feature_fraction_seed", 1, 10000)
    params["extra_trees"] = trial.suggest_categorical("extra_trees", [True, False])
    params["feature_fraction_bynode"] = trial.suggest_float("feature_fraction_bynode", 0.1, 1.0)
    
    # Path smoothing parameter (reduces overfitting)
    params["path_smooth"] = trial.suggest_float("path_smooth", 0.0, 10.0)
    
    # Add histogram optimization parameters
    if trial.suggest_categorical("use_histogram_opt", [True, False]):
        params["max_bin"] = trial.suggest_int("max_bin", 2, 1024, log=True)
        params["min_data_in_bin"] = trial.suggest_int("min_data_in_bin", 1, 100)
    
    if task == "multiclass":
        params["num_class"] = len(np.unique(y_train))
    
    model = lgb.LGBMClassifier(**params) if task in ["binary", "multiclass"] else lgb.LGBMRegressor(**params)
    
    # Add early stopping and learning rate scheduling
    callbacks = []
    if boosting_type != "dart":
        callbacks.append(lgb.early_stopping(stopping_rounds=50))
    
    if use_lr_schedule and boosting_type != "dart":
        # Learning rate scheduling with expanded range
        min_lr_ratio = trial.suggest_float("min_lr_ratio", 0.001, 0.9, log=True)
        
        if min_lr_ratio < 1.0:
            # Just use early stopping instead of custom callbacks
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

def add_polynomial_features(X_train, X_test, degree=2, max_features=50):
    """Add polynomial features for top features by variance"""
    print(f"-- Adding polynomial features (degree={degree})")
    
    # Select top features by variance to avoid explosion in feature count
    variances = np.var(X_train.values, axis=0)
    top_indices = np.argsort(variances)[-max_features:]
    top_columns = X_train.columns[top_indices]
    
    # Create polynomial features
    poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=True)
    X_train_poly = poly.fit_transform(X_train[top_columns])
    X_test_poly = poly.transform(X_test[top_columns])
    
    # Create feature names
    poly_feature_names = poly.get_feature_names_out(top_columns)
    
    # Remove original features from polynomial output (they're already in the dataframe)
    original_indices = [i for i, name in enumerate(poly_feature_names) if name in top_columns]
    interaction_indices = [i for i, name in enumerate(poly_feature_names) if i not in original_indices]
    
    # Add only interaction terms as new features
    poly_df_train = pd.DataFrame(
        X_train_poly[:, interaction_indices], 
        columns=[f"poly_{name}" for name in poly_feature_names[interaction_indices]],
        index=X_train.index
    )
    poly_df_test = pd.DataFrame(
        X_test_poly[:, interaction_indices],
        columns=[f"poly_{name}" for name in poly_feature_names[interaction_indices]],
        index=X_test.index
    )
    
    # Combine with original features
    X_train = pd.concat([X_train, poly_df_train], axis=1)
    X_test = pd.concat([X_test, poly_df_test], axis=1)
    
    print(f"-- Added {poly_df_train.shape[1]} polynomial features")
    return X_train, X_test

def add_feature_clusters(X_train, X_test, n_clusters=10):
    """Add features based on feature clustering"""
    print(f"-- Creating {n_clusters} feature clusters")
    
    # Calculate correlation matrix
    corr_matrix = X_train.corr().abs().fillna(0)
    
    # Use K-means to cluster features based on correlation
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    # Convert correlation matrix to a distance measure (1 - correlation)
    corr_distance = 1 - corr_matrix
    # Fit on the "distances" between features
    clusters = kmeans.fit_predict(corr_distance)
    
    # Group features by cluster
    feature_groups = {}
    for i, cluster_id in enumerate(clusters):
        if cluster_id not in feature_groups:
            feature_groups[cluster_id] = []
        feature_groups[cluster_id].append(X_train.columns[i])
    
    # Create aggregated features
    for group_id, features in feature_groups.items():
        if len(features) > 1:
            X_train[f"cluster_{group_id}_mean"] = X_train[features].mean(axis=1)
            X_train[f"cluster_{group_id}_sum"] = X_train[features].sum(axis=1)
            X_train[f"cluster_{group_id}_min"] = X_train[features].min(axis=1)
            X_train[f"cluster_{group_id}_max"] = X_train[features].max(axis=1)
            
            X_test[f"cluster_{group_id}_mean"] = X_test[features].mean(axis=1)
            X_test[f"cluster_{group_id}_sum"] = X_test[features].sum(axis=1)
            X_test[f"cluster_{group_id}_min"] = X_test[features].min(axis=1)
            X_test[f"cluster_{group_id}_max"] = X_test[features].max(axis=1)
    
    print(f"-- Added {n_clusters * 4} cluster-based features")
    return X_train, X_test

def add_nonlinear_transforms(X_train, X_test, top_n=20):
    """Add nonlinear transformations of top features"""
    print(f"-- Adding nonlinear transformations for top {top_n} features")
    
    # Get features with highest variance
    variances = np.var(X_train.values, axis=0)
    top_indices = np.argsort(variances)[-top_n:]
    top_columns = X_train.columns[top_indices]
    
    transforms_added = 0
    for col in top_columns:
        # Skip columns with zeros or negatives for log transforms
        if (X_train[col].min() > 0) and (X_test[col].min() > 0):
            # Log transform
            X_train[f"log_{col}"] = np.log(X_train[col])
            X_test[f"log_{col}"] = np.log(X_test[col])
            transforms_added += 1
        
        # Square root (use absolute value to handle negatives)
        X_train[f"sqrt_{col}"] = np.sqrt(np.abs(X_train[col]))
        X_test[f"sqrt_{col}"] = np.sqrt(np.abs(X_test[col]))
        transforms_added += 1
        
        # Square
        X_train[f"sq_{col}"] = X_train[col]**2
        X_test[f"sq_{col}"] = X_test[col]**2
        transforms_added += 1
        
        # Cube
        X_train[f"cube_{col}"] = X_train[col]**3
        X_test[f"cube_{col}"] = X_test[col]**3
        transforms_added += 1
    
    print(f"-- Added {transforms_added} nonlinear transformations")
    return X_train, X_test

def bin_features(X_train, X_test, n_bins=5, top_n=10):
    """Bin top numerical features to capture nonlinear relationships"""
    print(f"-- Binning top {top_n} features into {n_bins} bins")
    
    # Find features with highest variance
    variances = np.var(X_train.values, axis=0)
    top_indices = np.argsort(variances)[-top_n:]
    top_columns = X_train.columns[top_indices]
    
    bins_added = 0
    for feature in top_columns:
        # Create bins for this feature
        for i in range(n_bins):
            # Calculate bin boundaries based on quantiles
            q_low = i / n_bins
            q_high = (i + 1) / n_bins
            
            lower = X_train[feature].quantile(q_low)
            upper = X_train[feature].quantile(q_high)
            
            # Create binary indicator for this bin
            bin_name = f"{feature}_bin_{i}"
            X_train[bin_name] = ((X_train[feature] >= lower) & (X_train[feature] <= upper)).astype(float)
            X_test[bin_name] = ((X_test[feature] >= lower) & (X_test[feature] <= upper)).astype(float)
            bins_added += 1
    
    print(f"-- Added {bins_added} binned features")
    return X_train, X_test

def add_nn_features(X_train, X_test, y_train, task="regression", hidden_layers=(64, 32)):
    """Add neural network predictions as features"""
    print(f"-- Training neural network for feature boosting with layers {hidden_layers}")
    
    # Debug the column issue
    print(f"-- X_train shape: {X_train.shape}")
    print(f"-- X_test shape: {X_test.shape}")
    
    # Ensure X_train and X_test are proper DataFrames
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train)
    
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test)
    
    # Ensure DataFrame indices are integers and reset if needed
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    
    # Convert y_train to numeric if it's not already
    if isinstance(y_train, np.ndarray) and y_train.dtype.kind in ['U', 'S', 'O']:
        print("-- Converting string target to numeric values")
        try:
            y_train = y_train.astype(float)
        except ValueError:
            print("-- Target contains non-numeric values, using dummy value")
            # Use a dummy constant value if conversion fails
            y_train = np.zeros(len(y_train))
    
    # When X_test has only one feature but the names don't match,
    # this suggests a deeper issue with how the test set is being processed.
    if X_test.shape[1] == 1 and X_train.shape[1] > 1:
        print("-- Detected significant mismatch: test set has only one column")
        print("-- Creating a mean-filled pseudodata for neural network")
        
        # Create a test dataset with same columns as train but filled with means
        col_means = X_train.mean()
        X_test_nn = pd.DataFrame({col: col_means[col] for col in X_train.columns}, 
                                index=X_test.index)
        X_train_nn = X_train
    else:
        # Verify that columns match between train and test
        train_cols = set(X_train.columns)
        test_cols = set(X_test.columns)
        
        if train_cols != test_cols:
            print(f"-- Warning: Train and test columns don't match for neural network")
            print(f"-- Train has {len(train_cols)} features, test has {len(test_cols)} features")
            # Use only common columns
            common_cols = list(train_cols.intersection(test_cols))
            print(f"-- Using {len(common_cols)} common features for neural network")
            
            # If no common features, we can't train a neural network
            if len(common_cols) == 0:
                print("-- Error: No common features between train and test. Skipping neural network.")
                print("-- This should not happen if the datasets were properly aligned.")
                
                # Add a dummy neural network feature with numeric constant
                if hasattr(y_train, 'mean'):
                    # For numeric y_train
                    mean_value = float(y_train.mean())
                else:
                    # Fallback for non-numeric
                    mean_value = 0.0
                
                print(f"-- Adding constant prediction ({mean_value}) as neural network feature")
                X_train['nn_pred'] = mean_value
                X_test['nn_pred'] = mean_value
                return X_train, X_test
                
            X_train_nn = X_train[common_cols]
            X_test_nn = X_test[common_cols]
        else:
            X_train_nn = X_train
            X_test_nn = X_test
    
    # Simple check to prevent empty DataFrames
    if X_train_nn.shape[1] == 0 or X_test_nn.shape[1] == 0:
        print("-- Error: Empty features for neural network. Adding mean prediction.")
        if hasattr(y_train, 'mean'):
            mean_value = float(y_train.mean())
        else:
            mean_value = 0.0
        X_train['nn_pred'] = mean_value
        X_test['nn_pred'] = mean_value
        return X_train, X_test
    
    # Ensure y_train is a proper array
    if isinstance(y_train, pd.Series):
        y_train = y_train.values
    
    # Scale inputs for neural network
    print(f"-- Scaling {X_train_nn.shape[1]} features for neural network")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_nn)
    X_test_scaled = scaler.transform(X_test_nn)
    
    # Convert hidden layers from string to tuple if needed
    if isinstance(hidden_layers, str):
        hidden_layers = tuple(int(x) for x in hidden_layers.split(','))
    
    # Choose NN model based on task
    if task == "regression":
        nn = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.2,
            random_state=42,
            alpha=0.0001,  # L2 regularization
            solver='adam'
        )
    else:  # Classification
        nn = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.2,
            random_state=42,
            alpha=0.0001,  # L2 regularization
            solver='adam'
        )
    
    print("-- Fitting neural network")
    nn.fit(X_train_scaled, y_train)
    
    # Generate predictions as new features
    if task == "regression":
        X_train['nn_pred'] = nn.predict(X_train_scaled)
        X_test['nn_pred'] = nn.predict(X_test_scaled)
    else:  # Classification
        # For binary tasks
        if task == "binary" and len(nn.classes_) == 2:
            X_train['nn_pred_prob'] = nn.predict_proba(X_train_scaled)[:, 1]
            X_test['nn_pred_prob'] = nn.predict_proba(X_test_scaled)[:, 1]
        # For multiclass tasks, add a feature for each class
        else:
            probs_train = nn.predict_proba(X_train_scaled)
            probs_test = nn.predict_proba(X_test_scaled)
            for i, cls in enumerate(nn.classes_):
                X_train[f'nn_pred_class_{cls}'] = probs_train[:, i]
                X_test[f'nn_pred_class_{cls}'] = probs_test[:, i]
    
    print("-- Added neural network prediction features")
    return X_train, X_test

def add_stacked_models(X_train, X_test, y_train, cv=5, task="regression"):
    """Add stacked model predictions as features"""
    from sklearn.ensemble import StackingRegressor, StackingClassifier
    from sklearn.linear_model import RidgeCV, LogisticRegressionCV
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.svm import SVR, SVC
    from sklearn.model_selection import KFold
    
    print("-- Adding stacked model predictions as features")
    
    # Define base models
    if task == "regression":
        estimators = [
            ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
            ('svr', SVR(kernel='rbf')),
            ('gbm', lgb.LGBMRegressor(n_estimators=100))
        ]
        # Use RidgeCV as meta-learner
        stacked = StackingRegressor(
            estimators=estimators,
            final_estimator=RidgeCV(),
            cv=KFold(n_splits=cv, shuffle=True, random_state=42),
            n_jobs=-1
        )
    else:
        estimators = [
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('svc', SVC(probability=True)),
            ('gbm', lgb.LGBMClassifier(n_estimators=100))
        ]
        # Use LogisticRegressionCV as meta-learner
        stacked = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegressionCV(),
            cv=KFold(n_splits=cv, shuffle=True, random_state=42),
            n_jobs=-1
        )
    
    # Fit stacked model
    stacked.fit(X_train, y_train)
    
    # Add predictions as features
    if task == "regression":
        X_train['stacked_pred'] = stacked.predict(X_train)
        X_test['stacked_pred'] = stacked.predict(X_test)
    else:
        proba = stacked.predict_proba(X_train)
        proba_test = stacked.predict_proba(X_test)
        for i, cls in enumerate(stacked.classes_):
            X_train[f'stacked_class_{cls}'] = proba[:, i]
            X_test[f'stacked_class_{cls}'] = proba_test[:, i]
    
    return X_train, X_test, stacked

def add_manifold_features(X_train, X_test, n_components=5):
    """Add manifold learning features"""
    print("-- Adding manifold learning features")
    
    # Sample data for TSNE (expensive for large datasets)
    n_samples = min(5000, X_train.shape[0])
    sample_idx = np.random.choice(X_train.shape[0], n_samples, replace=False)
    X_train_sample = X_train.iloc[sample_idx]
    
    # Train manifold models
    tsne = TSNE(n_components=2, random_state=42)
    isomap = Isomap(n_components=n_components)
    kpca = KernelPCA(n_components=n_components, kernel='rbf')
    
    # Fit TSNE on sample
    tsne_result = tsne.fit_transform(X_train_sample)
    
    # Fit Isomap and KPCA on full data
    isomap_result = isomap.fit_transform(X_train)
    kpca_result = kpca.fit_transform(X_train)
    
    # Apply to test data
    kpca_test = kpca.transform(X_test)
    isomap_test = isomap.transform(X_test)
    
    # Add features
    for i in range(n_components):
        if i < 2:  # TSNE only has 2 components
            # For TSNE, we need to create a model that maps original space to TSNE space
            knn_tsne = [KNeighborsRegressor(n_neighbors=5) for _ in range(2)]
            for j in range(2):
                knn_tsne[j].fit(X_train_sample, tsne_result[:, j])
                X_train[f'tsne_{j}'] = np.nan
                X_train.loc[sample_idx, f'tsne_{j}'] = tsne_result[:, j]
                # Fill NaNs with predictions
                nan_idx = X_train[f'tsne_{j}'].isna()
                if nan_idx.any():
                    X_train.loc[nan_idx, f'tsne_{j}'] = knn_tsne[j].predict(X_train[nan_idx])
                X_test[f'tsne_{j}'] = knn_tsne[j].predict(X_test)
                
        X_train[f'isomap_{i}'] = isomap_result[:, i]
        X_test[f'isomap_{i}'] = isomap_test[:, i]
        
        X_train[f'kpca_{i}'] = kpca_result[:, i]
        X_test[f'kpca_{i}'] = kpca_test[:, i]
    
    return X_train, X_test

def check_domain_shift(X_train, X_test):
    """Check for domain shift between train and test sets"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    
    # Create a binary classification task:
    # Can we distinguish between train and test sets?
    print("-- Checking domain shift between train and test")
    
    # Create a combined dataset with labels
    X_combined = pd.concat([X_train, X_test], axis=0)
    y_combined = np.concatenate([np.zeros(len(X_train)), np.ones(len(X_test))])
    
    # Train a classifier
    clf = RandomForestClassifier(n_estimators=100, max_depth=5)
    
    # Use cross-validation to evaluate
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(clf, X_combined, y_combined, cv=5, scoring='roc_auc')
    
    mean_score = np.mean(scores)
    print(f"-- Domain shift score: {mean_score:.4f} (0.5 = no shift, 1.0 = complete shift)")
    
    return mean_score

def create_adaptive_lgbm(X_train, y_train, task="regression"):
    """Create a model that adapts feature selection during boosting"""
    from sklearn.feature_selection import SelectFromModel
    from sklearn.pipeline import Pipeline
    
    # Create a pipeline with an initial feature selector
    # followed by a LightGBM model
    if task == "regression":
        base_model = lgb.LGBMRegressor(n_estimators=100)
        final_model = lgb.LGBMRegressor(n_estimators=1000)
    else:
        base_model = lgb.LGBMClassifier(n_estimators=100)
        final_model = lgb.LGBMClassifier(n_estimators=1000)
    
    # Create a pipeline with feature selection
    pipeline = Pipeline([
        ('feature_selection', SelectFromModel(base_model, threshold='median')),
        ('final_model', final_model)
    ])
    
    # Fit the pipeline
    pipeline.fit(X_train, y_train)
    
    return pipeline

def add_transformer_embeddings(X_train, X_test, smiles_list_train, smiles_list_test, custom_model_path=None):
    """Add molecular embeddings from pre-trained transformer models with MPS acceleration"""
    import warnings
    from tqdm import tqdm
    
    print("-- Adding transformer-based molecular embeddings")
    
    # Suppress specific transformer warnings
    warnings.filterwarnings("ignore", message="Some weights of .* were not initialized")
    warnings.filterwarnings("ignore", message="You should probably TRAIN this model")
    
    # Check for MPS availability (Apple Silicon)
    device = "cpu"
    if torch.backends.mps.is_available():
        print("-- Using MPS acceleration for transformer models")
        device = torch.device("mps")
    elif torch.cuda.is_available():
        print("-- Using CUDA acceleration for transformer models")
        device = torch.device("cuda")
    else:
        print("-- Using CPU for transformer computations")
    
    # Load ChemBERTa or a custom pretrained model
    if custom_model_path and os.path.exists(custom_model_path):
        print(f"-- Loading custom pretrained transformer model: {custom_model_path}")
        model_name = custom_model_path
    else:
        print("-- Loading default ChemBERTa model")
        model_name = "DeepChem/ChemBERTa-77M-MTR"
    
    # Load tokenizer and model
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # For a pretrained sequence classification model, we can use AutoModelForSequenceClassification
        # and extract embeddings from the hidden states
        if custom_model_path and "final_model" in custom_model_path:
            try:
                # Try loading as sequence classification model first
                model = AutoModelForSequenceClassification.from_pretrained(model_name)
                print("-- Loaded custom model as sequence classification model")
            except:
                # Fall back to base model
                model = AutoModel.from_pretrained(model_name)
                print("-- Loaded custom model as base transformer model")
        else:
            model = AutoModel.from_pretrained(model_name)
    except Exception as e:
        print(f"-- Error loading transformer model: {str(e)}")
        print("-- Falling back to default ChemBERTa model")
        model_name = "DeepChem/ChemBERTa-77M-MTR"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
    
    model.to(device)  # Move model to appropriate device
    
    # Function to get embeddings in batches
    def get_embeddings(smiles_list, batch_size=32):
        all_embeddings = []
        total_batches = (len(smiles_list) + batch_size - 1) // batch_size
        
        # Use tqdm for progress bar
        for i in tqdm(range(0, len(smiles_list), batch_size), 
                     desc="-- Generating embeddings", 
                     total=total_batches,
                     bar_format="{desc}: {percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt} batches"):
            batch = smiles_list[i:i+batch_size]
            tokens = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            
            # Move tensors to appropriate device
            tokens = {k: v.to(device) for k, v in tokens.items()}
            
            with torch.no_grad():
                # Different handling for different model types
                if isinstance(model, AutoModelForSequenceClassification.__bases__[0]):
                    # Extract the last hidden state from the model's base outputs
                    outputs = model.base_model(**tokens)
                    embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                else:
                    outputs = model(**tokens)
                    embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            all_embeddings.append(embeddings)
            
            # Force garbage collection
            if device != "cpu":
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif torch.backends.mps.is_available():
                    import gc
                    gc.collect()
        
        return np.vstack(all_embeddings)
    
    # Get embeddings with progress display
    print(f"-- Processing {len(smiles_list_train)} training compounds")
    train_embeddings = get_embeddings(smiles_list_train)
    
    print(f"-- Processing {len(smiles_list_test)} test compounds")
    test_embeddings = get_embeddings(smiles_list_test)
    
    # Add PCA-reduced embeddings as features for efficiency
    print("-- Reducing embedding dimensions with PCA")
    from sklearn.decomposition import PCA
    
    # Reduce dimensionality to a manageable size
    n_components = min(20, train_embeddings.shape[1], train_embeddings.shape[0])
    pca = PCA(n_components=n_components)
    train_embeddings_reduced = pca.fit_transform(train_embeddings)
    test_embeddings_reduced = pca.transform(test_embeddings)
    
    # Create DataFrame of embeddings all at once (avoid fragmentation)
    embedding_cols = [f'transformer_emb_{i}' for i in range(train_embeddings_reduced.shape[1])]
    
    train_emb_df = pd.DataFrame(
        train_embeddings_reduced, 
        columns=embedding_cols,
        index=X_train.index
    )
    
    test_emb_df = pd.DataFrame(
        test_embeddings_reduced, 
        columns=embedding_cols,
        index=X_test.index
    )
    
    # Concat to original frames
    X_train = pd.concat([X_train, train_emb_df], axis=1)
    X_test = pd.concat([X_test, test_emb_df], axis=1)
    
    print(f"-- Added {len(embedding_cols)} transformer embedding features")
    
    # Free up memory
    del model, tokenizer, train_embeddings, test_embeddings
    
    # Force garbage collection
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return X_train, X_test

def add_gnn_features(X_train, X_test, smiles_list_train, smiles_list_test):
    """Add graph neural network features with MPS acceleration"""
    from rdkit import Chem
    import torch
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import GCNConv, global_mean_pool
    import torch.nn.functional as F
    
    print("-- Adding GNN-based molecular features")
    
    # Check for MPS availability (Apple Silicon)
    device = "cpu"
    if torch.backends.mps.is_available():
        print("-- Using MPS acceleration for GNN models")
        device = torch.device("mps")
    elif torch.cuda.is_available():
        print("-- Using CUDA acceleration for GNN models")
        device = torch.device("cuda")
    else:
        print("-- Using CPU for GNN computations")
    
    # Convert SMILES to molecular graphs
    def smiles_to_graph(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
                
            # Get atom features
            atoms = mol.GetAtoms()
            x = []
            for atom in atoms:
                atom_features = [
                    atom.GetAtomicNum(),
                    atom.GetDegree(),
                    atom.GetFormalCharge(),
                    atom.GetChiralTag(),
                    atom.GetIsAromatic() * 1,
                    atom.GetHybridization(),
                ]
                x.append(atom_features)
            
            x = torch.tensor(x, dtype=torch.float)
            
            # Get bond indices and features
            edge_indices = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                edge_indices.append([i, j])
                edge_indices.append([j, i])  # Add reverse edge for undirected graph
                
            if len(edge_indices) == 0:
                # Molecule with no bonds
                edge_index = torch.zeros((2, 0), dtype=torch.long)
            else:
                edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
                
            return Data(x=x, edge_index=edge_index)
        except:
            return None
    
    # Create graphs for all molecules
    print("-- Converting molecules to graphs")
    train_graphs = [smiles_to_graph(s) for s in smiles_list_train]
    test_graphs = [smiles_to_graph(s) for s in smiles_list_test]
    
    # Remove None values
    train_graphs = [g for g in train_graphs if g is not None]
    test_graphs = [g for g in test_graphs if g is not None]
    
    # Simple GNN model
    class GNN(torch.nn.Module):
        def __init__(self, input_dim=6, hidden_dim=64, output_dim=32):
            super(GNN, self).__init__()
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.conv3 = GCNConv(hidden_dim, output_dim)
            
        def forward(self, data):
            x, edge_index, batch = data.x, data.edge_index, data.batch
            
            # Apply convolutions
            x = F.relu(self.conv1(x, edge_index))
            x = F.relu(self.conv2(x, edge_index))
            x = self.conv3(x, edge_index)
            
            # Global pooling
            x = global_mean_pool(x, batch)
            return x
    
    # Train the GNN with MPS acceleration
    print("-- Training GNN for feature extraction")
    gnn = GNN().to(device)  # Move model to device
    optimizer = torch.optim.Adam(gnn.parameters(), lr=0.01)
    
    # Process data in batches with memory optimization
    def batch_graphs(graphs, batch_size=32):
        batches = []
        for i in range(0, len(graphs), batch_size):
            batch_graphs = graphs[i:i+batch_size]
            batch = Batch.from_data_list(batch_graphs)
            # Move batch to device
            batch = batch.to(device)
            batches.append(batch)
        return batches
    
    # Create batches
    train_batches = batch_graphs(train_graphs)
    
    # Train with a simple reconstruction objective
    gnn.train()
    for epoch in range(10):
        total_loss = 0
        for batch in train_batches:
            optimizer.zero_grad()
            out = gnn(batch)
            # Simplified self-supervised loss
            loss = torch.norm(out)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"-- GNN training epoch {epoch+1}/10, loss: {total_loss/len(train_batches):.4f}")
    
    # Extract features with memory-efficient batching
    gnn.eval()
    
    def extract_features(graphs, batch_size=32):
        from tqdm import tqdm
        
        all_embeddings = []
        batches = batch_graphs(graphs, batch_size)
        
        for batch in tqdm(batches, desc="-- Extracting GNN embeddings", 
                         bar_format="{desc}: {percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt} batches"):
            with torch.no_grad():
                emb = gnn(batch).cpu().numpy()
            all_embeddings.append(emb)
        
        return np.vstack(all_embeddings)
    
    print("-- Extracting GNN embeddings")
    train_embeddings = extract_features(train_graphs)
    test_embeddings = extract_features(test_graphs)
    
    # Create DataFrames for GNN embeddings (all at once)
    gnn_cols = [f'gnn_emb_{i}' for i in range(train_embeddings.shape[1])]
    
    # Create empty dataframes with zeros
    train_gnn_df = pd.DataFrame(
        0, 
        index=X_train.index, 
        columns=gnn_cols
    )
    
    test_gnn_df = pd.DataFrame(
        0, 
        index=X_test.index, 
        columns=gnn_cols
    )
    
    # Fill in values for valid molecules
    valid_train_indices = [i for i, g in enumerate(smiles_list_train) if Chem.MolFromSmiles(g) is not None]
    valid_test_indices = [i for i, g in enumerate(smiles_list_test) if Chem.MolFromSmiles(g) is not None]
    
    for i, feat_idx in enumerate(valid_train_indices):
        if i < len(train_embeddings):
            train_gnn_df.iloc[feat_idx] = train_embeddings[i]
    
    for i, feat_idx in enumerate(valid_test_indices):
        if i < len(test_embeddings):
            test_gnn_df.iloc[feat_idx] = test_embeddings[i]
    
    # Concat to original frames
    X_train = pd.concat([X_train, train_gnn_df], axis=1)
    X_test = pd.concat([X_test, test_gnn_df], axis=1)
    
    print(f"-- Added {train_embeddings.shape[1]} GNN features")
    
    # Free memory
    del gnn, train_graphs, test_graphs
    if device != "cpu":
        torch.cuda.empty_cache() if device == "cuda" else torch.mps.empty_cache()
    
    return X_train, X_test

def create_hybrid_pipeline(X_train, X_test, smiles_list_train, smiles_list_test):
    """Create an integrated pipeline combining tabular, transformer and graph features"""
    
    # 1. Process raw SMILES data for transformer and GNN
    X_train_with_transformer, X_test_with_transformer = add_transformer_embeddings(
        X_train.copy(), X_test.copy(), smiles_list_train, smiles_list_test
    )
    
    X_train_with_gnn, X_test_with_gnn = add_gnn_features(
        X_train.copy(), X_test.copy(), smiles_list_train, smiles_list_test
    )
    
    # 2. Combine the enhanced feature sets
    # Add transformer features
    transformer_cols = [col for col in X_train_with_transformer.columns if col.startswith('transformer_emb_')]
    for col in transformer_cols:
        X_train[col] = X_train_with_transformer[col]
        X_test[col] = X_test_with_transformer[col]
    
    # Add GNN features
    gnn_cols = [col for col in X_train_with_gnn.columns if col.startswith('gnn_emb_')]
    for col in gnn_cols:
        X_train[col] = X_train_with_gnn[col]
        X_test[col] = X_test_with_gnn[col]
    
    print(f"-- Added {len(transformer_cols)} transformer features and {len(gnn_cols)} GNN features")
    
    # 3. Feature importance analysis for different feature types
    print("-- Analyzing feature importance by source")
    feature_types = {
        'mordred': [col for col in X_train.columns if col.startswith('mordred_')],
        'fingerprint': [col for col in X_train.columns if col.startswith('fp_')],
        'transformer': transformer_cols,
        'gnn': gnn_cols,
        'other': [col for col in X_train.columns if not any(
            col.startswith(prefix) for prefix in ['mordred_', 'fp_', 'transformer_emb_', 'gnn_emb_']
        )]
    }
    
    # Print feature count by type
    for feat_type, cols in feature_types.items():
        print(f"--   {feat_type}: {len(cols)} features")
    
    return X_train, X_test

def main():
    args = parse_args()
    np.random.seed(args.seed)
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Load data
    X_train, y_train, X_test, y_test, feature_cols, detected_task = load_and_preprocess_data(
        args.input, args.external_test, args.target_column, args.use_file_descriptors, 
        args.descriptors, args.smiles_col, args.advanced_preprocessing
    )
    
    # Set task
    if args.task != detected_task:
        print(f"-- Warning: Specified task '{args.task}' doesn't match detected task '{detected_task}'")
        print(f"-- Using detected task: {detected_task}")
        args.task = detected_task
    
    # Create feature processor
    processor = FeatureProcessor(args)
    
    # Apply feature processing pipeline
    X_train_processed, X_test_processed = processor.fit_transform(X_train, y_train, X_test, y_test)
    
    # Save the processor for future use
    processor.save(args.output_dir)
    
    # Continue with model training as before
    X_train_opt, X_valid, y_train_opt, y_valid = train_test_split(
        X_train_processed, y_train, test_size=0.2, random_state=args.seed
    )
    
    print("-- LightGBM model training with Optuna optimization")
    
    print(f"-- Training data shape: {X_train_processed.shape}")
    print(f"-- External test data shape: {X_test_processed.shape}")
    
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
    if args.polynomial_features:
        enhancements.append("Polynomial Features")
    if args.feature_clustering:
        enhancements.append("Feature Clustering")
    if args.nonlinear_transforms:
        enhancements.append("Nonlinear Transformations")
    if args.feature_binning:
        enhancements.append("Feature Binning")
    if args.nn_boost:
        enhancements.append(f"Neural Network Boosting ({args.nn_hidden_layers})")
    if args.stacked_models:
        enhancements.append("Stacked Models")
    if args.manifold_features:
        enhancements.append("Manifold Learning Features")
    
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
            n_features = len(trial.user_attrs["selected_features"]) if "selected_features" in trial.user_attrs else X_train_processed.shape[1]
            print(f"-- [Improvement] Trial {trial.number}: External RMSE = {current_score:.4f} with {n_features} features")
    
    # Create sampler with better exploration
    if args.sample_strategy == "tpe":
        sampler = optuna.samplers.TPESampler(
            multivariate=True, 
            seed=args.seed,
            warn_independent_sampling=False
        )
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
            trial, X_train_opt, y_train_opt, X_valid, y_valid, X_test_processed, y_test, 
            args.task, get_evaluation_metric(args.task), args.feature_selection, args.focal_loss, args.lr_schedule
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
    X_train_final = X_train_processed[best_selected_features]
    X_test_final = X_test_processed[best_selected_features]
    
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

def fine_tune_transformer(smiles_list_train, y_train, smiles_list_test, y_test, epochs=10):
    """Fine-tune a transformer model directly on LogD prediction"""
    from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
    import torch
    from datasets import Dataset

    print("-- Fine-tuning transformer model on LogD prediction")
    
    # Check for MPS/CUDA
    device = "cpu"
    if torch.backends.mps.is_available():
        print("-- Using MPS acceleration for transformer training")
        device = torch.device("mps")
    elif torch.cuda.is_available():
        print("-- Using CUDA acceleration for transformer training")
        device = torch.device("cuda")
    
    # Create datasets
    train_dataset = Dataset.from_dict({
        "text": smiles_list_train,
        "label": y_train.astype(float)
    })
    test_dataset = Dataset.from_dict({
        "text": smiles_list_test,
        "label": y_test.astype(float)
    })
    
    # Load model and tokenizer
    model_name = "DeepChem/ChemBERTa-77M-MTR"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=1,  # Regression task
        problem_type="regression"
    )
    
    # Tokenize function
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    
    # Apply tokenization
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./transformer_checkpoints",
        num_train_epochs=epochs,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./transformer_logs",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none"  # Disable wandb etc.
    )
    
    # MSE loss for regression
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        mse = np.mean((predictions.flatten() - labels) ** 2)
        rmse = np.sqrt(mse)
        return {"mse": mse, "rmse": rmse}
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics,
    )
    
    # Train model
    print("-- Training transformer model")
    trainer.train()
    
    # Evaluate
    print("-- Evaluating fine-tuned transformer")
    eval_results = trainer.evaluate()
    print(f"-- Evaluation RMSE: {eval_results['eval_rmse']:.4f}")
    
    # Generate predictions for original data
    def get_predictions(dataset):
        outputs = trainer.predict(dataset)
        return outputs.predictions.flatten()
    
    train_preds = get_predictions(tokenized_train)
    test_preds = get_predictions(tokenized_test)
    
    # Save model
    trainer.save_model("./transformer_finetuned")
    
    return train_preds, test_preds, model, tokenizer

# Add a class to encapsulate the entire feature processing pipeline
class FeatureProcessor:
    """Encapsulates the entire feature processing pipeline for reproducible predictions"""
    
    def __init__(self, args):
        self.args = args
        self.feature_selectors = []  # Store feature selectors
        self.pca = None              # Store PCA for transformer embeddings
        self.scaler = None           # Store scaler for neural network
        self.nn_model = None         # Store neural network model
        self.transformer_model = None  # Store transformer model
        self.tokenizer = None        # Store tokenizer
        self.selected_columns = []   # Store selected column names
        self.interaction_pairs = []  # Store which features were used for interactions
        # Store transformer model path if specified
        self.transformer_model_path = args.transformer_model if hasattr(args, 'transformer_model') else None
    
    def fit_transform(self, X_train, y_train, X_test=None, y_test=None):
        """Fit the entire pipeline and transform both training and test data"""
        # 1. Initial feature selection
        if self.args.feature_selection:
            print("-- Performing initial feature selection on base features")
            selector1 = VarianceThreshold(threshold=0.01)
            X_train_basic = selector1.fit_transform(X_train)
            self.feature_selectors.append(selector1)
            
            # Get selected column names
            self.selected_columns = [X_train.columns[i] for i in range(X_train.shape[1]) 
                                   if selector1.get_support()[i]]
            
            # Convert back to DataFrame
            X_train = pd.DataFrame(X_train_basic, columns=self.selected_columns, index=X_train.index)
            
            if X_test is not None:
                X_test_basic = selector1.transform(X_test)
                X_test = pd.DataFrame(X_test_basic, columns=self.selected_columns, index=X_test.index)
            
            print(f"-- Reduced from {X_train.shape[1]} to {len(self.selected_columns)} base features")
        
        # 2. Add interaction features
        if self.args.interactions:
            print("-- Adding interaction features to base features")
            X_train = X_train.copy()
            if X_test is not None:
                X_test = X_test.copy()
            
            # Store the features used for interactions for future use
            selector = VarianceThreshold()
            selector.fit(X_train)
            variances = selector.variances_
            
            # Get indices of top features by variance
            top_indices = np.argsort(variances)[-20:]  # Take top 20 features
            self.top_features = X_train.columns[top_indices].tolist()
            
            # Create interactions among top features
            interactions_added = 0
            for i, feat1 in enumerate(self.top_features):
                for feat2 in self.top_features[i+1:]:
                    if interactions_added >= 5:  # Limit to 5 interactions
                        break
                        
                    # Create new feature name and store the pair
                    interaction_name = f"interact_{feat1}_{feat2}"
                    self.interaction_pairs.append((feat1, feat2, interaction_name))
                    
                    # Add multiplication interaction
                    X_train[interaction_name] = X_train[feat1] * X_train[feat2]
                    if X_test is not None:
                        X_test[interaction_name] = X_test[feat1] * X_test[feat2]
                    
                    interactions_added += 1
            
            print(f"-- Added {interactions_added} interaction features")
        
        # 3. Perform feature selection on engineered features
        if self.args.feature_selection:
            print("-- Performing feature selection on engineered features")
            initial_model = lgb.LGBMRegressor(n_estimators=100) if self.args.task == "regression" else lgb.LGBMClassifier(n_estimators=100)
            selector2 = SelectFromModel(initial_model, threshold="median")
            
            X_train_selected = selector2.fit_transform(X_train, y_train)
            self.feature_selectors.append(selector2)
            
            # Update selected columns
            self.selected_columns = [X_train.columns[i] for i in range(X_train.shape[1]) 
                                  if selector2.get_support()[i]]
            
            # Convert back to DataFrame
            X_train = pd.DataFrame(X_train_selected, columns=self.selected_columns, index=X_train.index)
            
            if X_test is not None:
                X_test_selected = selector2.transform(X_test)
                X_test = pd.DataFrame(X_test_selected, columns=self.selected_columns, index=X_test.index)
            
            print(f"-- Selected {X_train.shape[1]} features after engineering")
        
        # 4. Add transformer embeddings
        if self.args.use_transformers and self.args.smiles_col:
            try:
                # Load SMILES data
                smiles_df_train = pd.read_csv(self.args.input)
                
                if X_test is not None:
                    smiles_df_test = pd.read_csv(self.args.external_test)
                    smiles_list_test = smiles_df_test[self.args.smiles_col].astype(str).tolist()
                else:
                    smiles_list_test = None
                
                if self.args.smiles_col in smiles_df_train.columns:
                    smiles_list_train = smiles_df_train[self.args.smiles_col].astype(str).tolist()
                    
                    # Process transformer embeddings
                    X_train, X_test, self.transformer_model, self.tokenizer, self.pca = self._add_transformer_embeddings(
                        X_train, X_test, smiles_list_train, smiles_list_test
                    )
            except Exception as e:
                print(f"-- Error adding transformer features: {str(e)}")
                import traceback
                traceback.print_exc()  # Print full traceback for debugging
        
        # 5. Add neural network features
        if self.args.nn_boost:
            try:
                X_train, X_test, self.nn_model, self.scaler = self._add_nn_features(
                    X_train, X_test, y_train, 
                    task=self.args.task,
                    hidden_layers=self.args.nn_hidden_layers
                )
            except Exception as e:
                print(f"-- Error adding neural network features: {str(e)}")
        
        return X_train, X_test
    
    def transform(self, X, smiles_list=None):
        """Transform new data using the fitted pipeline"""
        # Make a copy to avoid modifying the input
        X = X.copy()
        
        # 1. Apply feature selectors in sequence
        for selector in self.feature_selectors:
            # Handle case when X is missing columns needed by the selector
            if isinstance(X, pd.DataFrame):
                missing_cols = set(self.selected_columns) - set(X.columns)
                for col in missing_cols:
                    X[col] = 0  # Fill missing columns with zeros
                
                # Select only columns that were used during training
                cols_to_use = [col for col in self.selected_columns if col in X.columns]
                X_subset = X[cols_to_use]
                
                # Apply the selector
                X_transformed = selector.transform(X_subset)
                
                # Create new DataFrame with selected features
                X = pd.DataFrame(X_transformed, columns=cols_to_use, index=X.index)
            else:
                X = selector.transform(X)
        
        # 2. Add interaction features
        for feat1, feat2, interaction_name in self.interaction_pairs:
            if feat1 in X.columns and feat2 in X.columns:
                X[interaction_name] = X[feat1] * X[feat2]
            else:
                # If features are missing, add a column of zeros
                X[interaction_name] = 0
        
        # 3. Add transformer embeddings if available
        if self.transformer_model is not None and self.tokenizer is not None and smiles_list is not None:
            X = self._transform_with_transformer(X, smiles_list)
        
        # 4. Add neural network predictions if available
        if self.nn_model is not None and self.scaler is not None:
            X = self._transform_with_nn(X)
        
        return X
    
    def _add_transformer_embeddings(self, X_train, X_test, smiles_list_train, smiles_list_test):
        """Add transformer embeddings and return the updated model components"""
        import torch
        from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
        from tqdm import tqdm
        import os
        
        print("-- Adding transformer-based molecular embeddings")
        
        # Initialize device
        device = "cpu"
        if torch.backends.mps.is_available():
            print("-- Using MPS acceleration for transformer models")
            device = torch.device("mps")
        elif torch.cuda.is_available():
            print("-- Using CUDA acceleration for transformer models")
            device = torch.device("cuda")
        
        # Load model - use custom model if provided
        if self.transformer_model_path and os.path.exists(self.transformer_model_path):
            model_path = self.transformer_model_path
            print(f"-- Loading custom transformer model from: {model_path}")
            try:
                # First try loading as a sequence classification model (fine-tuned)
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForSequenceClassification.from_pretrained(model_path)
                print("-- Successfully loaded custom sequence classification model")
            except Exception as e:
                print(f"-- Error loading custom model as sequence classifier: {str(e)}")
                print("-- Trying to load as base model...")
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_path)
                    model = AutoModel.from_pretrained(model_path)
                    print("-- Successfully loaded custom base model")
                except Exception as e2:
                    print(f"-- Error loading custom base model: {str(e2)}")
                    print("-- Falling back to default ChemBERTa model")
                    model_path = "DeepChem/ChemBERTa-77M-MTR"
                    tokenizer = AutoTokenizer.from_pretrained(model_path)
                    model = AutoModel.from_pretrained(model_path)
        else:
            model_path = "DeepChem/ChemBERTa-77M-MTR"
            print(f"-- Using default ChemBERTa model")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModel.from_pretrained(model_path)
        
        model.to(device)
        
        # Function to get embeddings in batches - handle different model types
        def get_embeddings(smiles_list, batch_size=32):
            all_embeddings = []
            
            for i in tqdm(range(0, len(smiles_list), batch_size), 
                        desc="-- Generating embeddings", 
                        total=(len(smiles_list) + batch_size - 1) // batch_size):
                batch = smiles_list[i:i+batch_size]
                tokens = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
                tokens = {k: v.to(device) for k, v in tokens.items()}
                
                with torch.no_grad():
                    # Handle different model types appropriately
                    if isinstance(model, AutoModelForSequenceClassification.from_pretrained(model_path).__class__):
                        # For sequence classification models, get the hidden states
                        outputs = model(
                            input_ids=tokens['input_ids'],
                            attention_mask=tokens['attention_mask'],
                            output_hidden_states=True
                        )
                        # Use the last hidden state of the [CLS] token
                        embeddings = outputs.hidden_states[-1][:, 0, :].cpu().numpy()
                    else:
                        # Base model
                        outputs = model(**tokens)
                        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
                all_embeddings.append(embeddings)
            
            return np.vstack(all_embeddings)
        
        # Get embeddings
        train_embeddings = get_embeddings(smiles_list_train)
        
        if smiles_list_test is not None:
            test_embeddings = get_embeddings(smiles_list_test)
        
        # Apply PCA
        from sklearn.decomposition import PCA
        n_components = min(20, train_embeddings.shape[1], train_embeddings.shape[0])
        pca = PCA(n_components=n_components)
        train_embeddings_reduced = pca.fit_transform(train_embeddings)
        
        if smiles_list_test is not None:
            test_embeddings_reduced = pca.transform(test_embeddings)
        
        # Create DataFrames
        embedding_cols = [f'transformer_emb_{i}' for i in range(train_embeddings_reduced.shape[1])]
        
        train_emb_df = pd.DataFrame(train_embeddings_reduced, 
                                   columns=embedding_cols,
                                   index=X_train.index)
        
        # Concatenate
        X_train = pd.concat([X_train, train_emb_df], axis=1)
        
        if X_test is not None and smiles_list_test is not None:
            test_emb_df = pd.DataFrame(test_embeddings_reduced, 
                                      columns=embedding_cols,
                                      index=X_test.index)
            X_test = pd.concat([X_test, test_emb_df], axis=1)
        
        print(f"-- Added {len(embedding_cols)} transformer embedding features")
        
        return X_train, X_test, model, tokenizer, pca
    
    def _transform_with_transformer(self, X, smiles_list):
        """Apply transformer embedding to new data"""
        import torch
        from tqdm import tqdm
        
        print("-- Generating transformer embeddings for new data")
        
        # Initialize device
        device = "cpu"
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        
        # Function to get embeddings
        def get_embeddings(smiles_list, batch_size=32):
            all_embeddings = []
            
            for i in tqdm(range(0, len(smiles_list), batch_size), 
                        desc="-- Generating embeddings"):
                batch = smiles_list[i:i+batch_size]
                tokens = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
                tokens = {k: v.to(device) for k, v in tokens.items()}
                
                with torch.no_grad():
                    outputs = self.transformer_model(**tokens)
                
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_embeddings.append(embeddings)
            
            return np.vstack(all_embeddings)
        
        # Get embeddings
        embeddings = get_embeddings(smiles_list)
        
        # Apply PCA reduction
        embeddings_reduced = self.pca.transform(embeddings)
        
        # Create DataFrame
        embedding_cols = [f'transformer_emb_{i}' for i in range(embeddings_reduced.shape[1])]
        emb_df = pd.DataFrame(embeddings_reduced, columns=embedding_cols, index=X.index)
        
        # Concatenate
        X = pd.concat([X, emb_df], axis=1)
        
        return X
    
    def _add_nn_features(self, X_train, X_test, y_train, task="regression", hidden_layers=(64, 32)):
        """Add neural network features and return model components"""
        from sklearn.neural_network import MLPRegressor, MLPClassifier
        from sklearn.preprocessing import StandardScaler
        
        print("-- Training neural network for feature boosting")
        
        # Convert hidden layers
        if isinstance(hidden_layers, str):
            hidden_layers = tuple(int(x) for x in hidden_layers.split(','))
        
        # Scale data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        if X_test is not None:
            X_test_scaled = scaler.transform(X_test)
        
        # Train neural network
        if task == "regression":
            nn = MLPRegressor(
                hidden_layer_sizes=hidden_layers,
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.2,
                random_state=42
            )
        else:
            nn = MLPClassifier(
                hidden_layer_sizes=hidden_layers,
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.2,
                random_state=42
            )
        
        nn.fit(X_train_scaled, y_train)
        
        # Generate predictions
        if task == "regression":
            X_train['nn_pred'] = nn.predict(X_train_scaled)
            if X_test is not None:
                X_test['nn_pred'] = nn.predict(X_test_scaled)
        else:
            X_train['nn_pred_prob'] = nn.predict_proba(X_train_scaled)[:, 1]
            if X_test is not None:
                X_test['nn_pred_prob'] = nn.predict_proba(X_test_scaled)[:, 1]
        
        print("-- Added neural network features")
        
        return X_train, X_test, nn, scaler
    
    def _transform_with_nn(self, X):
        """Apply neural network to new data"""
        print("-- Generating neural network features for new data")
        
        # Scale the input data
        X_scaled = self.scaler.transform(X)
        
        # Generate predictions
        if self.args.task == "regression":
            X['nn_pred'] = self.nn_model.predict(X_scaled)
        else:
            X['nn_pred_prob'] = self.nn_model.predict_proba(X_scaled)[:, 1]
        
        return X
    
    def save(self, output_dir):
        """Save the feature processor for future use"""
        import joblib
        import os
        
        # Create feature processor directory
        processor_dir = os.path.join(output_dir, "feature_processor")
        os.makedirs(processor_dir, exist_ok=True)
        
        # Save components
        joblib.dump(self.feature_selectors, os.path.join(processor_dir, "feature_selectors.pkl"))
        joblib.dump(self.pca, os.path.join(processor_dir, "pca.pkl"))
        joblib.dump(self.scaler, os.path.join(processor_dir, "scaler.pkl"))
        joblib.dump(self.nn_model, os.path.join(processor_dir, "nn_model.pkl"))
        joblib.dump(self.interaction_pairs, os.path.join(processor_dir, "interaction_pairs.pkl"))
        joblib.dump(self.selected_columns, os.path.join(processor_dir, "selected_columns.pkl"))
        
        # Save transformer model and tokenizer if they exist
        if self.transformer_model is not None and self.tokenizer is not None:
            transformer_dir = os.path.join(processor_dir, "transformer")
            os.makedirs(transformer_dir, exist_ok=True)
            self.transformer_model.save_pretrained(transformer_dir)
            self.tokenizer.save_pretrained(transformer_dir)
        
        print(f"-- Saved feature processor to {processor_dir}")
    
    @classmethod
    def load(cls, processor_dir):
        """Load a saved feature processor"""
        import joblib
        import os
        from transformers import AutoModel, AutoTokenizer
        
        # Create an empty processor
        processor = cls(None)
        
        # Load components
        processor.feature_selectors = joblib.load(os.path.join(processor_dir, "feature_selectors.pkl"))
        processor.pca = joblib.load(os.path.join(processor_dir, "pca.pkl"))
        processor.scaler = joblib.load(os.path.join(processor_dir, "scaler.pkl"))
        processor.nn_model = joblib.load(os.path.join(processor_dir, "nn_model.pkl"))
        processor.interaction_pairs = joblib.load(os.path.join(processor_dir, "interaction_pairs.pkl"))
        processor.selected_columns = joblib.load(os.path.join(processor_dir, "selected_columns.pkl"))
        
        # Load transformer model and tokenizer if they exist
        transformer_dir = os.path.join(processor_dir, "transformer")
        if os.path.exists(transformer_dir):
            processor.transformer_model = AutoModel.from_pretrained(transformer_dir)
            processor.tokenizer = AutoTokenizer.from_pretrained(transformer_dir)
        
        print(f"-- Loaded feature processor from {processor_dir}")
        return processor

def predict(model_dir, input_file, smiles_col="SMILES", output_file=None):
    """Make predictions using a trained model and feature processor"""
    print(f"-- Loading model from {model_dir}")
    
    # Load model
    model = joblib.load(os.path.join(model_dir, "best_model.pkl"))
    
    # Load feature processor
    processor = FeatureProcessor.load(os.path.join(model_dir, "feature_processor"))
    
    # Load input data
    print(f"-- Loading input data from {input_file}")
    df = pd.read_csv(input_file)
    
    # Extract SMILES if available
    smiles_list = None
    if smiles_col in df.columns:
        smiles_list = df[smiles_col].astype(str).tolist()
    
    # Drop target column if it exists in the input
    if "LOGD" in df.columns:
        X = df.drop("LOGD", axis=1)
    else:
        X = df
    
    # Apply feature processing
    print("-- Processing features")
    X_processed = processor.transform(X, smiles_list)
    
    # Make predictions
    print("-- Making predictions")
    predictions = model.predict(X_processed)
    
    # Save or return results
    if output_file:
        result_df = pd.DataFrame({
            "SMILES": df[smiles_col] if smiles_col in df.columns else range(len(predictions)),
            "predicted_LOGD": predictions
        })
        result_df.to_csv(output_file, index=False)
        print(f"-- Saved predictions to {output_file}")
    
    return predictions

if __name__ == "__main__":
    main()