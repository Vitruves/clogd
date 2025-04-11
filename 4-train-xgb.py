#!/usr/bin/env python3
import os
import sys
import time
import json
import yaml
import joblib
import argparse
import datetime
import platform
import subprocess
import numpy as np
import pandas as pd
import xgboost as xgb
import sklearn.metrics
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

try:
    import optuna
except ImportError:
    pass

def parse_arguments():
    parser = argparse.ArgumentParser(description="XGBoost Training Tool")
    parser.add_argument("--input", type=str, help="Path to input CSV file")
    parser.add_argument("--output_dir", type=str, default="./models", help="Directory for output files")
    parser.add_argument("--target", type=str, help="Target column name")
    parser.add_argument("--id_col", type=str, default=None, help="ID column name")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set size")
    parser.add_argument("--task", type=str, default="regression", choices=["regression", "classification", "ranking", "survival"])
    parser.add_argument("--objective", type=str, default=None)
    parser.add_argument("--smiles_cols", type=str, default="SMILES", help="SMILES column names, comma-separated")
    parser.add_argument("--keep_cols", type=str, default="", help="Additional columns to preserve, comma-separated")
    parser.add_argument("--missing_threshold", type=float, default=0.1, help="Maximum fraction of missing values allowed")
    parser.add_argument("--n_estimators", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--metric", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--config", type=str)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "gpu", "cuda", "mps"])
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--auto_tune", action="store_true")
    parser.add_argument("--tune_trials", type=int, default=10)
    parser.add_argument("--early_stopping", type=int, default=10)
    parser.add_argument("--subsample", type=float, default=1.0)
    parser.add_argument("--colsample_bytree", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0)
    parser.add_argument("--min_child_weight", type=float, default=1)
    parser.add_argument("--reg_alpha", type=float, default=0)
    parser.add_argument("--reg_lambda", type=float, default=1)
    parser.add_argument("--ensemble", action="store_true")
    parser.add_argument("--ensemble_size", type=int, default=5)
    parser.add_argument("--feature_selection", action="store_true")
    parser.add_argument("--max_features", type=int, default=100)
    parser.add_argument("--scale_features", action="store_true")
    parser.add_argument("--export_validation", action="store_true")
    parser.add_argument("--external_test_set", type=str, default=None)
    parser.add_argument("--tune_optuna", action="store_true")
    parser.add_argument("--optuna_trials", type=int, default=50)
    parser.add_argument("--optuna_timeout", type=int, default=None)
    parser.add_argument("--optuna_sampler", choices=["tpe", "random"], default="tpe")
    parser.add_argument("--optuna_pruner", choices=["median", "none"], default="median")
    args = parser.parse_args()
    if args.cv < 0:
        parser.error("--cv must be >= 0")
    return args

def log_message(msg):
    print(f"-- {msg}")
    sys.stdout.flush()

def detect_device(device=None, use_gpu=False):
    if device:
        if device == "cpu":
            log_message("Using CPU as specified")
            return "hist", None
        elif device in ["gpu", "cuda", "mps"]:
            pass
        else:
            log_message(f"Unknown device '{device}', falling back to auto-detection")
    if use_gpu or device in ["gpu", "cuda", "mps"]:
        gpu_available = False
        try:
            test_model = xgb.XGBRegressor(tree_method='gpu_hist')
            sample_data = np.array([[1, 2], [3, 4]])
            sample_labels = np.array([0, 1])
            test_model.fit(sample_data, sample_labels)
            gpu_available = True
        except xgb.core.XGBoostError as e:
            if "not compiled with GPU support" in str(e):
                log_message("WARNING: XGBoost not compiled with GPU support")
            else:
                pass
        except:
            pass
        if not gpu_available:
            log_message("XGBoost was not compiled with GPU support, falling back to CPU")
            return "hist", None
        is_apple_silicon = (platform.system() == "Darwin" and (platform.processor() == "arm" or "Apple M" in platform.processor()))
        if is_apple_silicon:
            try:
                import torch
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    log_message("Apple Silicon GPU detected with Metal support")
                    return "gpu_hist", 0
                else:
                    log_message("Apple Silicon detected but Metal acceleration not available")
            except ImportError:
                pass
        try:
            n_gpus = xgb.config.get_config().get("n_gpus", 0)
            if n_gpus > 0:
                log_message(f"NVIDIA GPU detected with {n_gpus} devices")
                return "gpu_hist", 0
        except:
            pass
        if device in ["gpu", "cuda", "mps"]:
            log_message(f"WARNING: {device} was requested but could not be configured")
            log_message("Proceeding with CPU training")
    log_message("Using CPU for training")
    return "hist", None

def get_objectives_by_task():
    return {
        "regression": [
            "reg:squarederror",
            "reg:pseudohubererror",
            "reg:logistic",
            "reg:absoluteerror",
            "reg:gamma",
            "reg:tweedie"
        ],
        "classification": [
            "binary:logistic",
            "multi:softmax",
            "multi:softprob",
            "binary:hinge",
            "binary:logitraw"
        ],
        "ranking": [
            "rank:pairwise",
            "rank:ndcg",
            "rank:map"
        ]
    }

def get_default_objective(task):
    objectives = get_objectives_by_task()
    if task in objectives:
        return objectives[task][0]
    return "reg:squarederror"

def get_default_metric(task, objective=None):
    metrics_map = {
        "regression": "rmse",
        "classification": "logloss" if objective == "binary:logistic" else "mlogloss",
        "ranking": "ndcg"
    }
    return metrics_map.get(task, "rmse")

def load_data(file_path):
    log_message(f"Loading data from {file_path}")
    try:
        extension = os.path.splitext(file_path)[1].lower()
        if extension == ".csv":
            df = pd.read_csv(file_path)
        elif extension == ".parquet":
            df = pd.read_parquet(file_path)
        elif extension in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path)
        elif extension == ".json":
            df = pd.read_json(file_path)
        elif extension == ".feather":
            df = pd.read_feather(file_path)
        else:
            raise ValueError(f"Unsupported file extension: {extension}")
        log_message(f"Loaded dataset with shape: {df.shape}")
        return df
    except Exception as e:
        log_message(f"Error loading data: {str(e)}")
        sys.exit(1)

def analyze_data_quality(df):
    stats = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "missing_values": {},
        "data_types": {}
    }
    for col in df.columns:
        missing = df[col].isna().sum()
        if missing > 0:
            stats["missing_values"][col] = {"count": int(missing), "percentage": float(missing / len(df) * 100)}
        stats["data_types"][col] = str(df[col].dtype)
    return stats

def clean_dataset(df, config):
    original_shape = df.shape
    cleaning_report = {
        "original_shape": original_shape,
        "operations": [],
        "removed_columns": [],
        "removed_rows": 0
    }
    missing_threshold = config.get("missing_threshold", 0.1)
    cols_to_remove = []
    for col in df.columns:
        missing_pct = df[col].isna().sum() / len(df)
        if missing_pct > missing_threshold:
            cols_to_remove.append(col)
    for col in cols_to_remove:
        df = df.drop(columns=[col])
        cleaning_report["operations"].append({"operation": "remove_column", "column": col, "reason": "high_missing"})
        cleaning_report["removed_columns"].append(col)
    constant_columns = []
    for col in df.columns:
        if df[col].nunique() == 1:
            constant_columns.append(col)
    for col in constant_columns:
        df = df.drop(columns=[col])
        cleaning_report["operations"].append({"operation": "remove_column", "column": col, "reason": "constant_column"})
        cleaning_report["removed_columns"].append(col)
    rows_with_na = df.isna().any(axis=1).sum()
    if rows_with_na > 0:
        numeric_cols = df.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
    cleaning_report["final_shape"] = df.shape
    return df, cleaning_report

def preprocess_data(df, target_col, preserve_cols, task_type, scale_features):
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found")
    preserved_data = {}
    for col in preserve_cols:
        if col in df.columns:
            preserved_data[col] = df[col].copy()
    y = df[target_col].values
    X_df = df.drop(columns=[target_col])
    for col in preserve_cols:
        if col in X_df.columns:
            X_df = X_df.drop(columns=[col])
    numeric_cols = X_df.select_dtypes(include=np.number).columns.tolist()
    non_numeric_cols = [col for col in X_df.columns if col not in numeric_cols]
    if non_numeric_cols:
        X_df = X_df.drop(columns=non_numeric_cols)
    label_encoder = None
    if task_type == "classification":
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
    scaler = None
    if scale_features and X_df.shape[1] > 0:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_df)
        X_df = pd.DataFrame(X_scaled, columns=X_df.columns, index=X_df.index)
    return X_df, y, preserved_data, label_encoder, scaler

def align_features(X_train_features, X_test_df):
    """
    Aligns test features with training features by:
    1. Adding missing columns with zeros
    2. Removing extra columns not present in training
    3. Ensuring column order matches
    
    Args:
        X_train_features: List of feature names used during model training
        X_test_df: DataFrame containing test features
        
    Returns:
        DataFrame with aligned features
    """
    log_message(f"Aligning test features with training features")
    
    # Get the current test features
    test_features = list(X_test_df.columns)
    
    # Identify missing and extra features
    missing_features = set(X_train_features) - set(test_features)
    extra_features = set(test_features) - set(X_train_features)
    
    # Log the feature differences
    if missing_features:
        log_message(f"Adding {len(missing_features)} missing features to test set")
    if extra_features:
        log_message(f"Removing {len(extra_features)} extra features from test set")
    
    # Create a dictionary to hold all feature data
    feature_dict = {}
    
    # Add features in the correct order
    for feature in X_train_features:
        if feature in X_test_df.columns:
            feature_dict[feature] = X_test_df[feature]
        else:
            feature_dict[feature] = pd.Series(0.0, index=X_test_df.index)
    
    # Create DataFrame all at once to avoid fragmentation
    aligned_df = pd.DataFrame(feature_dict, index=X_test_df.index)
    
    return aligned_df

def preprocess_external_test(external_df, features, categorical_features, scaler=None, label_encoder=None):
    """
    Preprocess external test data in the same way as the training data
    
    Args:
        external_df: DataFrame containing external test data
        features: List of feature names used in training
        categorical_features: List of categorical features
        scaler: StandardScaler fitted on training data (or None)
        label_encoder: LabelEncoder for target (optional)
        
    Returns:
        Tuple of (X_test_scaled, X_test_df, preserved_data)
    """
    # Check if required SMILES column exists
    if 'SMILES' not in external_df.columns:
        raise ValueError("External test data must have a 'SMILES' column")
    
    # Extract features
    X_test_df = pd.DataFrame()
    for feature in features:
        if feature in external_df.columns:
            X_test_df[feature] = external_df[feature]
    
    # Handle categorical features if any
    if categorical_features and label_encoder:
        for cat_feature in categorical_features:
            if cat_feature in X_test_df.columns and cat_feature in label_encoder:
                # Apply label encoding using the same encoder as during training
                X_test_df[cat_feature] = label_encoder[cat_feature].transform(
                    external_df[cat_feature].astype(str).fillna('unknown')
                )
    
    # Align test features with training features
    X_test_df = align_features(features, X_test_df)
    
    # Apply the same scaling as for training data if scaler is provided
    if scaler is not None:
        X_test_scaled = scaler.transform(X_test_df)
    else:
        # If no scaler, just convert to numpy array
        log_message("No scaler available, using raw feature values")
        X_test_scaled = X_test_df.values
    
    # Create preserved data dict with SMILES
    preserved_data = {'SMILES': external_df['SMILES'].values}
    
    # Return the preprocessed test data
    return X_test_scaled, X_test_df, preserved_data

def check_data_sanity(X, y, name):
    log_message(f"Data sanity check for {name}")
    log_message(f"Shape: {X.shape}")
    log_message(f"Target stats -> mean: {float(np.mean(y)):.4f}, min: {float(np.min(y)):.4f}, max: {float(np.max(y)):.4f}")
    log_message(f"Feature NaN count: {X.isna().sum().sum()}")

def select_features(X, y, max_features=100, task_type="regression"):
    if X.shape[1] <= max_features:
        return X
    if task_type == "regression":
        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, subsample=0.8, tree_method="hist", verbosity=0)
    else:
        model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, subsample=0.8, tree_method="hist", verbosity=0)
    model.fit(X, y)
    importance = model.feature_importances_
    feature_importance = pd.Series(importance, index=X.columns)
    top_features = feature_importance.nlargest(max_features).index.tolist()
    return X[top_features]

def evaluate_classification_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    metrics = {"accuracy": accuracy}
    if len(np.unique(y_test)) == 2:
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics["auc"] = float(roc_auc_score(y_test, y_prob))
        metrics["f1"] = float(f1_score(y_test, y_pred))
        metrics["precision"] = float(precision_score(y_test, y_pred))
        metrics["recall"] = float(recall_score(y_test, y_pred))
    return metrics

def evaluate_regression_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}

def train_xgboost_model(X_train, y_train, X_test, y_test, params, task_type):
    eval_set = [(X_test, y_test)]
    def attempt_training(current_params):
        if task_type == "regression":
            model = xgb.XGBRegressor(**current_params)
            try:
                model.fit(X_train, y_train, eval_set=eval_set, verbose=current_params.get("verbosity"))
            except:
                model.fit(X_train, y_train)
        else:
            model = xgb.XGBClassifier(**current_params)
            try:
                model.fit(X_train, y_train, eval_set=eval_set, verbose=current_params.get("verbosity"))
            except:
                model.fit(X_train, y_train)
        return model
    try:
        model = attempt_training(params)
    except xgb.core.XGBoostError as e:
        if "GPU" in str(e) or "CUDA" in str(e) or "device" in str(e):
            fallback = params.copy()
            fallback["tree_method"] = "hist"
            fallback.pop("gpu_id", None)
            fallback.pop("device", None)
            model = attempt_training(fallback)
        else:
            fallback = {"n_estimators": params.get("n_estimators", 100), "learning_rate": params.get("learning_rate", 0.1),
                        "max_depth": params.get("max_depth", 6), "objective": params.get("objective", "reg:squarederror" if task_type=="regression" else "binary:logistic"),
                        "verbosity": 1, "tree_method": "hist"}
            model = attempt_training(fallback)
    if task_type == "regression":
        metrics = evaluate_regression_model(model, X_test, y_test)
    else:
        metrics = evaluate_classification_model(model, X_test, y_test)
    return model, metrics

def train_ensemble(X_train, y_train, X_test, y_test, params, ensemble_size, task_type):
    models = []
    predictions = []
    ensemble_params = params.copy()
    ensemble_params["subsample"] = min(0.8, ensemble_params.get("subsample", 1.0))
    ensemble_params["colsample_bytree"] = min(0.8, ensemble_params.get("colsample_bytree", 1.0))
    for i in range(ensemble_size):
        seed = 42 + i
        ensemble_params["random_state"] = seed
        model, _ = train_xgboost_model(X_train, y_train, X_test, y_test, ensemble_params, task_type)
        models.append(model)
        predictions.append(model.predict(X_test))
    ensemble_pred = np.mean(predictions, axis=0)
    if task_type == "regression":
        rmse = float(np.sqrt(mean_squared_error(y_test, ensemble_pred)))
        mae = float(mean_absolute_error(y_test, ensemble_pred))
        r2 = float(r2_score(y_test, ensemble_pred))
        metrics = {"rmse": rmse, "mae": mae, "r2": r2}
    else:
        if len(np.unique(y_test)) == 2:
            ensemble_class = (ensemble_pred > 0.5).astype(int)
        else:
            ensemble_class = np.round(ensemble_pred).astype(int)
        accuracy = float(accuracy_score(y_test, ensemble_class))
        metrics = {"accuracy": accuracy}
        if len(np.unique(y_test)) == 2:
            metrics["auc"] = float(roc_auc_score(y_test, ensemble_pred))
            metrics["f1"] = float(f1_score(y_test, ensemble_class))
            metrics["precision"] = float(precision_score(y_test, ensemble_class))
            metrics["recall"] = float(recall_score(y_test, ensemble_class))
    return models, metrics, ensemble_pred

def perform_cross_validation(X, y, params, cv, task_type):
    if task_type == "classification" and len(np.unique(y)) > 1:
        kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    else:
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    metrics_list = []
    fold_models = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        model, metrics = train_xgboost_model(X_train, y_train, X_val, y_val, params, task_type)
        metrics_list.append(metrics)
        fold_models.append(model)
    agg_metrics = {}
    for metric in metrics_list[0].keys():
        vals = [m[metric] for m in metrics_list]
        agg_metrics[metric] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "values": vals}
    return fold_models, agg_metrics

def evaluate_external_test_set(model, X_test, y_test, task_type, model_dir, preserved_data):
    y_pred = model.predict(X_test)
    if task_type == "regression":
        metrics = evaluate_regression_model(model, X_test, y_test)
    else:
        metrics = evaluate_classification_model(model, X_test, y_test)
    results_df = pd.DataFrame({"true": y_test, "predicted": y_pred})
    if preserved_data:
        for col, values in preserved_data.items():
            if len(values) == len(results_df):
                results_df[col] = values.values
    test_results_file = os.path.join(model_dir, "external_test_results.csv")
    results_df.to_csv(test_results_file, index=False)
    test_metrics_file = os.path.join(model_dir, "external_test_metrics.json")
    with open(test_metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    try:
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        mn = min(np.min(y_test), np.min(y_pred))
        mx = max(np.max(y_test), np.max(y_pred))
        plt.plot([mn, mx], [mn, mx], "r--")
        plt.xlabel("True Values")
        plt.ylabel("Predictions")
        plt.title("External Test: True vs Predicted")
        plt.grid(True, alpha=0.3)
        if task_type == "regression":
            plt.figtext(0.15, 0.85, f"RMSE: {metrics['rmse']:.4f}", fontsize=12)
            plt.figtext(0.15, 0.82, f"MAE: {metrics['mae']:.4f}", fontsize=12)
            plt.figtext(0.15, 0.79, f"R²: {metrics['r2']:.4f}", fontsize=12)
        else:
            plt.figtext(0.15, 0.85, f"Accuracy: {metrics['accuracy']:.4f}", fontsize=12)
            if "auc" in metrics:
                plt.figtext(0.15, 0.82, f"AUC: {metrics['auc']:.4f}", fontsize=12)
        plot_file = os.path.join(model_dir, "external_test_plot.png")
        plt.tight_layout()
        plt.savefig(plot_file)
        plt.close()
    except:
        pass
    return metrics

def optimize_xgboost_optuna(X_train, y_train, basic_params, args, task_type):
    try:
        import optuna
    except ImportError:
        log_message("Optuna not installed, skipping tuning")
        return basic_params, None, []
    best_score_so_far = float("inf")
    trial_results = []
    if args.optuna_sampler == "random":
        sampler = optuna.samplers.RandomSampler(seed=42)
    else:
        sampler = optuna.samplers.TPESampler(seed=42)
    if args.optuna_pruner == "none":
        pruner = optuna.pruners.NopPruner()
    else:
        n_startup = max(5, args.optuna_trials // 5)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=n_startup, n_warmup_steps=0, interval_steps=1)
    cv_folds = 5
    if task_type == "classification" and len(np.unique(y_train)) > 1:
        kf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    else:
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    fold_timeout = 60
    if args.optuna_timeout and args.optuna_trials > 0:
        per_trial_time = args.optuna_timeout * 0.8 / args.optuna_trials
        fold_timeout = max(30, int(per_trial_time / (cv_folds * 1.2)))
    def objective(trial):
        nonlocal best_score_so_far, trial_results
        param_suggestions = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        current_params = {**basic_params, **param_suggestions}
        current_params["objective"] = basic_params.get("objective", get_default_objective(task_type))
        current_params["tree_method"] = basic_params.get("tree_method", "hist")
        if "eval_metric" in current_params:
            del current_params["eval_metric"]
        scores = []
        import time
        trial_start_time = time.time()
        try:
            for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
                fold_start_time = time.time()
                X_fold_train = X_train.iloc[train_idx] if hasattr(X_train, "iloc") else X_train[train_idx]
                X_fold_val = X_train.iloc[val_idx] if hasattr(X_train, "iloc") else X_train[val_idx]
                y_fold_train = y_train[train_idx]
                y_fold_val = y_train[val_idx]
                if task_type == "regression":
                    dtrain = xgb.DMatrix(X_fold_train, label=y_fold_train)
                    dval = xgb.DMatrix(X_fold_val, label=y_fold_val)
                    booster_params = {
                        "objective": current_params["objective"],
                        "eta": current_params["learning_rate"],
                        "max_depth": current_params["max_depth"],
                        "subsample": current_params["subsample"],
                        "colsample_bytree": current_params["colsample_bytree"],
                        "gamma": current_params["gamma"],
                        "min_child_weight": current_params["min_child_weight"],
                        "alpha": current_params["reg_alpha"],
                        "lambda": current_params["reg_lambda"],
                        "tree_method": current_params["tree_method"],
                        "verbosity": 0
                    }
                    booster_params = {k: v for k, v in booster_params.items() if v is not None}
                    evals = [(dtrain, "train"), (dval, "val")]
                    num_boost_round = min(current_params.get("n_estimators", 500), 300)
                    bst = xgb.train(
                        booster_params,
                        dtrain,
                        num_boost_round=num_boost_round,
                        evals=evals,
                        early_stopping_rounds=10,
                        verbose_eval=False
                    )
                    y_pred = bst.predict(dval)
                    score = float(np.sqrt(mean_squared_error(y_fold_val, y_pred)))
                else:
                    dtrain = xgb.DMatrix(X_fold_train, label=y_fold_train)
                    dval = xgb.DMatrix(X_fold_val, label=y_fold_val)
                    booster_params = {
                        "objective": current_params["objective"],
                        "eta": current_params["learning_rate"],
                        "max_depth": current_params["max_depth"],
                        "subsample": current_params["subsample"],
                        "colsample_bytree": current_params["colsample_bytree"],
                        "gamma": current_params["gamma"],
                        "min_child_weight": current_params["min_child_weight"],
                        "alpha": current_params["reg_alpha"],
                        "lambda": current_params["reg_lambda"],
                        "tree_method": current_params["tree_method"],
                        "verbosity": 0
                    }
                    booster_params = {k: v for k, v in booster_params.items() if v is not None}
                    evals = [(dtrain, "train"), (dval, "val")]
                    num_boost_round = min(current_params.get("n_estimators", 500), 300)
                    bst = xgb.train(
                        booster_params,
                        dtrain,
                        num_boost_round=num_boost_round,
                        evals=evals,
                        early_stopping_rounds=10,
                        verbose_eval=False
                    )
                    y_pred = bst.predict(dval)
                    if len(np.unique(y_fold_val)) == 2:
                        score = float(sklearn.metrics.log_loss(y_fold_val, y_pred))
                    else:
                        score = 1.0
                scores.append(score)
                fold_end_time = time.time()
                if (time.time() - trial_start_time) + (fold_end_time - fold_start_time) * (cv_folds - fold - 1) > args.optuna_timeout if args.optuna_timeout else 999999:
                    break
            if scores:
                avg_score = float(np.mean(scores))
            else:
                avg_score = float("inf")
            if avg_score < best_score_so_far:
                best_score_so_far = avg_score
            trial_results.append({"trial": trial.number+1, "score": avg_score, "params": param_suggestions.copy()})
            return avg_score
        except optuna.TrialPruned:
            return float("inf")
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    try:
        study.optimize(objective, n_trials=args.optuna_trials, timeout=args.optuna_timeout)
    except KeyboardInterrupt:
        pass
    if study.best_trial:
        best_params = study.best_trial.params
        best_score = study.best_value
        final_params = {**basic_params, **best_params}
    else:
        final_params = basic_params
    return final_params, study, trial_results

def main():
    start_time = time.time()
    args = parse_arguments()
    log_message(f"Python version: {platform.python_version()}")
    log_message(f"Operating system: {platform.system()} {platform.release()}")
    if args.config:
        try:
            with open(args.config, "r") as f:
                cfg = yaml.safe_load(f)
            for k, v in cfg.items():
                if hasattr(args, k) and getattr(args, k) is None:
                    setattr(args, k, v)
        except Exception as e:
            log_message(f"Error loading config: {e}")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    df = load_data(args.input)
    external_df = None
    if args.external_test_set and os.path.isfile(args.external_test_set):
        external_df = load_data(args.external_test_set)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{args.target.lower()}_{timestamp}" if args.target else f"model_{timestamp}"
    model_dir = os.path.join(args.output_dir, run_id)
    os.makedirs(model_dir, exist_ok=True)
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, "w") as f:
        dcfg = vars(args).copy()
        for kk, vv in dcfg.items():
            if not isinstance(vv, (str, int, float, bool, list, dict, type(None))):
                dcfg[kk] = str(vv)
        json.dump(dcfg, f, indent=2)
    data_stats = analyze_data_quality(df)
    data_stats_path = os.path.join(model_dir, "data_stats.json")
    with open(data_stats_path, "w") as f:
        json.dump(data_stats, f, indent=2)
    if args.target is None:
        log_message("Error: Target column must be specified")
        return 1
    if args.target not in df.columns:
        log_message(f"Error: Target column '{args.target}' not found in dataset")
        return 1
    config_dict = vars(args)
    df, cleaning_report = clean_dataset(df, config_dict)
    cleaning_report_path = os.path.join(model_dir, "cleaning_report.json")
    with open(cleaning_report_path, "w") as f:
        json.dump(cleaning_report, f, indent=2)
    preserve_cols = []
    if args.id_col and args.id_col in df.columns:
        preserve_cols.append(args.id_col)
    if args.smiles_cols:
        for c in [x.strip() for x in args.smiles_cols.split(",") if x.strip()]:
            if c in df.columns and c not in preserve_cols:
                preserve_cols.append(c)
    if args.keep_cols:
        for c in [x.strip() for x in args.keep_cols.split(",") if x.strip()]:
            if c in df.columns and c not in preserve_cols:
                preserve_cols.append(c)
    if external_df is not None:
        ext_stats = analyze_data_quality(external_df)
        ext_stats_path = os.path.join(model_dir, "external_data_stats.json")
        with open(ext_stats_path, "w") as f:
            json.dump(ext_stats, f, indent=2)
        ext_cols_removed = []
        for rc in cleaning_report["removed_columns"]:
            if rc in external_df.columns:
                external_df = external_df.drop(columns=[rc])
                ext_cols_removed.append(rc)
        final_cols = list(df.columns)
        drop_in_ext = []
        for c in external_df.columns:
            if c not in final_cols:
                drop_in_ext.append(c)
        if drop_in_ext:
            external_df = external_df.drop(columns=drop_in_ext)
    X, y, preserved_data, label_encoder, scaler = preprocess_data(df, args.target, preserve_cols, args.task, args.scale_features)
    check_data_sanity(X, y, "main_dataset")
    if external_df is not None and args.target in external_df.columns:
        external_y = external_df[args.target].values
        ext_preserve = {}
        for pc in preserve_cols:
            if pc in external_df.columns:
                ext_preserve[pc] = external_df[pc].copy()
        X_external = external_df.drop(columns=[args.target])
        for pc in preserve_cols:
            if pc in X_external.columns:
                X_external = X_external.drop(columns=[pc])
        numeric_cols_ext = X_external.select_dtypes(include=np.number).columns.tolist()
        non_numeric_ext = [c for c in X_external.columns if c not in numeric_cols_ext]
        if non_numeric_ext:
            X_external = X_external.drop(columns=non_numeric_ext)
        if scaler is not None:
            X_external = pd.DataFrame(scaler.transform(X_external), columns=X_external.columns, index=X_external.index)
        if label_encoder is not None and args.task == "classification":
            try:
                external_y = label_encoder.transform(external_y)
            except:
                pass
        X_external = align_features(X.columns.tolist(), X_external)
        check_data_sanity(X_external, external_y, "external_dataset")
    else:
        X_external = None
        external_y = None
        ext_preserve = {}
    if args.feature_selection and X.shape[1] > 0:
        X = select_features(X, y, args.max_features, args.task)
    feature_list = X.columns.tolist()
    feature_path = os.path.join(model_dir, "features.json")
    with open(feature_path, "w") as f:
        json.dump({"features": feature_list, "n_features": len(feature_list)}, f, indent=2)
    if args.cv > 0:
        log_message(f"Performing {args.cv}-fold cross-validation")
        cv_models, cv_metrics = perform_cross_validation(
            X, y, params, cv=args.cv, task_type=args.task
        )
        
        # Save CV results
        cv_results_path = os.path.join(model_dir, "cv_results.json")
        with open(cv_results_path, 'w') as f:
            json.dump(cv_metrics, f, indent=2)
        log_message(f"Saved cross-validation results to {cv_results_path}")
    else:
        log_message("Cross-validation disabled (--cv 0)")
        cv_models = []
        cv_metrics = {"info": "Cross-validation disabled (--cv 0)"}
        
        # Save empty CV results to document the skipped step
        cv_results_path = os.path.join(model_dir, "cv_results.json")
        with open(cv_results_path, 'w') as f:
            json.dump(cv_metrics, f, indent=2)
        log_message(f"Saved cross-validation status to {cv_results_path}")
    if args.ensemble:
        models, metrics, ensemble_pred = train_ensemble(X_train, y_train, X_test, y_test, params, args.ensemble_size, args.task)
        for i, m in enumerate(models):
            mp = os.path.join(model_dir, f"model_{i+1}.json")
            m.save_model(mp)
        model = models[0]
    else:
        model, metrics = train_xgboost_model(X_train, y_train, X_test, y_test, params, args.task)
        model_path = os.path.join(model_dir, "model.json")
        model.save_model(model_path)
        joblib_path = os.path.join(model_dir, "model.joblib")
        joblib.dump(model, joblib_path)
    fi = pd.Series(model.feature_importances_, index=X.columns)
    fi_df = pd.DataFrame({"feature": fi.index, "importance": fi.values}).sort_values("importance", ascending=False)
    fi_df.to_csv(os.path.join(model_dir, "feature_importance.csv"), index=False)
    try:
        top20 = fi.nlargest(20)
        plt.figure(figsize=(10, 8))
        top20.plot(kind="barh")
        plt.title("Top 20 Feature Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, "feature_importance.png"))
        plt.close()
    except:
        pass
    if args.export_validation:
        y_pred = model.predict(X_test)
        pred_df = pd.DataFrame({"true": y_test, "pred": y_pred})
        if args.task == "classification" and hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_test)
            if y_pred_proba.shape[1] == 2:
                pred_df["prob"] = y_pred_proba[:, 1]
            else:
                for i in range(y_pred_proba.shape[1]):
                    pred_df[f"prob_class_{i}"] = y_pred_proba[:, i]
        for c, vals in preserved_data.items():
            test_idx = X_test.index
            pred_df[c] = vals.iloc[test_idx].values
        pred_path = os.path.join(model_dir, "validation_predictions.csv")
        pred_df.to_csv(pred_path, index=False)
        if args.task == "regression":
            try:
                plt.figure(figsize=(10, 6))
                plt.scatter(y_test, y_pred, alpha=0.5)
                mn = min(y_test.min(), y_pred.min())
                mx = max(y_test.max(), y_pred.max())
                plt.plot([mn, mx], [mn, mx], "r--")
                plt.xlabel("True Values")
                plt.ylabel("Predictions")
                plt.title("True vs Predicted")
                plt.figtext(0.15, 0.8, f"RMSE: {metrics['rmse']:.4f}", fontsize=12)
                plt.figtext(0.15, 0.75, f"MAE: {metrics['mae']:.4f}", fontsize=12)
                plt.figtext(0.15, 0.7, f"R²: {metrics['r2']:.4f}", fontsize=12)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(model_dir, "prediction_plot.png"))
                plt.close()
            except:
                pass
    if X_external is not None and external_y is not None:
        if args.ensemble:
            preds = []
            for m in models:
                preds.append(m.predict(X_external))
            ext_pred = np.mean(preds, axis=0)
            if args.task == "regression":
                rmse = float(np.sqrt(mean_squared_error(external_y, ext_pred)))
                mae = float(mean_absolute_error(external_y, ext_pred))
                r2v = float(r2_score(external_y, ext_pred))
                external_metrics = {"rmse": rmse, "mae": mae, "r2": r2v}
            else:
                if len(np.unique(external_y)) == 2:
                    ext_class = (ext_pred > 0.5).astype(int)
                else:
                    ext_class = np.round(ext_pred).astype(int)
                acc = float(accuracy_score(external_y, ext_class))
                external_metrics = {"accuracy": acc}
                if len(np.unique(external_y)) == 2:
                    external_metrics["auc"] = float(roc_auc_score(external_y, ext_pred))
                    external_metrics["f1"] = float(f1_score(external_y, ext_class))
                    external_metrics["precision"] = float(precision_score(external_y, ext_class))
                    external_metrics["recall"] = float(recall_score(external_y, ext_class))
            edf = pd.DataFrame({"true": external_y, "predicted": ext_pred})
            for c, vals in ext_preserve.items():
                edf[c] = vals.values
            extf = os.path.join(model_dir, "external_test_results.csv")
            edf.to_csv(extf, index=False)
            extm = os.path.join(model_dir, "external_test_metrics.json")
            with open(extm, "w") as f:
                json.dump(external_metrics, f, indent=2)
        else:
            external_metrics = evaluate_external_test_set(model, X_external, external_y, args.task, model_dir, ext_preserve)
    metrics_summary = {
        "model_type": "ensemble" if args.ensemble else "single",
        "task": args.task,
        "features": {
            "count": len(feature_list),
            "top_10": fi.nlargest(10).index.tolist()
        },
        "performance": metrics
    }
    metrics_path = os.path.join(model_dir, "metrics_summary.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    preprocessor_info = {"feature_list": feature_list, "task_type": args.task}
    if scaler is not None:
        preprocessor_info["scaler"] = {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}
    if label_encoder is not None:
        preprocessor_info["label_encoder"] = {"classes": label_encoder.classes_.tolist()}
    preprocessor_path = os.path.join(model_dir, "preprocessor_info.json")
    with open(preprocessor_path, "w") as f:
        json.dump(preprocessor_info, f, indent=2)
    if scaler is not None or label_encoder is not None:
        joblib.dump({"scaler": scaler, "label_encoder": label_encoder}, os.path.join(model_dir, "preprocessor.joblib"))
    total_time = time.time() - start_time
    return 0

if __name__ == "__main__":
    sys.exit(main())