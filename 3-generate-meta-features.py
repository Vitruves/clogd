#!/usr/bin/env python3

import os
import argparse
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import AgglomerativeClustering
import pickle
from datetime import datetime
import warnings
import multiprocessing
from multiprocessing import Pool, cpu_count
import time
import sys
import gc

try:
    import torch
    import torch.mps
    HAS_MPS = torch.backends.mps.is_available()
except ImportError:
    HAS_MPS = False

def log_message(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"-- [{timestamp}] {msg}")

def print_success(msg):
    print(f"-- \033[92m{msg}\033[0m")

def print_warning(msg):
    print(f"-- \033[93m{msg}\033[0m")

def print_error(msg):
    print(f"-- \033[91m{msg}\033[0m")

def parse_args():
    parser = argparse.ArgumentParser(description="XGBoost Meta Feature Generator (MPS-Optimized)", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", required=True, help="Input CSV file")
    parser.add_argument("--output", required=True, help="Output CSV file")
    parser.add_argument("--output-dir", default="meta_features", help="Output directory")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--smiles-col", default="SMILES", help="SMILES column name")
    parser.add_argument("--n-meta", type=int, default=10, help="Number of meta features to generate")
    parser.add_argument("--correlation-threshold", type=float, default=0.3, help="Minimum correlation with target")
    parser.add_argument("--intercorrelation-threshold", type=float, default=0.7, help="Maximum correlation among meta features")
    parser.add_argument("--correlation-method", choices=["r2", "pearson", "spearman"], default="pearson", help="Correlation method")
    parser.add_argument("--max-attempts", type=int, default=5, help="Maximum attempts for uncorrelated meta features")
    parser.add_argument("--cv-folds", type=int, default=5, help="Cross-validation folds")
    parser.add_argument("--xgb-folds", type=int, default=0, help="Folds for XGBoost CV (0=no CV)")
    parser.add_argument("--min-features", type=int, default=5, help="Minimum number of features per cluster")
    parser.add_argument("--learning-rate", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n-trees", type=int, default=200, help="Number of trees")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test size fraction")
    parser.add_argument("--nan-threshold", type=float, default=0.3, help="Drop columns exceeding NaN fraction")
    parser.add_argument("--drop-rows-with-nan", action="store_true", help="Drop rows with NaNs")
    parser.add_argument("--n-jobs", type=int, default=None, help="Number of parallel jobs")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--initial-clusters", type=int, default=None, help="Initial number of clusters")
    parser.add_argument("--compact-output", action="store_true", help="Compact console output")
    parser.add_argument("--reproduce", action="store_true", help="Reproduce meta features from saved models")
    parser.add_argument("--json-path", type=str, default=None, help="Path to meta_feature_info.json for reproduction")
    parser.add_argument("--model-dir", type=str, default=None, help="Directory of saved models")
    parser.add_argument("--use-mps", action="store_true", help="Use Apple MPS acceleration if available")
    parser.add_argument("--batch-size", type=int, default=4096, help="Batch size for MPS correlation")
    return parser.parse_args()

def load_data(filepath, target_col, smiles_col, nan_threshold=0.3, drop_rows_with_nan=False):
    try:
        df = pd.read_csv(filepath)
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(filepath, encoding="latin1")
        except UnicodeDecodeError:
            df = pd.read_csv(filepath, encoding="utf-8", errors="ignore")
    if target_col not in df.columns:
        print_error(f"Target column '{target_col}' not found")
        exit(1)
    if smiles_col not in df.columns:
        print_error(f"SMILES column '{smiles_col}' not found")
        exit(1)
    if df[target_col].isna().any():
        df = df.dropna(subset=[target_col])
    feature_cols = [c for c in df.columns if c not in [target_col, smiles_col]]
    non_numeric = []
    for c in feature_cols:
        if not pd.api.types.is_numeric_dtype(df[c]):
            non_numeric.append(c)
    if non_numeric:
        for c in non_numeric:
            feature_cols.remove(c)
    high_nan = []
    for c in feature_cols:
        nan_count = df[c].isna().sum()
        nan_frac = nan_count / len(df)
        if nan_frac > nan_threshold:
            high_nan.append(c)
    if high_nan:
        for c in high_nan:
            feature_cols.remove(c)
    if drop_rows_with_nan:
        df = df.dropna(subset=feature_cols)
    df_clean = df[[smiles_col, target_col] + feature_cols].copy()
    if not drop_rows_with_nan:
        for c in feature_cols:
            if df_clean[c].isna().any():
                if df_clean[c].isna().all():
                    df_clean[c] = 0
                else:
                    m = df_clean[c].median()
                    if pd.isna(m):
                        m2 = df_clean[c].mean()
                        if pd.isna(m2):
                            df_clean[c] = df_clean[c].fillna(0)
                        else:
                            df_clean[c] = df_clean[c].fillna(m2)
                    else:
                        df_clean[c] = df_clean[c].fillna(m)
    if df_clean.isna().any().any():
        df_clean = df_clean.fillna(0)
    features = df_clean[feature_cols]
    target = df_clean[target_col]
    smiles = df_clean[smiles_col]
    return df_clean, features, target, smiles, feature_cols

def compute_correlation_matrix_np(features_np, method="pearson"):
    n_features = features_np.shape[1]
    corr_matrix = np.zeros((n_features, n_features))
    if method == "pearson":
        centered = features_np - np.mean(features_np, axis=0)
        std = np.std(features_np, axis=0, ddof=1)
        std[std == 0] = 1
        normed = centered / std
        n = features_np.shape[0]
        corr_matrix = (normed.T @ normed) / (n - 1)
    elif method == "spearman":
        from scipy.stats import rankdata
        ranked = np.array([rankdata(features_np[:, j]) for j in range(n_features)]).T
        rc = ranked - np.mean(ranked, axis=0)
        rstd = np.std(ranked, axis=0, ddof=1)
        rstd[rstd == 0] = 1
        rn = rc / rstd
        n = ranked.shape[0]
        corr_matrix = (rn.T @ rn) / (n - 1)
    np.clip(corr_matrix, -1.0, 1.0, out=corr_matrix)
    np.fill_diagonal(corr_matrix, 1.0)
    return corr_matrix

def compute_correlation_matrix_mps(features_df, method="pearson", batch_size=4096):
    try:
        import torch
        arr = features_df.values
        n_features = arr.shape[1]
        if n_features > 10000:
            return pd.DataFrame(compute_correlation_matrix_np(arr, method), index=features_df.columns, columns=features_df.columns)
            
        # Create a safer tensor creation with error handling
        try:
            x = torch.tensor(arr, dtype=torch.float32)
            # Check if MPS is available before using it
            if not torch.backends.mps.is_available():
                raise ValueError("MPS not available")
                
            # Move to MPS with error handling
            try:
                x = x.to("mps")
            except Exception as e:
                # If moving to MPS fails, fall back to CPU
                print_warning(f"Error moving tensor to MPS: {str(e)}")
                return pd.DataFrame(compute_correlation_matrix_np(arr, method), 
                                   index=features_df.columns, columns=features_df.columns)
        except Exception as e:
            print_warning(f"Error creating tensor: {str(e)}")
            return pd.DataFrame(compute_correlation_matrix_np(arr, method), 
                               index=features_df.columns, columns=features_df.columns)
            
        if method == "pearson":
            corr_matrix = torch.zeros((n_features, n_features), dtype=torch.float32).to("mps")
            mean = x.mean(dim=0, keepdim=True)
            std = x.std(dim=0, keepdim=True, unbiased=True)
            std[std == 0] = 1.0
            xn = (x - mean) / std
            if n_features <= batch_size:
                try:
                    corr_matrix = torch.mm(xn.T, xn) / (arr.shape[0] - 1)
                except Exception as e:
                    print_warning(f"Error in MPS matrix multiplication: {str(e)}")
                    return pd.DataFrame(compute_correlation_matrix_np(arr, method), 
                                       index=features_df.columns, columns=features_df.columns)
            else:
                try:
                    for i in range(0, n_features, batch_size):
                        end = min(i + batch_size, n_features)
                        batch_x = xn[:, i:end]
                        cb = torch.mm(batch_x.T, xn) / (arr.shape[0] - 1)
                        corr_matrix[i:end, :] = cb
                    lt = torch.tril(corr_matrix, -1)
                    corr_matrix = corr_matrix - lt + lt.T
                except Exception as e:
                    print_warning(f"Error in MPS batched processing: {str(e)}")
                    return pd.DataFrame(compute_correlation_matrix_np(arr, method), 
                                       index=features_df.columns, columns=features_df.columns)
        elif method == "spearman":
            try:
                from scipy.stats import rankdata
                ranked = np.array([rankdata(arr[:, j]) for j in range(n_features)]).T
                xr = torch.tensor(ranked, dtype=torch.float32).to("mps")
                mean = xr.mean(dim=0, keepdim=True)
                std = xr.std(dim=0, keepdim=True, unbiased=True)
                std[std == 0] = 1.0
                xrn = (xr - mean) / std
                corr_matrix = torch.zeros((n_features, n_features), dtype=torch.float32).to("mps")
                if n_features <= batch_size:
                    corr_matrix = torch.mm(xrn.T, xrn) / (arr.shape[0] - 1)
                else:
                    for i in range(0, n_features, batch_size):
                        end = min(i + batch_size, n_features)
                        batch_x = xrn[:, i:end]
                        cb = torch.mm(batch_x.T, xrn) / (arr.shape[0] - 1)
                        corr_matrix[i:end, :] = cb
                    lt = torch.tril(corr_matrix, -1)
                    corr_matrix = corr_matrix - lt + lt.T
            except Exception as e:
                print_warning(f"Error in Spearman correlation computation with MPS: {str(e)}")
                return pd.DataFrame(compute_correlation_matrix_np(arr, method), 
                                   index=features_df.columns, columns=features_df.columns)
                
        # Safely process the correlation matrix
        try:
            corr_matrix = torch.clamp(corr_matrix, -1.0, 1.0)
            corr_matrix.fill_diagonal_(1.0)
            cnp = corr_matrix.cpu().numpy()
            return pd.DataFrame(cnp, index=features_df.columns, columns=features_df.columns)
        except Exception as e:
            print_warning(f"Error finalizing MPS correlation matrix: {str(e)}")
            return pd.DataFrame(compute_correlation_matrix_np(arr, method), 
                               index=features_df.columns, columns=features_df.columns)
    except Exception as e:
        print_warning(f"MPS correlation computation failed: {str(e)}")
        return pd.DataFrame(compute_correlation_matrix_np(features_df.values, method), 
                           index=features_df.columns, columns=features_df.columns)

def compute_correlation_matrix(features_df, method="pearson", use_mps=False, batch_size=4096):
    if features_df.isna().any().any():
        features_df = features_df.fillna(0)
    try:
        if use_mps and HAS_MPS:
            try:
                print_success("Using MPS acceleration for correlation matrix")
                result = compute_correlation_matrix_mps(features_df, method, batch_size)
                print_success("MPS correlation matrix calculation completed")
                return result
            except Exception as e:
                print_warning(f"MPS acceleration failed, falling back to NumPy implementation: {str(e)}")
        
        arr = features_df.values
        print_success("Using optimized NumPy implementation for correlation")
        c = compute_correlation_matrix_np(arr, method)
        df = pd.DataFrame(c, index=features_df.columns, columns=features_df.columns)
        if df.isna().any().any():
            df = df.fillna(0)
        return df
    except:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                if method == "pearson":
                    df = features_df.corr(method="pearson")
                else:
                    df = features_df.corr(method="spearman")
                if df.isna().any().any():
                    df = df.fillna(0)
                return df
            except:
                cols = features_df.columns
                return pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)

def calculate_correlation(x, y, method="pearson"):
    try:
        if method == "pearson":
            c, _ = pearsonr(x, y)
            return c
        elif method == "spearman":
            c, _ = spearmanr(x, y)
            return c
        elif method == "r2":
            return r2_score(y, x)
        return 0
    except:
        return 0

def cluster_features(features, corr_matrix, n_clusters, seed=42, min_features=5):
    feature_cols = features.columns.tolist()
    np.random.seed(seed)
    if len(feature_cols) <= n_clusters:
        return [[col] for col in feature_cols[:n_clusters]]
    distance_matrix = 1 - abs(corr_matrix)
    try:
        from sklearn.cluster import SpectralClustering
        from sklearn.neighbors import kneighbors_graph
        affinity = abs(corr_matrix).values
        clustering = SpectralClustering(n_clusters=n_clusters, affinity="precomputed", random_state=seed, n_init=10, assign_labels="kmeans", n_jobs=-1)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Graph is not fully connected")
            labels = clustering.fit_predict(affinity)
        unique = np.unique(labels)
        clusters = []
        for u in unique:
            idx = np.where(labels == u)[0]
            if len(idx) > 0:
                fc = [feature_cols[i] for i in idx]
                clusters.append(fc)
        small = [i for i, c in enumerate(clusters) if len(c) < min_features]
        if small:
            pool = []
            for idx in small:
                pool.extend(clusters[idx])
                clusters[idx] = []
            for c in clusters:
                if not c:
                    continue
                while len(c) < min_features and pool:
                    c.append(pool.pop())
            if pool:
                clusters = [c for c in clusters if c]
                while pool:
                    sm = min(range(len(clusters)), key=lambda i: len(clusters[i]))
                    clusters[sm].append(pool.pop())
                clusters = [c for c in clusters if c]
            return clusters
    except:
        try:
            from sklearn.cluster import AgglomerativeClustering
            clustering = AgglomerativeClustering(n_clusters=n_clusters, metric="precomputed", linkage="average", compute_full_tree="auto", distance_threshold=None)
            labels = clustering.fit_predict(distance_matrix.values)
            unique = np.unique(labels)
            clusters = []
            for u in unique:
                idx = np.where(labels == u)[0]
                if len(idx) > 0:
                    fc = [feature_cols[i] for i in idx]
                    clusters.append(fc)
            small = [i for i, c in enumerate(clusters) if len(c) < min_features]
            if small:
                pool = []
                for idx in small:
                    pool.extend(clusters[idx])
                    clusters[idx] = []
                for c in clusters:
                    if not c:
                        continue
                    while len(c) < min_features and pool:
                        c.append(pool.pop())
                if pool:
                    clusters = [c for c in clusters if c]
                    while pool:
                        sm = min(range(len(clusters)), key=lambda i: len(clusters[i]))
                        clusters[sm].append(pool.pop())
                    clusters = [c for c in clusters if c]
            return clusters
        except:
            try:
                std = np.std(features.values, axis=0)
                var = np.sqrt(abs(np.var(features.values, axis=0)))
                imp = std * var
                imp_map = dict(zip(feature_cols, imp))
                sorted_feats = sorted(imp_map.items(), key=lambda x: x[1], reverse=True)
                top_feats = [f[0] for f in sorted_feats[: min(len(sorted_feats), n_clusters * 3)]]
                num_valid = min(n_clusters, len(feature_cols) // min_features)
                clusters = [[] for _ in range(num_valid)]
                seeds = [top_feats[0]]
                clusters[0].append(top_feats[0])
                for i in range(1, len(clusters)):
                    cand = [f for f in top_feats if f not in seeds]
                    if not cand:
                        break
                    diss = []
                    for candidate in cand:
                        sims = [abs(corr_matrix.loc[candidate, s]) for s in seeds]
                        avg = sum(sims) / len(sims)
                        diss.append((candidate, 1 - avg))
                    bestf = max(diss, key=lambda x: x[1])[0]
                    seeds.append(bestf)
                    clusters[i].append(bestf)
                remaining = np.array([f for f in feature_cols if f not in seeds])
                np.random.shuffle(remaining)
                needed = max(min_features - 1, (len(feature_cols) - len(seeds)) // len(clusters))
                for i, c in enumerate(clusters):
                    s = c[0]
                    for _ in range(min(needed, len(remaining))):
                        if len(remaining) == 0:
                            break
                        sims = np.array([abs(corr_matrix.loc[s, f]) for f in remaining])
                        midx = np.argmin(sims)
                        bf = remaining[midx]
                        c.append(bf)
                        remaining = np.delete(remaining, midx)
                while len(remaining) > 0:
                    sm = min(range(len(clusters)), key=lambda i: len(clusters[i]))
                    clusters[sm].append(remaining[0])
                    remaining = remaining[1:]
                return clusters
            except:
                np.random.seed(seed)
                sf = np.array(feature_cols)
                np.random.shuffle(sf)
                mx = len(sf) // min_features
                nc = min(n_clusters, mx)
                arrs = np.array_split(sf, nc)
                return [list(a) for a in arrs]

def train_meta_feature_model(X_train, y_train, X_test, y_test, subset, idx, n_trees=200, seed=42, xgb_folds=0, lr=0.1, verbose=False):
    params = {
        "objective": "reg:squarederror",
        "n_estimators": n_trees,
        "learning_rate": lr,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "random_state": seed + idx,
        "verbosity": 0,
        "missing": 0,
        "tree_method": "hist",
        "base_score": float(y_train.mean()),
    }
    Xtr = X_train[subset]
    Xts = X_test[subset]
    try:
        model = xgb.XGBRegressor(**params)
        if xgb_folds > 1:
            model.fit(Xtr, y_train, eval_set=[(Xts, y_test)], verbose=False)
        else:
            model.fit(Xtr, y_train)
        yp = model.predict(Xts)
        rmse = np.sqrt(mean_squared_error(y_test, yp))
        r2v = r2_score(y_test, yp)
        if hasattr(model, "feature_importances_"):
            fi = dict(zip(subset, model.feature_importances_))
        else:
            sc = model.get_booster().get_score(importance_type="gain")
            mp = {f"f{i}": f for i, f in enumerate(subset)}
            fi = {mp.get(k, k): v for k, v in sc.items()}
    except:
        model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=50, max_depth=3, learning_rate=lr, verbosity=0, missing=0, random_state=seed + idx, tree_method="hist")
        model.fit(Xtr, y_train)
        yp = model.predict(Xts)
        rmse = np.sqrt(mean_squared_error(y_test, yp))
        r2v = r2_score(y_test, yp)
        fi = dict(zip(subset, model.feature_importances_))
    return model, {"rmse": rmse, "r2": r2v, "feature_subset": subset, "feature_importances": fi}, yp

def train_model_parallel(item, X_train, y_train, X_test, y_test, n_trees, seed, xgb_folds, lr, verbose):
    idx, subset = item
    np.random.seed(seed + idx + os.getpid())
    try:
        model, metrics, preds = train_meta_feature_model(X_train, y_train, X_test, y_test, subset, idx, n_trees, seed, xgb_folds, lr, verbose)
        return idx, model, metrics, preds
    except:
        return idx, None, None, None

def perform_cross_validation(mid, subset, X, y, folds=5, n_trees=200, seed=42, lr=0.1):
    try:
        kf = KFold(n_splits=min(folds, 3), shuffle=True, random_state=seed + mid)
        y_out = np.zeros(X.shape[0])
        params = {
            "objective": "reg:squarederror",
            "n_estimators": n_trees,
            "learning_rate": lr,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
            "random_state": seed + mid,
            "verbosity": 0,
            "missing": 0,
            "tree_method": "hist",
        }
        Xs = X[subset]
        for tr, va in kf.split(Xs):
            Xt = Xs.iloc[tr]
            Xv = Xs.iloc[va]
            yt = y.iloc[tr]
            yv = y.iloc[va]
            m = xgb.XGBRegressor(**params)
            if Xt.shape[0] > 10000 or Xt.shape[1] > 100:
                dtr = xgb.DMatrix(Xt, yt)
                bst = xgb.train(params, dtr, num_boost_round=n_trees)
                dv = xgb.DMatrix(Xv)
                yp = bst.predict(dv)
            else:
                m.fit(Xt, yt)
                yp = m.predict(Xv)
            y_out[va] = yp
        return mid, y_out
    except:
        return mid, None
    
class DynamicDisplay:
    def __init__(self):
        self.start_time = 0
        self.last_update = 0
        self.stage_name = ""
        
    def start_stage(self, name):
        self.stage_name = name
        self.start_time = time.time()
        self.last_update = 0
        print(f"-- [{time.strftime('%H:%M:%S')}] {name} ", end="", flush=True)
        
    def update_progress(self, current, total, info=""):
        now = time.time()
        if now - self.last_update < 0.2 and current < total:
            return
        p = int(100 * current / total)
        w = 20
        f = int(w * current / total)
        b = "█" * f + "░" * (w - f)
        et = now - self.start_time
        if current > 0:
            eta = et * (total - current) / current
            if eta > 60:
                es = f"{int(eta//60)}m {int(eta%60)}s"
            else:
                es = f"{eta:.1f}s"
        else:
            es = "?"
        print(f"\r\033[K-- [{time.strftime('%H:%M:%S')}] {self.stage_name} [{current}/{total}] |{b}| {p}% {info} ETA: {es}", end="", flush=True)
        self.last_update = now
        if current >= total:
            print()
            
    def print_info(self, msg):
        print(f"\r\033[K-- {msg}")
        
    def print_status_table(self, headers, rows):
        print(f"\r\033[K")
        widths = [max(len(str(row[i])) for row in rows + [headers]) for i in range(len(headers))]
        hline = "-- " + " | ".join(f"{h:{w}}" for h, w in zip(headers, widths))
        print(hline)
        print("-- " + "-|-".join("-" * w for w in widths))
        for row in rows:
            line = "-- " + " | ".join(f"{str(c):{w}}" for c, w in zip(row, widths))
            print(line)

display = DynamicDisplay()

def generate_meta_features_from_clusters(df, features, target, smiles, feature_cols, clusters, args):
    X_train, X_test, y_train, y_test, sm_train, sm_test = train_test_split(features, target, smiles, test_size=args.test_size, random_state=args.seed)
    items = [(i, c) for i, c in enumerate(clusters) if c]
    display.start_stage("Training models")
    n_jobs = args.n_jobs if args.n_jobs else max(1, cpu_count() - 1)
    n_jobs = min(n_jobs, cpu_count())
    
    # MPS acceleration for feature standardization if available and requested
    if args.use_mps and HAS_MPS:
        try:
            print_success("Using MPS for feature preprocessing")
            # Check if dataset is not too large to avoid memory issues
            if X_train.shape[0] * X_train.shape[1] < 100000000:  # Limit to ~100M elements
                import torch
                
                # Process features in smaller batches to prevent memory issues
                batch_size = min(args.batch_size, 2048)  # Limit batch size
                
                # Process X_train
                X_train_values = torch.tensor(X_train.values, dtype=torch.float32)
                X_test_values = torch.tensor(X_test.values, dtype=torch.float32)
                
                # Initialize empty arrays for normalized data
                X_train_norm = np.zeros_like(X_train.values)
                X_test_norm = np.zeros_like(X_test.values)
                
                # Process in batches
                for start_idx in range(0, X_train.shape[1], batch_size):
                    end_idx = min(start_idx + batch_size, X_train.shape[1])
                    try:
                        # Create device tensors
                        X_train_batch = X_train_values[:, start_idx:end_idx].to("mps")
                        X_test_batch = X_test_values[:, start_idx:end_idx].to("mps")
                        
                        # Calculate statistics
                        means = X_train_batch.mean(dim=0, keepdim=True)
                        stds = X_train_batch.std(dim=0, keepdim=True)
                        stds[stds < 1e-8] = 1.0  # Prevent division by zero
                        
                        # Normalize
                        X_train_batch_norm = ((X_train_batch - means) / stds).cpu().numpy()
                        X_test_batch_norm = ((X_test_batch - means) / stds).cpu().numpy()
                        
                        # Store normalized data
                        X_train_norm[:, start_idx:end_idx] = X_train_batch_norm
                        X_test_norm[:, start_idx:end_idx] = X_test_batch_norm
                        
                    except Exception as batch_err:
                        print_warning(f"Failed to process batch {start_idx}-{end_idx}: {str(batch_err)}")
                        # Skip normalization for this batch
                        X_train_norm[:, start_idx:end_idx] = X_train.values[:, start_idx:end_idx]
                        X_test_norm[:, start_idx:end_idx] = X_test.values[:, start_idx:end_idx]
                
                # Convert back to pandas DataFrames
                X_train = pd.DataFrame(X_train_norm, index=X_train.index, columns=X_train.columns)
                X_test = pd.DataFrame(X_test_norm, index=X_test.index, columns=X_test.columns)
                
                print_success("MPS feature preprocessing completed")
            else:
                print_warning("Dataset too large for MPS preprocessing - skipping normalization")
        except Exception as e:
            print_warning(f"MPS preprocessing error: {str(e)}")
    
    # Don't use lambda for multiprocessing - define parameters directly
    results = []
    if n_jobs > 1 and len(items) > 1:
        try:
            # Set start method to avoid issues
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Already set
        
        with Pool(processes=n_jobs) as pool:
            total = len(items)
            i = 0
            # Use starmap with explicit arguments instead of lambda
            parallel_tasks = [(item, X_train, y_train, X_test, y_test, args.n_trees, args.seed, args.xgb_folds, args.learning_rate, args.compact_output) for item in items]
            for r in pool.starmap(train_model_parallel_wrapper, parallel_tasks):
                i += 1
                results.append(r)
                display.update_progress(i, total, f"(Model {r[0]+1})")
    else:
        total = len(items)
        for i, it in enumerate(items):
            r = train_model_parallel(it, X_train, y_train, X_test, y_test, args.n_trees, args.seed, args.xgb_folds, args.learning_rate, args.compact_output)
            results.append(r)
            display.update_progress(i + 1, total, f"(Model {r[0]+1})")
    results.sort(key=lambda x: x[0])
    results = [r for r in results if r[1] is not None]
    if not results:
        print_error("All models failed to train")
        return None
    cv_map = {}
    tasks = []
    for i, m, mm, yp in results:
        tasks.append((i, mm["feature_subset"]))
    display.start_stage("Cross-validating models")
    if n_jobs > 1 and len(tasks) > 1:
        try:
            # Create explicit task list for cross-validation
            cv_tasks = [(task[0], task[1], X_train, y_train, args.cv_folds, args.n_trees, args.seed, args.learning_rate) for task in tasks]
            with Pool(processes=n_jobs) as pool:
                total = len(tasks)
                i = 0
                for r in pool.starmap(perform_cross_validation, cv_tasks):
                    i += 1
                    if r[1] is not None:
                        cv_map[r[0]] = r[1]
                    display.update_progress(i, total, f"(Model {r[0]+1})")
        except Exception as e:
            print_warning(f"Parallel cross-validation failed: {str(e)}")
            # Fall back to sequential
            for i, (idx, s) in enumerate(tasks):
                mid, arr = perform_cross_validation(idx, s, X_train, y_train, args.cv_folds, args.n_trees, args.seed, args.learning_rate)
                if arr is not None:
                    cv_map[mid] = arr
                display.update_progress(i + 1, total, f"(Model {idx+1})")
    else:
        total = len(tasks)
        for i, (idx, s) in enumerate(tasks):
            mid, arr = perform_cross_validation(idx, s, X_train, y_train, args.cv_folds, args.n_trees, args.seed, args.learning_rate)
            if arr is not None:
                cv_map[mid] = arr
            display.update_progress(i + 1, total, f"(Model {idx+1})")
    evaluated = []
    for i, model, mtr, preds_test in results:
        if i in cv_map:
            cvpred = cv_map[i]
        else:
            try:
                cvpred = model.predict(X_train[mtr["feature_subset"]])
            except:
                continue
        c = calculate_correlation(cvpred, y_train, args.correlation_method)
        cc = abs(c)
        if args.correlation_method == "r2":
            cc = c
        evaluated.append({"idx": i, "model": model, "metrics": mtr, "cvpred": cvpred, "testpred": preds_test, "cor": c, "abs_cor": cc})
    evaluated.sort(key=lambda x: x["abs_cor"], reverse=True)
    meta_models = []
    meta_metrics = []
    meta_tr_preds = []
    meta_ts_preds = []
    idxs = []
    display.print_info("Adaptive Meta-Feature Selection:")
    hdr = ["Rank", "ID", "Target Corr", "Max Inter-Corr", "RMSE", "Features", "Status"]
    rows = []
    rnk = 0
    for c in evaluated:
        rnk += 1
        mid = c["idx"]
        t = c["abs_cor"] >= args.correlation_threshold
        mc = 0
        pass_ic = True
        if meta_tr_preds:
            tmp = pd.DataFrame({f"m{i+1}": v for i, v in enumerate(meta_tr_preds)})
            tmp["cand"] = c["cvpred"]
            corrs = tmp.corr().abs()
            mc = corrs.loc["cand"].drop("cand").max()
            if pd.isna(mc):
                mc = 0
            pass_ic = mc <= args.intercorrelation_threshold
        status = "✓ Accepted" if (t and pass_ic) else ("✗ Low Target Corr" if not t else "✗ High Inter-Corr")
        rows.append([f"{rnk}", f"{mid+1}", f"{c['cor']:+.4f}", f"{mc:.4f}", f"{c['metrics']['rmse']:.4f}", f"{len(c['metrics']['feature_subset'])}", status])
        if t and pass_ic and len(meta_models) < args.n_meta:
            meta_models.append(c["model"])
            c["metrics"]["correlation_with_target"] = c["cor"]
            c["metrics"]["max_correlation_with_meta"] = mc
            meta_metrics.append(c["metrics"])
            meta_tr_preds.append(c["cvpred"])
            meta_ts_preds.append(c["testpred"])
            idxs.append(mid)
        if len(meta_models) >= args.n_meta:
            display.print_info(f"Reached target of {args.n_meta} meta-features.")
            break
    display.print_status_table(hdr, rows)
    if not meta_models:
        print_error("No meta features passed thresholds")
        return None
    trdf = pd.DataFrame({f"meta_feature_{i+1}": p for i, p in enumerate(meta_tr_preds)})
    trdf[args.target] = y_train.values
    trdf[args.smiles_col] = sm_train.values
    tsdf = pd.DataFrame({f"meta_feature_{i+1}": p for i, p in enumerate(meta_ts_preds)})
    tsdf[args.target] = y_test.values
    tsdf[args.smiles_col] = sm_test.values
    trdf["set"] = "train"
    tsdf["set"] = "test"
    meta_df = pd.concat([trdf, tsdf], axis=0)
    mcols = [f"meta_feature_{i+1}" for i in range(len(meta_models))]
    corr = meta_df[mcols + [args.target]].corr()
    return {"meta_df": meta_df, "meta_models": meta_models, "meta_metrics": meta_metrics, "meta_corr_matrix": corr, "X_train": X_train, "X_test": X_test, "meta_cols": mcols}

# Add a wrapper function for multiprocessing
def train_model_parallel_wrapper(item, X_train, y_train, X_test, y_test, n_trees, seed, xgb_folds, lr, verbose):
    return train_model_parallel(item, X_train, y_train, X_test, y_test, n_trees, seed, xgb_folds, lr, verbose)

def check_meta_feature_correlations(meta_results, threshold, method):
    mcols = meta_results["meta_cols"]
    mdf = meta_results["meta_df"]
    cmx = meta_results["meta_corr_matrix"]
    combos = []
    for i in range(len(mcols)):
        for j in range(i+1, len(mcols)):
            c = abs(cmx.loc[mcols[i], mcols[j]])
            if c > threshold:
                combos.append((mcols[i], mcols[j], c))
    if combos:
        return False
        return True

def save_results(meta_results, args, feature_cols):
    meta_df = meta_results["meta_df"]
    models = meta_results["meta_models"]
    mets = meta_results["meta_metrics"]
    cm = meta_results["meta_corr_matrix"]
    mcols = meta_results["meta_cols"]
    os.makedirs(args.output_dir, exist_ok=True)
    outdf = meta_df[[args.smiles_col, args.target] + mcols].copy()
    outdf.to_csv(args.output, index=False)
    print_success(f"Saved meta features to {args.output}")
    try:
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm.loc[mcols + [args.target], mcols + [args.target]], annot=True, cmap="coolwarm", vmin=-1, vmax=1, center=0, fmt=".2f")
        plt.title("Meta Feature Correlations")
        plt.tight_layout()
        cp = os.path.join(args.output_dir, "meta_feature_correlations.png")
        plt.savefig(cp, dpi=300)
        plt.close()
        print_success(f"Saved correlation heatmap to {cp}")
    except:
        pass
    minfo = []
    for i, (model, metric) in enumerate(zip(models, mets)):
        minfo.append({
            "meta_feature_id": i+1,
            "rmse": metric["rmse"],
            "r2": metric["r2"],
            "correlation_with_target": metric.get("correlation_with_target", 0),
            "max_correlation_with_meta": metric.get("max_correlation_with_meta", 0),
            "feature_subset": metric["feature_subset"],
            "top_features": sorted(metric["feature_importances"].items(), key=lambda x: x[1], reverse=True)[:10],
        })
    try:
        def js(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        with open(os.path.join(args.output_dir, "meta_feature_info.json"), "w") as f:
            json.dump(minfo, f, default=js, indent=2)
        print_success("Saved model info JSON")
    except:
        pass
    try:
        for i, m in enumerate(models):
            p = os.path.join(args.output_dir, f"meta_feature_model_{i+1}.pkl")
            with open(p, "wb") as f:
                pickle.dump(m, f)
        print_success("Saved model files")
    except:
        pass
    try:
        for i, (m, mt) in enumerate(zip(models, mets)):
            fi = mt["feature_importances"]
            sf = sorted(fi.items(), key=lambda x: x[1], reverse=True)
            nf = min(20, len(sf))
            fts, vals = zip(*sf[:nf])
            plt.figure(figsize=(10, 8))
            plt.barh(range(nf), vals, align="center")
            plt.yticks(range(nf), fts)
            plt.xlabel("Importance")
            plt.title(f"Feature Importance (Meta {i+1})")
            plt.tight_layout()
            ip = os.path.join(args.output_dir, f"meta_feature_{i+1}_importance.png")
            plt.savefig(ip, dpi=300)
            plt.close()
    except:
        pass
    try:
        fi = {"total_features": len(feature_cols), "features": feature_cols}
        def js2(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        with open(os.path.join(args.output_dir, "feature_info.json"), "w") as f:
            json.dump(fi, f, default=js2, indent=2)
        print_success("Saved feature info")
    except:
        pass
    try:
        with open(os.path.join(args.output_dir, "summary_report.txt"), "w") as f:
            f.write("Meta Feature Generation Summary\n")
            f.write("============================\n\n")
            f.write(f"Input file: {args.input}\n")
            f.write(f"Output file: {args.output}\n")
            f.write(f"Target variable: {args.target}\n\n")
            f.write(f"Data:\n")
            f.write(f"Total rows: {len(meta_df)}\n")
            f.write(f"Train set: {meta_df[meta_df['set'] == 'train'].shape[0]}\n")
            f.write(f"Test set: {meta_df[meta_df['set'] == 'test'].shape[0]}\n")
            f.write(f"Input features: {len(feature_cols)}\n")
            f.write(f"Generated meta features: {len(models)}\n\n")
            f.write("Meta Feature Performance:\n")
            for i, m in enumerate(mets):
                f.write(f"Meta {i+1}:\n")
                f.write(f"  RMSE: {m['rmse']:.4f}\n")
                f.write(f"  R²: {m['r2']:.4f}\n")
                f.write(f"  Corr with target: {m.get('correlation_with_target', 0):.4f}\n")
                f.write(f"  Max corr with other meta: {m.get('max_correlation_with_meta', 0):.4f}\n")
                f.write(f"  Features used: {len(m['feature_subset'])}\n\n")
        print_success("Summary report saved")
    except:
        pass

def reproduce_meta_features(args):
    try:
        print_success(f"Loading meta feature information from {args.json_path}")
        # Load and parse JSON file
        with open(args.json_path, "r") as f:
            meta_json = json.load(f)
        
        # Determine if the JSON is an array of models or feature info
        if isinstance(meta_json, dict) and 'features' in meta_json:
            # This is a feature_info.json file
            print_success(f"Detected feature_info.json format with {len(meta_json.get('features', []))} features")
            # We need to locate meta_feature_info.json in the same directory
            json_dir = os.path.dirname(args.json_path)
            meta_info_path = os.path.join(json_dir, 'meta_feature_info.json')
            
            if os.path.exists(meta_info_path):
                print_success(f"Found meta_feature_info.json at {meta_info_path}")
                with open(meta_info_path, "r") as mf:
                    meta_info = json.load(mf)
            else:
                print_warning(f"meta_feature_info.json not found at {meta_info_path}")
                print_warning("Will attempt to use model files directly")
                # Try to find model files directly
                model_dir = args.model_dir if args.model_dir else args.output_dir
                model_files = [f for f in os.listdir(model_dir) if f.startswith('meta_feature_model_') and f.endswith('.pkl')]
                if not model_files:
                    print_error(f"No model files found in {model_dir}")
                    return False
                
                # Create minimal meta_info from model filenames
                meta_info = []
                for model_file in sorted(model_files):
                    try:
                        model_id = int(model_file.split('_')[-1].split('.')[0])
                        meta_info.append({"meta_feature_id": model_id})
                    except:
                        meta_info.append({"meta_feature_id": len(meta_info) + 1})
        else:
            # Assume it's the correct meta_feature_info.json format
            meta_info = meta_json
            
        print_success(f"Successfully loaded information for {len(meta_info)} models")
    except Exception as e:
        print_error(f"Failed to load meta feature information: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return False
    
    try:
        df, feats, targ, smi, fcols = load_data(args.input, args.target, args.smiles_col, args.nan_threshold, args.drop_rows_with_nan)
        print_success(f"Loaded dataset with {df.shape[0]} rows and {len(fcols)} features")
    except Exception as e:
        print_error(f"Failed to load input data: {str(e)}")
        return False
    
    model_dir = args.model_dir if args.model_dir else args.output_dir
    res = {}
    successful_metas = []  # Only track meta features that were successfully generated
    display.start_stage("Loading models and generating meta features")
    
    for i, info in enumerate(meta_info):
        # Handle both string and dict format for model info
        if isinstance(info, str):
            # If info is a string, use default values
            mid = i + 1
            fs = []
        else:
            # Get meta feature ID
            mid = info.get("meta_feature_id", i+1)
            # Get feature subset if available
            fs = info.get("feature_subset", [])
        
        modp = os.path.join(model_dir, f"meta_feature_model_{mid}.pkl")
        mc = f"meta_feature_{mid}"
        
        try:
            # Check if model file exists
            if not os.path.exists(modp):
                print_warning(f"Model file not found: {modp}")
                continue
                
            # Check for missing features
            if fs:
                missing = [x for x in fs if x not in fcols]
                if missing:
                    print_warning(f"Model {mid} missing {len(missing)} features")
                    if len(missing) <= 10:
                        print_warning(f"Missing features: {missing}")
                    else:
                        print_warning(f"First 10 missing features: {missing[:10]}...")
                    continue  # Skip this model if features are missing
            
            # Load model
            with open(modp, "rb") as ff:
                model = pickle.load(ff)
                
            # Generate predictions
            if fs:
                # Use specified features
                Xs = feats[fs]
            else:
                # If no features specified, try to infer from model
                try:
                    if hasattr(model, 'feature_names_in_'):
                        model_features = model.feature_names_in_
                        # Check if these features exist in our dataset
                        missing = [x for x in model_features if x not in fcols]
                        if missing:
                            print_warning(f"Model {mid} missing features from feature_names_in_")
                            continue  # Skip this model if features are missing
                        Xs = feats[model_features]
                    else:
                        print_warning(f"Model {mid} has no feature subset or feature_names_in_")
                        continue
                except:
                    print_warning(f"Failed to determine features for model {mid}")
                    continue
            
            # Make predictions
            p = model.predict(Xs)
            res[mc] = p
            successful_metas.append(mc)  # Only add to list if successfully generated
            display.update_progress(i+1, len(meta_info), f"(Model {mid})")
        except Exception as e:
            print_warning(f"Error with model {mid}: {str(e)}")
            continue
    
    if not res:
        print_error("Failed to generate any meta features")
        return False
    
    # Create output dataframe
    dfout = pd.DataFrame(res)
    dfout[args.target] = targ.values
    dfout[args.smiles_col] = smi.values
    
    # Only include successfully generated meta features in the output
    output_cols = [args.smiles_col, args.target] + successful_metas
    print_success(f"Generated {len(successful_metas)} meta features:")
    for meta in successful_metas:
        print_success(f"  - {meta}")
    
    # Save to output file
    dfout[output_cols].to_csv(args.output, index=False)
    print_success(f"Successfully reproduced {len(successful_metas)} meta features and saved to {args.output}")
    
    return True

def main():
    args = parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    start = time.time()
    
    # Check for MPS availability and set up appropriate configuration
    global HAS_MPS
    if args.use_mps:
        try:
            import torch
            if not torch.backends.mps.is_available():
                print_warning("MPS requested but not available on this system")
                print_warning("  - Running on CPU instead")
                HAS_MPS = False
            else:
                # Set torch MPS allocation mode to delayed to prevent OOM errors
                try:
                    torch.mps.set_allocation_mode("delayed")
                    print_success("MPS is available and configured")
                    HAS_MPS = True
                except Exception as mps_err:
                    print_warning(f"Failed to configure MPS: {str(mps_err)}")
                    print_warning("  - Running on CPU instead")
                    HAS_MPS = False
        except ImportError:
            print_warning("PyTorch not found, MPS acceleration disabled")
            HAS_MPS = False
    
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        if args.reproduce:
            if not args.json_path:
                print_error("--json-path required in reproduce mode")
                exit(1)
            if not reproduce_meta_features(args):
                print_error("Reproduction failed")
                exit(1)
            et = time.time() - start
            print_success(f"Reproduction time: {et:.2f}s")
            exit(0)
        df, feats, targ, smi, fcols = load_data(args.input, args.target, args.smiles_col, args.nan_threshold, args.drop_rows_with_nan)
        if len(fcols) < 2:
            print_error("Not enough features")
            exit(1)
        log_message("Computing correlation matrix")
        cmat = compute_correlation_matrix(feats, args.correlation_method, args.use_mps, args.batch_size)
        atp = 0
        mr = None
        ncl = args.initial_clusters or (args.n_meta * 2)
        while atp < args.max_attempts:
            atp += 1
            log_message(f"Attempt {atp}/{args.max_attempts}")
            clus = cluster_features(feats, cmat, ncl, args.seed + atp, args.min_features)
            mr = generate_meta_features_from_clusters(df, feats, targ, smi, fcols, clus, args)
            if mr is None:
                ncl = int(ncl * 1.5)
                continue
            if check_meta_feature_correlations(mr, args.intercorrelation_threshold, args.correlation_method):
                break
            ncl = int(ncl * 1.5)
        if mr is None:
            print_error("No valid meta features after attempts")
            exit(1)
        if atp == args.max_attempts and not check_meta_feature_correlations(mr, args.intercorrelation_threshold, args.correlation_method):
            print_warning("Intercorrelation threshold not met; using best found")
        save_results(mr, args, fcols)
        et = time.time() - start
        print_success(f"Total execution time: {et:.2f}s")
    except KeyboardInterrupt:
        print_error("Interrupted by user")
        exit(1)
    except Exception as e:
        print_error(str(e))
        if args.verbose:
            import traceback
            traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    main() 