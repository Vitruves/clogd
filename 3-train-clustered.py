#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import joblib
import json
import time
from tqdm import tqdm
import multiprocessing
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import xgboost as xgb
from xgboost import XGBRegressor
import shap
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors, rdMolDescriptors, MACCSkeys, Scaffolds
from rdkit import DataStructs
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning) # Ignore potential NaN warnings during metrics
import optuna
from optuna.samplers import TPESampler
from optuna.integration import XGBoostPruningCallback
# Check if visualization modules are available before importing
try:
    from optuna.visualization import plot_optimization_history, plot_param_importances
    _optuna_vis_available = True
except ImportError:
    _optuna_vis_available = False
    print_status("Warning: optuna.visualization is not available. Tuning plots will be skipped.")


import pickle
import logging
import inspect
import random
import traceback
from copy import deepcopy

class ProgressTracker:
    def __init__(self, total, description="Processing"):
        self.total = total
        self.description = description
        self.current = 0
        self.start_time = time.time()

    def update(self, n=1):
        self.current += n
        pct = min(100, 100 * self.current / self.total if self.total > 0 else 100)
        elapsed = time.time() - self.start_time
        rate = self.current / elapsed if elapsed > 0 else 0
        eta = (self.total - self.current) / rate if rate > 0 else 0
        eta_str = f"ETA: {eta:.1f}s" if eta < 60 else f"ETA: {eta/60:.1f}m"

        sys.stdout.write(f"\r-- [{pct:6.2f}%] {self.description} | {self.current}/{self.total} | {rate:.1f} items/s | {eta_str}   ")
        sys.stdout.flush()

    def finalize(self):
        elapsed = time.time() - self.start_time
        time_str = f"{elapsed:.1f}s" if elapsed < 60 else f"{elapsed/60:.1f}m"
        sys.stdout.write(f"\r-- [100.00%] {self.description} | {self.current}/{self.total} | Completed in {time_str}               \n")
        sys.stdout.flush()

def print_status(message):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_message = f"-- {timestamp} -- {message}"
    print(log_message)

    if hasattr(print_status, 'log_file') and print_status.log_file is not None:
        try:
            with open(print_status.log_file, 'a') as f:
                f.write(f"{log_message}\n")
        except Exception as e:
            # Use standard print for logging errors to avoid recursion
            print(f"-- Warning: Could not write to log file: {e}")

    sys.stdout.flush()

# --- Predictor Classes for Clustering Methods ---

class BasePredictor:
    def __init__(self, cluster_model, **kwargs):
        self.cluster_model = cluster_model
        self.__dict__.update(kwargs)

    def predict(self, X):
        raise NotImplementedError("Subclasses must implement predict method.")

class KMeansPredictor(BasePredictor):
    def predict(self, X):
        return self.cluster_model.predict(X)

class GaussianMixturePredictor(BasePredictor):
    def predict(self, X):
        return self.cluster_model.predict(X)

class RandomClusterPredictor(BasePredictor):
    def __init__(self, n_clusters, seed):
        super().__init__(cluster_model=None)
        self.n_clusters = n_clusters
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def predict(self, X):
        n_samples = X.shape[0]
        return self.rng.randint(0, self.n_clusters, size=n_samples)

class NearestNeighborPredictor(BasePredictor):
    def __init__(self, cluster_model, trained_clusters, training_data):
        super().__init__(cluster_model)
        self.clusters = trained_clusters
        self.training_data = training_data
        n_neighbors = 1
        if training_data.shape[0] < n_neighbors + 1 :
             print_status(f"Warning: training data size ({training_data.shape[0]}) < n_neighbors+1 ({n_neighbors+1}). NN predictor might be unstable.")

        self.nn = NearestNeighbors(n_neighbors=n_neighbors)
        try:
             # Ensure training data is not empty
             if training_data.shape[0] > 0:
                  self.nn.fit(training_data)
             else:
                  raise ValueError("Cannot fit NearestNeighbors predictor with empty training data.")
        except ValueError as e:
             print_status(f"Error fitting NearestNeighbors: {e}")
             print_status("Training data shape:", training_data.shape)
             raise RuntimeError(f"Failed to fit NearestNeighbors predictor: {e}")

    def predict(self, X):
        if not hasattr(self.nn, '_fit_method') or self.nn._fit_method is None:
             raise RuntimeError("NearestNeighbors predictor was not fitted successfully.")
        # Check if input X is empty
        if X.shape[0] == 0:
             return np.array([], dtype=self.clusters.dtype)
        _, indices = self.nn.kneighbors(X)
        return self.clusters[indices.flatten()]


class AgglomerativePredictor(NearestNeighborPredictor):
    pass

class SpectralPredictor(NearestNeighborPredictor):
    pass

class DBSCANWithPredict(NearestNeighborPredictor):
     def __init__(self, cluster_model, trained_clusters, training_data):
        core_samples_mask = np.zeros_like(cluster_model.labels_, dtype=bool)
        core_labels = np.array([], dtype=trained_clusters.dtype)
        core_training_data = np.array([]).reshape(0, training_data.shape[1])

        if hasattr(cluster_model, 'core_sample_indices_') and len(cluster_model.core_sample_indices_) > 0:
            core_indices = cluster_model.core_sample_indices_
            # Ensure core_indices are within bounds of training_data and trained_clusters
            if np.max(core_indices) < len(trained_clusters) and np.max(core_indices) < training_data.shape[0]:
                 core_samples_mask[core_indices] = True
                 core_training_data = training_data[core_samples_mask]
                 core_labels = trained_clusters[core_samples_mask]
            else:
                 print_status("Warning: DBSCAN core_sample_indices_ out of bounds. Using all points.")
                 core_training_data = training_data
                 core_labels = trained_clusters

            if core_training_data.shape[0] == 0:
                 print_status("Warning: DBSCAN has core_sample_indices_ but resulted in 0 core samples. Using all points.")
                 core_training_data = training_data
                 core_labels = trained_clusters
        else:
            print_status("Warning: DBSCAN model missing 'core_sample_indices_' or no core samples found. Using all points for prediction.")
            core_training_data = training_data
            core_labels = trained_clusters

        super().__init__(cluster_model, trained_clusters, core_training_data) # Fit NN on core samples
        self.core_labels = core_labels
        self.noise_label = -1

     def predict(self, X):
        if not hasattr(self.nn, '_fit_method') or self.nn._fit_method is None:
             raise RuntimeError("DBSCAN predictor (NearestNeighbors) was not fitted successfully.")
        if X.shape[0] == 0: return np.array([], dtype=self.core_labels.dtype)
        if self.training_data.shape[0] == 0: # No data to predict from
            print_status("Warning: DBSCAN Predictor has no training data. Returning noise label.")
            return np.full(X.shape[0], self.noise_label, dtype=self.core_labels.dtype)

        _, indices = self.nn.kneighbors(X)
        # Ensure indices are valid for core_labels
        valid_indices = indices.flatten()
        if len(valid_indices) > 0 and np.max(valid_indices) >= len(self.core_labels):
             print_status("Warning: NN indices out of bounds for core_labels in DBSCAN predictor. Clipping.")
             valid_indices = np.clip(valid_indices, 0, len(self.core_labels) - 1)

        if len(self.core_labels) > 0:
            predicted_labels = self.core_labels[valid_indices]
        else: # No core labels available
            predicted_labels = np.full(X.shape[0], self.noise_label, dtype=int)


        return predicted_labels


class ScaffoldPredictor(BasePredictor):
    def __init__(self, scaffold_to_cluster, default_cluster=0):
        super().__init__(cluster_model=None)
        self.scaffold_to_cluster = scaffold_to_cluster
        self.default_cluster = default_cluster

    @staticmethod
    def _process_scaffold(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                scaffold = Scaffolds.MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
                return scaffold
        except Exception as e:
            pass
        return ""

    def predict(self, smiles_list):
        if not isinstance(smiles_list, (list, np.ndarray, pd.Series)):
             raise TypeError("Input must be a list, numpy array, or pandas Series of SMILES strings.")

        clusters = np.full(len(smiles_list), self.default_cluster, dtype=int)
        for i, smiles in enumerate(smiles_list):
            if not isinstance(smiles, str):
                clusters[i] = self.default_cluster
                continue
            scaffold = self._process_scaffold(smiles)
            clusters[i] = self.scaffold_to_cluster.get(scaffold, self.default_cluster)

        return clusters

class FunctionalGroupPredictor(BasePredictor):
    _functional_groups = {
        'acid': '[CX3](=O)[OX2H1]', 'base_amine': '[NX3;H2,H1,H0;!$(NC=O)]', 'amide': '[NX3][CX3](=[OX1])',
        'ester': '[#6][CX3](=O)[OX2][#6]', 'ketone': '[#6][CX3](=O)[#6]', 'aldehyde': '[CX3H1](=O)',
        'alcohol': '[OX2H]', 'phenol': '[OX2H][cX3]', 'ether': '[OX2]([#6])[#6]',
        'sulfonamide': '[SX4](=[OX1])(=[OX1])([#6,#1])[NX3]', 'sulfonyl': '[SX4](=[OX1])(=[OX1])([#6])[#6]',
        'sulfoxide': '[SX3](=[OX1])([#6])[#6]', 'nitro': '[$([NX3](=O)=O),$([NX3+](=O)[O-])]',
        'nitrile': '[NX1]#[CX2]', 'halogen': '[F,Cl,Br,I]', 'aromatic': 'a', 'heterocycle': '[a;!c]',
        'carbamate': '[NX3][CX3](=[OX1])[OX2]', 'phosphate': '[PX4](=[OX1])([OX2])([OX2])[OX2]',
    }
    _patterns = {}
    for name, smarts in _functional_groups.items():
        try:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern: _patterns[name] = pattern
        except Exception: pass

    def __init__(self, vectorizer, cluster_model):
        super().__init__(cluster_model)
        if not hasattr(vectorizer, 'transform'):
             raise TypeError("Vectorizer object must have a 'transform' method.")
        self.vectorizer = vectorizer

    @classmethod
    def _detect_functional_groups(cls, smiles):
        features = {}
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            for name, pattern in cls._patterns.items():
                try: features[name] = 1 if mol.HasSubstructMatch(pattern) else 0
                except Exception: features[name] = 0
        else:
            for name in cls._patterns.keys(): features[name] = 0
        return features

    def predict(self, smiles_list):
        if isinstance(smiles_list, str): smiles_list = [smiles_list]
        elif not isinstance(smiles_list, (list, np.ndarray, pd.Series)):
             raise TypeError("Input must be a string or list/array/Series of SMILES strings.")
        if len(smiles_list) == 0: return np.array([], dtype=int)

        features_list = [self._detect_functional_groups(smi) for smi in smiles_list]
        if not hasattr(self.vectorizer, 'vocabulary_') or not self.vectorizer.vocabulary_:
             raise RuntimeError("FunctionalGroupPredictor's vectorizer must be fitted before prediction.")
        features_vector = self.vectorizer.transform(features_list)
        clusters = self.cluster_model.predict(features_vector)
        return clusters

class ChemicalPropertyPredictor(BasePredictor):
    _property_calculators = {
        "carbon_mw_percent": lambda mol: (sum(12.01 for a in mol.GetAtoms() if a.GetAtomicNum() == 6) / Descriptors.MolWt(mol)) if Descriptors.MolWt(mol) > 0 else 0,
        "hydrophobicity": Descriptors.MolLogP,
        "carbon_percent": lambda mol: (sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6) / mol.GetNumAtoms()) if mol.GetNumAtoms() > 0 else 0,
        "halogen_percent": lambda mol: (sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() in [9, 17, 35, 53]) / mol.GetNumAtoms()) if mol.GetNumAtoms() > 0 else 0,
        "halogen_carbon_percent": lambda mol: (sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() in [9, 17, 35, 53]) / sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6)) if sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6) > 0 else 0,
        "heteroatom_percent": lambda mol: (sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() not in [1, 6]) / mol.GetNumAtoms()) if mol.GetNumAtoms() > 0 else 0,
        "halogen_mw_percent": lambda mol: (sum({9: 18.998, 17: 35.453, 35: 79.904, 53: 126.904}.get(a.GetAtomicNum(), 0) for a in mol.GetAtoms()) / Descriptors.MolWt(mol)) if Descriptors.MolWt(mol) > 0 else 0,
        "heteroatom_mw_percent": lambda mol: (sum({7: 14.007, 8: 15.999, 9: 18.998, 15: 30.974, 16: 32.06, 17: 35.453, 35: 79.904, 53: 126.904}.get(a.GetAtomicNum(), 0) for a in mol.GetAtoms() if a.GetAtomicNum() not in [1, 6]) / Descriptors.MolWt(mol)) if Descriptors.MolWt(mol) > 0 else 0,
        "polarity_index": lambda mol: (sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() in [7, 8, 15, 16]) / mol.GetNumAtoms()) if mol.GetNumAtoms() > 0 else 0,
        "electronegativity_sum": lambda mol: (sum({1: 2.2, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98, 15: 2.19, 16: 2.58, 17: 3.16, 35: 2.96, 53: 2.66}.get(a.GetAtomicNum(), 0) for a in mol.GetAtoms()) / mol.GetNumAtoms()) if mol.GetNumAtoms() > 0 else 0,
        "acid_base": lambda mol: sum(1 for _ in mol.GetSubstructMatches(Chem.MolFromSmarts('[NX3;H2,H1,H0;!$(NC=O)]'), uniquify=True)) - sum(1 for _ in mol.GetSubstructMatches(Chem.MolFromSmarts('[CX3](=O)[OX2H1]'), uniquify=True))
    }

    def __init__(self, property_type, scaler, cluster_model):
        super().__init__(cluster_model)
        if property_type not in self._property_calculators:
            raise ValueError(f"Unsupported property type: {property_type}")
        if not hasattr(scaler, 'transform'):
            raise TypeError("Scaler object must have a 'transform' method.")
        self.property_type = property_type
        self.scaler = scaler
        self.calculator = self._property_calculators[property_type]

    def _calculate_property(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return 0.0
        try:
            return self.calculator(mol)
        except Exception as e:
            return 0.0

    def predict(self, smiles_list):
        if isinstance(smiles_list, str): smiles_list = [smiles_list]
        elif not isinstance(smiles_list, (list, np.ndarray, pd.Series)):
             raise TypeError("Input must be a string or list/array/Series of SMILES strings.")
        if len(smiles_list) == 0: return np.array([], dtype=int)

        property_features = np.array([self._calculate_property(smi) for smi in smiles_list]).reshape(-1, 1)
        if not hasattr(self.scaler, 'mean_'):
             raise RuntimeError("Scaler must be fitted before prediction.")
        property_features = np.nan_to_num(property_features, nan=0.0, posinf=0.0, neginf=0.0)
        scaled_features = self.scaler.transform(property_features)
        clusters = self.cluster_model.predict(scaled_features)
        return clusters


class StratifiedPredictor(BasePredictor):
    def __init__(self, pca, bins, n_bins_per_component, n_clusters):
        super().__init__(cluster_model=None)
        if not hasattr(pca, 'transform'): raise TypeError("PCA object must have transform method.")
        self.pca = pca
        self.bins = bins
        self.n_bins_per_component = n_bins_per_component
        self.n_clusters = n_clusters

    def predict(self, X):
        if not hasattr(self.pca, 'mean_'):
            raise RuntimeError("PCA must be fitted before prediction.")
        if X.shape[0] == 0: return np.array([], dtype=int)
        X_pca = self.pca.transform(X)
        clusters = np.zeros(X.shape[0], dtype=int)
        n_components = X_pca.shape[1]

        for i in range(X.shape[0]):
            cluster_id = 0
            multiplier = 1
            for j in range(n_components):
                component_val = X_pca[i, j]
                # Ensure bins[j] exists and is valid
                if j >= len(self.bins) or not isinstance(self.bins[j], np.ndarray):
                    raise RuntimeError(f"Invalid bins configuration for component {j} in StratifiedPredictor.")
                bin_idx = np.digitize(component_val, self.bins[j])
                cluster_id += bin_idx * multiplier
                multiplier *= self.n_bins_per_component

            clusters[i] = min(cluster_id, self.n_clusters - 1)

        return clusters


class ErrorBasedPredictor(BasePredictor):
    def __init__(self, scaler, cluster_model):
        super().__init__(cluster_model)
        if not hasattr(scaler, 'transform'): raise TypeError("Scaler must have transform method.")
        self.scaler = scaler

    def predict(self, X_with_error):
        if X_with_error.shape[1] < 2:
             raise ValueError("Input for ErrorBasedPredictor needs at least features and error column.")
        if X_with_error.shape[0] == 0: return np.array([], dtype=int)
        error_features = X_with_error[:, -1].reshape(-1, 1)
        if not hasattr(self.scaler, 'mean_'):
             raise RuntimeError("Scaler must be fitted before prediction.")
        scaled_errors = self.scaler.transform(error_features)
        return self.cluster_model.predict(scaled_errors)


# --- Fingerprint Calculation ---

def _calc_fingerprints(smiles, fp_types=None):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None

    results = {}
    fp_types = fp_types if fp_types is not None else ["morgan"]

    fp_type_map = {
        "morgan": ("Morgan", lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024), 1024),
        "maccs": ("MACCS", MACCSkeys.GenMACCSKeys, 167),
        "rdkit": ("RDKit", lambda m: Chem.RDKFingerprint(m, fpSize=1024), 1024),
        "atom_pairs": ("AtomPair", lambda m: AllChem.GetHashedAtomPairFingerprintAsBitVect(m, nBits=1024), 1024),
        "topological_torsion": ("Torsion", lambda m: AllChem.GetHashedTopologicalTorsionFingerprintAsBitVect(m, nBits=1024), 1024),
        "pattern": ("Pattern", lambda m: Chem.PatternFingerprint(m, fpSize=1024), 1024),
    }
    if "avalon" in fp_types:
        try:
            from rdkit.Avalon import pyAvalonTools
            fp_type_map["avalon"] = ("Avalon", lambda m: pyAvalonTools.GetAvalonFP(m, 1024), 1024)
        except ImportError:
            if "avalon" in fp_types: fp_types.remove("avalon")

    for fp_type_key in fp_types:
        fp_type_key_lower = fp_type_key.lower()
        if fp_type_key_lower in fp_type_map:
            name, generator, n_bits = fp_type_map[fp_type_key_lower]
            try:
                fp = generator(mol)
                fp_arr = np.zeros((n_bits,))
                DataStructs.ConvertToNumpyArray(fp, fp_arr)
                results[name] = fp_arr
            except Exception as e:
                results[name] = np.zeros((n_bits,))

    return results if results else None


# --- Structural Feature Detection ---
_functional_group_patterns = {}
_functional_group_definitions = {
    'acid': '[CX3](=O)[OX2H1]', 'base_amine': '[NX3;H2,H1,H0;!$(NC=O)]', 'amide': '[NX3][CX3](=[OX1])',
    'ester': '[#6][CX3](=O)[OX2][#6]', 'ketone': '[#6][CX3](=O)[#6]', 'aldehyde': '[CX3H1](=O)',
    'alcohol': '[OX2H]', 'phenol': '[OX2H][cX3]', 'ether': '[OX2]([#6])[#6]',
    'sulfonamide': '[SX4](=[OX1])(=[OX1])([#6,#1])[NX3]', 'sulfonyl': '[SX4](=[OX1])(=[OX1])([#6])[#6]',
    'sulfoxide': '[SX3](=[OX1])([#6])[#6]', 'nitro': '[$([NX3](=O)=O),$([NX3+](=O)[O-])]',
    'nitrile': '[NX1]#[CX2]', 'halogen': '[F,Cl,Br,I]', 'aromatic': 'a', 'heterocycle': '[a;!c]',
    'carbamate': '[NX3][CX3](=[OX1])[OX2]', 'phosphate': '[PX4](=[OX1])([OX2])([OX2])[OX2]',
}
for name, smarts in _functional_group_definitions.items():
    try:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern: _functional_group_patterns[name] = pattern
    except Exception: pass

def detect_structural_features(smiles):
    features = {name: 0 for name in _functional_group_patterns.keys()}
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        for name, pattern in _functional_group_patterns.items():
            try: features[name] = 1 if mol.HasSubstructMatch(pattern) else 0
            except Exception: features[name] = 0
    return features

# --- Scaffold Processing ---
def process_scaffold(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            scaffold = Scaffolds.MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            return scaffold
    except Exception: pass
    return ""


# --- Main Ensemble Class ---

class ClusteredXGBoostEnsemble:
    def __init__(self, output_dir="models/clustered_ensemble", seed=42):
        self.output_dir = Path(output_dir)
        self.seed = seed
        self.cluster_models = {}
        self.global_model = None
        self.meta_model = None
        self.cluster_model_predictor = None
        self.cluster_model = None # Underlying sklearn cluster model instance
        self.scaler = None
        self.property_scaler = None
        self.functional_group_vectorizer = None
        self.pca = None
        self.feature_cols = None
        self.important_indices = None
        self.important_features = None
        self.feature_importance_scores = None
        self.clusters = None
        self.n_clusters = None
        self.bins = None
        self.n_bins_per_component = None
        self.scaffold_to_cluster = None
        self.default_cluster = 0
        self.clustering_method = 'none'
        self.cluster_property = None
        self.n_features_used_in_model = None
        self.xgb_params = {}
        self.min_cluster_size = 10
        self.fp_types = ["morgan"]
        self.skip_visualization = False
        self.skip_meta_features = False # Ensure default is False
        self.replace_nan = None
        self.cluster_feature_cols = None
        self.smiles_col = "SMILES"
        self.target_col = "LogD"
        self.feature_method = "xgb"
        self.n_features = None
        self.use_model_mean = False
        self.use_weighted_mean = False
        self.group_feature_per_cluster = False
        self.cluster_specific_feature_indices = {}
        self.df_columns_ = None
        self.cluster_stats = {}
        self.model_comparison = {}
        self.error_analysis = {}
        self.meta_feature_structure = {}
        self.meta_feature_names = []
        self.user_specified_params = set()
        self.tuned_params = None

        self.output_dir.mkdir(parents=True, exist_ok=True)
        np.random.seed(seed)
        random.seed(seed)
        print_status(f"ClusteredXGBoostEnsemble initialized. Output dir: {self.output_dir}, Seed: {self.seed}")

    def _get_n_jobs(self, n_jobs_arg):
        if n_jobs_arg == -1:
            cpu_count = os.cpu_count()
            return cpu_count if cpu_count is not None else 1
        elif n_jobs_arg is None:
             return 1
        return max(1, int(n_jobs_arg))


    def _generate_fingerprints_and_features(self, smiles_list, fp_types, n_jobs=1):
        actual_n_jobs = self._get_n_jobs(n_jobs)
        # print_status(f"Generating features for {len(smiles_list)} compounds using {actual_n_jobs} processes.") # Reduce noise
        # if fp_types: print_status(f"Requested fingerprint types: {', '.join(fp_types)}")
        # else: print_status("No fingerprint types requested.")

        fp_results = [None] * len(smiles_list)
        if fp_types:
            # tracker_fp = ProgressTracker(len(smiles_list), "Calculating fingerprints") # Reduce noise
            with ProcessPoolExecutor(max_workers=actual_n_jobs) as executor:
                fp_compute_partial = partial(_calc_fingerprints, fp_types=fp_types)
                fp_results = list(executor.map(fp_compute_partial, smiles_list)) # Collect results
                # for i, result in enumerate(results_iter): fp_results[i] = result; tracker_fp.update()
            # tracker_fp.finalize()

        # tracker_sf = ProgressTracker(len(smiles_list), "Analyzing structural features") # Reduce noise
        struct_results = []
        with ProcessPoolExecutor(max_workers=actual_n_jobs) as executor:
            results_iter = executor.map(detect_structural_features, smiles_list)
            struct_results = list(results_iter)
            # for result in results_iter: struct_results.append(result); tracker_sf.update()
        # tracker_sf.finalize()

        # print_status("Combining features into matrix") # Reduce noise
        feature_names = []
        feature_name_set = set()
        fp_arrays = []
        fp_order = []
        ref_result_fp = next((r for r in fp_results if r is not None), None)
        if ref_result_fp:
            fp_order = list(ref_result_fp.keys())
            fp_sizes = {name: len(ref_result_fp[name]) for name in fp_order}
            default_fps = {name: np.zeros(fp_sizes[name]) for name in fp_order}
            fp_data_dict = {name: [] for name in fp_order}
            for result in fp_results:
                item = result if result else {}
                for fp_name in fp_order:
                    fp_data_dict[fp_name].append(item.get(fp_name, default_fps[fp_name]))
            for fp_name in fp_order:
                 current_fp_names = [f"{fp_name}_{i}" for i in range(fp_sizes[fp_name])]
                 feature_names.extend(current_fp_names)
                 feature_name_set.update(current_fp_names)
                 fp_arrays.append(np.array(fp_data_dict[fp_name]))

        struct_df = pd.DataFrame(struct_results).fillna(0)
        struct_feature_names_orig = list(struct_df.columns)
        unique_struct_names = []
        for name in struct_feature_names_orig:
            final_name = name
            suffix = 1
            while final_name in feature_name_set: final_name = f"{name}_{suffix}"; suffix += 1
            unique_struct_names.append(final_name)
            feature_name_set.add(final_name)
        feature_names.extend(unique_struct_names)
        struct_array = struct_df.values

        if fp_arrays and struct_array.size > 0:
             all_features_array = np.concatenate(fp_arrays + [struct_array], axis=1)
        elif fp_arrays: all_features_array = np.concatenate(fp_arrays, axis=1)
        elif struct_array.size > 0: all_features_array = struct_array
        else: all_features_array = np.zeros((len(smiles_list), 0))

        # print_status(f"Generated feature matrix with shape: {all_features_array.shape}") # Reduce noise
        if len(feature_names) != all_features_array.shape[1]:
             print_status(f"Critical Warning: Mismatch feature names ({len(feature_names)}) vs array cols ({all_features_array.shape[1]}).")
             feature_names = [f"feature_{i}" for i in range(all_features_array.shape[1])] if all_features_array.shape[1] > 0 else []

        return all_features_array, feature_names

    def prepare_data(self, df, smiles_col="SMILES", target_col="LogD", fp_types=None, n_jobs=1):
        print_status("Preparing data...")
        self.smiles_col = smiles_col; self.target_col = target_col
        self.df_columns_ = list(df.columns)
        if smiles_col not in df.columns: raise ValueError(f"SMILES column '{smiles_col}' not found.")
        if target_col not in df.columns: raise ValueError(f"Target column '{target_col}' not found.")

        original_len = len(df)
        df = df.dropna(subset=[smiles_col])
        if len(df) < original_len: print_status(f"Dropped {original_len - len(df)} rows with missing SMILES.")
        if len(df) == 0: raise ValueError("No valid data after dropping missing SMILES.")

        smiles_list = df[smiles_col].tolist()
        y = df[target_col].values

        nan_indices_y = np.isnan(y)
        if np.any(nan_indices_y):
             num_nan_y = np.sum(nan_indices_y)
             print_status(f"Warning: Found {num_nan_y} NaN values in target column '{target_col}'.")
             if self.replace_nan is not None:
                 print_status(f"Replacing NaN target values with {self.replace_nan}.")
                 y = np.nan_to_num(y, nan=self.replace_nan)
             else:
                 print_status("Removing rows with NaN target values.")
                 df = df[~nan_indices_y]; smiles_list = df[smiles_col].tolist(); y = df[target_col].values
                 print_status(f"Data reduced to {len(df)} rows.")
                 if len(df) == 0: raise ValueError("No valid data after removing NaN target values.")

        descriptor_columns = [col for col in df.columns if col not in [smiles_col, target_col]]
        if descriptor_columns:
            print_status(f"Using {len(descriptor_columns)} existing descriptor columns.")
            X = df[descriptor_columns].values
            self.feature_cols = descriptor_columns
        else:
            print_status("Generating fingerprints and structural features.")
            fp_types = fp_types if fp_types is not None else self.fp_types
            X, self.feature_cols = self._generate_fingerprints_and_features(smiles_list, fp_types, n_jobs)

        if np.isnan(X).any():
            nan_count_X = np.isnan(X).sum()
            print_status(f"Found {nan_count_X} NaN values in feature matrix.")
            replace_val = self.replace_nan if self.replace_nan is not None else 0.0
            print_status(f"Replacing NaN feature values with {replace_val}.")
            X = np.nan_to_num(X, nan=replace_val)

        print_status(f"Final data shape: X = {X.shape}, y = {y.shape}")
        if X.shape[0] != y.shape[0]: raise ValueError(f"Row mismatch: Features={X.shape[0]}, Target={y.shape[0]}.")
        return X, y, smiles_list, self.feature_cols


    def identify_important_features(self, X, y, n_features=None, method='xgb', n_jobs=1):
        start_time = time.time()
        actual_n_jobs = self._get_n_jobs(n_jobs)
        # print_status(f"Identifying important features using '{method}' (n_jobs={actual_n_jobs}).") # Reduce noise
        self.n_features = n_features; self.feature_method = method

        if X.shape[1] == 0:
             print_status("Warning: Feature matrix empty. Cannot select features.")
             self.important_indices = []; self.important_features = []; self.feature_importance_scores = np.array([])
             self.n_features_used_in_model = 0; return np.array([])

        if n_features is None or n_features >= X.shape[1] or method == 'all':
            # print_status(f"Using all {X.shape[1]} features.") # Reduce noise
            selected_indices = np.arange(X.shape[1])
            importances = np.ones(X.shape[1]) # Placeholder importance
            method_used = 'all'; n_features_selected = X.shape[1]
        else:
            # print_status(f"Selecting top {n_features} features from {X.shape[1]}.") # Reduce noise
            method_used = method; n_features_selected = n_features
            selected_indices = np.array([], dtype=int) # Initialize
            importances = np.zeros(X.shape[1]) # Initialize
            try:
                 if method == 'xgb':
                      model = XGBRegressor(random_state=self.seed, n_jobs=actual_n_jobs).fit(X, y)
                      importances = model.feature_importances_
                 elif method == 'rf':
                      model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=self.seed, n_jobs=actual_n_jobs).fit(X, y)
                      importances = model.feature_importances_
                 elif method == 'mutual_info':
                      selector = SelectKBest(mutual_info_regression, k=min(n_features, X.shape[1])) # Ensure k <= n_features
                      selector.fit(X, y.ravel()); selected_indices = selector.get_support(indices=True)
                      importances[selected_indices] = selector.scores_[selected_indices] # Assign scores correctly
                 elif method == 'permutation':
                      model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=self.seed, n_jobs=actual_n_jobs).fit(X, y)
                      perm_importance = permutation_importance(model, X, y, n_repeats=5, random_state=self.seed, n_jobs=actual_n_jobs)
                      importances = perm_importance.importances_mean
                 elif method == 'shap':
                      model = XGBRegressor(random_state=self.seed, n_jobs=actual_n_jobs).fit(X, y)
                      explainer = shap.TreeExplainer(model); shap_values = explainer.shap_values(X)
                      importances = np.abs(shap_values).mean(axis=0)
                 elif method == 'random':
                      selected_indices = np.random.choice(X.shape[1], size=n_features, replace=False)
                      importances[selected_indices] = 1.0
                 else: raise ValueError(f"Unknown feature selection method: {method}")

                 # If indices weren't set by selector (e.g., for xgb, rf), get them now
                 if len(selected_indices) == 0 and method != 'random':
                      selected_indices = np.argsort(importances)[::-1][:n_features]

            except Exception as e:
                 print_status(f"Warning: Feature selection method '{method}' failed ({e}). Falling back to random.")
                 method_used = 'random'
                 selected_indices = np.random.choice(X.shape[1], size=n_features, replace=False)
                 importances = np.zeros(X.shape[1]); importances[selected_indices] = 1.0


        self.important_indices = selected_indices.tolist()
        if self.feature_cols and len(self.feature_cols) == X.shape[1]:
            self.important_features = [self.feature_cols[i] for i in self.important_indices]
            self.feature_importance_scores = importances
        else:
             # print_status("Warning: Feature names unavailable/mismatched. Using generic names.") # Reduce noise
             self.important_features = [f"feature_{i}" for i in self.important_indices]
             self.feature_importance_scores = importances

        self.n_features_used_in_model = len(self.important_indices)

        end_time = time.time()
        # print_status(f"Feature selection ('{method_used}') completed in {end_time - start_time:.2f}s. Selected {self.n_features_used_in_model} features.") # Reduce noise

        # Return only the selected columns/features
        # Handle empty selection case
        if not self.important_indices:
             return np.zeros((X.shape[0], 0))
        # Ensure indices are valid before slicing
        if max(self.important_indices) >= X.shape[1]:
             raise IndexError(f"Feature selection indices {self.important_indices} out of bounds for X shape {X.shape}")
        return X[:, self.important_indices]


    def _get_clustering_data(self, X, smiles_list=None, method='kmeans', n_clusters=5, cluster_property=None):
        if method == 'functional':
            if smiles_list is None: raise ValueError("Functional group clustering requires SMILES list.")
            features_list = [FunctionalGroupPredictor._detect_functional_groups(smi) for smi in smiles_list]
            self.functional_group_vectorizer = DictVectorizer(sparse=False)
            return self.functional_group_vectorizer.fit_transform(features_list)
        elif method == 'scaffold': return None
        elif method in ChemicalPropertyPredictor._property_calculators:
             if smiles_list is None: raise ValueError(f"Property clustering '{method}' requires SMILES.")
             prop_predictor = ChemicalPropertyPredictor(method, StandardScaler(), None)
             clustering_data = np.array([prop_predictor._calculate_property(smi) for smi in smiles_list]).reshape(-1, 1)
             clustering_data = np.nan_to_num(clustering_data, nan=0.0, posinf=0.0, neginf=0.0)
             self.property_scaler = StandardScaler(); return self.property_scaler.fit_transform(clustering_data)
        elif method == 'property':
             if cluster_property and cluster_property in ChemicalPropertyPredictor._property_calculators:
                  self.cluster_property = cluster_property
                  return self._get_clustering_data(X, smiles_list, method=cluster_property)
             else:
                  self.scaler = StandardScaler(); X_scaled = self.scaler.fit_transform(X)
                  n_components = min(5, X_scaled.shape[1])
                  if n_components < 1: raise ValueError("Cannot perform PCA with < 1 features.")
                  self.pca = PCA(n_components=n_components, random_state=self.seed)
                  return self.pca.fit_transform(X_scaled)
        elif method == 'error_based': raise NotImplementedError("Error-based clustering requires integration.")
        else: # Methods using main features (kmeans, dbscan, etc.)
             if self.scaler is None or not hasattr(self.scaler, 'mean_'): # Fit scaler only if needed
                 # print_status("Standardizing features before clustering.") # Reduce noise
                 self.scaler = StandardScaler()
                 return self.scaler.fit_transform(X)
             else: # Assume scaler already fitted or not needed
                  if hasattr(self.scaler, 'mean_'): return self.scaler.transform(X)
                  else: return X # Return unscaled if no fitted scaler


    def cluster_compounds(self, X, y=None, smiles_list=None, n_clusters=5, method='kmeans', cluster_property=None, min_cluster_size=10):
        # print_status(f"Clustering {X.shape[0]} compounds into {n_clusters} groups using '{method}' method.") # Reduce noise
        self.n_clusters = n_clusters; self.clustering_method = method; self.min_cluster_size = min_cluster_size
        self.cluster_model = None; self.cluster_model_predictor = None # Reset

        X_cluster_visualization = X # Default visualization input

        if method == 'scaffold':
            if smiles_list is None: raise ValueError("Scaffold clustering requires SMILES list.")
            actual_n_jobs = self._get_n_jobs(self.xgb_params.get('n_jobs', 1))
            if len(smiles_list) > 10000 and actual_n_jobs > 1:
                 with ProcessPoolExecutor(max_workers=actual_n_jobs) as executor: scaffolds = list(executor.map(process_scaffold, smiles_list))
            else: scaffolds = [process_scaffold(smi) for smi in smiles_list]
            unique_scaffolds, scaffold_counts = np.unique(scaffolds, return_counts=True)
            # print_status(f"Found {len(unique_scaffolds)} unique scaffolds.") # Reduce noise
            self.scaffold_to_cluster = {}; sorted_indices = np.argsort(scaffold_counts)[::-1]
            self.default_cluster = n_clusters - 1
            for i, idx in enumerate(sorted_indices): self.scaffold_to_cluster[unique_scaffolds[idx]] = min(i, self.default_cluster)
            self.scaffold_to_cluster.setdefault("", self.default_cluster)
            clusters = np.array([self.scaffold_to_cluster.get(sc, self.default_cluster) for sc in scaffolds], dtype=int)
            self.cluster_model_predictor = ScaffoldPredictor(self.scaffold_to_cluster, self.default_cluster)
            X_cluster_visualization = X

        else:
             try:
                 clustering_data = self._get_clustering_data(X, smiles_list, method, n_clusters, cluster_property)
                 X_cluster_visualization = clustering_data # Use data used for clustering for visualization
             except (NotImplementedError, ValueError) as e:
                 print_status(f"Error preparing data for clustering method '{method}': {e}")
                 return None

             if clustering_data.shape[0] < n_clusters:
                 print_status(f"Warning: Number of samples ({clustering_data.shape[0]}) is less than n_clusters ({n_clusters}). Reducing n_clusters.")
                 n_clusters = max(1, clustering_data.shape[0]) # At least 1 cluster
                 self.n_clusters = n_clusters


             actual_model = None
             predictor = None

             if method == 'kmeans':
                 actual_model = KMeans(n_clusters=n_clusters, random_state=self.seed, n_init='auto') # Use 'auto'
                 clusters = actual_model.fit_predict(clustering_data)
                 predictor = KMeansPredictor(actual_model)
             elif method == 'random':
                 predictor = RandomClusterPredictor(n_clusters, self.seed)
                 clusters = predictor.predict(clustering_data)
             elif method == 'agglomerative':
                 actual_model = AgglomerativeClustering(n_clusters=n_clusters)
                 clusters = actual_model.fit_predict(clustering_data)
                 predictor = AgglomerativePredictor(actual_model, clusters, clustering_data)
             elif method == 'spectral':
                 affinity = 'nearest_neighbors' if clustering_data.shape[0] > 2000 else 'rbf'
                 n_neighbors = min(10, max(2, clustering_data.shape[0]-1)) # Ensure 2 <= n_neighbors < n_samples
                 if clustering_data.shape[0] <= n_neighbors: affinity = 'rbf' # Fallback if too few samples

                 try:
                     actual_model = SpectralClustering(n_clusters=n_clusters, random_state=self.seed,
                                                    affinity=affinity, n_neighbors=n_neighbors, assign_labels='kmeans')
                     clusters = actual_model.fit_predict(clustering_data)
                     predictor = SpectralPredictor(actual_model, clusters, clustering_data)
                 except Exception as e:
                    print_status(f"Spectral clustering failed: {e}. Falling back to KMeans.")
                    method = 'kmeans'
                    self.clustering_method = method # Update state if falling back
                    actual_model = KMeans(n_clusters=n_clusters, random_state=self.seed, n_init='auto')
                    clusters = actual_model.fit_predict(clustering_data)
                    predictor = KMeansPredictor(actual_model)

             elif method == 'gaussian':
                 actual_model = GaussianMixture(n_components=n_clusters, random_state=self.seed, covariance_type='full')
                 actual_model.fit(clustering_data)
                 clusters = actual_model.predict(clustering_data)
                 predictor = GaussianMixturePredictor(actual_model)
             elif method == 'stratified':
                 n_components = min(3, clustering_data.shape[1])
                 if n_components < 1:
                     print_status("Warning: Cannot perform PCA for stratified clustering with < 1 features. Falling back to random.")
                     method = 'random'; self.clustering_method = method
                     predictor = RandomClusterPredictor(n_clusters, self.seed)
                     clusters = predictor.predict(clustering_data)
                 else:
                    pca_strat = PCA(n_components=n_components, random_state=self.seed)
                    X_pca = pca_strat.fit_transform(clustering_data)
                    n_bins_p_comp = max(2, int(np.ceil(n_clusters**(1.0 / n_components))))
                    self.n_bins_per_component = n_bins_p_comp
                    self.bins = [] # Store list of bin edges
                    for i in range(n_components):
                        component = X_pca[:, i]
                        percentiles = np.linspace(0, 100, n_bins_p_comp + 1)[1:-1]
                        bin_edges = np.percentile(component, percentiles)
                        self.bins.append(bin_edges)

                    predictor = StratifiedPredictor(pca_strat, self.bins, n_bins_p_comp, n_clusters)
                    clusters = predictor.predict(clustering_data)
                    self.pca = pca_strat # Store fitted PCA needed by predictor

             elif method == 'dbscan':
                k = min(5, max(1, clustering_data.shape[0] - 1)) # Ensure k is valid
                nn = NearestNeighbors(n_neighbors=k)
                nn.fit(clustering_data)
                distances, _ = nn.kneighbors(clustering_data)
                k_distances = np.sort(distances[:, -1])
                # Simple percentile heuristic for eps
                eps_percentile = 95
                eps = np.percentile(k_distances, eps_percentile) if len(k_distances) > 0 else 0.5
                print_status(f"Estimated DBSCAN eps={eps:.4f} based on {k}-NN distances ({eps_percentile}th percentile).")
                dbscan_min_samples = max(5, int(clustering_data.shape[0] * 0.01))
                print_status(f"Using DBSCAN min_samples={dbscan_min_samples}")

                actual_model = DBSCAN(eps=eps, min_samples=dbscan_min_samples)
                clusters = actual_model.fit_predict(clustering_data)

                n_noise = np.sum(clusters == -1)
                n_found_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
                print_status(f"DBSCAN found {n_found_clusters} clusters and {n_noise} noise points.")

                if n_noise > 0 and hasattr(actual_model, 'core_sample_indices_') and len(actual_model.core_sample_indices_) > 0:
                     print_status("Assigning noise points to the nearest core sample cluster.")
                     noise_mask = (clusters == -1)
                     core_indices = actual_model.core_sample_indices_
                     nn_noise = NearestNeighbors(n_neighbors=1)
                     nn_noise.fit(clustering_data[core_indices])
                     _, indices = nn_noise.kneighbors(clustering_data[noise_mask])
                     nearest_core_labels = actual_model.labels_[core_indices[indices.flatten()]]
                     clusters[noise_mask] = nearest_core_labels
                     print_status(f"Reassigned {n_noise} noise points.")
                     n_found_clusters = len(set(clusters)) # Update count

                print_status(f"DBSCAN resulted in {n_found_clusters} final clusters.")
                predictor = DBSCANWithPredict(actual_model, clusters, clustering_data)

             elif method == 'functional':
                 kmeans_func = KMeans(n_clusters=n_clusters, random_state=self.seed, n_init='auto')
                 clusters = kmeans_func.fit_predict(clustering_data)
                 # Store the fitted cluster model itself if needed later (e.g. for saving)
                 self.cluster_model = kmeans_func # Store the underlying KMeans model
                 predictor = FunctionalGroupPredictor(self.functional_group_vectorizer, kmeans_func)
             elif method in ChemicalPropertyPredictor._property_calculators:
                 kmeans_prop = KMeans(n_clusters=n_clusters, random_state=self.seed, n_init='auto')
                 clusters = kmeans_prop.fit_predict(clustering_data)
                 self.cluster_model = kmeans_prop # Store underlying model
                 predictor = ChemicalPropertyPredictor(method, self.property_scaler, kmeans_prop)
             elif method == 'property': # Default property (PCA) case
                  kmeans_prop_pca = KMeans(n_clusters=n_clusters, random_state=self.seed, n_init='auto')
                  clusters = kmeans_prop_pca.fit_predict(clustering_data)
                  self.cluster_model = kmeans_prop_pca # Store underlying model
                  predictor = KMeansPredictor(kmeans_prop_pca) # Predicts on PCA data
             else:
                 raise ValueError(f"Clustering method '{method}' is defined but not fully implemented.")

             self.cluster_model_predictor = predictor
             # Store the underlying sklearn model if it exists (needed for saving/loading state)
             if actual_model:
                  self.cluster_model = actual_model


        self.clusters = clusters
        self._analyze_clusters(clusters, y, smiles_list)
        self._visualize_clusters(X_cluster_visualization, clusters, method)

        return clusters

    def _visualize_clusters(self, X_vis, clusters, method_name):
        if self.skip_visualization:
            print_status("Skipping cluster visualization (--skip-visualization).")
            return

        print_status(f"Visualizing clustering results ({method_name})")
        if X_vis is None or X_vis.shape[0] < 2 or X_vis.shape[1] < 1:
             print_status("Warning: Insufficient data for cluster visualization.")
             return
        unique_clusters_vis = np.unique(clusters)
        if len(unique_clusters_vis) < 1:
             print_status("Warning: No clusters found. Skipping visualization.")
             return
        if len(unique_clusters_vis) == 1:
             print_status("Warning: Only one cluster found. Skipping scatter plot visualization.")
             # Optionally plot histogram if 1D
             if X_vis.shape[1] == 1:
                 # (Add 1D plot code here if desired)
                 pass
             return


        try:
            if X_vis.shape[1] > 2:
                print_status("Reducing dimensions using t-SNE for visualization (may take time)...")
                n_components_pca = min(50, X_vis.shape[1])
                if n_components_pca > 1 : # Only do PCA if > 1 feature
                     if X_vis.shape[1] > n_components_pca:
                         print_status(f"Applying PCA to {n_components_pca} components before t-SNE...")
                         pca_vis = PCA(n_components=n_components_pca, random_state=self.seed)
                         X_vis_pca = pca_vis.fit_transform(X_vis)
                     else:
                         X_vis_pca = X_vis # No PCA needed if already <= 50
                else:
                     X_vis_pca = X_vis # Cannot do PCA on 1 feature


                # Adjust t-SNE perplexity based on sample size
                perplexity_val = min(30, max(5, X_vis_pca.shape[0] - 1))
                tsne = TSNE(n_components=2, random_state=self.seed, perplexity=perplexity_val, n_iter=300, init='pca', learning_rate='auto')
                X_2d = tsne.fit_transform(X_vis_pca)
            elif X_vis.shape[1] == 1:
                 print_status("Data is 1D, creating a histogram/density plot for visualization.")
                 plt.figure(figsize=(12, 6))
                 palette_hist = sns.color_palette("husl", len(unique_clusters_vis))
                 for i, cluster in enumerate(sorted(unique_clusters_vis)):
                      cluster_mask = (clusters == cluster)
                      if np.sum(cluster_mask) > 1: # Need >1 point for kdeplot
                           sns.kdeplot(X_vis[cluster_mask, 0], label=f'Cluster {cluster}', color=palette_hist[i], fill=True, alpha=0.5)
                      elif np.sum(cluster_mask) == 1: # Plot vertical line for single point
                           plt.axvline(X_vis[cluster_mask, 0], color=palette_hist[i], linestyle='--', label=f'Cluster {cluster} (1 pt)')

                 plt.xlabel('Feature Value (or Property / PCA Comp 1)')
                 plt.ylabel('Density')
                 plt.title(f'Cluster Distribution ({method_name})')
                 plt.legend()
                 plot_filename = self.output_dir / f"clusters_{method_name}_dist.png"
                 plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
                 plt.close()
                 print_status(f"Cluster distribution plot saved to {plot_filename}")
                 return
            else: # Already 2D
                X_2d = X_vis

            # --- Scatter Plot ---
            plt.figure(figsize=(12, 10))
            palette = sns.color_palette("husl", len(unique_clusters_vis))
            for i, cluster in enumerate(sorted(unique_clusters_vis)):
                cluster_mask = (clusters == cluster)
                plt.scatter(X_2d[cluster_mask, 0], X_2d[cluster_mask, 1],
                            label=f'Cluster {cluster} ({np.sum(cluster_mask)})',
                            color=palette[i], alpha=0.7, s=10)

            plt.legend(title="Clusters (Size)", loc='best', fontsize='small')
            plt.xlabel("Dimension 1 (e.g., t-SNE 1 / PCA 1)")
            plt.ylabel("Dimension 2 (e.g., t-SNE 2 / PCA 2)")
            plt.title(f'Compound Clusters ({method_name})')
            plt.tight_layout()
            plot_filename = self.output_dir / f"clusters_{method_name}_scatter.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            print_status(f"Cluster scatter plot saved to {plot_filename}")

        except Exception as e:
            print_status(f"Error during cluster visualization: {e}")
            print_status(traceback.format_exc())


    def _analyze_clusters(self, clusters, y, smiles_list=None):
        print_status("Analyzing cluster properties...")
        self.cluster_stats = {}
        if clusters is None:
             print_status("Warning: No clusters assigned. Skipping analysis.")
             return
        unique_cluster_ids = sorted(np.unique(clusters))

        if y is None: print_status("Warning: Target variable 'y' not provided for cluster analysis.")

        for cluster_id in unique_cluster_ids:
            cluster_mask = (clusters == cluster_id)
            cluster_size = int(np.sum(cluster_mask))
            stats = {'size': cluster_size}

            if y is not None and cluster_size > 0:
                y_cluster = y[cluster_mask]
                stats['target_mean'] = float(np.mean(y_cluster))
                stats['target_std'] = float(np.std(y_cluster))
                stats['target_min'] = float(np.min(y_cluster))
                stats['target_max'] = float(np.max(y_cluster))
            else:
                 stats.update({'target_mean': np.nan, 'target_std': np.nan, 'target_min': np.nan, 'target_max': np.nan})

            if smiles_list is not None and cluster_size > 0:
                 cluster_smiles = np.array(smiles_list)[cluster_mask]
                 # Use a try-except for scaffold processing within list comprehension
                 scaffolds = [process_scaffold(smi) for smi in cluster_smiles if isinstance(smi, str)]
                 unique_scaffolds = set(s for s in scaffolds if s)
                 stats['num_unique_scaffolds'] = len(unique_scaffolds)
                 stats['scaffold_diversity'] = len(unique_scaffolds) / cluster_size if cluster_size > 0 else 0
            else:
                 stats.update({'num_unique_scaffolds': 0, 'scaffold_diversity': 0})


            self.cluster_stats[int(cluster_id)] = stats
            # Format N/A properly if target is missing
            target_mean_str = f"{stats['target_mean']:.3f}" if not np.isnan(stats['target_mean']) else "N/A"
            print_status(f"Cluster {cluster_id}: Size={stats['size']}, Target Mean={target_mean_str}, Num Scaffolds={stats.get('num_unique_scaffolds', 'N/A')}")


        try:
            stats_df = pd.DataFrame.from_dict(self.cluster_stats, orient='index')
            stats_df.index.name = 'ClusterID'
            stats_filename = self.output_dir / "cluster_analysis_stats.csv"
            stats_df.to_csv(stats_filename)
            print_status(f"Cluster analysis stats saved to {stats_filename}")
        except Exception as e:
            print_status(f"Error saving cluster analysis stats: {e}")

        print_status("Cluster analysis completed.")


    def create_model(self, params=None):
        """ Creates an XGBoost regressor instance. """
        model_params = params if params is not None else self.xgb_params
        # Define defaults robustly
        final_params = {
            'objective': 'reg:squarederror',
            'random_state': self.seed,
            'n_jobs': self._get_n_jobs(self.xgb_params.get('n_jobs', -1)), # Use stored config
            'max_depth': 6, 'learning_rate': 0.1, 'n_estimators': 100,
            'subsample': 1.0, 'colsample_bytree': 1.0, 'min_child_weight': 1,
            'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1,
        }
        # Update with provided or stored params, ensuring types are correct
        for key, value in model_params.items():
            if key in final_params: # Only update known params
                try:
                     # Attempt type conversion based on default type
                     default_type = type(final_params[key])
                     final_params[key] = default_type(value)
                except (ValueError, TypeError):
                     print_status(f"Warning: Could not convert param '{key}' value '{value}' to {default_type}. Using default.")
            # else: Ignore unknown params passed in

        return xgb.XGBRegressor(**final_params)


    def train_cluster_models(self, X, y, clusters):
        print_status("Training models for each cluster...")
        self.cluster_models = {}
        actual_n_jobs = self._get_n_jobs(self.xgb_params.get('n_jobs', 1))

        # Use selected features if available
        if self.important_indices is not None:
            print_status(f"Using {len(self.important_indices)} pre-selected important features for cluster models.")
            X_train_base = X[:, self.important_indices]
            base_features = self.important_features if self.important_features else [f"feature_{i}" for i in self.important_indices]
        else:
            print_status("Using all available features for cluster models.")
            X_train_base = X
            base_features = self.feature_cols if self.feature_cols else [f"feature_{i}" for i in range(X.shape[1])]

        if X_train_base.shape[1] == 0:
             print_status("Warning: Base feature matrix for cluster models is empty. Skipping training.")
             return {}


        unique_cluster_ids = sorted(np.unique(clusters))
        trained_count = 0
        self.cluster_specific_feature_indices = {} # Reset

        for cluster_id in unique_cluster_ids:
            cluster_mask = (clusters == cluster_id)
            cluster_size = np.sum(cluster_mask)

            print_status(f"Processing Cluster {cluster_id}: {cluster_size} samples.")

            if cluster_size < self.min_cluster_size:
                print_status(f"  Skipping training: Size ({cluster_size}) < min_cluster_size ({self.min_cluster_size}).")
                self.cluster_models[cluster_id] = None
                self.cluster_specific_feature_indices[cluster_id] = None
                continue

            X_cluster_base = X_train_base[cluster_mask]
            y_cluster = y[cluster_mask]

            cluster_feature_indices_rel_to_base = None # Indices relative to X_train_base
            X_cluster_final = X_cluster_base

            if self.group_feature_per_cluster:
                 # Default n_features for cluster-specific selection if main n_features not set
                 cluster_n_features = self.n_features if self.n_features is not None else min(100, X_cluster_base.shape[1]) # e.g., max 100
                 cluster_n_features = min(cluster_n_features, X_cluster_base.shape[1]) # Ensure <= available

                 if cluster_n_features < X_cluster_base.shape[1] and cluster_n_features > 0 :
                      print_status(f"  Performing feature selection for Cluster {cluster_id} (target: {cluster_n_features})...")
                      # Use a relatively fast method like RF or mutual_info
                      fs_method_cluster = 'rf'
                      try:
                           if fs_method_cluster == 'rf':
                                rf_c = RandomForestRegressor(n_estimators=50, random_state=self.seed, n_jobs=actual_n_jobs, max_depth=10)
                                rf_c.fit(X_cluster_base, y_cluster.ravel())
                                importances_c = rf_c.feature_importances_
                                cluster_feature_indices_rel_to_base = np.argsort(importances_c)[::-1][:cluster_n_features]
                           elif fs_method_cluster == 'mutual_info':
                                selector_c = SelectKBest(mutual_info_regression, k=cluster_n_features)
                                selector_c.fit(X_cluster_base, y_cluster.ravel())
                                cluster_feature_indices_rel_to_base = selector_c.get_support(indices=True)

                           X_cluster_final = X_cluster_base[:, cluster_feature_indices_rel_to_base]
                           self.cluster_specific_feature_indices[cluster_id] = cluster_feature_indices_rel_to_base # Store indices
                           print_status(f"  Selected {X_cluster_final.shape[1]} features for Cluster {cluster_id} using {fs_method_cluster}.")
                      except Exception as e_fs_c:
                           print_status(f"  Warning: Feature selection for Cluster {cluster_id} failed ({e_fs_c}). Using base features.")
                           self.cluster_specific_feature_indices[cluster_id] = None # Indicate failure / use base
                           X_cluster_final = X_cluster_base
                 else:
                      print_status(f"  Using base {X_cluster_base.shape[1]} features for Cluster {cluster_id} (selection not applicable).")
                      self.cluster_specific_feature_indices[cluster_id] = None # No specific selection done
            else:
                 self.cluster_specific_feature_indices[cluster_id] = None # Not using per-cluster selection

            if X_cluster_final.shape[1] == 0:
                 print_status(f"  Warning: Final feature set for Cluster {cluster_id} is empty. Skipping training.")
                 self.cluster_models[cluster_id] = None
                 continue


            model = self.create_model()
            model.fit(X_cluster_final, y_cluster)
            self.cluster_models[cluster_id] = model
            trained_count += 1

            y_pred_cluster_train = model.predict(X_cluster_final)
            rmse = np.sqrt(mean_squared_error(y_cluster, y_pred_cluster_train))
            r2 = r2_score(y_cluster, y_pred_cluster_train)
            mae = mean_absolute_error(y_cluster, y_pred_cluster_train)
            print_status(f"  Cluster {cluster_id} (In-Sample) - RMSE: {rmse:.4f}, R²: {r2:.4f}, MAE: {mae:.4f}")

        print_status(f"Trained models for {trained_count} clusters (out of {len(unique_cluster_ids)}).")
        return self.cluster_models

    def train_global_model(self, X, y):
         print_status("Training global model...")
         # Use selected features if available
         if self.important_indices is not None:
             print_status(f"Using {len(self.important_indices)} pre-selected important features.")
             # Ensure indices are valid for X
             if max(self.important_indices) >= X.shape[1]:
                  raise ValueError(f"Important feature indices ({max(self.important_indices)}) out of bounds for input X ({X.shape[1]}).")
             X_train = X[:, self.important_indices]
             current_n_features = X_train.shape[1]
         else:
             print_status("Using all available features.")
             X_train = X
             current_n_features = X.shape[1]

         if X_train.shape[1] == 0:
              print_status("Warning: Feature matrix for global model is empty. Skipping training.")
              self.global_model = None
              self.n_features_used_in_model = 0
              return None

         self.global_model = self.create_model()
         self.global_model.fit(X_train, y)
         # Store the actual number of features the final global model was trained on
         self.n_features_used_in_model = current_n_features
         print_status(f"Global model trained on {self.n_features_used_in_model} features.")

         y_pred_global_train = self.global_model.predict(X_train)
         rmse = np.sqrt(mean_squared_error(y, y_pred_global_train))
         r2 = r2_score(y, y_pred_global_train)
         mae = mean_absolute_error(y, y_pred_global_train)
         print_status(f"Global Model (In-Sample) - RMSE: {rmse:.4f}, R²: {r2:.4f}, MAE: {mae:.4f}")

         return self.global_model


    def train_ensemble_model(self, X, y, clusters):
        """ Trains a meta-model to combine predictions. """
        if self.skip_meta_features:
            print_status("Skipping meta-model training (--skip-meta-features).")
            self.meta_model = None
            return None

        print_status("Training ensemble meta-model...")
        actual_n_jobs = self._get_n_jobs(self.xgb_params.get('n_jobs', 1))
        n_folds = 5
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=self.seed)

        # Use globally selected features for base model training in CV
        if self.important_indices is not None:
            X_base = X[:, self.important_indices]
            print_status(f"Using {X_base.shape[1]} selected features for base model predictions in CV.")
        else:
            X_base = X
            print_status(f"Using all {X_base.shape[1]} features for base model predictions in CV.")

        if X_base.shape[1] == 0:
             print_status("Warning: Feature matrix for meta-feature generation is empty. Skipping meta-model.")
             self.meta_model = None
             return None

        oof_global_preds = np.zeros_like(y, dtype=float)
        oof_cluster_preds = np.zeros_like(y, dtype=float)

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_base, y)):
            print_status(f"  Processing Fold {fold+1}/{n_folds}...")
            X_train_fold, X_val_fold = X_base[train_idx], X_base[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            clusters_train_fold, clusters_val_fold = clusters[train_idx], clusters[val_idx]

            # Train global model on fold
            fold_global_model = self.create_model()
            fold_global_model.fit(X_train_fold, y_train_fold)
            oof_global_preds[val_idx] = fold_global_model.predict(X_val_fold)

            # Train cluster models on fold
            fold_cluster_models = {} # Store (model, indices_used)
            for cluster_id in np.unique(clusters_train_fold):
                fold_cluster_mask_train = (clusters_train_fold == cluster_id)
                if np.sum(fold_cluster_mask_train) >= self.min_cluster_size:
                     X_fold_cluster_base = X_train_fold[fold_cluster_mask_train]
                     y_fold_cluster = y_train_fold[fold_cluster_mask_train]

                     X_fold_cluster_final = X_fold_cluster_base
                     indices_used_for_fold_cluster = None # Default: use base features

                     # Apply per-cluster feature selection if enabled *for this specific fold*
                     if self.group_feature_per_cluster:
                          # Use cluster_specific_feature_indices if already computed on full data,
                          # otherwise, perform selection on the fold data.
                          # Re-selecting on fold might be better but slower. Let's use pre-computed for speed.
                          precomputed_indices = self.cluster_specific_feature_indices.get(cluster_id, None)
                          if precomputed_indices is not None:
                               # Ensure indices are valid for X_fold_cluster_base
                               if max(precomputed_indices) < X_fold_cluster_base.shape[1]:
                                    X_fold_cluster_final = X_fold_cluster_base[:, precomputed_indices]
                                    indices_used_for_fold_cluster = precomputed_indices
                               else:
                                    print_status(f"Warning: Fold {fold+1}, Cluster {cluster_id} - Invalid precomputed indices.")
                                    # Fallback to using base features for this fold
                          # else: No specific features for this cluster, use base

                     fold_cluster_model = self.create_model()
                     fold_cluster_model.fit(X_fold_cluster_final, y_fold_cluster)
                     fold_cluster_models[cluster_id] = (fold_cluster_model, indices_used_for_fold_cluster)


            # Predict cluster models on validation fold
            for cluster_id in np.unique(clusters_val_fold):
                fold_cluster_mask_val = (clusters_val_fold == cluster_id)
                if np.sum(fold_cluster_mask_val) > 0:
                    if cluster_id in fold_cluster_models:
                        model, indices = fold_cluster_models[cluster_id]
                        X_val_cluster_base = X_val_fold[fold_cluster_mask_val]
                        X_val_cluster_final = X_val_cluster_base
                        if indices is not None: # Apply specific features if used
                             if max(indices) < X_val_cluster_base.shape[1]:
                                  X_val_cluster_final = X_val_cluster_base[:, indices]
                             else:
                                   print_status(f"Warning: Fold {fold+1}, Val Cluster {cluster_id} - Invalid feature indices.")

                        # Check feature compatibility before predicting
                        if hasattr(model, 'n_features_in_') and X_val_cluster_final.shape[1] != model.n_features_in_:
                             print_status(f"Warning: Fold {fold+1}, Val Cluster {cluster_id} - Feature mismatch. Using global pred.")
                             oof_cluster_preds[val_idx[fold_cluster_mask_val]] = fold_global_model.predict(X_val_cluster_base) # Use base features for global pred
                        else:
                              oof_cluster_preds[val_idx[fold_cluster_mask_val]] = model.predict(X_val_cluster_final)

                    else: # Fallback to global prediction
                         oof_cluster_preds[val_idx[fold_cluster_mask_val]] = fold_global_model.predict(X_val_fold[fold_cluster_mask_val])


        print_status("Assembling final meta-features...")
        meta_features_list = []
        self.meta_feature_names = [] # Reset meta feature names

        meta_features_list.append(oof_global_preds.reshape(-1, 1)); self.meta_feature_names.append("oof_global_pred")
        meta_features_list.append(oof_cluster_preds.reshape(-1, 1)); self.meta_feature_names.append("oof_cluster_pred")
        oof_global_error = np.abs(oof_global_preds - y)
        oof_cluster_error = np.abs(oof_cluster_preds - y)
        meta_features_list.append(oof_global_error.reshape(-1, 1)); self.meta_feature_names.append("oof_global_error_abs")
        meta_features_list.append(oof_cluster_error.reshape(-1, 1)); self.meta_feature_names.append("oof_cluster_error_abs")
        meta_features_list.append((oof_cluster_preds - oof_global_preds).reshape(-1, 1)); self.meta_feature_names.append("oof_pred_diff")

        unique_clusters_train = sorted(np.unique(clusters))
        n_unique_clusters_train = len(unique_clusters_train)
        cluster_map_train = {cid: i for i, cid in enumerate(unique_clusters_train)}
        cluster_one_hot = np.zeros((len(y), n_unique_clusters_train))
        for i, cluster_id in enumerate(clusters):
            if cluster_id in cluster_map_train:
                 cluster_one_hot[i, cluster_map_train[cluster_id]] = 1
        meta_features_list.append(cluster_one_hot)
        self.meta_feature_names.extend([f"cluster_{cid}_ohe" for cid in unique_clusters_train])

        n_orig_features_meta = 10
        orig_feat_indices_for_meta = []
        orig_feat_names_for_meta = []
        if self.important_indices is not None and len(self.important_indices) > 0:
             n_to_include = min(n_orig_features_meta, len(self.important_indices), X.shape[1])
             top_orig_indices_global = self.important_indices[:n_to_include] # Indices relative to original X
             # Ensure indices are valid
             if max(top_orig_indices_global) < X.shape[1]:
                  meta_features_list.append(X[:, top_orig_indices_global])
                  orig_feat_indices_for_meta = top_orig_indices_global
                  if self.important_features and len(self.important_features) >= n_to_include:
                       orig_feat_names_for_meta = [f"orig_{self.important_features[i]}" for i in range(n_to_include)]
                  else:
                       orig_feat_names_for_meta = [f"orig_feat_{idx}" for idx in top_orig_indices_global]
                  self.meta_feature_names.extend(orig_feat_names_for_meta)
             else:
                  print_status(f"Warning: Cannot include top {n_to_include} original features in meta-model due to index mismatch.")
        n_included_orig = len(orig_feat_indices_for_meta)


        combined_meta_features = np.hstack(meta_features_list)
        print_status(f"Created meta-feature matrix with shape: {combined_meta_features.shape}")

        print_status("Training meta-model...")
        meta_model_params = {
            'objective': 'reg:squarederror', 'random_state': self.seed,
            'n_estimators': 150, 'learning_rate': 0.05, 'max_depth': 4,
            'subsample': 0.7, 'colsample_bytree': 0.7, 'min_child_weight': 2,
            'n_jobs': actual_n_jobs, 'reg_alpha': 0.1, 'reg_lambda': 1.0
        }
        self.meta_model = self.create_model(params=meta_model_params)
        self.meta_model.fit(combined_meta_features, y)

        self.meta_feature_structure = {
            'oof_global_pred': 1, 'oof_cluster_pred': 1, 'oof_global_error': 1,
            'oof_cluster_error': 1, 'oof_pred_diff': 1, 'cluster_ohe': n_unique_clusters_train,
            'original_features': n_included_orig,
            'total_meta_features': combined_meta_features.shape[1]
        }


        y_pred_meta_train = self.meta_model.predict(combined_meta_features)
        rmse = np.sqrt(mean_squared_error(y, y_pred_meta_train))
        r2 = r2_score(y, y_pred_meta_train)
        mae = mean_absolute_error(y, y_pred_meta_train)
        print_status(f"Meta Model (In-Sample using OOF feats) - RMSE: {rmse:.4f}, R²: {r2:.4f}, MAE: {mae:.4f}")

        return self.meta_model


    def train(self, X, y, smiles_list=None):
        """ Main training orchestrator """
        start_time = time.time()
        print_status(f"Starting full training pipeline...")
        print_status(f"Input shape: X={X.shape}, y={len(y)}")
        self.df = None

        X_train = X # Start with original features
        # 1. Feature Selection (applied to X before clustering or global model)
        if self.n_features is not None and self.n_features < X.shape[1]:
             X_train = self.identify_important_features(X, y, self.n_features, self.feature_method, n_jobs=self._get_n_jobs(self.xgb_params.get('n_jobs')))
             # After selection, self.important_indices and self.important_features are set
        else:
            print_status("Skipping feature selection (using all features or n_features >= total).")
            self.important_indices = None
            # If feature_cols were loaded/generated, use them as important_features
            self.important_features = self.feature_cols if self.feature_cols else [f"feature_{i}" for i in range(X.shape[1])]
            self.n_features_used_in_model = X.shape[1] # Model will use all features

        if X_train.shape[1] == 0:
            raise ValueError("Feature selection resulted in 0 features. Cannot train.")

        # 2. Train Global Model (uses selected features internally)
        self.train_global_model(X, y) # Pass original X, method handles selection

        # 3. Clustering
        if self.clustering_method != 'none' and self.n_clusters is not None and self.n_clusters > 0:
             # Cluster using the *potentially selected* features (X_train)
             cluster_input_X = X_train
             self.clusters = self.cluster_compounds(cluster_input_X, y, smiles_list,
                                                   n_clusters=self.n_clusters,
                                                   method=self.clustering_method,
                                                   cluster_property=self.cluster_property,
                                                   min_cluster_size=self.min_cluster_size)
             if self.clusters is None: # Handle clustering failure
                  print_status("Warning: Clustering failed. Proceeding without cluster models.")
                  self.clustering_method = 'none' # Disable further cluster steps
                  self.clusters = np.zeros(len(y), dtype=int) # Default cluster 0

             # 4. Train Cluster-Specific Models (uses selected features internally)
             self.train_cluster_models(X, y, self.clusters) # Pass original X

             # 5. Train Ensemble Meta-Model (optional)
             if not self.skip_meta_features:
                 self.train_ensemble_model(X, y, self.clusters) # Pass original X

        else:
             print_status("Skipping clustering and cluster-specific models.")
             self.clusters = np.zeros(len(y), dtype=int)
             self.cluster_models = {}
             self.meta_model = None


        self._perform_training_analysis(X, y) # Compare models on training data

        end_time = time.time()
        print_status(f"Training pipeline completed in {end_time - start_time:.2f} seconds.")
        return self


    def _perform_training_analysis(self, X, y):
        """ Compares Global, Cluster, and Ensemble model performance on the training data. """
        print_status("Performing final analysis on training data...")
        self.model_comparison = {}
        self.error_analysis = {}

        if self.global_model is None:
            print_status("Warning: Global model not trained. Cannot perform analysis.")
            return

        # --- Predictions on Training Data ---
        # Use the same feature set the global model was trained on
        X_eval = X
        if self.important_indices is not None:
             if max(self.important_indices) < X.shape[1]:
                  X_eval = X[:, self.important_indices]
             else:
                  print_status("Warning: Stored important_indices are invalid for training data X. Using all features for analysis.")


        if not hasattr(self.global_model, 'n_features_in_') or X_eval.shape[1] != self.global_model.n_features_in_:
             print_status(f"Warning: Mismatch between evaluation features ({X_eval.shape[1]}) and global model expected features ({getattr(self.global_model, 'n_features_in_', 'N/A')}). Analysis might be incorrect.")
             # Attempt to proceed, but results may be unreliable


        y_pred_global = self.global_model.predict(X_eval)

        y_pred_clusters = np.zeros_like(y, dtype=float)
        if self.cluster_models and self.clusters is not None:
            unique_clusters = sorted(np.unique(self.clusters))
            for cluster_id in unique_clusters:
                cluster_mask = (self.clusters == cluster_id)
                if np.sum(cluster_mask) > 0:
                    if cluster_id in self.cluster_models and self.cluster_models[cluster_id] is not None:
                        model = self.cluster_models[cluster_id]
                        X_cluster_eval_base = X_eval[cluster_mask] # Base features for this cluster
                        X_cluster_eval_final = X_cluster_eval_base

                        # Apply cluster-specific features if used
                        cluster_feature_indices = self.cluster_specific_feature_indices.get(cluster_id, None)
                        if cluster_feature_indices is not None:
                            if max(cluster_feature_indices) < X_cluster_eval_base.shape[1]:
                                X_cluster_eval_final = X_cluster_eval_base[:, cluster_feature_indices]
                            else:
                                print_status(f"Warning: Cluster {cluster_id} analysis - invalid feature indices.")

                        if hasattr(model, 'n_features_in_') and X_cluster_eval_final.shape[1] == model.n_features_in_:
                             y_pred_clusters[cluster_mask] = model.predict(X_cluster_eval_final)
                        else:
                             print_status(f"Warning: Cluster {cluster_id} analysis - feature mismatch ({X_cluster_eval_final.shape[1]} vs {getattr(model, 'n_features_in_', 'N/A')}). Using global prediction.")
                             y_pred_clusters[cluster_mask] = y_pred_global[cluster_mask] # Fallback
                    else:
                         y_pred_clusters[cluster_mask] = y_pred_global[cluster_mask] # Fallback
        else:
             y_pred_clusters = y_pred_global # No clustering, cluster preds = global preds


        y_pred_ensemble = None
        if self.meta_model is not None and not self.skip_meta_features and self.clusters is not None:
            try:
                 # Need original X here for meta feature generation if it used original features
                 meta_features_full = self._get_meta_features_for_prediction(X, y_pred_global, y_pred_clusters, self.clusters)
                 if meta_features_full.shape[1] == self.meta_model.n_features_in_:
                      y_pred_ensemble = self.meta_model.predict(meta_features_full)
                 else:
                      print_status(f"Warning: Meta-feature mismatch for training analysis ({meta_features_full.shape[1]} vs {self.meta_model.n_features_in_}). Skipping ensemble analysis.")
            except Exception as e_meta_analysis:
                 print_status(f"Warning: Error generating meta-features for training analysis: {e_meta_analysis}. Skipping ensemble.")

        elif self.use_model_mean:
             print_status("Using simple mean of global and cluster predictions for analysis.")
             y_pred_clusters_clean = np.nan_to_num(y_pred_clusters, nan=np.nanmean(y_pred_clusters))
             y_pred_global_clean = np.nan_to_num(y_pred_global, nan=np.nanmean(y_pred_global))
             y_pred_ensemble = (y_pred_global_clean + y_pred_clusters_clean) / 2.0
        elif self.use_weighted_mean:
            print_status("Using weighted mean for analysis (requires weights - using default 0.5 for now).")
            # Implement actual weighting later if needed
            y_pred_clusters_clean = np.nan_to_num(y_pred_clusters, nan=np.nanmean(y_pred_clusters))
            y_pred_global_clean = np.nan_to_num(y_pred_global, nan=np.nanmean(y_pred_global))
            y_pred_ensemble = (y_pred_global_clean + y_pred_clusters_clean) / 2.0 # Placeholder


        def calculate_metrics(y_true, y_pred):
            if len(y_true) == 0 or len(y_pred) == 0: return {'rmse': np.nan, 'r2': np.nan, 'mae': np.nan}
            # Ensure no NaNs in predictions before metric calculation
            y_pred_clean = np.nan_to_num(y_pred, nan=np.nanmean(y_pred))
            rmse = np.sqrt(mean_squared_error(y_true, y_pred_clean))
            # R2 score can be negative, handle cases with single value prediction or poor fit
            try:
                 r2 = r2_score(y_true, y_pred_clean)
            except ValueError: r2 = np.nan # Handle constant prediction case
            mae = mean_absolute_error(y_true, y_pred_clean)
            return {'rmse': rmse, 'r2': r2, 'mae': mae}

        self.model_comparison['global'] = calculate_metrics(y, y_pred_global)
        self.model_comparison['cluster'] = calculate_metrics(y, y_pred_clusters)
        if y_pred_ensemble is not None:
             self.model_comparison['ensemble'] = calculate_metrics(y, y_pred_ensemble)
             final_predictions = y_pred_ensemble
             final_pred_label = "Ensemble"
        elif self.clustering_method != 'none':
             final_predictions = y_pred_clusters
             final_pred_label = "Cluster"
        else:
             final_predictions = y_pred_global
             final_pred_label = "Global"


        print_status("--- Training Set Performance Comparison ---")
        print_status(f"Global:   RMSE={self.model_comparison['global']['rmse']:.4f}, R²={self.model_comparison['global']['r2']:.4f}, MAE={self.model_comparison['global']['mae']:.4f}")
        print_status(f"Cluster:  RMSE={self.model_comparison['cluster']['rmse']:.4f}, R²={self.model_comparison['cluster']['r2']:.4f}, MAE={self.model_comparison['cluster']['mae']:.4f}")
        if 'ensemble' in self.model_comparison:
             print_status(f"Ensemble: RMSE={self.model_comparison['ensemble']['rmse']:.4f}, R²={self.model_comparison['ensemble']['r2']:.4f}, MAE={self.model_comparison['ensemble']['mae']:.4f}")

        final_errors = final_predictions - y

        print_status(f"--- Error Analysis by Cluster using '{final_pred_label}' Predictions (Training Set) ---")
        if self.clusters is not None:
             for cluster_id in sorted(np.unique(self.clusters)):
                 cluster_mask = (self.clusters == cluster_id)
                 if np.sum(cluster_mask) > 0:
                     cluster_y_true = y[cluster_mask]
                     cluster_y_pred_final = final_predictions[cluster_mask]
                     cluster_errors = final_errors[cluster_mask]
                     cluster_metrics = calculate_metrics(cluster_y_true, cluster_y_pred_final)

                     cluster_stats = {
                         'samples': int(np.sum(cluster_mask)),
                         'rmse': cluster_metrics['rmse'], 'mae': cluster_metrics['mae'], 'r2': cluster_metrics['r2'],
                         'mean_error': np.mean(cluster_errors), 'std_error': np.std(cluster_errors),
                         # Add comparison metrics
                         'global_rmse': calculate_metrics(cluster_y_true, y_pred_global[cluster_mask])['rmse'],
                         'cluster_rmse': calculate_metrics(cluster_y_true, y_pred_clusters[cluster_mask])['rmse']
                     }
                     # Convert cluster_id to int for JSON compatibility
                     self.error_analysis[int(cluster_id)] = cluster_stats
                     print_status(f"Cluster {cluster_id}: N={cluster_stats['samples']}, Final RMSE={cluster_stats['rmse']:.4f}, MAE={cluster_stats['mae']:.4f}, MeanErr={cluster_stats['mean_error']:.4f}")
        else:
            print_status("Skipping cluster error analysis (no clusters).")


        # Save analysis results
        try:
             analysis_data = {
                 'model_comparison': self.model_comparison,
                 # Ensure keys are standard Python ints or strings for JSON
                 'error_analysis_per_cluster': {int(k): v for k, v in self.error_analysis.items()},
                 'meta_feature_structure': self.meta_feature_structure,
                 # Save feature importance scores correctly
                 'feature_importance': {feat: float(score) for feat, score in zip(self.feature_cols, self.feature_importance_scores) if score != -np.inf} if self.feature_cols and self.feature_importance_scores is not None and len(self.feature_cols)==len(self.feature_importance_scores) else {}
             }
             analysis_filename = self.output_dir / "training_analysis.json"
             with open(analysis_filename, 'w') as f:
                 json.dump(analysis_data, f, indent=2, cls=NpEncoder)
             print_status(f"Training analysis saved to {analysis_filename}")
        except Exception as e:
            print_status(f"Error saving training analysis: {e}")


    def _get_meta_features_for_prediction(self, X, y_pred_global, y_pred_clusters, clusters):
        """ Generates meta-features for new data (prediction time). """
        meta_features_list = []

        if not hasattr(self, 'meta_feature_names') or not self.meta_feature_names:
             raise RuntimeError("Meta-model feature names not found. Cannot generate meta-features for prediction.")

        # Check if original features are needed by meta model
        n_orig_needed = self.meta_feature_structure.get('original_features', 0)
        orig_feature_indices_meta = []
        if n_orig_needed > 0:
            meta_orig_names = [name.replace("orig_", "", 1) for name in self.meta_feature_names if name.startswith("orig_")]
            if self.feature_cols: # Use original feature names from training
                 feat_to_idx_map = {name: i for i, name in enumerate(self.feature_cols)}
                 for name in meta_orig_names:
                     if name in feat_to_idx_map:
                          idx = feat_to_idx_map[name]
                          if idx < X.shape[1]: # Ensure index is valid for input X
                               orig_feature_indices_meta.append(idx)
                     elif name.startswith("feat_") and name.split('_')[1].isdigit(): # Handle generic names
                          idx = int(name.split('_')[1])
                          if idx < X.shape[1]: orig_feature_indices_meta.append(idx)

                 if len(orig_feature_indices_meta) != n_orig_needed:
                      print_status(f"Warning: Could only find {len(orig_feature_indices_meta)}/{n_orig_needed} required original features for meta-model prediction.")
                      # Cannot proceed reliably if original features are missing
                      raise ValueError(f"Missing required original features for meta-model prediction.")
            else:
                 raise ValueError("Missing feature_cols needed to identify original features for meta-model.")


        # Generate meta-features in the order defined by self.meta_feature_names
        current_orig_feat_idx = 0
        current_cluster_ohe_idx = 0
        # Find cluster IDs used during meta training from names
        trained_cluster_ids_meta = sorted([int(c.split('_')[1]) for c in self.meta_feature_names if c.startswith("cluster_") and c.endswith("_ohe")])
        cluster_map_meta = {cid: i for i, cid in enumerate(trained_cluster_ids_meta)}
        n_unique_clusters_meta = len(trained_cluster_ids_meta)

        for feature_name in self.meta_feature_names:
            if feature_name == "oof_global_pred":
                meta_features_list.append(y_pred_global.reshape(-1, 1))
            elif feature_name == "oof_cluster_pred":
                meta_features_list.append(y_pred_clusters.reshape(-1, 1))
            elif feature_name == "oof_global_error_abs":
                # Proxy at prediction time (abs difference)
                meta_features_list.append(np.abs(y_pred_clusters - y_pred_global).reshape(-1, 1))
            elif feature_name == "oof_cluster_error_abs":
                # Proxy at prediction time (abs difference)
                meta_features_list.append(np.abs(y_pred_clusters - y_pred_global).reshape(-1, 1))
            elif feature_name == "oof_pred_diff":
                meta_features_list.append((y_pred_clusters - y_pred_global).reshape(-1, 1))
            elif feature_name.startswith("cluster_") and feature_name.endswith("_ohe"):
                # Generate the OHE column for this specific cluster ID
                cluster_id_meta = int(feature_name.split('_')[1])
                if cluster_id_meta in cluster_map_meta:
                     col_idx = cluster_map_meta[cluster_id_meta]
                     ohe_col = (clusters == cluster_id_meta).astype(int).reshape(-1, 1)
                     meta_features_list.append(ohe_col)
                else: # This should not happen if names are consistent
                     print_status(f"Warning: Mismatch in OHE cluster ID {cluster_id_meta} during meta-feature generation.")
                     meta_features_list.append(np.zeros((X.shape[0], 1)))

            elif feature_name.startswith("orig_"):
                if current_orig_feat_idx < len(orig_feature_indices_meta):
                     orig_idx = orig_feature_indices_meta[current_orig_feat_idx]
                     meta_features_list.append(X[:, orig_idx].reshape(-1, 1))
                     current_orig_feat_idx += 1
                else: # Should not happen if indices were validated
                     raise RuntimeError("Mismatch processing original features for meta-model.")
            else:
                raise RuntimeError(f"Unknown feature name '{feature_name}' encountered during meta-feature generation.")

        if not meta_features_list:
             return np.zeros((X.shape[0], 0)) # Return empty if no features

        combined_meta_features = np.hstack(meta_features_list)

        expected_meta_features = self.meta_feature_structure.get('total_meta_features', -1)
        if expected_meta_features != -1 and combined_meta_features.shape[1] != expected_meta_features:
             print_status(f"CRITICAL WARNING: Generated meta-feature shape ({combined_meta_features.shape[1]}) does not match expected shape ({expected_meta_features}) during prediction.")
             # Attempt to pad or truncate? Or error out. Let's error.
             raise ValueError("Meta-feature shape mismatch during prediction generation.")

        return combined_meta_features


    def predict(self, X, smiles=None):
        print_status(f"Making predictions on {X.shape[0]} samples...")

        if not self.global_model and not self.cluster_models:
            raise RuntimeError("No models available for prediction. Train the model first.")

        # 0. Handle NaNs in input X
        if self.replace_nan is not None and np.isnan(X).any():
             X = np.nan_to_num(X, nan=self.replace_nan)
        elif np.isnan(X).any():
             # print_status(f"Warning: NaN values detected in prediction input. Replacing with 0.") # Reduce noise
             X = np.nan_to_num(X, nan=0.0)


        # 1. Feature Selection/Alignment
        X_input_orig = X # Keep original X for meta features if needed
        X_pred_base = X # Features for base models (global/cluster)

        if self.important_indices is not None:
            # Check if input X needs alignment to the original training feature set
            if self.feature_cols and X.shape[1] != len(self.feature_cols):
                 print_status(f"Warning: Prediction input feature count ({X.shape[1]}) differs from training feature count ({len(self.feature_cols)}). Performing alignment.")
                 
                 # Create properly sized array with default values
                 X_aligned = np.full((X.shape[0], len(self.feature_cols)), self.replace_nan if self.replace_nan is not None else 0.0)
                 
                 if X.shape[1] < len(self.feature_cols):
                     # Case 1: Input has fewer features than expected - direct copy what we have
                     print_status(f"Input has fewer features than expected. Copying available {X.shape[1]} features and using default values for the rest.")
                     X_aligned[:, :X.shape[1]] = X
                 else:
                     # Case 2: Input has more features than expected - needs more careful selection
                     print_status(f"Input has more features than expected. Selecting first {len(self.feature_cols)} features.")
                     X_aligned = X[:, :len(self.feature_cols)]
                 
                 print_status(f"Aligned prediction input to {X_aligned.shape[1]} features before feature selection.")
                 X_input_orig = X_aligned # Use aligned version if original features needed later
                 
                 # Select important features from the aligned matrix
                 if max(self.important_indices) < X_aligned.shape[1]:
                     X_pred_base = X_aligned[:, self.important_indices]
                     print_status(f"Selected {X_pred_base.shape[1]} important features from aligned matrix.")
                 else:
                      print_status(f"Warning: Important feature indices (max={max(self.important_indices)}) exceed aligned matrix dimensions ({X_aligned.shape[1]}).")
                      raise ValueError("Important indices out of bounds after aligning prediction input.")

            else: # Input feature count matches training, just select
                 if max(self.important_indices) < X.shape[1]:
                     X_pred_base = X[:, self.important_indices]
                     print_status(f"Feature counts match. Selected {X_pred_base.shape[1]} important features for prediction.")
                 else:
                      raise ValueError(f"Important feature indices (max={max(self.important_indices)}) are out of bounds for input matrix (columns={X.shape[1]}).")

            # Check against number of features model was actually trained on
            if self.n_features_used_in_model is not None and X_pred_base.shape[1] != self.n_features_used_in_model:
                 print_status(f"Warning: Selected features for prediction ({X_pred_base.shape[1]}) don't match number used in trained models ({self.n_features_used_in_model}).")

        else:
            # No feature selection during training, use all features
            X_pred_base = X
            print_status(f"No feature selection applied. Using all {X.shape[1]} input features.")
            if self.n_features_used_in_model is not None and X.shape[1] != self.n_features_used_in_model:
                 print_status(f"Warning: Prediction input features ({X.shape[1]}) don't match number used in trained models ({self.n_features_used_in_model}).")

        if X_pred_base.shape[1] == 0:
             print_status("Warning: Feature matrix for prediction is empty after selection/alignment.")
             return np.full(X.shape[0], np.nan)


        # 2. Global Model Prediction
        if self.global_model:
             if hasattr(self.global_model, 'n_features_in_') and X_pred_base.shape[1] != self.global_model.n_features_in_:
                  raise ValueError(f"Global model expects {self.global_model.n_features_in_} features, but input has {X_pred_base.shape[1]}.")
             y_pred_global = self.global_model.predict(X_pred_base)
        else:
             y_pred_global = np.zeros(X.shape[0]) # Fallback


        # 3. Clustering Prediction
        predicted_clusters = np.zeros(X.shape[0], dtype=int) # Default to cluster 0
        if self.clustering_method != 'none' and self.cluster_model_predictor:
             try:
                 # print_status(f"Predicting clusters using '{self.clustering_method}' predictor...") # Reduce noise
                 # Use the appropriate data for the predictor
                 if isinstance(self.cluster_model_predictor, (FunctionalGroupPredictor, ScaffoldPredictor, ChemicalPropertyPredictor)):
                     if smiles is None: raise ValueError(f"Predictor {type(self.cluster_model_predictor).__name__} requires SMILES.")
                     predicted_clusters = self.cluster_model_predictor.predict(smiles)
                 elif isinstance(self.cluster_model_predictor, StratifiedPredictor):
                      # Stratified predictor expects the data that PCA was fit on (potentially scaled base features)
                      if self.scaler and hasattr(self.scaler, 'mean_'): # Check if scaler was used and fitted
                           cluster_predict_data = self.scaler.transform(X_pred_base)
                      else: # Assume PCA was fit on unscaled base features
                           cluster_predict_data = X_pred_base
                      predicted_clusters = self.cluster_model_predictor.predict(cluster_predict_data) # Predictor handles PCA inside
                 elif isinstance(self.cluster_model_predictor, KMeansPredictor) and self.clustering_method == 'property' and self.pca:
                      # Property clustering using PCA + KMeans
                      if not self.scaler or not hasattr(self.scaler, 'mean_'): raise RuntimeError("Scaler required for PCA property clustering prediction.")
                      X_scaled_pred = self.scaler.transform(X_pred_base)
                      X_pca_pred = self.pca.transform(X_scaled_pred)
                      predicted_clusters = self.cluster_model_predictor.predict(X_pca_pred)

                 else: # kmeans, dbscan(nn), agglomerative(nn), spectral(nn), gaussian, random
                      # These usually operate on scaled base features (except random)
                      if self.clustering_method == 'random':
                           cluster_predict_data = X_pred_base # Input doesn't matter
                      elif self.scaler and hasattr(self.scaler, 'mean_'):
                           cluster_predict_data = self.scaler.transform(X_pred_base)
                      else: # Assume no scaling was used or needed
                           cluster_predict_data = X_pred_base
                      predicted_clusters = self.cluster_model_predictor.predict(cluster_predict_data)

                 # print_status(f"Predicted cluster distribution: {np.bincount(predicted_clusters)}") # Reduce noise

             except Exception as e:
                 print_status(f"Error during cluster prediction: {e}. Using default cluster 0.")
                 # Keep default predicted_clusters = 0

        # 4. Cluster Model Predictions
        y_pred_clusters = np.zeros_like(y_pred_global)
        if self.cluster_models:
            unique_pred_clusters = sorted(np.unique(predicted_clusters))
            for cluster_id in unique_pred_clusters:
                cluster_mask = (predicted_clusters == cluster_id)
                if np.sum(cluster_mask) == 0: continue

                if cluster_id in self.cluster_models and self.cluster_models[cluster_id] is not None:
                    model = self.cluster_models[cluster_id]
                    X_cluster_pred_base = X_pred_base[cluster_mask] # Base features for this cluster
                    X_cluster_pred_final = X_cluster_pred_base

                    # Apply cluster-specific features if used
                    cluster_feature_indices = self.cluster_specific_feature_indices.get(cluster_id, None)
                    if cluster_feature_indices is not None:
                         if max(cluster_feature_indices) < X_cluster_pred_base.shape[1]:
                              X_cluster_pred_final = X_cluster_pred_base[:, cluster_feature_indices]
                         else:
                              print_status(f"Warning: Predict Cluster {cluster_id} - invalid feature indices.")

                    if hasattr(model, 'n_features_in_') and X_cluster_pred_final.shape[1] == model.n_features_in_:
                         y_pred_clusters[cluster_mask] = model.predict(X_cluster_pred_final)
                    else:
                         # print_status(f"Warning: Cluster {cluster_id} pred feature mismatch. Using global.") # Reduce noise
                         y_pred_clusters[cluster_mask] = y_pred_global[cluster_mask] # Fallback

                else:
                    # print_status(f"No model for predicted Cluster {cluster_id}. Using global.") # Reduce noise
                    y_pred_clusters[cluster_mask] = y_pred_global[cluster_mask] # Fallback
        else:
             y_pred_clusters = y_pred_global # No cluster models trained

        # 5. Final Prediction Logic
        final_predictions = None
        if self.meta_model is not None and not self.skip_meta_features:
            # print_status("Using meta-model for final prediction.") # Reduce noise
            try:
                 # Use X_input_orig (potentially aligned original features) for meta feature generation
                 meta_features = self._get_meta_features_for_prediction(X_input_orig, y_pred_global, y_pred_clusters, predicted_clusters)
                 if hasattr(self.meta_model, 'n_features_in_') and meta_features.shape[1] == self.meta_model.n_features_in_:
                      final_predictions = self.meta_model.predict(meta_features)
                 else:
                      print_status(f"Warning: Meta-feature mismatch ({meta_features.shape[1]} vs {getattr(self.meta_model, 'n_features_in_', 'N/A')}). Falling back.")
                      final_predictions = (y_pred_global + y_pred_clusters) / 2.0 # Simple average fallback
            except Exception as e:
                 print_status(f"Error predicting with meta-model: {e}. Falling back.")
                 final_predictions = (y_pred_global + y_pred_clusters) / 2.0

        if final_predictions is None: # If meta-model wasn't used or failed
             if self.use_model_mean:
                 # print_status("Using simple mean of global and cluster predictions.") # Reduce noise
                 y_pred_clusters = np.nan_to_num(y_pred_clusters, nan=np.nanmean(y_pred_clusters))
                 y_pred_global = np.nan_to_num(y_pred_global, nan=np.nanmean(y_pred_global))
                 final_predictions = (y_pred_global + y_pred_clusters) / 2.0
             elif self.use_weighted_mean:
                 # print_status("Using weighted mean.") # Reduce noise
                 cluster_weights = getattr(self, 'cluster_weights', {})
                 final_predictions = np.zeros_like(y_pred_global)
                 for i in range(len(final_predictions)):
                      cluster_id = predicted_clusters[i]
                      weight = cluster_weights.get(cluster_id, 0.5) # Default 0.5
                      global_pred_i = np.nan_to_num(y_pred_global[i], nan=0)
                      cluster_pred_i = np.nan_to_num(y_pred_clusters[i], nan=global_pred_i)
                      final_predictions[i] = weight * cluster_pred_i + (1 - weight) * global_pred_i
             elif self.clustering_method != 'none' and self.cluster_models:
                 # print_status("Using cluster-specific model predictions.") # Reduce noise
                 final_predictions = y_pred_clusters
             else: # Fallback to global if no clustering or other methods specified
                  # print_status("Using global model predictions.") # Reduce noise
                  final_predictions = y_pred_global

        # print_status("Prediction complete.") # Reduce noise
        return final_predictions


    def evaluate(self, X, y, smiles=None):
        print_status("Evaluating model...")
        y_pred = self.predict(X, smiles)

        # --- Overall Performance ---
        # Ensure y_pred does not contain NaNs before metric calculation
        y_pred_clean = np.nan_to_num(y_pred, nan=np.nanmean(y)) # Replace NaN preds with mean of true values for eval
        if np.isnan(y_pred_clean).any(): # If mean was NaN
             y_pred_clean = np.nan_to_num(y_pred_clean, nan=0.0)


        mse = mean_squared_error(y, y_pred_clean)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred_clean)
        try:
            r2 = r2_score(y, y_pred_clean)
        except ValueError: r2 = np.nan # Handle constant prediction case

        print_status(f"Overall Performance - RMSE: {rmse:.4f}, R²: {r2:.4f}, MAE: {mae:.4f}")

        results = {
            'overall': {'mse': float(mse), 'rmse': float(rmse), 'mae': float(mae), 'r2': float(r2)},
            'y_true': y.tolist(),
            'y_pred': y_pred.tolist() # Store original predictions (might have NaN)
        }

        # --- Per-Cluster Performance ---
        eval_clusters = None
        if self.clustering_method != 'none' and hasattr(self, 'cluster_model_predictor') and self.cluster_model_predictor:
             print_status("Calculating performance metrics per cluster...")
             try:
                 # Replicate cluster prediction logic from predict() for consistency
                 if isinstance(self.cluster_model_predictor, (FunctionalGroupPredictor, ScaffoldPredictor, ChemicalPropertyPredictor)):
                     if smiles is None: raise ValueError("Predictor needs SMILES.")
                     eval_clusters = self.cluster_model_predictor.predict(smiles)
                 elif isinstance(self.cluster_model_predictor, StratifiedPredictor):
                      X_eval_base = X[:, self.important_indices] if self.important_indices else X
                      if self.scaler and hasattr(self.scaler, 'mean_'):
                           cluster_eval_data = self.scaler.transform(X_eval_base)
                      else: cluster_eval_data = X_eval_base
                      eval_clusters = self.cluster_model_predictor.predict(cluster_eval_data)
                 elif isinstance(self.cluster_model_predictor, KMeansPredictor) and self.clustering_method == 'property' and self.pca:
                      X_eval_base = X[:, self.important_indices] if self.important_indices else X
                      if not self.scaler or not hasattr(self.scaler, 'mean_'): raise RuntimeError("Scaler required.")
                      X_scaled_eval = self.scaler.transform(X_eval_base)
                      X_pca_eval = self.pca.transform(X_scaled_eval)
                      eval_clusters = self.cluster_model_predictor.predict(X_pca_eval)
                 else: # kmeans, dbscan(nn), etc.
                      X_eval_base = X[:, self.important_indices] if self.important_indices else X
                      if self.clustering_method == 'random': cluster_eval_data = X_eval_base
                      elif self.scaler and hasattr(self.scaler, 'mean_'): cluster_eval_data = self.scaler.transform(X_eval_base)
                      else: cluster_eval_data = X_eval_base
                      eval_clusters = self.cluster_model_predictor.predict(cluster_eval_data)

                 results['clusters'] = eval_clusters.tolist()
                 cluster_results = {}
                 unique_eval_clusters = sorted(np.unique(eval_clusters))

                 for cluster_id in unique_eval_clusters:
                     mask = (eval_clusters == cluster_id)
                     n_samples_cluster = np.sum(mask)
                     if n_samples_cluster == 0: continue

                     y_cluster = y[mask]
                     y_pred_cluster = y_pred_clean[mask] # Use cleaned predictions for metrics

                     if len(y_cluster) > 0:
                        cluster_mse = mean_squared_error(y_cluster, y_pred_cluster)
                        cluster_rmse = np.sqrt(cluster_mse)
                        cluster_mae = mean_absolute_error(y_cluster, y_pred_cluster)
                        try: cluster_r2 = r2_score(y_cluster, y_pred_cluster)
                        except ValueError: cluster_r2 = np.nan
                     else: # Should not happen if mask sum > 0
                        cluster_mse, cluster_rmse, cluster_mae, cluster_r2 = np.nan, np.nan, np.nan, np.nan

                     print_status(f"  Cluster {cluster_id} - RMSE: {cluster_rmse:.4f}, R²: {cluster_r2:.4f}, MAE: {cluster_mae:.4f}, Samples: {n_samples_cluster}")

                     cluster_results[int(cluster_id)] = {
                         'mse': float(cluster_mse), 'rmse': float(cluster_rmse),
                         'mae': float(cluster_mae), 'r2': float(cluster_r2),
                         'n_samples': n_samples_cluster
                     }
                 results['cluster_metrics'] = cluster_results

             except Exception as e:
                 print_status(f"Warning: Error calculating per-cluster metrics during evaluation: {e}")
                 results['clusters'] = None
                 results['cluster_metrics'] = None

        # --- Error Analysis ---
        errors = y_pred_clean - y # Use cleaned predictions
        abs_errors = np.abs(errors)
        results['error_stats'] = {
            'mean_error': float(np.mean(errors)), 'median_error': float(np.median(errors)),
            'std_error': float(np.std(errors)), 'mean_abs_error': float(np.mean(abs_errors)),
            'median_abs_error': float(np.median(abs_errors)), 'max_abs_error': float(np.max(abs_errors)),
            'min_abs_error': float(np.min(abs_errors)),
            'q1_abs_error': float(np.percentile(abs_errors, 25)), 'q3_abs_error': float(np.percentile(abs_errors, 75))
        }
        print_status(f"Error Analysis - MAE: {results['error_stats']['mean_abs_error']:.4f}, Mean Error: {results['error_stats']['mean_error']:.4f} +/- {results['error_stats']['std_error']:.4f}")

        print_status("Evaluation complete.")
        return results


    def tune_hyperparameters(self, X, y, smiles=None, n_trials=50, cv_folds=3, tune_clusters=False, external_test_data=None, args=None):
        """ Tune hyperparameters using Optuna. """
        print_status(f"Tuning hyperparameters with Optuna ({n_trials} trials, {cv_folds}-fold CV)")
        print_status(f"Input data shape: X={X.shape}, y={len(y)}")

        local_args = args if args is not None else argparse.Namespace()
        actual_n_jobs = self._get_n_jobs(getattr(local_args, 'n_jobs', -1))

        ext_X, ext_y, ext_smiles = None, None, None
        if external_test_data:
             ext_X = external_test_data.get('X_test')
             ext_y = external_test_data.get('y_true')
             ext_smiles = external_test_data.get('smiles')
             if ext_X is not None and ext_y is not None:
                  print_status(f"Using provided external test set ({len(ext_y)} samples) for optimization.")
             else:
                  print_status("Warning: External test data provided but incomplete. Using CV.")
                  ext_X, ext_y, ext_smiles = None, None, None

        # Define param_map here, within the method scope
        param_map = {
            'learning_rate': ('learning_rate', lambda t: t.suggest_float('learning_rate', 0.01, 0.3, log=True)),
            'max_depth': ('max_depth', lambda t: t.suggest_int('max_depth', 3, 12)),
            'n_estimators': ('n_estimators', lambda t: t.suggest_int('n_estimators', 50, 500, step=50)),
            'subsample': ('subsample', lambda t: t.suggest_float('subsample', 0.5, 1.0)),
            'colsample_bytree': ('colsample_bytree', lambda t: t.suggest_float('colsample_bytree', 0.5, 1.0)),
            'min_child_weight': ('min_child_weight', lambda t: t.suggest_int('min_child_weight', 1, 10)),
            'gamma': ('gamma', lambda t: t.suggest_float('gamma', 0.0, 1.0)),
            'reg_alpha': ('reg_alpha', lambda t: t.suggest_float('reg_alpha', 1e-8, 1.0, log=True)),
            'reg_lambda': ('reg_lambda', lambda t: t.suggest_float('reg_lambda', 1e-8, 1.0, log=True)),
        }

        def objective(trial):
            # Temporary instance for trial to avoid modifying self
            trial_ensemble = ClusteredXGBoostEnsemble(seed=self.seed)
            trial_ensemble.feature_cols = self.feature_cols # Use original feature names
            trial_ensemble.min_cluster_size = self.min_cluster_size
            trial_ensemble.group_feature_per_cluster = self.group_feature_per_cluster
            trial_ensemble.replace_nan = self.replace_nan # Use consistent NaN handling

            # 1. Feature Selection Parameters
            if 'n_features' in self.user_specified_params:
                n_features = self.n_features
            elif getattr(local_args, 'tune_full_pipeline', False):
                min_features = min(10, X.shape[1]) if X.shape[1] > 0 else 0
                n_features = trial.suggest_int('n_features', min_features, X.shape[1]) if min_features < X.shape[1] else X.shape[1]
            else: # Not tuning features
                 n_features = self.n_features # Use original setting

            if 'feature_method' in self.user_specified_params:
                 current_feature_method = self.feature_method
            elif getattr(local_args, 'tune_full_pipeline', False):
                 current_feature_method = trial.suggest_categorical('feature_method', ['xgb', 'rf', 'mutual_info'])
            else:
                 current_feature_method = self.feature_method

            # Select features for this trial (use temporary instance's method)
            # Pass actual_n_jobs for feature selection methods that support it
            X_selected = trial_ensemble.identify_important_features(X, y, n_features, current_feature_method, n_jobs=actual_n_jobs)
            # Store selected indices on the trial instance
            trial_ensemble.important_indices = trial_ensemble.important_indices # identify_important_features sets this

            ext_X_selected = None
            if ext_X is not None:
                 if trial_ensemble.important_indices is not None:
                      if max(trial_ensemble.important_indices) < ext_X.shape[1]:
                           ext_X_selected = ext_X[:, trial_ensemble.important_indices]
                      else: # Alignment failed
                           print_status(f"Warning: Trial {trial.number} - Cannot align external features. Using CV.")
                           ext_X_selected, ext_y_trial = None, None # Use CV
                 else: # No feature selection, use all
                      ext_X_selected = ext_X

            # 2. Clustering Parameters
            if 'clustering_method' in self.user_specified_params:
                 current_clustering_method = self.clustering_method
            elif tune_clusters or getattr(local_args, 'tune_full_pipeline', False):
                 clustering_choices = ["none", "kmeans", "random", "agglomerative", "gaussian"]
                 if smiles is not None:
                     clustering_choices.extend(["functional", "scaffold"])
                     clustering_choices.extend(list(ChemicalPropertyPredictor._property_calculators.keys()))
                 current_clustering_method = trial.suggest_categorical('clustering_method', clustering_choices)
            else:
                 current_clustering_method = self.clustering_method
            trial_ensemble.clustering_method = current_clustering_method # Set on trial instance

            if 'n_clusters' in self.user_specified_params:
                 current_n_clusters = self.n_clusters
            elif current_clustering_method != 'none' and (tune_clusters or getattr(local_args, 'tune_full_pipeline', False)):
                 current_n_clusters = trial.suggest_int('n_clusters', 2, 15)
            else: # No clustering or not tuning clusters
                 current_n_clusters = 0 if current_clustering_method == 'none' else self.n_clusters
            trial_ensemble.n_clusters = current_n_clusters

            # 3. XGBoost Parameters
            xgb_trial_params = {'n_jobs': actual_n_jobs}
            for cli_param, (xgb_param, suggest_func) in param_map.items():
                 if cli_param in self.user_specified_params:
                      xgb_trial_params[xgb_param] = self.xgb_params.get(xgb_param)
                 else:
                      xgb_trial_params[xgb_param] = suggest_func(trial)
            trial_ensemble.xgb_params = xgb_trial_params # Set on trial instance


            # --- Evaluation ---
            scores = []
            try:
                if ext_X_selected is not None and ext_y is not None:
                     # Evaluate on external set
                     print_status(f"Trial {trial.number}: Evaluating on external test set.")
                     # Train trial model on full (X, y) data using selected features
                     trial_ensemble.train(X, y, smiles) # train() handles internal feature selection logic
                     # Predict on external set (handles feature selection inside predict)
                     ext_y_pred = trial_ensemble.predict(ext_X, ext_smiles)
                     score = np.sqrt(mean_squared_error(ext_y, np.nan_to_num(ext_y_pred, nan=np.nanmean(ext_y)))) # Handle potential NaNs in prediction
                     scores.append(score)
                     print_status(f"Trial {trial.number}: External RMSE = {score:.5f}")

                else:
                     # Use Cross-Validation on X_selected
                     print_status(f"Trial {trial.number}: Evaluating using {cv_folds}-fold CV.")
                     kf = KFold(n_splits=cv_folds, shuffle=True, random_state=self.seed)
                     fold_scores = []
                     for fold, (train_idx, val_idx) in enumerate(kf.split(X_selected, y)):
                          X_train_fold, X_val_fold = X_selected[train_idx], X_selected[val_idx]
                          y_train_fold, y_val_fold = y[train_idx], y[val_idx]
                          smiles_train_fold = [smiles[i] for i in train_idx] if smiles is not None else None
                          smiles_val_fold = [smiles[i] for i in val_idx] if smiles is not None else None

                          # Create and train model for the fold
                          fold_model = ClusteredXGBoostEnsemble(seed=self.seed)
                          # Copy relevant settings from trial_ensemble
                          fold_model.xgb_params = trial_ensemble.xgb_params
                          fold_model.clustering_method = trial_ensemble.clustering_method
                          fold_model.n_clusters = trial_ensemble.n_clusters
                          fold_model.min_cluster_size = trial_ensemble.min_cluster_size
                          fold_model.group_feature_per_cluster = trial_ensemble.group_feature_per_cluster
                          fold_model.feature_cols = trial_ensemble.important_features # Use names of selected features
                          fold_model.important_indices = np.arange(X_train_fold.shape[1]) # Indices relative to fold input (X_selected)
                          fold_model.n_features_used_in_model = X_train_fold.shape[1]
                          fold_model.replace_nan = trial_ensemble.replace_nan
                          fold_model.cluster_property = trial_ensemble.cluster_property


                          # Train fold model (using already selected features X_train_fold)
                          # Need to adapt train() to accept pre-selected features or refactor
                          # Quick adaptation: Temporarily set n_features=None to avoid re-selection inside train()
                          original_n_features = fold_model.n_features
                          fold_model.n_features = None # Prevent re-selection
                          fold_model.train(X_train_fold, y_train_fold, smiles_train_fold)
                          fold_model.n_features = original_n_features # Restore

                          # Predict on validation fold (X_val_fold uses selected features)
                          # predict() needs adaptation or careful input
                          # Pass X_val_fold as if it's the original X, and rely on internal logic with important_indices = None
                          y_val_pred = fold_model.predict(X_val_fold, smiles_val_fold)
                          fold_rmse = np.sqrt(mean_squared_error(y_val_fold, np.nan_to_num(y_val_pred, nan=np.nanmean(y_val_fold))))
                          fold_scores.append(fold_rmse)

                     mean_cv_score = np.mean(fold_scores) if fold_scores else float('inf')
                     scores.append(mean_cv_score)
                     print_status(f"Trial {trial.number}: Mean CV RMSE = {mean_cv_score:.5f}")

            except Exception as e:
                 print_status(f"Trial {trial.number} failed: {e}")
                 print_status(traceback.format_exc())
                 return float('inf')

            final_score = np.mean(scores) if scores else float('inf')
            # Handle Optuna pruning
            trial.report(final_score, trial.number)
            if trial.should_prune():
                raise optuna.TrialPruned()

            return final_score


        # --- Run Optuna Study ---
        sampler = TPESampler(seed=self.seed, multivariate=True, group=True, constant_liar=True)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials, n_jobs=1) # n_jobs=1 for Optuna to avoid nested parallelism issues

        # --- Process Best Trial ---
        print_status("\n--- Optuna Tuning Results ---")
        if study.best_trial:
             print_status(f"Best trial number: {study.best_trial.number}")
             print_status(f"Best RMSE: {study.best_value:.5f}")
             print_status("Best parameters:")
             best_params = study.best_params
             for key, value in best_params.items():
                 print_status(f"  {key}: {value}")

             self.tuned_params = best_params
             # Update self with best parameters
             if 'n_features' in best_params: self.n_features = best_params['n_features']
             if 'feature_method' in best_params: self.feature_method = best_params['feature_method']
             if 'n_clusters' in best_params: self.n_clusters = best_params['n_clusters']
             if 'clustering_method' in best_params: self.clustering_method = best_params['clustering_method']
             # Update XGBoost params
             xgb_best_params = {k: v for k, v in best_params.items() if k in [p[0] for p in param_map.values()]}
             self.xgb_params.update(xgb_best_params)

             print_status("Retraining final model with best parameters found...")
             self.train(X, y, smiles) # Rerun the main train method with updated self state

             # Save tuning results plots if optuna.visualization is available
             if _optuna_vis_available:
                 try:
                     fig_history = plot_optimization_history(study)
                     fig_history.write_image(self.output_dir / "tuning_history.png")
                     fig_params = plot_param_importances(study)
                     fig_params.write_image(self.output_dir / "tuning_param_importances.png")
                     print_status("Tuning plots saved.")
                 except Exception as e:
                     print_status(f"Warning: Could not generate Optuna plots: {e}")
             else:
                  print_status("Skipping Optuna plots: optuna.visualization not available.")

             return best_params
        else:
             print_status("Optuna study finished without finding a best trial (all might have failed).")
             return None


    def save(self, output_dir=None):
        """ Saves the entire ensemble object and components. """
        save_dir = Path(output_dir) if output_dir else self.output_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        print_status(f"Saving model components to {save_dir}")

        # --- Save the main ensemble object via Pickle ---
        # This is the primary way state is saved.
        try:
            with open(save_dir / "ensemble_model.pkl", "wb") as f:
                pickle.dump(self, f)
            print_status("Saved main ensemble object state via pickle.")
        except Exception as e:
             print_status(f"ERROR: Could not pickle main ensemble object: {e}. Model saving incomplete.")
             # Fallback: Try saving components individually

        # --- Save individual model components (as backup/alternative) ---
        try:
            if self.global_model: joblib.dump(self.global_model, save_dir / "global_model.joblib")
            if self.meta_model: joblib.dump(self.meta_model, save_dir / "meta_model.joblib")

            if self.cluster_models:
                cluster_models_dir = save_dir / "cluster_models"
                cluster_models_dir.mkdir(exist_ok=True)
                for cluster_id, model in self.cluster_models.items():
                    if model: joblib.dump(model, cluster_models_dir / f"cluster_{cluster_id}.joblib")

            if self.scaler: joblib.dump(self.scaler, save_dir / "scaler.joblib")
            if self.pca: joblib.dump(self.pca, save_dir / "pca.joblib")
            if self.functional_group_vectorizer: joblib.dump(self.functional_group_vectorizer, save_dir / "functional_vectorizer.joblib")
            if self.property_scaler: joblib.dump(self.property_scaler, save_dir / "property_scaler.joblib")
            # Save other potentially fitted objects like cluster_model if needed
            if hasattr(self, 'cluster_model') and self.cluster_model:
                 try: joblib.dump(self.cluster_model, save_dir / "cluster_model_core.joblib")
                 except Exception: print_status("Could not save core cluster model via joblib.")


            print_status("Saved individual components via joblib/xgboost save.")
        except Exception as e_comp:
             print_status(f"Warning: Error saving individual components: {e_comp}")

        # --- Save Metadata ---
        metadata = {
            'timestamp': datetime.now().isoformat(), 'seed': self.seed,
            'feature_cols': self.feature_cols, # List of original feature names
            'important_indices': self.important_indices, # Indices relative to feature_cols
            'important_features': self.important_features, # Names of selected features
            'n_features_used_in_model': self.n_features_used_in_model, # Actual number used
            'feature_method': self.feature_method, 'n_features_requested': self.n_features,
            'clustering_method': self.clustering_method, 'n_clusters': self.n_clusters,
            'min_cluster_size': self.min_cluster_size, 'cluster_property': self.cluster_property,
            'fp_types': self.fp_types, 'smiles_col': self.smiles_col, 'target_col': self.target_col,
            'replace_nan': self.replace_nan,
            'skip_meta_features': getattr(self, 'skip_meta_features', False),
            'use_model_mean': self.use_model_mean, 'use_weighted_mean': self.use_weighted_mean,
            'group_feature_per_cluster': self.group_feature_per_cluster,
            'cluster_specific_feature_indices': {str(k): v.tolist() if isinstance(v, np.ndarray) else v for k, v in self.cluster_specific_feature_indices.items()},
            'xgb_params': self.xgb_params, 'tuned_params': self.tuned_params,
            'model_comparison': self.model_comparison, 'error_analysis_per_cluster': self.error_analysis,
            'meta_feature_structure': self.meta_feature_structure, 'meta_feature_names': self.meta_feature_names,
            'cluster_stats': self.cluster_stats, 'df_columns_original': self.df_columns_,
            'user_specified_cli_params': list(self.user_specified_params),
            # Save state needed for predictors if not pickled
            'scaffold_clustering_state': {'map': self.scaffold_to_cluster, 'default': self.default_cluster} if self.clustering_method == 'scaffold' else None,
            'stratified_clustering_state': {'bins': self.bins, 'n_bins': self.n_bins_per_component} if self.clustering_method == 'stratified' else None,
        }
        try:
            with open(save_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2, cls=NpEncoder)
            print_status("Saved model metadata.")
        except Exception as e:
            print_status(f"Error saving metadata.json: {e}")

        print_status("Model saving process completed.")

    @classmethod
    def load(cls, load_dir):
        """ Loads the ensemble model primarily via pickle, with fallback for components. """
        load_dir = Path(load_dir)
        print_status(f"Loading model from {load_dir}")

        ensemble_pkl_path = load_dir / "ensemble_model.pkl"
        if not ensemble_pkl_path.exists():
             raise FileNotFoundError(f"Main ensemble file not found: {ensemble_pkl_path}")

        try:
             with open(ensemble_pkl_path, "rb") as f:
                 model = pickle.load(f)
             print_status("Loaded main ensemble object state via pickle.")
             if not isinstance(model, cls):
                  print_status(f"Warning: Loaded object type ({type(model)}) != expected ({cls}).")

             # --- Post-load checks and potential component reloading ---
             # Check if essential models loaded correctly via pickle
             models_ok = True
             if not model.global_model:
                  print_status("Pickled global_model is missing. Trying joblib load...")
                  global_joblib = load_dir / "global_model.joblib"
                  if global_joblib.exists(): model.global_model = joblib.load(global_joblib)
                  else: models_ok = False; print_status("ERROR: Cannot load global_model.")

             if model.clustering_method != 'none' and not model.cluster_models:
                  print_status("Pickled cluster_models are missing. Trying joblib load...")
                  cluster_models_dir = load_dir / "cluster_models"
                  if cluster_models_dir.is_dir():
                      loaded_any_cluster = False
                      model.cluster_models = {} # Ensure it's a dict
                      for model_file in cluster_models_dir.glob("cluster_*.joblib"):
                           try:
                               cluster_id = int(model_file.stem.split('_')[-1])
                               model.cluster_models[cluster_id] = joblib.load(model_file)
                               loaded_any_cluster = True
                           except Exception as e_c: print_status(f"Warn: Failed load {model_file}: {e_c}")
                      if not loaded_any_cluster: print_status("No cluster models found in joblib files.")
                  else: print_status("Cluster models directory not found.")


             if not model.meta_model and not getattr(model, 'skip_meta_features', False):
                   meta_joblib = load_dir / "meta_model.joblib"
                   if meta_joblib.exists():
                       print_status("Loading meta_model from joblib...")
                       model.meta_model = joblib.load(meta_joblib)
                   # else: Meta model might be optional or skipped

             # Reload auxiliary objects if they weren't pickled correctly
             if not model.scaler and (load_dir / "scaler.joblib").exists(): model.scaler = joblib.load(load_dir / "scaler.joblib")
             if not model.pca and (load_dir / "pca.joblib").exists(): model.pca = joblib.load(load_dir / "pca.joblib")
             if not model.functional_group_vectorizer and (load_dir / "functional_vectorizer.joblib").exists(): model.functional_group_vectorizer = joblib.load(load_dir / "functional_vectorizer.joblib")
             if not model.property_scaler and (load_dir / "property_scaler.joblib").exists(): model.property_scaler = joblib.load(load_dir / "property_scaler.joblib")
             if not hasattr(model, 'cluster_model') or not model.cluster_model: # Core cluster model (e.g., KMeans)
                   core_cluster_path = load_dir / "cluster_model_core.joblib"
                   if core_cluster_path.exists(): model.cluster_model = joblib.load(core_cluster_path)


             # Reconstruct predictor based on loaded state
             model._reconstruct_predictor()

             print_status("Model loading complete.")
             return model

        except Exception as e:
            print_status(f"ERROR: Critical error during model loading: {e}")
            print_status(traceback.format_exc())
            raise # Re-raise error

    def _reconstruct_predictor(self):
        """ Reconstructs the appropriate predictor instance after loading. """
        # print_status("Reconstructing cluster predictor...") # Reduce noise
        method = getattr(self, 'clustering_method', 'none')
        self.cluster_model_predictor = None # Reset predictor

        try:
            if method == 'none': pass # Predictor remains None
            elif method == 'kmeans' and hasattr(self, 'cluster_model') and isinstance(self.cluster_model, KMeans):
                self.cluster_model_predictor = KMeansPredictor(self.cluster_model)
            elif method == 'gaussian' and hasattr(self, 'cluster_model') and isinstance(self.cluster_model, GaussianMixture):
                 self.cluster_model_predictor = GaussianMixturePredictor(self.cluster_model)
            elif method == 'random':
                if hasattr(self, 'n_clusters') and self.n_clusters is not None:
                     self.cluster_model_predictor = RandomClusterPredictor(self.n_clusters, self.seed)
                else: print_status("Warn: Cannot reconstruct Random predictor - n_clusters missing.")
            # --- NN Predictors: Require training data which isn't saved ---
            elif method in ['agglomerative', 'spectral', 'dbscan']:
                 print_status(f"Warning: Cannot fully reconstruct predictor for NN-based method '{method}' without training data. Prediction will likely fail if clustering is needed.")
                 # Attempt partial reconstruction if core model exists (e.g. DBSCAN)
                 if method == 'dbscan' and hasattr(self, 'cluster_model') and isinstance(self.cluster_model, DBSCAN):
                       # Cannot create DBSCANWithPredict without training data reference
                       pass
                 elif method == 'agglomerative' and hasattr(self, 'cluster_model') and isinstance(self.cluster_model, AgglomerativeClustering):
                       pass # Cannot create AgglomerativePredictor
                 elif method == 'spectral' and hasattr(self, 'cluster_model') and isinstance(self.cluster_model, SpectralClustering):
                       pass # Cannot create SpectralPredictor

            elif method == 'scaffold':
                # Try loading state from metadata if not pickled directly
                scaffold_map = getattr(self, 'scaffold_to_cluster', None)
                default_c = getattr(self, 'default_cluster', 0)
                if scaffold_map is None and hasattr(self, 'metadata') and self.metadata: # Check if metadata was loaded
                    sc_state = self.metadata.get('scaffold_clustering_state')
                    if sc_state: scaffold_map = sc_state.get('map'); default_c = sc_state.get('default', 0)
                if scaffold_map is not None:
                    self.cluster_model_predictor = ScaffoldPredictor(scaffold_map, default_c)
                else: print_status("Warn: Missing scaffold map for predictor.")

            elif method == 'functional':
                vectorizer = getattr(self, 'functional_group_vectorizer', None)
                core_model = getattr(self, 'cluster_model', None) # e.g., KMeans fitted on func groups
                if vectorizer and core_model:
                     self.cluster_model_predictor = FunctionalGroupPredictor(vectorizer, core_model)
                else: print_status("Warn: Missing vectorizer or core model for functional predictor.")

            elif method in ChemicalPropertyPredictor._property_calculators:
                prop_scaler = getattr(self, 'property_scaler', None)
                core_model = getattr(self, 'cluster_model', None) # e.g., KMeans fitted on property
                if prop_scaler and core_model:
                     self.cluster_model_predictor = ChemicalPropertyPredictor(method, prop_scaler, core_model)
                else: print_status(f"Warn: Missing scaler or core model for property predictor '{method}'.")

            elif method == 'property': # PCA case
                 pca = getattr(self, 'pca', None)
                 scaler = getattr(self, 'scaler', None) # General feature scaler
                 core_model = getattr(self, 'cluster_model', None) # e.g., KMeans fitted on PCA
                 if pca and scaler and core_model:
                     # Predictor needs the KMeans model. PCA/Scaler applied before calling predictor.
                     self.cluster_model_predictor = KMeansPredictor(core_model)
                 else: print_status("Warn: Missing PCA, scaler or core model for PCA property predictor.")

            elif method == 'stratified':
                 pca = getattr(self, 'pca', None)
                 bins = getattr(self, 'bins', None)
                 n_bins = getattr(self, 'n_bins_per_component', None)
                 n_clus = getattr(self, 'n_clusters', None)
                 if pca and bins is not None and n_bins is not None and n_clus is not None:
                      self.cluster_model_predictor = StratifiedPredictor(pca, bins, n_bins, n_clus)
                 else: print_status("Warn: Missing components for stratified predictor.")

            else:
                 print_status(f"Warning: Predictor reconstruction not implemented for method '{method}'.")


            if self.cluster_model_predictor:
                 print_status(f"Reconstructed predictor: {type(self.cluster_model_predictor).__name__}")
            elif method != 'none':
                  print_status(f"Failed to reconstruct predictor for method '{method}'.")

        except Exception as e_reconstruct:
             print_status(f"Error during predictor reconstruction: {e_reconstruct}")


# --- Helper Classes ---
class NpEncoder(json.JSONEncoder):
    """ Custom JSON encoder for NumPy data types """
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating):
            if np.isnan(obj): return None # Represent NaN as null in JSON
            if np.isinf(obj): return str(obj) # Represent Inf as "Infinity" or "-Infinity" string
            return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.bool_, bool)): return bool(obj)
        if isinstance(obj, set): return list(obj)
        # Handle Path objects
        if isinstance(obj, Path): return str(obj)
        # Handle functions/lambdas (store name or placeholder)
        if callable(obj): return f"<function {obj.__name__}>"

        try: # General fallback for objects with to_dict
             if hasattr(obj, 'to_dict') and callable(obj.to_dict):
                  return obj.to_dict()
        except Exception: pass # Ignore if to_dict fails

        try: # General fallback for objects with __dict__
             return obj.__dict__
        except Exception: pass # Ignore if __dict__ fails


        return super(NpEncoder, self).default(obj)


# --- Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description="Train or evaluate a clustered XGBoost ensemble model for LogD prediction.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    io_group = parser.add_argument_group("Input/Output")
    io_group.add_argument("--input", required=False, help="Input CSV file with SMILES and target values (required for training).")
    io_group.add_argument("--output-dir", required=True, help="Output directory for models, results, and logs.")
    io_group.add_argument("--smiles-col", default="SMILES", help="Column name for SMILES strings.")
    io_group.add_argument("--target", default="LogD", help="Column name for the target property (e.g., LogD).")
    io_group.add_argument("--external-test", help="Optional external test set CSV file for evaluation/tuning.")
    io_group.add_argument("--auto-evaluate-external", action="store_true", default=True, 
                        help="Automatically evaluate on external test set (if provided) after training. Default: True")
    io_group.add_argument("--export-validation", action="store_true", help="Export validation/test set predictions and errors to CSV.")
    io_group.add_argument("--replace-nan", type=float, default=None, help="Value to replace NaN values with in features or target (e.g., 0). If None, rows with NaN target are dropped, feature NaNs become 0.")
    io_group.add_argument("--load-model", help="Path to directory containing a previously saved model to load.", default=None)
    io_group.add_argument("--log-file", help="If specified, save console output to this file name inside output-dir.", default=None)

    action_group = parser.add_argument_group("Actions")
    action_group.add_argument("--train", action="store_true", help="Train a new model (requires --input).")
    action_group.add_argument("--evaluate", action="store_true", help="Evaluate a loaded model (--load-model) on data (--input or --external-test).")
    action_group.add_argument("--predict-only", action="store_true", help="Load model (--load-model) and predict on input data (--input or --external-test), saving predictions.")

    model_group = parser.add_argument_group("XGBoost Model Parameters")
    model_group.add_argument("--max-depth", type=int, default=6, help="Maximum depth of trees.")
    model_group.add_argument("--learning-rate", type=float, default=0.1, help="Learning rate (eta).")
    model_group.add_argument("--n-estimators", type=int, default=100, help="Number of boosting rounds/trees.")
    model_group.add_argument("--subsample", type=float, default=1.0, help="Subsample ratio of the training instance.")
    model_group.add_argument("--colsample-bytree", type=float, default=1.0, help="Subsample ratio of columns when constructing each tree.")
    model_group.add_argument("--min-child-weight", type=int, default=1, help="Minimum sum of instance weight needed in a child.")
    model_group.add_argument("--gamma", type=float, default=0, help="Minimum loss reduction required to make a further partition on a leaf node (min_split_loss).")
    model_group.add_argument("--reg-alpha", type=float, default=0, help="L1 regularization term on weights (alpha).")
    model_group.add_argument("--reg-lambda", type=float, default=1, help="L2 regularization term on weights (lambda).")

    cluster_group = parser.add_argument_group("Clustering Parameters")
    cluster_group.add_argument("--clustering-method", default="kmeans",
                                choices=["none", "kmeans", "dbscan", "random", "agglomerative", "spectral", "gaussian",
                                        "stratified", "functional", "scaffold", "property"] + list(ChemicalPropertyPredictor._property_calculators.keys()),
                                help="Method for clustering compounds ('none' to disable clustering). 'property' uses PCA of features unless --cluster-property is set.")
    cluster_group.add_argument("--n-clusters", type=int, default=5, help="Number of clusters to create (target).")
    cluster_group.add_argument("--min-cluster-size", type=int, default=10, help="Minimum number of samples required to train a cluster-specific model.")
    cluster_group.add_argument("--cluster-property", default=None, choices=list(ChemicalPropertyPredictor._property_calculators.keys()),
                                help="Specific chemical property to use when --clustering-method is set to a property name or 'property'.")


    feature_group = parser.add_argument_group("Feature Engineering & Selection")
    feature_group.add_argument("--fingerprint-types", nargs="+", default=["morgan"],
                                choices=["morgan", "maccs", "rdkit", "atom_pairs", "topological_torsion", "pattern", "avalon"],
                                help="Types of fingerprints to generate if not using existing features.")
    feature_group.add_argument("--n-features", type=int, default=None, help="Number of top features to select (default: use all available features).")
    feature_group.add_argument("--feature-method", default="xgb", choices=["xgb", "rf", "mutual_info", "permutation", "shap", "random", "all"],
                                help="Method for selecting important features if --n-features is set.")

    tuning_group = parser.add_argument_group("Hyperparameter Tuning (Optuna)")
    tuning_group.add_argument("--tune", action="store_true", help="Enable hyperparameter tuning for XGBoost params.")
    tuning_group.add_argument("--tune-full-pipeline", action="store_true", help="Tune features, clustering, and XGBoost parameters together (implies --tune).")
    tuning_group.add_argument("--n-trials", type=int, default=20, help="Number of trials for Optuna tuning.")
    tuning_group.add_argument("--cv-folds", type=int, default=3, help="Number of cross-validation folds used during tuning (if external test not used).")

    other_group = parser.add_argument_group("Other Parameters")
    other_group.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    other_group.add_argument("--n-jobs", type=int, default=-1, help="Number of parallel jobs (-1 uses all available cores). Applies to feature generation, model training, etc.")
    other_group.add_argument("--max-rows", type=int, default=None, help="Maximum number of rows to load from input CSV (for debugging/quick tests).")
    other_group.add_argument("--skip-meta-features", action="store_true", help="Do not train or use the ensemble meta-model.")
    other_group.add_argument("--skip-visualization", action="store_true", help="Skip generating visualization plots (e.g., cluster plots, tuning plots).")
    other_group.add_argument("--use-model-mean", action="store_true", help="If meta-model is skipped/disabled, use simple average of global and cluster predictions instead of cluster-only.")
    other_group.add_argument("--use-weighted-mean", action="store_true", help="Use weighted average of global/cluster predictions (requires weights - currently uses 0.5 default).")
    other_group.add_argument("--group-feature-per-cluster", action="store_true", help="Perform feature selection separately for each cluster's model (experimental).")

    # --- Argument Validation ---
    # Use a dummy parser to get defaults without triggering help/errors prematurely
    default_args = {}
    try:
        dummy_parser = argparse.ArgumentParser(add_help=False)
        # Re-add all arguments to the dummy parser to capture defaults
        for group in [io_group, action_group, model_group, cluster_group, feature_group, tuning_group, other_group]:
             for action in group._group_actions:
                 # Reconstruct add_argument call (simplified)
                 kwargs = {'default': action.default, 'type': action.type, 'help': argparse.SUPPRESS}
                 if action.choices: kwargs['choices'] = action.choices
                 if action.nargs: kwargs['nargs'] = action.nargs
                 if isinstance(action, argparse._StoreTrueAction): kwargs['action'] = 'store_true'; del kwargs['type']
                 elif isinstance(action, argparse._StoreFalseAction): kwargs['action'] = 'store_false'; del kwargs['type']
                 # Add other action types if needed
                 dummy_parser.add_argument(*action.option_strings, **kwargs)
        default_args = vars(dummy_parser.parse_args([]))
    except Exception as e:
         print(f"Warning: Could not get default arguments for validation: {e}")


    # Parse actual arguments
    parsed_args = parser.parse_args()
    passed_args_dict = vars(parsed_args)

    # Store user-specified params based on comparison with defaults
    user_specified = {
        k for k, v in passed_args_dict.items()
        if k in default_args and v != default_args[k]
    }
    # Add action flags that were triggered (True when default is False)
    for k, v in passed_args_dict.items():
         if isinstance(parser._option_string_actions.get('--'+k.replace('_','-')), argparse._StoreTrueAction) and v is True:
              user_specified.add(k)

    parsed_args.user_specified_params = user_specified # Attach for later use

    # Perform validation checks
    if not parsed_args.load_model and not parsed_args.train:
        parser.error("Action required: Specify either --train or --load-model (with --evaluate or --predict-only).")
    if parsed_args.train and not parsed_args.input:
        parser.error("--train requires --input file.")
    if parsed_args.evaluate and not parsed_args.load_model:
        parser.error("--evaluate requires --load-model.")
    if parsed_args.evaluate and not (parsed_args.input or parsed_args.external_test):
        parser.error("--evaluate requires data from --input or --external-test.")
    if parsed_args.predict_only and not parsed_args.load_model:
        parser.error("--predict-only requires --load-model.")
    if parsed_args.predict_only and not (parsed_args.input or parsed_args.external_test):
         parser.error("--predict-only requires data from --input or --external-test.")

    if parsed_args.tune_full_pipeline:
         parsed_args.tune = True # Ensure base tuning is enabled

    if parsed_args.clustering_method != 'property' and parsed_args.clustering_method not in ChemicalPropertyPredictor._property_calculators and parsed_args.cluster_property:
         print_status(f"Warning: --cluster-property ('{parsed_args.cluster_property}') is ignored because --clustering-method is '{parsed_args.clustering_method}'.")
         parsed_args.cluster_property = None # Clear it if not applicable
    elif parsed_args.clustering_method == 'property' and not parsed_args.cluster_property:
         print_status("Info: --clustering-method='property' without --cluster-property will use PCA of features.")
    elif parsed_args.clustering_method in ChemicalPropertyPredictor._property_calculators:
         if parsed_args.cluster_property and parsed_args.cluster_property != parsed_args.clustering_method:
              parser.error(f"Cannot specify conflicting --cluster-property='{parsed_args.cluster_property}' when --clustering-method is already '{parsed_args.clustering_method}'.")
         parsed_args.cluster_property = parsed_args.clustering_method # Use method name as property
         print_status(f"Using chemical property '{parsed_args.cluster_property}' for clustering.")

    if parsed_args.group_feature_per_cluster and parsed_args.n_features is None and 'n_features' not in user_specified:
        print_status("Warning: --group-feature-per-cluster enabled but --n-features not set. Feature selection per cluster might use a default limit or all features.")

    return parsed_args


# --- Plotting Functions ---
def plot_results(y_true, y_pred, filename, title='True vs Predicted Target'):
    if y_true is None or y_pred is None or len(y_true) == 0 or len(y_true) != len(y_pred):
        print_status("Warning: Cannot plot results due to missing or mismatched data.")
        return
    try:
        # Ensure numpy arrays and handle potential NaNs before plotting/metrics
        y_true_np = np.array(y_true)
        y_pred_np = np.array(y_pred)
        valid_mask = ~np.isnan(y_true_np) & ~np.isnan(y_pred_np)
        y_true_valid = y_true_np[valid_mask]
        y_pred_valid = y_pred_np[valid_mask]

        if len(y_true_valid) < 2:
             print_status("Warning: Too few valid data points (<2) to generate meaningful scatter plot.")
             return

        r2 = r2_score(y_true_valid, y_pred_valid)
        rmse = np.sqrt(mean_squared_error(y_true_valid, y_pred_valid))

        plt.figure(figsize=(8, 8))
        min_val = min(np.min(y_true_valid), np.min(y_pred_valid)) - 0.5
        max_val = max(np.max(y_true_valid), np.max(y_pred_valid)) + 0.5
        plt.scatter(y_true_valid, y_pred_valid, alpha=0.5, s=10, label=f'R² = {r2:.3f}\nRMSE = {rmse:.3f}')
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title(title)
        plt.legend(loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.axis('equal')
        plt.xlim(min_val, max_val)
        plt.ylim(min_val, max_val)
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
        print_status(f"Scatter plot saved to {filename}")
    except Exception as e:
        print_status(f"Error creating results plot '{filename}': {e}")

def plot_cluster_metrics(cluster_metrics, filename, metric='rmse', title='RMSE by Cluster'):
     if not cluster_metrics:
         print_status("Warning: No cluster metrics data to plot.")
         return
     try:
         metrics_df = pd.DataFrame.from_dict(cluster_metrics, orient='index')
         if metric not in metrics_df.columns:
              alt_metrics = [m for m in ['rmse', 'mae', 'r2'] if m in metrics_df.columns]
              if not alt_metrics:
                   print_status(f"Warning: Metric '{metric}' (and alternatives) not found in cluster metrics. Cannot plot.")
                   return
              metric = alt_metrics[0] # Use first available alternative
              title = f"{metric.upper()} by Cluster"
              print_status(f"Using alternative metric '{metric}' for cluster plot.")


         metrics_df.index.name = 'ClusterID'
         metrics_df = metrics_df.sort_index()

         # Filter out clusters with NaN metric (e.g., if only 1 sample)
         metrics_df = metrics_df.dropna(subset=[metric])
         if metrics_df.empty:
              print_status("Warning: No valid cluster metrics to plot after removing NaNs.")
              return


         plt.figure(figsize=(max(6, len(metrics_df)*0.5), 5))
         bars = plt.bar(metrics_df.index.astype(str), metrics_df[metric])
         plt.xlabel('Cluster ID')
         plt.ylabel(metric.upper())
         plt.title(title)
         if 'n_samples' in metrics_df.columns:
              for bar, n_samples in zip(bars, metrics_df['n_samples']):
                   yval = bar.get_height()
                   plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'N={int(n_samples)}', va='bottom' if yval >= 0 else 'top', ha='center', fontsize=8)

         plt.tight_layout()
         plt.savefig(filename, dpi=300)
         plt.close()
         print_status(f"Cluster metrics plot saved to {filename}")
     except Exception as e:
        print_status(f"Error creating cluster metrics plot '{filename}': {e}")


# --- Main Execution Logic ---
def main():
    args = parse_args()
    start_time_main = time.time()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.log_file:
        log_path = output_dir / args.log_file
        print(f"-- Logging console output to {log_path}")
        try:
            with open(log_path, 'w') as f:
                f.write(f"-- LogD Prediction Log - {time.strftime('%Y-%m-%d %H:%M:%S')} --\n")
                f.write("Command: " + " ".join(sys.argv) + "\n")
                f.write("Arguments:\n")
                for arg, value in vars(args).items():
                     if arg != 'user_specified_params': # Don't log the set itself
                          f.write(f"  --{arg}: {value}\n")
                f.write(f"User specified: {args.user_specified_params}\n")
                f.write("-" * 30 + "\n")
            print_status.log_file = log_path
        except Exception as e:
            print(f"-- Warning: Could not create log file '{log_path}': {e}")
            print_status.log_file = None
    else:
        print_status.log_file = None


    print_status("Starting Process...")
    print_status("Arguments used:")
    for arg, value in vars(args).items():
        if arg != 'user_specified_params': print_status(f"  --{arg}: {value}")
    print_status(f"User explicitly set: {args.user_specified_params}")


    model = None
    try:
        if args.load_model:
            model = ClusteredXGBoostEnsemble.load(args.load_model)
            model.output_dir = output_dir # Use current output dir
            model.skip_visualization = args.skip_visualization # Update from CLI
            # Avoid overriding key loaded params unless intended
            model.seed = args.seed
            model.xgb_params['n_jobs'] = args.n_jobs # Update n_jobs

        elif args.train:
            model = ClusteredXGBoostEnsemble(output_dir=output_dir, seed=args.seed)
            # Transfer all relevant args
            model.xgb_params = {k: getattr(args, k) for k in ['max_depth', 'learning_rate', 'n_estimators', 'subsample', 'colsample_bytree', 'min_child_weight', 'gamma', 'reg_alpha', 'reg_lambda']}
            model.xgb_params['n_jobs'] = args.n_jobs # Crucial: pass n_jobs here
            model.n_clusters = args.n_clusters
            model.min_cluster_size = args.min_cluster_size
            model.clustering_method = args.clustering_method
            model.cluster_property = args.cluster_property
            model.n_features = args.n_features
            model.feature_method = args.feature_method
            model.fp_types = args.fingerprint_types
            model.skip_meta_features = args.skip_meta_features
            model.skip_visualization = args.skip_visualization
            model.replace_nan = args.replace_nan
            model.use_model_mean = args.use_model_mean
            model.use_weighted_mean = args.use_weighted_mean
            model.group_feature_per_cluster = args.group_feature_per_cluster
            model.smiles_col = args.smiles_col
            model.target_col = args.target
            model.user_specified_params = args.user_specified_params # Store the set

        else:
             raise ValueError("No valid action specified.")

        # --- Load Data ---
        data_df = None
        smiles, y, X = None, None, None
        feature_cols = None
        eval_smiles, eval_y, eval_X = None, None, None # Data for evaluate/predict actions

        input_source = args.input
        # Determine primary data source for evaluate/predict if input not specified
        if not input_source and (args.evaluate or args.predict_only) and args.external_test:
             input_source = args.external_test
             print_status(f"Using external test set as primary data for action: {input_source}")

        if input_source:
            print_status(f"Loading data from: {input_source}")
            data_df = pd.read_csv(input_source, nrows=args.max_rows)
            print_status(f"Loaded {len(data_df)} rows.")

            # Prepare data using the model instance
            X, y, smiles, feature_cols = model.prepare_data(
                 data_df, smiles_col=args.smiles_col, target_col=args.target,
                 fp_types=args.fingerprint_types, n_jobs=args.n_jobs # Pass n_jobs
                 )
            model.feature_cols = feature_cols # Ensure model has feature names

            # Assign data for evaluate/predict actions
            if args.evaluate or args.predict_only:
                 eval_X, eval_y, eval_smiles = X, y, smiles
                 # Check if target 'y' is actually present for evaluation
                 if args.evaluate and y is None:
                      print_status(f"Warning: Target column '{args.target}' not found or invalid in '{input_source}'. Cannot perform evaluation.")
                      args.evaluate = False # Disable evaluation if target missing


        # --- Handle External Test Data (if specified AND not used as primary source) ---
        if args.external_test and args.external_test != input_source:
             print_status(f"Loading external test data from: {args.external_test}")
             ext_df = pd.read_csv(args.external_test)
             print_status(f"Loaded {len(ext_df)} rows from external test set.")
             try:
                  ext_smiles_list = ext_df[model.smiles_col].tolist()
                  ext_y_true = ext_df[model.target_col].values if model.target_col in ext_df else None

                  if not model.feature_cols:
                       raise RuntimeError("Cannot process external test data: Model is missing training feature information (feature_cols). Load a trained model or run --train first.")

                  # Align features: Generate or select based on training feature_cols
                  existing_ext_feats = [col for col in model.feature_cols if col in ext_df.columns]
                  if len(existing_ext_feats) == len(model.feature_cols):
                       print_status(f"External test set has all {len(model.feature_cols)} feature columns matching the training data.")
                       ext_X_prepared = ext_df[model.feature_cols].values
                  else:
                       print_status(f"External test set has {len(existing_ext_feats)}/{len(model.feature_cols)} matching feature columns. Generating/aligning features...")
                       # Generate features based on smiles
                       ext_X_generated, gen_names = model._generate_fingerprints_and_features(ext_smiles_list, model.fp_types, model._get_n_jobs(args.n_jobs))
                       # Align generated features to model.feature_cols
                       if gen_names == model.feature_cols:
                            ext_X_prepared = ext_X_generated
                            print_status("Generated feature names match training features exactly.")
                       else: # Need alignment based on names
                            print_status("Aligning generated external features with model features...")
                            ext_X_prepared = np.full((len(ext_smiles_list), len(model.feature_cols)), model.replace_nan if model.replace_nan is not None else 0.0)
                            gen_name_to_idx = {name: i for i, name in enumerate(gen_names)}
                            matched_count = 0
                            for model_idx, model_name in enumerate(model.feature_cols):
                                 if model_name in gen_name_to_idx:
                                      gen_idx = gen_name_to_idx[model_name]
                                      ext_X_prepared[:, model_idx] = ext_X_generated[:, gen_idx]
                                      matched_count += 1
                                 # else: Keep default fill value (NaN replacement)
                            print_status(f"Feature alignment complete. Matched {matched_count}/{len(model.feature_cols)} features.")


                  # Handle NaNs
                  replace_val_ext = model.replace_nan if model.replace_nan is not None else 0.0
                  if np.isnan(ext_X_prepared).any():
                       ext_nan_count = np.isnan(ext_X_prepared).sum()
                       print_status(f"Replacing {ext_nan_count} NaN values in external features with {replace_val_ext}.")
                       ext_X_prepared = np.nan_to_num(ext_X_prepared, nan=replace_val_ext)

                  # Store the external data for use after training
                  model.external_eval_data = {'X': ext_X_prepared, 'y': ext_y_true, 'smiles': ext_smiles_list}
                  
                  # Use this external data if evaluate/predict action is set and primary data wasn't external
                  if (args.evaluate or args.predict_only) and (eval_X is None):
                       eval_X, eval_smiles = ext_X_prepared, ext_smiles_list
                       if args.evaluate:
                            if ext_y_true is not None: eval_y = ext_y_true
                            else: print_status(f"Warning: Target '{args.target}' missing in external test set. Cannot evaluate."); args.evaluate = False
                       print_status(f"Using external test set for {'evaluation' if args.evaluate else 'prediction'}.")
                  # Store prepared external data for potential use in tuning
                  elif args.tune or args.tune_full_pipeline:
                       if ext_y_true is not None:
                            # Store under a different name to avoid conflict if primary data is also eval data
                            model.external_tune_data = {'X_test': ext_X_prepared, 'y_true': ext_y_true, 'smiles': ext_smiles_list}
                            print_status("External test data prepared for use in tuning.")
                       else: print_status("External test data missing target, cannot be used for tuning validation.")


             except Exception as e:
                  print_status(f"Error processing external test data: {e}. Skipping.")
                  print_status(traceback.format_exc())


        # --- Perform Actions ---
        if args.train:
            if X is None or y is None:
                 raise ValueError("--train action requires valid data from --input.")

            # Prepare data for training/validation split or tuning
            X_train_final, y_train_final, smiles_train_final = X, y, smiles
            X_val, y_val, smiles_val = None, None, None # Internal validation set

            external_tune_data_dict = getattr(model, 'external_tune_data', None)

            if not (args.tune or args.tune_full_pipeline) or not external_tune_data_dict:
                 # Need internal validation split if not tuning with external data
                 X_train_final, X_val, y_train_final, y_val, smiles_train_final, smiles_val = train_test_split(
                      X, y, smiles, test_size=0.2, random_state=args.seed
                 )
                 print_status(f"Internal split: Training set={len(X_train_final)}, Validation set={len(X_val)}.")
            elif external_tune_data_dict:
                 print_status("Using full input data for training, external set for tuning validation.")


            # Tuning (Optional)
            if args.tune or args.tune_full_pipeline:
                 model.tune_hyperparameters(X_train_final, y_train_final, smiles_train_final,
                                          n_trials=args.n_trials, cv_folds=args.cv_folds,
                                          tune_clusters=args.tune_full_pipeline,
                                          external_test_data=external_tune_data_dict,
                                          args=args) # Pass args for context
            else:
                 # Train without tuning
                 model.train(X_train_final, y_train_final, smiles_train_final)


            # Evaluate on internal validation set (if it exists)
            if X_val is not None and y_val is not None:
                 print_status("Evaluating model on internal validation set...")
                 val_results = model.evaluate(X_val, y_val, smiles_val)
                 if not args.skip_visualization:
                      plot_results(np.array(val_results['y_true']), np.array(val_results['y_pred']),
                                   output_dir / "validation_scatter.png", title="Internal Validation Set")
                      if 'cluster_metrics' in val_results and val_results['cluster_metrics']:
                           plot_cluster_metrics(val_results['cluster_metrics'], output_dir / "validation_cluster_metrics.png", title="Cluster Metrics (Internal Validation)")

                 if args.export_validation:
                      val_df = pd.DataFrame({
                           model.smiles_col: smiles_val,
                           model.target_col + '_true': val_results['y_true'],
                           model.target_col + '_pred': val_results['y_pred'],
                           'cluster': val_results.get('clusters', [-1]*len(smiles_val))
                      })
                      val_df.to_csv(output_dir / "internal_validation_predictions.csv", index=False)
                      print_status("Internal validation predictions saved.")

            # Evaluate on external test set (if provided and processed)
            if eval_X is not None and eval_y is not None:
                 print_status(f"\n-- Evaluating model on External Test Set ({input_source or args.external_test}) --")
                 ext_test_results = model.evaluate(eval_X, eval_y, eval_smiles)

                 if not args.skip_visualization:
                      plot_title_ext = f"External Test Set ({Path(input_source or args.external_test).name})"
                      plot_results(np.array(ext_test_results['y_true']), np.array(ext_test_results['y_pred']),
                                   output_dir / "external_test_scatter.png", title=plot_title_ext)
                      if 'cluster_metrics' in ext_test_results and ext_test_results['cluster_metrics']:
                           plot_cluster_metrics(ext_test_results['cluster_metrics'], output_dir / "external_test_cluster_metrics.png", title=f"Cluster Metrics ({plot_title_ext})")

                 if args.export_validation: # Re-use flag to export external results
                      ext_test_df = pd.DataFrame({
                           model.smiles_col: eval_smiles,
                           model.target_col + '_true': ext_test_results['y_true'],
                           model.target_col + '_pred': ext_test_results['y_pred'],
                           'cluster': ext_test_results.get('clusters', [-1]*len(eval_smiles))
                      })
                      # Add error column
                      if 'y_pred' in ext_test_results and 'y_true' in ext_test_results:
                           pred_clean_ext = np.nan_to_num(np.array(ext_test_results['y_pred']), nan=np.nan)
                           true_clean_ext = np.array(ext_test_results['y_true'])
                           ext_test_df['error'] = pred_clean_ext - true_clean_ext

                      ext_test_df.to_csv(output_dir / "external_test_predictions.csv", index=False)
                      print_status("External test predictions saved.")
            elif eval_X is not None and eval_y is None:
                 print_status("\n-- Skipping evaluation on external test set: Target data (y) not available --")

            # Save the trained/tuned model
            model.save()
            
            # After training and saving, automatically evaluate on the external test set if available
            if args.auto_evaluate_external and hasattr(model, 'external_eval_data') and model.external_eval_data.get('X') is not None:
                ext_X, ext_y, ext_smiles = model.external_eval_data['X'], model.external_eval_data['y'], model.external_eval_data['smiles']
                if ext_y is not None:
                    print_status(f"\n-- Automatically evaluating trained model on External Test Set ({args.external_test}) --")
                    ext_auto_results = model.evaluate(ext_X, ext_y, ext_smiles)
                    
                    if not args.skip_visualization:
                        plot_title_ext = f"External Test Set ({Path(args.external_test).name})"
                        plot_results(np.array(ext_auto_results['y_true']), np.array(ext_auto_results['y_pred']),
                                   output_dir / "external_test_scatter.png", title=plot_title_ext)
                        if 'cluster_metrics' in ext_auto_results and ext_auto_results['cluster_metrics']:
                            plot_cluster_metrics(ext_auto_results['cluster_metrics'], 
                                               output_dir / "external_test_cluster_metrics.png", 
                                               title=f"Cluster Metrics ({plot_title_ext})")
                    
                    if args.export_validation:
                        ext_auto_df = pd.DataFrame({
                            model.smiles_col: ext_smiles,
                            model.target_col + '_true': ext_auto_results['y_true'],
                            model.target_col + '_pred': ext_auto_results['y_pred'],
                            'cluster': ext_auto_results.get('clusters', [-1]*len(ext_smiles))
                        })
                        # Add error column
                        pred_clean_ext = np.nan_to_num(np.array(ext_auto_results['y_pred']), nan=np.nan)
                        true_clean_ext = np.array(ext_auto_results['y_true'])
                        ext_auto_df['error'] = pred_clean_ext - true_clean_ext
                        
                        ext_auto_df.to_csv(output_dir / "external_test_predictions.csv", index=False)
                        print_status("External test predictions saved.")
                else:
                    print_status("\n-- External test data available but missing target values. Cannot evaluate. --")
            elif hasattr(model, 'external_eval_data') and model.external_eval_data.get('X') is not None and not args.auto_evaluate_external:
                print_status("\n-- External test evaluation skipped (--auto-evaluate-external is disabled) --")


        if args.evaluate:
            if model is None: raise ValueError("--evaluate requires a loaded model.")
            if eval_X is None or eval_y is None: raise ValueError("--evaluate requires data with target column.")

            print_status(f"Evaluating loaded model on designated evaluation data...")
            eval_results = model.evaluate(eval_X, eval_y, eval_smiles)

            if not args.skip_visualization:
                 plot_title = f"Evaluation Results"
                 plot_results(np.array(eval_results['y_true']), np.array(eval_results['y_pred']),
                              output_dir / "evaluation_scatter.png", title=plot_title)
                 if 'cluster_metrics' in eval_results and eval_results['cluster_metrics']:
                      plot_cluster_metrics(eval_results['cluster_metrics'], output_dir / "evaluation_cluster_metrics.png", title=f"Cluster Metrics ({plot_title})")

            if args.export_validation:
                 eval_df = pd.DataFrame({
                      model.smiles_col: eval_smiles,
                      model.target_col + '_true': eval_results['y_true'],
                      model.target_col + '_pred': eval_results['y_pred'],
                      'cluster': eval_results.get('clusters', [-1]*len(eval_smiles))
                 })
                 eval_df.to_csv(output_dir / "evaluation_predictions.csv", index=False)
                 print_status("Evaluation predictions saved.")


        if args.predict_only:
            if model is None: raise ValueError("--predict-only requires a loaded model.")
            if eval_X is None: raise ValueError("--predict-only requires data.")

            print_status(f"Predicting on designated prediction data...")
            predictions = model.predict(eval_X, eval_smiles)

            pred_df = pd.DataFrame({
                model.smiles_col: eval_smiles,
                model.target_col + '_pred': predictions
            })
            # Add clusters if available and calculable
            eval_clusters_for_pred = None
            if model.clustering_method != 'none' and model.cluster_model_predictor:
                 try: # Predict clusters for the prediction set
                     if isinstance(model.cluster_model_predictor, (FunctionalGroupPredictor, ScaffoldPredictor, ChemicalPropertyPredictor)):
                          if eval_smiles is None: raise ValueError("Predictor needs SMILES.")
                          eval_clusters_for_pred = model.cluster_model_predictor.predict(eval_smiles)
                     # ... (add other predictor types similar to evaluate block) ...
                     else: # kmeans, etc. on features
                          X_eval_base = eval_X[:, model.important_indices] if model.important_indices else eval_X
                          if model.clustering_method == 'random': pred_cluster_data = X_eval_base
                          elif model.scaler and hasattr(model.scaler, 'mean_'): pred_cluster_data = model.scaler.transform(X_eval_base)
                          else: pred_cluster_data = X_eval_base
                          eval_clusters_for_pred = model.cluster_model_predictor.predict(pred_cluster_data)

                     if eval_clusters_for_pred is not None:
                          pred_df['cluster'] = eval_clusters_for_pred
                 except Exception as e_clus_pred:
                      print_status(f"Warning: Could not predict clusters for output file: {e_clus_pred}")

            pred_filename = output_dir / "predictions.csv"
            pred_df.to_csv(pred_filename, index=False)
            print_status(f"Predictions saved to {pred_filename}")


        end_time_main = time.time()
        duration = end_time_main - start_time_main
        print_status(f"Process finished in {duration:.2f} seconds.")
        return 0

    except FileNotFoundError as e:
         print_status(f"\n--- ERROR ---")
         print_status(f"File not found: {e}")
         if hasattr(print_status, 'log_file') and print_status.log_file is not None:
              with open(print_status.log_file, 'a') as f: f.write(f"\n--- ERROR ---\nFile not found: {e}\n")
         return 1
    except ValueError as e:
         print_status(f"\n--- ERROR ---")
         print_status(f"Input Error: {e}")
         if hasattr(print_status, 'log_file') and print_status.log_file is not None:
              with open(print_status.log_file, 'a') as f: f.write(f"\n--- ERROR ---\nInput Error: {e}\n")
         return 1
    except RuntimeError as e:
         print_status(f"\n--- ERROR ---")
         print_status(f"Runtime Error: {e}")
         if hasattr(print_status, 'log_file') and print_status.log_file is not None:
              with open(print_status.log_file, 'a') as f: f.write(f"\n--- ERROR ---\nRuntime Error: {e}\n")
         return 1
    except Exception as e: # Catch any other exceptions
        print_status(f"\n--- UNEXPECTED ERROR ---")
        print_status(f"An unexpected error occurred: {e}")
        print_status("Traceback:")
        print_status(traceback.format_exc())
        if hasattr(print_status, 'log_file') and print_status.log_file is not None:
             with open(print_status.log_file, 'a') as f:
                  f.write("\n--- UNEXPECTED ERROR ---\n")
                  f.write(f"Error: {e}\n")
                  f.write("Traceback:\n")
                  traceback.print_exc(file=f)
        return 1


if __name__ == "__main__":
    # Ensure RDKit uses multiple threads if available, controlled by environment variable or config
    # Set number of threads for RDKit descriptor calculations if applicable
    # Chem.SetNumThreads(multiprocessing.cpu_count()) # Use with caution, might conflict with ProcessPoolExecutor

    # Set Optuna logging level (optional)
    # optuna.logging.set_verbosity(optuna.logging.WARNING)

    sys.exit(main())