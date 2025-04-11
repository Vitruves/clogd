#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, cohen_kappa_score
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, rdFingerprintGenerator, Descriptors, Crippen, Lipinski
from tqdm import tqdm
import argparse
from torch.optim.lr_scheduler import ReduceLROnPlateau
import optuna
from optuna.trial import TrialState
import joblib
import time
import json
import sys
from scipy.stats import pearsonr, spearmanr
import multiprocessing
from functools import partial
import re
import gc
from rdkit import DataStructs

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else 
                     "mps" if torch.backends.mps.is_available() else 
                     "cpu")

class MolecularDataset(Dataset):
    def __init__(self, smiles_list, features, targets):
        self.smiles_list = smiles_list
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32).view(-1, 1)
        
    def __len__(self):
        return len(self.smiles_list)
    
    def __getitem__(self, idx):
        return {
            'smiles': self.smiles_list[idx],
            'features': self.features[idx],
            'target': self.targets[idx]
        }

def compute_rdkit_descriptors(mol):
    if mol is None:
        return [0] * 10
    
    descriptors = []
    descriptors.append(Descriptors.MolWt(mol))
    descriptors.append(Descriptors.MolLogP(mol))
    descriptors.append(Crippen.MolMR(mol))
    descriptors.append(Descriptors.NumHDonors(mol))
    descriptors.append(Descriptors.NumHAcceptors(mol))
    descriptors.append(Descriptors.TPSA(mol))
    descriptors.append(Lipinski.NumRotatableBonds(mol))
    descriptors.append(Descriptors.FractionCSP3(mol))
    descriptors.append(Descriptors.NumAromaticRings(mol))
    descriptors.append(Descriptors.NumAliphaticRings(mol))
    
    return descriptors

def _process_morgan_chunk(chunk_data, radius, nBits):
    smiles_chunk, chunk_idx = chunk_data
    fps = []
    
    for smiles in smiles_chunk:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                # Use the traditional Morgan fingerprint implementation
                fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
                # Convert to numpy array safely
                arr = np.zeros((1,))
                DataStructs.ConvertToNumpyArray(fp, arr)
                fps.append(arr)
            else:
                fps.append(np.zeros(nBits, dtype=np.int32))
        except Exception as e:
            print(f"-- Error processing SMILES: {smiles[:20]}... - {str(e)}")
            fps.append(np.zeros(nBits, dtype=np.int32))
    
    return (chunk_idx, fps)

def _process_maccs_chunk(chunk_data):
    smiles_chunk, chunk_idx = chunk_data
    fps = []
    
    for smiles in smiles_chunk:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = rdMolDescriptors.GetMACCSKeysFingerprint(mol)
            fps.append(np.array(fp))
        else:
            fps.append(np.zeros(167))
    
    return (chunk_idx, fps)

def _process_rdkit_chunk(chunk_data, nBits):
    smiles_chunk, chunk_idx = chunk_data
    fps = []
    
    for smiles in smiles_chunk:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = rdMolDescriptors.RDKFingerprint(mol, fpSize=nBits)
            fps.append(np.array(fp))
        else:
            fps.append(np.zeros(nBits))
    
    return (chunk_idx, fps)

def _process_descriptors_chunk(chunk_data):
    smiles_chunk, chunk_idx = chunk_data
    fps = []
    
    for smiles in smiles_chunk:
        mol = Chem.MolFromSmiles(smiles)
        fps.append(compute_rdkit_descriptors(mol))
    
    return (chunk_idx, fps)

def _process_atompair_chunk(chunk_data):
    smiles_chunk, chunk_idx = chunk_data
    fps = []
    fpgen = rdFingerprintGenerator.GetAtomPairGenerator()
    for smiles in smiles_chunk:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp_hashed = fpgen.GetHashedFingerprint(mol, nBits=2048)
            fps.append(np.array(fp_hashed))
        else:
            fps.append(np.zeros(2048))
    return (chunk_idx, fps)

def _process_torsion_chunk(chunk_data):
    smiles_chunk, chunk_idx = chunk_data
    fps = []
    fpgen = rdFingerprintGenerator.GetTopologicalTorsionGenerator()
    for smiles in smiles_chunk:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp_hashed = fpgen.GetHashedFingerprint(mol, nBits=2048)
            fps.append(np.array(fp_hashed))
        else:
            fps.append(np.zeros(2048))
    return (chunk_idx, fps)

def parse_fp_type(fp_type_str):
    match = re.match(r"morgan_(\d+)_(\d+)", fp_type_str, re.IGNORECASE)
    if match:
        return 'morgan', {'radius': int(match.group(1)), 'nBits': int(match.group(2))}
    
    match = re.match(r"rdkit_(\d+)", fp_type_str, re.IGNORECASE)
    if match:
        return 'rdkit', {'nBits': int(match.group(1))}
        
    if fp_type_str.lower() == 'morgan':
        return 'morgan', {'radius': 3, 'nBits': 2048}
    if fp_type_str.lower() == 'rdkit':
        return 'rdkit', {'nBits': 2048}
    if fp_type_str.lower() == 'maccs':
        return 'maccs', {}
    if fp_type_str.lower() == 'descriptors':
        return 'descriptors', {}
    if fp_type_str.lower() == 'atompair':
        return 'atompair', {'nBits': 2048}
    if fp_type_str.lower() == 'torsion':
        return 'torsion', {'nBits': 2048}

    print(f"-- Warning: Unrecognized fingerprint format '{fp_type_str}'. Skipping.")
    return None, None

def generate_fingerprints(smiles_list, fp_types, n_jobs=None, batch_size_fp=5000):
    if n_jobs is None:
        n_jobs = max(1, multiprocessing.cpu_count() - 1)
    else:
        n_jobs = min(n_jobs, multiprocessing.cpu_count())
    
    try_parallel = True
    if n_jobs == 1:
        try_parallel = False
        
    print(f"-- Using {'parallel' if try_parallel else 'sequential'} processing with {n_jobs if try_parallel else 1} worker(s)")
    
    total_smiles = len(smiles_list)
    num_batches = (total_smiles + batch_size_fp - 1) // batch_size_fp
    print(f"-- Processing {total_smiles} molecules in {num_batches} batches (batch size: {batch_size_fp})")
    
    processed_fp_types = []
    all_fps = []
    batch_results = []
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size_fp
        end_idx = min(start_idx + batch_size_fp, total_smiles)
        current_batch = smiles_list[start_idx:end_idx]
        
        batch_fps = []
        
        for fp_type_str in fp_types:
            fp_name, fp_params = parse_fp_type(fp_type_str)
            if fp_name is None:
                continue
            
            if batch_idx == 0:
                processed_fp_types.append({'name': fp_name, **fp_params})
            
            chunks = np.array_split(current_batch, min(n_jobs, len(current_batch)))
            chunk_data = [(chunk, i) for i, chunk in enumerate(chunks)]
            
            desc = f"Batch {batch_idx+1}/{num_batches}: {fp_name}"
            if fp_name == 'morgan':
                desc += f" (r={fp_params['radius']})"
                process_func = partial(_process_morgan_chunk, radius=fp_params['radius'], nBits=fp_params['nBits'])
            elif fp_name == 'rdkit':
                process_func = partial(_process_rdkit_chunk, nBits=fp_params['nBits'])
            elif fp_name == 'maccs':
                process_func = _process_maccs_chunk
            elif fp_name == 'descriptors':
                process_func = _process_descriptors_chunk
            elif fp_name == 'atompair':
                process_func = _process_atompair_chunk
            elif fp_name == 'torsion':
                process_func = _process_torsion_chunk
            else:
                print(f"-- Warning: Unknown fingerprint type '{fp_name}'")
                continue
            
            try:
                if try_parallel:
                    with multiprocessing.Pool(n_jobs) as pool:
                        results = list(tqdm(
                            pool.imap(process_func, chunk_data),
                            total=len(chunk_data),
                            desc=desc
                        ))
                else:
                    # Sequential fallback
                    results = []
                    for i, chunk in enumerate(tqdm(chunk_data, desc=f"{desc} (sequential)")):
                        results.append(process_func(chunk))
            except Exception as e:
                print(f"-- Error in parallel processing: {str(e)}")
                print("-- Falling back to sequential processing")
                try_parallel = False
                results = []
                for i, chunk in enumerate(tqdm(chunk_data, desc=f"{desc} (sequential fallback)")):
                    results.append(process_func(chunk))
            
            results.sort(key=lambda x: x[0])
            fp_list = []
            for _, chunk_fps in results:
                fp_list.extend(chunk_fps)
            
            batch_fps.append(np.array(fp_list))
        
        if batch_fps:
            batch_combined = np.hstack(batch_fps)
            batch_results.append(batch_combined)
        
        gc.collect()
    
    if batch_results:
        final_features = np.vstack(batch_results)
        print(f"-- Generated features shape: {final_features.shape}")
        return final_features, processed_fp_types
    else:
        print("-- No valid fingerprints generated.")
        return np.array([]), []

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout=0.2, activation_type='relu', 
                 batch_norm=True, residual_connections=False, layer_norm=False):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout
        self.activation_type = activation_type
        self.batch_norm = batch_norm
        self.residual_connections = residual_connections
        self.layer_norm = layer_norm
        
        layers = []
        prev_dim = input_dim
        
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.1),
            'silu': nn.SiLU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh()
        }
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            elif layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            
            if isinstance(activation_type, list):
                if i < len(activation_type):
                    layers.append(activations[activation_type[i]])
                else:
                    layers.append(activations['relu'])
            else:
                layers.append(activations[activation_type])
                
            layers.append(nn.Dropout(dropout))
            
            if residual_connections and i > 0 and hidden_dims[i-1] == hidden_dim:
                setattr(self, f'res_point_{i}', len(layers) - 3)
            
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.layers = nn.ModuleList(layers)
        
    def forward(self, x):
        if not self.residual_connections:
            for layer in self.layers:
                x = layer(x)
            return x
        else:
            residuals = {}
            for i, layer in enumerate(self.layers):
                if hasattr(self, f'res_point_{i//4}') and getattr(self, f'res_point_{i//4}') == i-3:
                    residuals[i//4] = x
                
                x = layer(x)
                
                if i > 0 and i % 4 == 3:
                    res_idx = i // 4
                    if res_idx in residuals:
                        x = x + residuals[res_idx]
            
            return x
            
    def __repr__(self):
        return (f"NeuralNetwork(input_dim={self.input_dim}, "
                f"hidden_dims={self.hidden_dims}, "
                f"dropout={self.dropout_rate}, "
                f"activation={self.activation_type}, "
                f"batch_norm={self.batch_norm}, "
                f"residual_connections={self.residual_connections}, "
                f"layer_norm={self.layer_norm})")

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, path='best_model.pt'):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_loss, model):
        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.save_checkpoint(model)
            self.counter = 0
            
    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)

def calculate_metrics(actual, predicted):
    metrics = {}
    
    mask = ~np.isnan(actual) & ~np.isnan(predicted)
    actual_clean = np.array(actual)[mask]
    predicted_clean = np.array(predicted)[mask]
    
    if len(actual_clean) == 0:
        return {"error": "No valid values for metrics calculation"}
    
    metrics["r2"] = r2_score(actual_clean, predicted_clean)
    metrics["rmse"] = np.sqrt(mean_squared_error(actual_clean, predicted_clean))
    metrics["mae"] = mean_absolute_error(actual_clean, predicted_clean)
    
    try:
        pearson_corr, pearson_p = pearsonr(actual_clean, predicted_clean)
        metrics["pearson_r"] = pearson_corr
        metrics["pearson_p"] = pearson_p
    except Exception:
        metrics["pearson_r"] = None
        metrics["pearson_p"] = None
    
    try:
        spearman_corr, spearman_p = spearmanr(actual_clean, predicted_clean)
        metrics["spearman_r"] = spearman_corr
        metrics["spearman_p"] = spearman_p
    except Exception:
        metrics["spearman_r"] = None
        metrics["spearman_p"] = None
    
    try:
        actual_bins = pd.qcut(actual_clean, 5, labels=False, duplicates='drop')
        predicted_bins = pd.qcut(predicted_clean, 5, labels=False, duplicates='drop')
        metrics["cohens_kappa"] = cohen_kappa_score(actual_bins, predicted_bins)
    except Exception:
        metrics["cohens_kappa"] = None
    
    return metrics

def load_model_with_metadata(model_dir):
    config_path = os.path.join(model_dir, 'model_config.json')
    if not os.path.exists(config_path):
        config_path = os.path.join(model_dir, 'best_config.json')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Model configuration file not found in {model_dir}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    model_path = os.path.join(model_dir, 'final_model.pt')
    if not os.path.exists(model_path):
        model_path = os.path.join(model_dir, 'best_model.pt')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights file not found in {model_dir}")
    
    model_metadata_path = os.path.join(model_dir, 'model_metadata.json')
    if os.path.exists(model_metadata_path):
        with open(model_metadata_path, 'r') as f:
            model_metadata = json.load(f)
    else:
        print("-- Warning: No model metadata found. Cannot reliably determine model input structure.")
        model_metadata = {
            'input_dim': None,
            'fingerprint_config': [{'name': 'morgan', 'radius': 3, 'nBits': 2048}],
            'use_input_descriptors': False,
            'descriptor_cols': [],
            'scaler_params': None,
            'scaler_mean': None,
            'scaler_scale': None,
            'imputer_params': None,
            'imputer_statistics': None,
        }
    
    input_dim = model_metadata.get('input_dim')
    if input_dim is None:
         if os.path.exists(model_metadata_path):
              raise ValueError("Could not determine input dimension from model metadata file.")
         else:
              raise ValueError("Model metadata file is missing. Cannot determine input dimension.")

    hidden_dims = config.get('hidden_dims')
    dropout = config.get('dropout', 0.2)
    activation = config.get('activation', 'relu')
    batch_norm = config.get('batch_norm', True)
    residual_connections = config.get('residual_connections', False)
    layer_norm = config.get('layer_norm', False)
    
    if hidden_dims is None:
         n_layers = config.get("n_layers")
         if n_layers is not None:
              hidden_dims = []
              for i in range(n_layers):
                  dim = config.get(f"hidden_dim_{i}")
                  if dim is not None:
                      hidden_dims.append(dim)
                  else:
                      raise ValueError(f"Missing 'hidden_dim_{i}' in configuration file '{config_path}' needed to reconstruct model.")
         else:
             raise ValueError(f"Could not find 'hidden_dims' or 'n_layers' in configuration file '{config_path}' to reconstruct model.")

    model = NeuralNetwork(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        activation_type=activation,
        batch_norm=batch_norm,
        residual_connections=residual_connections,
        layer_norm=layer_norm
    )
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    
    print(f"-- Model loaded from {model_dir}")
    print(f"-- Model architecture: {model}")
    print(f"-- Input dimension: {input_dim}")
    
    return model, model_metadata, config

def apply_progressive_unmasking(model, epoch, total_epochs, strategy='linear', freeze_layers=None):
    if strategy == 'none' or freeze_layers is None:
        return

    total_layers = len(model.layers)
    frozen_count = 0
    
    if strategy == 'linear':
        unfreeze_threshold = int((epoch / total_epochs) * total_layers)
        
        for i, layer in enumerate(model.layers):
            if i < unfreeze_threshold:
                for param in layer.parameters():
                    param.requires_grad = True
            else:
                for param in layer.parameters():
                    param.requires_grad = False
                    frozen_count += 1
    
    elif strategy == 'exponential':
        progress = epoch / total_epochs
        unfreeze_threshold = int((progress ** 0.5) * total_layers)
        
        for i, layer in enumerate(model.layers):
            if i < unfreeze_threshold:
                for param in layer.parameters():
                    param.requires_grad = True
            else:
                for param in layer.parameters():
                    param.requires_grad = False
                    frozen_count += 1
    
    elif strategy == 'scheduled':
        step_size = total_layers // 4
        unfreeze_count = min(total_layers, (epoch // (total_epochs // 4)) * step_size)
        
        for i, layer in enumerate(model.layers):
            if i < unfreeze_count:
                for param in layer.parameters():
                    param.requires_grad = True
            else:
                for param in layer.parameters():
                    param.requires_grad = False
                    frozen_count += 1
    
    elif strategy == 'custom':
        if isinstance(freeze_layers, list):
            for i, layer in enumerate(model.layers):
                if i in freeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
                        frozen_count += 1
                else:
                    for param in layer.parameters():
                        param.requires_grad = True
    
    print(f"-- Epoch {epoch+1}/{total_epochs}: {frozen_count} frozen parameters with strategy '{strategy}'")

def validate_datasets(model_metadata, train_df, test_df=None, smiles_col='SMILES', target_col='LOGD', new_target_col=None):
    required_descriptors = model_metadata.get('descriptor_cols', [])
    use_input_descriptors = model_metadata.get('use_input_descriptors', False)
    
    missing_descriptors = [col for col in required_descriptors if col not in train_df.columns]
    if missing_descriptors:
        raise ValueError(f"Required descriptor columns missing from training data: {missing_descriptors}")
    
    if new_target_col and new_target_col not in train_df.columns:
        raise ValueError(f"New target column '{new_target_col}' not found in training data")
    
    if smiles_col not in train_df.columns:
        potential_cols = [col for col in train_df.columns if 'smiles' in col.lower()]
        if potential_cols:
            print(f"-- SMILES column not found. Using {potential_cols[0]} instead")
        else:
            raise ValueError(f"SMILES column not found in training dataset")
    
    if test_df is not None:
        if smiles_col not in test_df.columns:
            potential_cols = [col for col in test_df.columns if 'smiles' in col.lower()]
            if potential_cols:
                print(f"-- SMILES column not found in test data. Using {potential_cols[0]} instead")
            else:
                raise ValueError(f"SMILES column not found in test dataset")
        
        missing_test_descriptors = [col for col in required_descriptors if col not in test_df.columns]
        if missing_test_descriptors:
            raise ValueError(f"Required descriptor columns missing from test data: {missing_test_descriptors}")
        
        if new_target_col and new_target_col not in test_df.columns:
            print(f"-- Warning: New target column '{new_target_col}' not found in test data.")
    
    print("-- Data validation successful")
    return True

def preprocess_data_for_finetuning(model_metadata, train_df, test_df=None, 
                                 smiles_col='SMILES', target_col=None, val_size=0.1,
                                 n_jobs=None, batch_size_fp=5000, new_target_col=None, 
                                 force_sequential=False, use_infile_fp=False, fp_prefixes=None):
    
    print(f"-- Preprocessing data for fine-tuning")
    
    use_input_descriptors = model_metadata.get('use_input_descriptors', False)
    input_dim = model_metadata.get('input_dim')
    fp_config = model_metadata.get('fingerprint_config', [])
    descriptor_cols = model_metadata.get('descriptor_cols', [])
    
    if not target_col and new_target_col:
        target_col = new_target_col
    elif not target_col:
        raise ValueError("No target column specified for fine-tuning")
    
    if target_col not in train_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")
    
    train_df = train_df.dropna(subset=[smiles_col, target_col])
    print(f"-- Dataset contains {len(train_df)} molecules after removing rows with missing data")
    
    # Always define fp_type_strings for potential use with test set
    fp_type_strings = []
    for fp in fp_config:
        if fp['name'] == 'morgan':
            fp_type_strings.append(f"morgan_{fp['radius']}_{fp['nBits']}")
        elif fp['name'] == 'rdkit':
            fp_type_strings.append(f"rdkit_{fp['nBits']}")
        else:
            fp_type_strings.append(fp['name'])
    
    train_fingerprints = np.array([])
    train_fp_columns = []
    
    # Use pre-computed fingerprints if requested
    if use_infile_fp and fp_prefixes:
        print(f"-- Using pre-computed fingerprints from input file with prefixes: {fp_prefixes}")
        for prefix in fp_prefixes:
            prefix_cols = [col for col in train_df.columns if col.startswith(prefix)]
            if prefix_cols:
                print(f"-- Found {len(prefix_cols)} columns with prefix '{prefix}'")
                train_fp_columns.extend(prefix_cols)
            else:
                print(f"-- Warning: No columns found with prefix '{prefix}'")
        
        if train_fp_columns:
            train_fingerprints = train_df[train_fp_columns].values
            print(f"-- Using {len(train_fp_columns)} fingerprint columns from input file")
        else:
            print(f"-- Warning: No fingerprint columns found with specified prefixes. Will generate fingerprints.")
    
    # If no pre-computed fingerprints were found or requested, generate them
    if train_fingerprints.size == 0:
        print(f"-- Generating fingerprints using: {fp_type_strings}")
        train_fingerprints, _ = generate_fingerprints(
            train_df[smiles_col].values, 
            fp_type_strings, 
            n_jobs=1 if force_sequential else n_jobs,
            batch_size_fp=batch_size_fp
        )
    
    train_feature_list = [train_fingerprints]
    
    if use_input_descriptors and descriptor_cols:
        print(f"-- Using {len(descriptor_cols)} input descriptors from dataset")
        descriptors = train_df[descriptor_cols].values.astype(float)
        
        if np.isnan(descriptors).any():
            print(f"-- Warning: NaNs found in input descriptors. Imputing with mean.")
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='mean')
            descriptors = imputer.fit_transform(descriptors)

        scaler_mean = model_metadata.get('scaler_mean')
        scaler_scale = model_metadata.get('scaler_scale')
        
        if scaler_mean and scaler_scale:
            scaled_descriptors = (descriptors - np.array(scaler_mean)) / np.array(scaler_scale)
            train_feature_list.append(scaled_descriptors)
        else:
            print("-- Warning: No scaler information found in model metadata. Using StandardScaler.")
            scaler = StandardScaler()
            scaled_descriptors = scaler.fit_transform(descriptors)
            train_feature_list.append(scaled_descriptors)
            
            model_metadata['scaler_mean'] = scaler.mean_.tolist()
            model_metadata['scaler_scale'] = scaler.scale_.tolist()
    
    train_features = np.hstack(train_feature_list)
    
    if train_features.shape[1] != input_dim:
        raise ValueError(f"Feature dimension mismatch. Model expects {input_dim}, generated {train_features.shape[1]}.")
    
    test_dataset = None
    
    if test_df is not None:
        test_df = test_df.dropna(subset=[smiles_col])
        has_targets = target_col in test_df.columns
        
        test_fingerprints = np.array([])
        
        # Try to use pre-computed fingerprints for test set too
        if use_infile_fp and fp_prefixes and train_fp_columns:
            print(f"-- Checking for pre-computed fingerprints in test set with prefixes: {fp_prefixes}")
            test_fp_columns = []
            for prefix in fp_prefixes:
                prefix_cols = [col for col in test_df.columns if col.startswith(prefix)]
                if prefix_cols:
                    print(f"-- Found {len(prefix_cols)} columns with prefix '{prefix}' in test set")
                    test_fp_columns.extend(prefix_cols)
                else:
                    print(f"-- Warning: No columns found with prefix '{prefix}' in test set")
            
            if test_fp_columns and len(test_fp_columns) == len(train_fp_columns):
                test_fingerprints = test_df[test_fp_columns].values
                print(f"-- Using {len(test_fp_columns)} fingerprint columns from test file")
            else:
                print(f"-- Warning: Test set fingerprints don't match training set. Will generate fingerprints.")
        
        # If no pre-computed fingerprints were found or requested, generate them
        if test_fingerprints.size == 0:
            print(f"-- Generating fingerprints for test set using: {fp_type_strings}")
            test_fingerprints, _ = generate_fingerprints(
                test_df[smiles_col].values, 
                fp_type_strings, 
                n_jobs=1 if force_sequential else n_jobs,
                batch_size_fp=batch_size_fp
            )
        
        test_feature_list = [test_fingerprints]
        
        if use_input_descriptors and descriptor_cols:
            available_descriptor_cols = [col for col in descriptor_cols if col in test_df.columns]
            if set(available_descriptor_cols) != set(descriptor_cols):
                missing_cols = set(descriptor_cols) - set(available_descriptor_cols)
                raise ValueError(f"Missing descriptor columns in test set: {missing_cols}")
            
            test_descriptors = test_df[descriptor_cols].values.astype(float)
            
            if np.isnan(test_descriptors).any():
                print(f"-- Warning: NaNs found in test descriptors. Imputing with training mean.")
                from sklearn.impute import SimpleImputer
                if 'imputer' in locals():
                    test_descriptors = imputer.transform(test_descriptors)
                else:
                    test_imputer = SimpleImputer(strategy='mean')
                    test_descriptors = test_imputer.fit_transform(test_descriptors)
            
            scaler_mean = model_metadata.get('scaler_mean')
            scaler_scale = model_metadata.get('scaler_scale')
            
            if scaler_mean and scaler_scale:
                test_scaled_descriptors = (test_descriptors - np.array(scaler_mean)) / np.array(scaler_scale)
                test_feature_list.append(test_scaled_descriptors)
            else:
                print("-- Warning: No scaler information found in model metadata.")
                if 'scaler' in locals():
                    test_scaled_descriptors = scaler.transform(test_descriptors)
                else:
                    test_scaler = StandardScaler()
                    test_scaled_descriptors = test_scaler.fit_transform(test_descriptors)
                test_feature_list.append(test_scaled_descriptors)
        
        test_features = np.hstack(test_feature_list)
        
        if test_features.shape[1] != input_dim:
            raise ValueError(f"Test feature dimension mismatch. Model expects {input_dim}, generated {test_features.shape[1]}.")
        
        test_targets = test_df[target_col].values if has_targets else np.zeros(len(test_df))
        test_dataset = MolecularDataset(test_df[smiles_col].values, test_features, test_targets)
    
    X_train, X_val, y_train, y_val, smiles_train, smiles_val = train_test_split(
        train_features, train_df[target_col].values, train_df[smiles_col].values, 
        test_size=val_size, random_state=SEED
    )
    
    train_dataset = MolecularDataset(smiles_train, X_train, y_train)
    val_dataset = MolecularDataset(smiles_val, X_val, y_val)
    
    print(f"-- Dataset split: {len(X_train)} train, {len(X_val)} validation" + 
          (f", {len(test_dataset)} test" if test_dataset else ""))
    
    return train_dataset, val_dataset, test_dataset

def finetune_model(model, train_loader, val_loader, criterion, optimizer, 
                  scheduler, device, epochs=100, patience=15, output_dir='model_output',
                  l1_lambda=0.0, gradient_clip=0.0, use_augmentation=False, augmentation_noise=0.01,
                  plot_dpi=150, plot_format='png', progressive_unmask=False, unmask_strategy='linear',
                  freeze_layers=None):
    
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'best_model.pt')
    early_stopping = EarlyStopping(patience=patience, path=model_path)
    
    train_losses = []
    val_losses = []
    val_r2_scores = []
    
    for epoch in range(epochs):
        if progressive_unmask:
            apply_progressive_unmasking(model, epoch, epochs, unmask_strategy, freeze_layers)
        
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            features = batch['features'].to(device)
            targets = batch['target'].to(device)
            
            if use_augmentation:
                noise = torch.randn_like(features) * augmentation_noise
                features = features + noise
                features = torch.clamp(features, 0, 1)
            
            outputs = model(features)
            mse_loss = criterion(outputs, targets)
            
            l1_reg = 0
            if l1_lambda > 0:
                for param in model.parameters():
                    if param.requires_grad:
                        l1_reg += torch.norm(param, 1)
                
            loss = mse_loss + l1_lambda * l1_reg
            
            optimizer.zero_grad()
            loss.backward()
            
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                
            optimizer.step()
            
            train_loss += mse_loss.item()
            
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                targets = batch['target'].to(device)
                
                outputs = model(features)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                val_preds.extend(outputs.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
                
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        val_r2 = r2_score(val_targets, val_preds)
        val_r2_scores.append(val_r2)
        
        scheduler.step(val_loss)
        
        print(f"-- Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val R²: {val_r2:.4f}")
        
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f"-- Early stopping at epoch {epoch+1}")
            break
    
    plt.figure(figsize=(12, 5))
    epochs_range = range(1, len(train_losses) + 1)
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs_range, val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, val_r2_scores, 'g-', label='Validation R²')
    plt.xlabel('Epochs')
    plt.ylabel('R²')
    plt.title('Validation R² Score')
    plt.legend()
    
    plt.tight_layout()
    plot_filename = os.path.join(output_dir, f'finetuning_curves.{plot_format}')
    plt.savefig(plot_filename, dpi=plot_dpi, format=plot_format)
    plt.close()
    
    model.load_state_dict(torch.load(model_path))
    
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'val_r2': val_r2_scores
    }
    
    return model, history, val_r2

def evaluate_model(model, test_loader, device, output_dir='model_output',
                  plot_dpi=150, plot_format='png', target_col_name='Target'):
    model.eval()
    test_preds = []
    test_targets = []
    test_smiles = []
    
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device)
            targets = batch['target'].to(device)
            smiles = batch['smiles']
            
            outputs = model(features)
            
            test_preds.extend(outputs.cpu().numpy())
            test_targets.extend(targets.cpu().numpy())
            test_smiles.extend(smiles)
    
    test_targets = np.array(test_targets).flatten()
    test_preds = np.array(test_preds).flatten()
    
    metrics = calculate_metrics(test_targets, test_preds)
    r2 = metrics.get("r2", float('nan'))
    rmse = metrics.get("rmse", float('nan'))
    mae = metrics.get("mae", float('nan'))
    
    print(f"\n-- Test Metrics:")
    print(f"   R²: {r2:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAE: {mae:.4f}")
    if "pearson_r" in metrics and metrics["pearson_r"] is not None:
        print(f"   Pearson r: {metrics['pearson_r']:.4f}")
    if "spearman_r" in metrics and metrics["spearman_r"] is not None:
        print(f"   Spearman r: {metrics['spearman_r']:.4f}")
    if "cohens_kappa" in metrics and metrics["cohens_kappa"] is not None:
        print(f"   Cohen's Kappa (5 bins): {metrics['cohens_kappa']:.4f}")

    results_df = pd.DataFrame({
        'SMILES': test_smiles,
        'Actual': test_targets,
        'Predicted': test_preds,
        'Error': np.abs(test_targets - test_preds)
    })
    
    results_csv_path = os.path.join(output_dir, 'test_predictions.csv')
    results_df.to_csv(results_csv_path, index=False)
    
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.scatter(test_targets, test_preds, alpha=0.5)
    
    min_val = min(np.min(test_targets), np.min(test_preds)) - 0.5
    max_val = max(np.max(test_targets), np.max(test_preds)) + 0.5
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')
    
    plt.xlabel(f'Actual {target_col_name}')
    plt.ylabel(f'Predicted {target_col_name}')
    plt.title(f'Predicted vs Actual (R²={r2:.3f})')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.legend()

    plt.subplot(1, 3, 2)
    residuals = test_targets - test_preds
    plt.scatter(test_preds, residuals, alpha=0.5)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel(f'Predicted {target_col_name}')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.title('Residual Plot')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.hist(np.abs(residuals), bins=30, alpha=0.7)
    plt.xlabel('Absolute Error')
    plt.ylabel('Frequency')
    plt.title(f'Error Distribution (MAE={mae:.3f})')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_filename = os.path.join(output_dir, f'test_evaluation_plots.{plot_format}')
    plt.savefig(plot_filename, dpi=plot_dpi, format=plot_format)
    print(f"-- Evaluation plots saved to {plot_filename}")
    plt.close()
    
    return r2, rmse, mae, results_df

def main():
    parser = argparse.ArgumentParser(description='Fine-tune Neural Network for Molecular Property Prediction')
    
    parser.add_argument('--model', type=str, required=True, 
                        help='Path to trained model directory containing model files')
    parser.add_argument('--input', type=str, required=True, 
                        help='Path to input CSV file for fine-tuning')
    parser.add_argument('--output', type=str, default='finetuned_model', 
                        help='Output directory for finetuned model')
    parser.add_argument('--external-test-set', type=str, 
                        help='Path to external test set CSV file')
    parser.add_argument('--smiles-col', type=str, default='SMILES', 
                        help='Column name for SMILES strings')
    parser.add_argument('--target-col', type=str, 
                        help='Column name for target values (use original if not specified)')
    parser.add_argument('--new-target-col', type=str, 
                        help='New target column name for fine-tuning (if different from original)')
    parser.add_argument('--epochs', type=int, default=50, 
                        help='Number of epochs')
    parser.add_argument('--patience', type=int, default=10, 
                        help='Patience for early stopping')
    parser.add_argument('--learning-rate', type=float, default=1e-4, 
                        help='Learning rate for fine-tuning')
    parser.add_argument('--batch-size', type=int, default=64, 
                        help='Batch size for training')
    parser.add_argument('--weight-decay', type=float, default=1e-6, 
                        help='Weight decay (L2 regularization)')
    parser.add_argument('--l1-lambda', type=float, default=0.0, 
                        help='L1 regularization coefficient')
    parser.add_argument('--gradient-clip', type=float, default=0.0, 
                        help='Gradient clipping norm (0 to disable)')
    parser.add_argument('--augmentation', action='store_true', 
                        help='Use data augmentation with random noise')
    parser.add_argument('--augmentation-noise', type=float, default=0.01, 
                        help='Standard deviation for data augmentation noise')
    parser.add_argument('--n-jobs', type=int, default=None, 
                        help='Number of parallel jobs (default: CPU count - 1)')
    parser.add_argument('--batch-size-fp', type=int, default=5000, 
                        help='Batch size for fingerprint computation')
    parser.add_argument('--plot-dpi', type=int, default=150, 
                        help='DPI for plots')
    parser.add_argument('--plot-format', type=str, default='png', 
                        choices=['png', 'pdf', 'svg', 'jpg'], 
                        help='Format for plots')
    parser.add_argument('--progressive-unmask', action='store_true', 
                        help='Use progressive unmasking during fine-tuning')
    parser.add_argument('--unmask-strategy', type=str, default='linear', 
                        choices=['linear', 'exponential', 'scheduled', 'custom', 'none'], 
                        help='Strategy for progressive unmasking')
    parser.add_argument('--freeze-layers', type=int, nargs='+', 
                        help='Layer indices to freeze in custom unmasking strategy')
    parser.add_argument('--force-sequential', action='store_true',
                       help='Force sequential fingerprint computation (no parallel processing)')
    parser.add_argument('--use-infile-fp', action='store_true',
                       help='Use pre-computed fingerprints from input file')
    parser.add_argument('--fp-prefixes', type=str, nargs='+',
                       help='Prefixes of fingerprint columns in input file')
    
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    try:
        model, model_metadata, config = load_model_with_metadata(args.model)
        print(f"-- Successfully loaded model from {args.model}")
        
        print(f"-- Loading data from {args.input}")
        train_df = pd.read_csv(args.input)
        
        test_df = None
        if args.external_test_set:
            print(f"-- Loading external test set from {args.external_test_set}")
            test_df = pd.read_csv(args.external_test_set)
        
        target_col = args.target_col
        if not target_col:
            orig_target = model_metadata.get('target_col', 'LOGD')
            if args.new_target_col:
                target_col = args.new_target_col
            else:
                target_col = orig_target
                print(f"-- Using original target column: {target_col}")
        
        validate_datasets(model_metadata, train_df, test_df, 
                         args.smiles_col, target_col, args.new_target_col)
        
        train_dataset, val_dataset, test_dataset = preprocess_data_for_finetuning(
            model_metadata, train_df, test_df,
            args.smiles_col, target_col, val_size=0.1,
            n_jobs=args.n_jobs, 
            batch_size_fp=args.batch_size_fp,
            new_target_col=args.new_target_col,
            force_sequential=args.force_sequential,
            use_infile_fp=args.use_infile_fp,
            fp_prefixes=args.fp_prefixes
        )
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
        
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=args.learning_rate, 
            weight_decay=args.weight_decay
        )
        criterion = nn.MSELoss()
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        print(f"\n-- Starting fine-tuning for {args.epochs} epochs")
        fine_tuned_model, history, val_r2 = finetune_model(
            model, train_loader, val_loader,
            criterion, optimizer, scheduler,
            device, epochs=args.epochs,
            patience=args.patience,
            output_dir=args.output,
            l1_lambda=args.l1_lambda,
            gradient_clip=args.gradient_clip,
            use_augmentation=args.augmentation,
            augmentation_noise=args.augmentation_noise,
            plot_dpi=args.plot_dpi,
            plot_format=args.plot_format,
            progressive_unmask=args.progressive_unmask,
            unmask_strategy=args.unmask_strategy,
            freeze_layers=args.freeze_layers
        )
        
        model_config = {
            'finetuned_from': args.model,
            'target_column': target_col,
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
            'weight_decay': args.weight_decay,
            'l1_lambda': args.l1_lambda,
            'gradient_clip': args.gradient_clip,
            'use_augmentation': args.augmentation,
            'augmentation_noise': args.augmentation_noise,
            'progressive_unmask': args.progressive_unmask,
            'unmask_strategy': args.unmask_strategy,
            'epochs': args.epochs,
            'val_r2': val_r2,
            'architecture': {
                'hidden_dims': model.hidden_dims,
                'dropout': model.dropout_rate,
                'activation': model.activation_type,
                'batch_norm': model.batch_norm,
                'residual_connections': model.residual_connections,
                'layer_norm': model.layer_norm
            }
        }
        
        with open(os.path.join(args.output, 'finetuned_config.json'), 'w') as f:
            json.dump(model_config, f, indent=4)
        
        model_metadata.update({
            'finetuned': True,
            'original_model': args.model,
            'target_col': target_col,
            'new_target_col': args.new_target_col
        })
        
        with open(os.path.join(args.output, 'model_metadata.json'), 'w') as f:
            json.dump(model_metadata, f, indent=4)
        
        torch.save(fine_tuned_model.state_dict(), os.path.join(args.output, 'final_model.pt'))
        print(f"-- Fine-tuned model saved to {args.output}")
        
        if test_dataset:
            print("\n-- Evaluating fine-tuned model on test set")
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
            r2, rmse, mae, results_df = evaluate_model(
                fine_tuned_model, test_loader, device,
                output_dir=args.output,
                plot_dpi=args.plot_dpi,
                plot_format=args.plot_format,
                target_col_name=target_col
            )
            
            model_config['test_metrics'] = {
                'r2': float(r2),
                'rmse': float(rmse),
                'mae': float(mae)
            }
            
            with open(os.path.join(args.output, 'finetuned_config.json'), 'w') as f:
                json.dump(model_config, f, indent=4)
            
            print("\n-- Most Accurate Predictions:")
            print(results_df.nsmallest(5, 'Error')[['SMILES', 'Actual', 'Predicted', 'Error']])
            
            print("\n-- Least Accurate Predictions:")
            print(results_df.nlargest(5, 'Error')[['SMILES', 'Actual', 'Predicted', 'Error']])
        
    except Exception as e:
        print(f"-- Error during fine-tuning: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    print(f"-- Using device: {device}")
    main()