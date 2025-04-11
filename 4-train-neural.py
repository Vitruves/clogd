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
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits)
    
    for smiles in smiles_chunk:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = fpgen.GetFingerprint(mol)
            fps.append(np.array(fp))
        else:
            fps.append(np.zeros(nBits))
    
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
            fp = fpgen.GetSparseFingerprint(mol)
            fp_counts = fpgen.GetCountFingerprint(mol).ToList()
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
    """Generate molecular fingerprints with memory-efficient batching"""
    
    if n_jobs is None:
        n_jobs = max(1, multiprocessing.cpu_count() - 1)
    else:
        n_jobs = min(n_jobs, multiprocessing.cpu_count())
    
    print(f"-- Using {n_jobs} cores for fingerprint computation")
    
    # Process SMILES in batches to manage memory usage
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
        
        # Process each fingerprint type for this batch
        for fp_type_str in fp_types:
            fp_name, fp_params = parse_fp_type(fp_type_str)
            if fp_name is None:
                continue
            
            # Only add to processed_fp_types on first batch
            if batch_idx == 0:
                processed_fp_types.append({'name': fp_name, **fp_params})
            
            # Create sub-batches for parallel processing
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
            
            # Process chunks in parallel
            with multiprocessing.Pool(n_jobs) as pool:
                results = list(tqdm(
                    pool.imap(process_func, chunk_data),
                    total=len(chunk_data),
                    desc=desc
                ))
            
            # Combine results from parallel processing
            results.sort(key=lambda x: x[0])
            fp_list = []
            for _, chunk_fps in results:
                fp_list.extend(chunk_fps)
            
            batch_fps.append(np.array(fp_list))
        
        # If we have fingerprints for this batch, combine them
        if batch_fps:
            # Horizontal stack of all fingerprint types for this batch
            batch_combined = np.hstack(batch_fps)
            batch_results.append(batch_combined)
        
        # Explicitly clean up to reduce memory usage
        gc.collect()
    
    # Combine all batches
    if batch_results:
        # Vertical stack of all batches
        final_features = np.vstack(batch_results)
        print(f"-- Generated features shape: {final_features.shape}")
        return final_features, processed_fp_types
    else:
        print("-- No valid fingerprints generated.")
        return np.array([]), []

def preprocess_data(data_path, target_col='LOGD', smiles_col='SMILES', test_size=0.1, val_size=0.1,
                    fingerprint_types=None, use_input_descriptors=False, external_test_file=None, n_jobs=None, 
                    batch_size_fp=5000, use_infile_fp=False, fp_prefixes=None):
    print(f"-- Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    if smiles_col not in df.columns:
        potential_cols = [col for col in df.columns if 'smiles' in col.lower()]
        if potential_cols:
            smiles_col = potential_cols[0]
            print(f"-- SMILES column not found. Using {smiles_col} instead")
        else:
            raise ValueError(f"SMILES column not found in dataset")
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")
        
    df = df.dropna(subset=[target_col, smiles_col])
    print(f"-- Dataset contains {len(df)} molecules after removing rows with missing data")
    
    fingerprints = np.array([])
    processed_fp_config = []
    
    # Use pre-computed fingerprints from input file if requested
    if use_infile_fp and fp_prefixes:
        print(f"-- Using pre-computed fingerprints from input file with prefixes: {fp_prefixes}")
        fp_columns = []
        for prefix in fp_prefixes:
            prefix_cols = [col for col in df.columns if col.startswith(prefix)]
            if prefix_cols:
                print(f"-- Found {len(prefix_cols)} columns with prefix '{prefix}'")
                fp_columns.extend(prefix_cols)
            else:
                print(f"-- Warning: No columns found with prefix '{prefix}'")
        
        if fp_columns:
            fingerprints = df[fp_columns].values
            print(f"-- Using {len(fp_columns)} fingerprint columns from input file")
            
            # Create metadata about fingerprint configuration
            for prefix in fp_prefixes:
                prefix_cols = [col for col in df.columns if col.startswith(prefix)]
                if prefix_cols:
                    # Try to determine fingerprint type from prefix
                    if 'morgan' in prefix.lower():
                        fp_type = 'morgan'
                        fp_params = {'radius': 3, 'nBits': len(prefix_cols)}
                    elif 'maccs' in prefix.lower():
                        fp_type = 'maccs'
                        fp_params = {}
                    elif 'rdkit' in prefix.lower():
                        fp_type = 'rdkit'
                        fp_params = {'nBits': len(prefix_cols)}
                    elif 'atompair' in prefix.lower():
                        fp_type = 'atompair'
                        fp_params = {'nBits': len(prefix_cols)}
                    elif 'torsion' in prefix.lower():
                        fp_type = 'torsion'
                        fp_params = {'nBits': len(prefix_cols)}
                    else:
                        fp_type = 'custom'
                        fp_params = {'nBits': len(prefix_cols)}
                    
                    processed_fp_config.append({'name': fp_type, 'prefix': prefix, **fp_params})
    # Generate fingerprints on the fly if not using pre-computed ones
    elif fingerprint_types is not None:
        print(f"-- Generating molecular fingerprints: {fingerprint_types}")
        fingerprints, processed_fp_config = generate_fingerprints(df[smiles_col].values, fingerprint_types, n_jobs=n_jobs, batch_size_fp=batch_size_fp)
    else:
        print("-- No fingerprint types specified and not using pre-computed fingerprints")
    
    if fingerprints.size == 0 and not use_input_descriptors:
        raise ValueError("No features could be generated (fingerprints or input descriptors).")

    feature_list = []
    if fingerprints.size > 0:
        feature_list.append(fingerprints)

    scaler = None
    descriptor_cols = []
    if use_input_descriptors:
        # Exclude fingerprint columns if using pre-computed fingerprints
        excluded_cols = [smiles_col, target_col]
        if use_infile_fp and fp_prefixes:
            for prefix in fp_prefixes:
                excluded_cols.extend([col for col in df.columns if col.startswith(prefix)])
        
        descriptor_cols = [col for col in df.columns if col not in excluded_cols]
        if descriptor_cols:
            print(f"-- Using {len(descriptor_cols)} input descriptors from dataset: {descriptor_cols}")
            descriptors = df[descriptor_cols].values.astype(float)
            
            if np.isnan(descriptors).any():
                print(f"-- Warning: NaNs found in input descriptors. Imputing with mean.")
                from sklearn.impute import SimpleImputer
                imputer = SimpleImputer(strategy='mean')
                descriptors = imputer.fit_transform(descriptors)

            scaler = StandardScaler()
            scaled_descriptors = scaler.fit_transform(descriptors)
            feature_list.append(scaled_descriptors)
        else:
            print("-- No additional descriptors found in dataset to use.")
    
    if not feature_list:
        raise ValueError("No features available for modeling.")
         
    all_features = np.hstack(feature_list)
    print(f"-- Final combined features shape: {all_features.shape}")

    if external_test_file:
        print(f"-- Loading external test set from {external_test_file}")
        test_df = pd.read_csv(external_test_file)
        
        if smiles_col not in test_df.columns:
            potential_cols = [col for col in test_df.columns if 'smiles' in col.lower()]
            if potential_cols:
                test_smiles_col = potential_cols[0]
                print(f"-- SMILES column not found in test set. Using {test_smiles_col} instead")
            else:
                raise ValueError(f"SMILES column not found in external test dataset")
        else:
            test_smiles_col = smiles_col
        
        has_targets = target_col in test_df.columns
        
        # Handle test set fingerprints
        test_fingerprints = np.array([])
        if use_infile_fp and fp_prefixes:
            # Use pre-computed fingerprints from test file
            fp_columns = []
            for prefix in fp_prefixes:
                prefix_cols = [col for col in test_df.columns if col.startswith(prefix)]
                if prefix_cols:
                    print(f"-- Found {len(prefix_cols)} fingerprint columns with prefix '{prefix}' in test set")
                    fp_columns.extend(prefix_cols)
                else:
                    print(f"-- Warning: No columns with prefix '{prefix}' found in test set")
            
            if fp_columns:
                test_fingerprints = test_df[fp_columns].values
                print(f"-- Using {len(fp_columns)} fingerprint columns from test file")
            else:
                print(f"-- Warning: No matching fingerprint columns found in test file")
        else:
            # Generate fingerprints for test set
            test_fingerprints, _ = generate_fingerprints(test_df[test_smiles_col].values, fingerprint_types, n_jobs=n_jobs, batch_size_fp=batch_size_fp)
        
        test_feature_list = []
        if test_fingerprints.size > 0:
            test_feature_list.append(test_fingerprints)

        if use_input_descriptors:
            # Get descriptor columns for test set, excluding fingerprint columns if needed
            excluded_test_cols = [test_smiles_col]
            if target_col in test_df.columns:
                excluded_test_cols.append(target_col)
            if use_infile_fp and fp_prefixes:
                for prefix in fp_prefixes:
                    excluded_test_cols.extend([col for col in test_df.columns if col.startswith(prefix)])
            
            test_descriptor_cols = [col for col in test_df.columns if col in descriptor_cols]
            if test_descriptor_cols:
                print(f"-- Using {len(test_descriptor_cols)} input descriptors from test dataset")
                test_descriptors_raw = test_df[test_descriptor_cols].values.astype(float)
                
                if np.isnan(test_descriptors_raw).any():
                    print(f"-- Warning: NaNs found in test descriptors. Imputing with mean (using training imputer if available).")
                    if 'imputer' in locals():
                         test_descriptors_imputed = imputer.transform(test_descriptors_raw)
                    else:
                         test_imputer = SimpleImputer(strategy='mean')
                         test_descriptors_imputed = test_imputer.fit_transform(test_descriptors_raw)
                else:
                    test_descriptors_imputed = test_descriptors_raw

                if scaler:
                    test_scaled_descriptors = scaler.transform(test_descriptors_imputed)
                    test_feature_list.append(test_scaled_descriptors)
                else:
                     print("-- Warning: Scaler not available from training phase, cannot scale test descriptors.")
            else:
                print("-- No matching descriptors found in test dataset.")
        
        if not test_feature_list:
            raise ValueError("No features available for the external test set.")
            
        test_features = np.hstack(test_feature_list)
        print(f"-- Final combined test features shape: {test_features.shape}")
        
        if test_features.shape[1] != all_features.shape[1]:
             raise ValueError(f"Feature dimension mismatch between training ({all_features.shape[1]}) and test ({test_features.shape[1]}) sets.")

        test_targets = test_df[target_col].values if has_targets else np.zeros(len(test_df))
        test_smiles = test_df[test_smiles_col].values
        
        X_train_val, y_train_val, smiles_train_val = all_features, df[target_col].values, df[smiles_col].values
        
        X_train, X_val, y_train, y_val, smiles_train, smiles_val = train_test_split(
            X_train_val, y_train_val, smiles_train_val, 
            test_size=val_size, random_state=SEED
        )
        
        X_test, y_test, smiles_test = test_features, test_targets, test_smiles
        
    else:
        X_train_val, X_test, y_train_val, y_test, smiles_train_val, smiles_test = train_test_split(
            all_features, df[target_col].values, df[smiles_col].values, 
            test_size=test_size, random_state=SEED
        )
        
        X_train, X_val, y_train, y_val, smiles_train, smiles_val = train_test_split(
            X_train_val, y_train_val, smiles_train_val, 
            test_size=val_size/(1-test_size), random_state=SEED
        )
    
    print(f"-- Dataset split: {len(X_train)} train, {len(X_val)} validation, {len(X_test)} test")
    
    train_dataset = MolecularDataset(smiles_train, X_train, y_train)
    val_dataset = MolecularDataset(smiles_val, X_val, y_val)
    test_dataset = MolecularDataset(smiles_test, X_test, y_test)
    
    target_mean = float(np.mean(y_train))
    target_std = float(np.std(y_train))
    print(f"-- Target statistics - Mean: {target_mean:.3f}, Std: {target_std:.3f}")
    
    model_metadata = {
        'fingerprint_config': processed_fp_config,
        'input_dim': all_features.shape[1],
        'use_input_descriptors': use_input_descriptors,
        'use_infile_fp': use_infile_fp,
        'fp_prefixes': fp_prefixes if fp_prefixes else [],
        'descriptor_cols': descriptor_cols,
        'target_mean': target_mean,
        'target_std': target_std,
        'scaler_state': scaler.get_params() if scaler else None,
        'scaler_mean': scaler.mean_.tolist() if scaler else None,
        'scaler_scale': scaler.scale_.tolist() if scaler else None,
    }
    
    return train_dataset, val_dataset, test_dataset, model_metadata

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
            
            # Store residual connection reference dimensions for forward pass
            if residual_connections and i > 0 and hidden_dims[i-1] == hidden_dim:
                setattr(self, f'res_point_{i}', len(layers) - 3)  # Store the position before activation
            
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.layers = nn.ModuleList(layers)
        
    def forward(self, x):
        if not self.residual_connections:
            # Simple sequential forward pass
            for layer in self.layers:
                x = layer(x)
            return x
        else:
            # Forward pass with residual connections where dimensions match
            residuals = {}
            for i, layer in enumerate(self.layers):
                # Check if we're at a position just after a potential residual start point
                if hasattr(self, f'res_point_{i//4}') and getattr(self, f'res_point_{i//4}') == i-3:
                    # Store the output after the linear layer for later use
                    residuals[i//4] = x
                
                # Apply the current layer
                x = layer(x)
                
                # Check if we're at a position to add a residual connection
                if i > 0 and i % 4 == 3:  # After dropout in each block
                    res_idx = i // 4
                    if res_idx in residuals:
                        x = x + residuals[res_idx]  # Add the residual connection
            
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

def train_model(model, train_loader, val_loader, criterion, optimizer, 
               scheduler, device, epochs=100, patience=15, output_dir='model_output',
               l1_lambda=0.0, gradient_clip=0.0, use_augmentation=False, augmentation_noise=0.01,
               plot_dpi=150, plot_format='png'):
    
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'best_model.pt')
    early_stopping = EarlyStopping(patience=patience, path=model_path)
    
    train_losses = []
    val_losses = []
    val_r2_scores = []
    
    for epoch in range(epochs):
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
    train_val_diff = np.array(train_losses) - np.array(val_losses)
    plt.plot(epochs_range, np.abs(train_val_diff), 'g-')
    plt.xlabel('Epochs')
    plt.ylabel('|Train Loss - Val Loss|')
    plt.title('Overfitting Gap')
    
    plt.tight_layout()
    plot_filename = os.path.join(output_dir, f'overfitting_analysis.{plot_format}')
    plt.savefig(plot_filename, dpi=plot_dpi, format=plot_format)
    print(f"-- Overfitting analysis plot saved to {plot_filename}")
    plt.close()
    
    model.load_state_dict(torch.load(model_path))
    
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'val_r2': val_r2_scores
    }
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(val_r2_scores, label='Validation R²')
    plt.xlabel('Epochs')
    plt.ylabel('R²')
    plt.title('Validation R² Score')
    plt.legend()
    
    plt.tight_layout()
    plot_filename = os.path.join(output_dir, f'training_curves.{plot_format}')
    plt.savefig(plot_filename, dpi=plot_dpi, format=plot_format)
    print(f"-- Training curves plot saved to {plot_filename}")
    plt.close()
    
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
    print(f"-- Test predictions saved to {results_csv_path}")
    
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

def create_model_and_train(trial, train_dataset, val_dataset, input_dim, output_dir, args):
    n_layers = trial.suggest_int("n_layers", 1, 5)
    hidden_dims = []
    
    for i in range(n_layers):
        hidden_dims.append(trial.suggest_int(f"hidden_dim_{i}", 32, 1024, log=True))
    
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    
    l1_lambda = trial.suggest_float("l1_lambda", 1e-8, 1e-3, log=True)
    
    gradient_clip = trial.suggest_float("gradient_clip", 0.0, 1.0)
    
    activation_options = ['relu', 'leaky_relu', 'silu', 'gelu', 'tanh']
    activation = trial.suggest_categorical("activation", activation_options)
    
    model = NeuralNetwork(
        input_dim=input_dim, 
        hidden_dims=hidden_dims,
        dropout=dropout,
        activation_type=activation
    ).to(device)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    patience = trial.suggest_int("patience", 5, 25)
    
    trial_output_dir = os.path.join(output_dir, f"trial_{trial.number}")
    os.makedirs(trial_output_dir, exist_ok=True)
    
    use_augmentation = trial.suggest_categorical("use_augmentation", [True, False])
    augmentation_noise = trial.suggest_float("augmentation_noise", 0.0, 0.1)
    
    model, history, val_r2 = train_model(
        model, train_loader, val_loader, 
        criterion, optimizer, scheduler, 
        device, epochs=args.epochs,
        patience=patience,
        output_dir=trial_output_dir,
        l1_lambda=l1_lambda,
        gradient_clip=gradient_clip,
        use_augmentation=use_augmentation,
        augmentation_noise=augmentation_noise,
        plot_dpi=args.plot_dpi,
        plot_format=args.plot_format
    )
    
    config = {
        'trial_number': trial.number,
        'n_layers': n_layers,
        'hidden_dims': hidden_dims,
        'dropout': dropout,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'batch_size': batch_size,
        'activation': activation,
        'val_r2': val_r2
    }
    
    with open(os.path.join(trial_output_dir, 'trial_config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    return val_r2

def predict_with_model(model, data_path, output_file, smiles_col, target_col=None, model_metadata=None, n_jobs=None,
                        plot_dpi=150, plot_format='png', batch_size=128, batch_size_fp=5000, 
                        use_infile_fp=False, fp_prefixes=None):
    df = pd.read_csv(data_path)
    
    if smiles_col not in df.columns:
        potential_cols = [col for col in df.columns if 'smiles' in col.lower()]
        if potential_cols:
            smiles_col = potential_cols[0]
            print(f"-- SMILES column not found. Using {smiles_col} instead")
        else:
            raise ValueError(f"SMILES column not found in dataset")
    
    has_targets = (target_col is not None) and (target_col in df.columns)
    
    print(f"-- Processing {len(df)} molecules from {data_path}")
    
    model.eval()
    
    if model_metadata is None:
        raise ValueError("Model metadata is required for prediction.")
        
    processed_fp_config = model_metadata.get('fingerprint_config', [{'name': 'morgan', 'radius': 3, 'nBits': 2048}])
    # Get metadata about input features
    use_infile_fp_meta = model_metadata.get('use_infile_fp', False)
    fp_prefixes_meta = model_metadata.get('fp_prefixes', [])
    fp_type_strings = [f"{fp['name']}_{fp['radius']}_{fp['nBits']}" if fp['name'] == 'morgan' else \
                       f"{fp['name']}_{fp['nBits']}" if fp['name'] == 'rdkit' else \
                       fp['name'] for fp in processed_fp_config]
    
    use_input_descriptors = model_metadata.get('use_input_descriptors', False)
    descriptor_cols = model_metadata.get('descriptor_cols', [])
    expected_input_dim = model_metadata.get('input_dim')

    # Determine whether to use pre-computed fingerprints or generate them
    fingerprints = np.array([])
    
    # Use pre-computed fingerprints if requested and prefixes provided
    if use_infile_fp and fp_prefixes:
        print(f"-- Using pre-computed fingerprints from input file with prefixes: {fp_prefixes}")
        fp_columns = []
        for prefix in fp_prefixes:
            prefix_cols = [col for col in df.columns if col.startswith(prefix)]
            if prefix_cols:
                print(f"-- Found {len(prefix_cols)} columns with prefix '{prefix}'")
                fp_columns.extend(prefix_cols)
            else:
                print(f"-- Warning: No columns found with prefix '{prefix}'")
        
        if fp_columns:
            fingerprints = df[fp_columns].values
            print(f"-- Using {len(fp_columns)} fingerprint columns from input file")
        else:
            print(f"-- Warning: No fingerprint columns found with specified prefixes. Will generate fingerprints.")
            print(f"-- Generating features using model's config: {fp_type_strings}")
            fingerprints, _ = generate_fingerprints(df[smiles_col].values, fp_type_strings, n_jobs=n_jobs, batch_size_fp=batch_size_fp)
    # Fall back to model's original settings
    elif use_infile_fp_meta and fp_prefixes_meta:
        print(f"-- Using pre-computed fingerprints based on model metadata with prefixes: {fp_prefixes_meta}")
        fp_columns = []
        for prefix in fp_prefixes_meta:
            prefix_cols = [col for col in df.columns if col.startswith(prefix)]
            if prefix_cols:
                print(f"-- Found {len(prefix_cols)} columns with prefix '{prefix}'")
                fp_columns.extend(prefix_cols)
            else:
                print(f"-- Warning: No columns found with prefix '{prefix}'")
        
        if fp_columns:
            fingerprints = df[fp_columns].values
            print(f"-- Using {len(fp_columns)} fingerprint columns from input file")
        else:
            print(f"-- Warning: No fingerprint columns found with model metadata prefixes. Will generate fingerprints.")
            print(f"-- Generating features using model's config: {fp_type_strings}")
            fingerprints, _ = generate_fingerprints(df[smiles_col].values, fp_type_strings, n_jobs=n_jobs, batch_size_fp=batch_size_fp)
    # Generate fingerprints if not using pre-computed ones
    else:
        print(f"-- Generating features using model's config: {fp_type_strings}")
        fingerprints, _ = generate_fingerprints(df[smiles_col].values, fp_type_strings, n_jobs=n_jobs, batch_size_fp=batch_size_fp)
    
    feature_list = []
    if fingerprints.size > 0:
        feature_list.append(fingerprints)

    if use_input_descriptors:
        # Exclude fingerprint columns if using pre-computed fingerprints
        excluded_cols = []
        if use_infile_fp and fp_prefixes:
            for prefix in fp_prefixes:
                excluded_cols.extend([col for col in df.columns if col.startswith(prefix)])
        elif use_infile_fp_meta and fp_prefixes_meta:
            for prefix in fp_prefixes_meta:
                excluded_cols.extend([col for col in df.columns if col.startswith(prefix)])
        
        # Get descriptor columns that are both in the model metadata and in the input file
        predict_descriptor_cols = [col for col in df.columns if col in descriptor_cols and col not in excluded_cols]
        if predict_descriptor_cols:
            if set(predict_descriptor_cols) != set(descriptor_cols):
                 print(f"-- Warning: Mismatch in descriptor columns. Expected: {descriptor_cols}, Found: {predict_descriptor_cols}")
            
            try:
                descriptors_raw = df[predict_descriptor_cols].values.astype(float)
            except KeyError as e:
                raise ValueError(f"Missing expected descriptor column in prediction input: {e}")

            print(f"-- Using {len(predict_descriptor_cols)} input descriptors from dataset")

            if np.isnan(descriptors_raw).any():
                print(f"-- Warning: NaNs found in prediction input descriptors. Imputing (using training stats if available).")
                scaler_mean = model_metadata.get('scaler_mean')
                if scaler_mean and len(scaler_mean) == descriptors_raw.shape[1]:
                    nan_mask = np.isnan(descriptors_raw)
                    for i, mean_val in enumerate(scaler_mean):
                        descriptors_raw[nan_mask[:, i], i] = mean_val
                else:
                     col_means = np.nanmean(descriptors_raw, axis=0)
                     inds = np.where(np.isnan(descriptors_raw))
                     descriptors_raw[inds] = np.take(col_means, inds[1])
            
            descriptors_imputed = descriptors_raw

            scaler_mean = model_metadata.get('scaler_mean')
            scaler_scale = model_metadata.get('scaler_scale')
            if scaler_mean and scaler_scale:
                 if len(scaler_mean) == descriptors_imputed.shape[1] and len(scaler_scale) == descriptors_imputed.shape[1]:
                     scaled_descriptors = (descriptors_imputed - np.array(scaler_mean)) / np.array(scaler_scale)
                     feature_list.append(scaled_descriptors)
                 else:
                     print("-- Warning: Scaler dimensions from metadata don't match descriptor count. Skipping scaling.")
            else:
                print("-- Warning: Scaler state not found in metadata. Cannot scale descriptors.")
        else:
             if descriptor_cols:
                 print("-- Warning: Expected input descriptors based on model metadata, but none found in prediction data.")

    if not feature_list:
        raise ValueError("No features generated for prediction.")

    features = np.hstack(feature_list)
    
    if expected_input_dim is not None and features.shape[1] != expected_input_dim:
        print(f"-- Error: Feature dimension mismatch. Model expects {expected_input_dim}, generated {features.shape[1]}.")
        print("-- This likely means the input data or fingerprint/descriptor settings differ from training.")
        raise ValueError("Feature dimension mismatch during prediction.")
    
    tensor_features = torch.tensor(features, dtype=torch.float32).to(device)
    
    predictions = []
    
    with torch.no_grad():
        for i in range(0, len(tensor_features), batch_size):
            batch = tensor_features[i:i+batch_size]
            batch_preds = model(batch).cpu().numpy().flatten()
            predictions.extend(batch_preds)
    
    df['Predicted'] = predictions
    
    if has_targets:
        actual_values = df[target_col].values
        predicted_values = np.array(predictions)
        
        metrics = calculate_metrics(actual_values, predicted_values)
        
        print("\n-- Performance Metrics on Prediction Set:")
        print(f"   R²: {metrics.get('r2', float('nan')):.4f}")
        print(f"   RMSE: {metrics.get('rmse', float('nan')):.4f}")
        print(f"   MAE: {metrics.get('mae', float('nan')):.4f}")
        if "pearson_r" in metrics and metrics["pearson_r"] is not None:
            print(f"   Pearson r: {metrics['pearson_r']:.4f}")
        if "spearman_r" in metrics and metrics["spearman_r"] is not None:
            print(f"   Spearman r: {metrics['spearman_r']:.4f}")
        
        df['Error'] = np.abs(df[target_col] - df['Predicted'])
        
        plt.figure(figsize=(18, 6))

        plt.subplot(1, 3, 1)
        plt.scatter(df[target_col], df['Predicted'], alpha=0.5)
        min_val = min(df[target_col].min(), df['Predicted'].min()) - 0.5
        max_val = max(df[target_col].max(), df['Predicted'].max()) + 0.5
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')
        plt.xlabel(f'Actual {target_col}')
        plt.ylabel(f'Predicted {target_col}')
        plt.title(f'Predictions (R²={metrics.get("r2", float("nan")):.3f})')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.legend()

        plt.subplot(1, 3, 2)
        residuals = df[target_col] - df['Predicted']
        plt.scatter(df['Predicted'], residuals, alpha=0.5)
        plt.axhline(0, color='r', linestyle='--')
        plt.xlabel(f'Predicted {target_col}')
        plt.ylabel('Residuals (Actual - Predicted)')
        plt.title('Residual Plot')
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 3, 3)
        plt.hist(df['Error'], bins=30, alpha=0.7)
        plt.xlabel('Absolute Error')
        plt.ylabel('Frequency')
        plt.title(f'Error Distribution (MAE={metrics.get("mae", float("nan")):.3f})')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        output_img = os.path.splitext(output_file)[0] + f'_evaluation_plots.{plot_format}'
        plt.savefig(output_img, dpi=plot_dpi, format=plot_format)
        print(f"-- Prediction evaluation plots saved to {output_img}")
        plt.close()
        
        print("\n-- Most Accurate Predictions:")
        print(df.nsmallest(5, 'Error')[[smiles_col, target_col, 'Predicted', 'Error']])
    
    df.to_csv(output_file, index=False)
    print(f"-- Predictions saved to {output_file}")
    
    return df

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
        # Attempt to load defaults, but this is risky and likely to fail prediction
        model_metadata = {
            'input_dim': None, # Cannot guess reliably
            'fingerprint_config': [{'name': 'morgan', 'radius': 3, 'nBits': 2048}], # Guess
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
         # If metadata existed but didn't have input_dim (shouldn't happen with new code)
         if os.path.exists(model_metadata_path):
              raise ValueError("Could not determine input dimension from model metadata file.")
         else: # If metadata file was missing entirely
              raise ValueError("Model metadata file is missing. Cannot determine input dimension.")

    # Get architecture params
    hidden_dims = config.get('hidden_dims')
    dropout = config.get('dropout', 0.2)
    activation = config.get('activation', 'relu')
    batch_norm = config.get('batch_norm', True)
    residual_connections = config.get('residual_connections', False)
    layer_norm = config.get('layer_norm', False)
    
    # Reconstruct hidden dims from best_config or model_config if necessary
    if hidden_dims is None:
        # Try reconstructing from optuna trial style params (often in best_config.json)
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
             # If neither hidden_dims nor n_layers are present
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
    model.eval()
    
    print(f"-- Model loaded from {model_dir}")
    print(f"-- Model architecture: {model}")
    print(f"-- Input dimension: {input_dim}")
    
    return model, model_metadata

def main():
    parser = argparse.ArgumentParser(description='Neural Network for Molecular Property Prediction')
    
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')
    subparsers.required = True
    
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument('--n-jobs', type=int, default=None, 
                              help='Number of parallel jobs for fingerprint generation (default: CPU count - 1)')
    common_parser.add_argument('--plot-dpi', type=int, default=150, 
                              help='Resolution (DPI) for saved plots')
    common_parser.add_argument('--plot-format', type=str, default='png', 
                              choices=['png', 'pdf', 'svg', 'jpg'], 
                              help='Format for saved plots')
    common_parser.add_argument('--batch-size', type=int, default=128,
                              help='Batch size for training and evaluation (default: 128)')
    common_parser.add_argument('--batch-size-fp', type=int, default=5000,
                              help='Batch size for fingerprint computation (default: 5000)')

    train_parser = subparsers.add_parser('train', help='Train a new model', parents=[common_parser])
    train_parser.add_argument('--input', type=str, required=True, help='Path to input CSV file')
    train_parser.add_argument('--output', type=str, default='nn_model', help='Output directory')
    train_parser.add_argument('--smiles-col', type=str, default='SMILES', 
                             help='Column name for SMILES strings')
    train_parser.add_argument('--target-col', type=str, default='LOGD', 
                             help='Column name for target values')
    train_parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    train_parser.add_argument('--patience', type=int, default=15, help='Patience for early stopping')
    train_parser.add_argument('--n-trials', type=int, default=25, help='Number of Optuna trials')
    train_parser.add_argument('--optimize', action='store_true', 
                             help='Run hyperparameter optimization')
    train_parser.add_argument('--no-optimization', dest='optimize', action='store_false', 
                             help='Skip hyperparameter optimization')
    train_parser.add_argument('--external-test-set', type=str, 
                             help='Path to external test set CSV file')
    train_parser.add_argument('--use-input-descriptors', action='store_true', 
                             help='Use all other non-target, non-smiles columns as input features (scaled, NaNs imputed with mean)')
    train_parser.add_argument('--use-infile-fp', action='store_true',
                             help='Use fingerprints already present in the input file instead of generating them')
    train_parser.add_argument('--fp-prefixes', type=str, nargs='+', default=[],
                             help='Column prefixes to identify fingerprint columns in the input file (e.g., morgan_fp_, maccs_)')
    train_parser.add_argument('--fingerprints', type=str, nargs='+', default=['morgan_3_2048'], 
                             help='Fingerprint types to use. Examples: morgan (default 3_2048), morgan_radius_nbits (e.g., morgan_4_4096), '
                                  'maccs, rdkit (default 2048), rdkit_nbits (e.g., rdkit_4096), descriptors, '
                                  'atompair (default hashed 2048), atompair_nbits (e.g., atompair_4096), '
                                  'torsion (default hashed 2048), torsion_nbits (e.g., torsion_4096)')
    
    # Architecture parameterization arguments
    train_parser.add_argument('--hidden-dims', type=int, nargs='+', 
                             help='List of hidden layer dimensions (e.g., 1024 512 256 128)')
    train_parser.add_argument('--dropout', type=float, default=0.2, 
                             help='Dropout rate (0-1) for regularization')
    train_parser.add_argument('--activation', type=str, default='relu', 
                             choices=['relu', 'leaky_relu', 'silu', 'gelu', 'tanh'], 
                             help='Activation function for hidden layers')
    train_parser.add_argument('--learning-rate', type=float, default=0.0001, 
                             help='Initial learning rate')
    train_parser.add_argument('--weight-decay', type=float, default=1e-5, 
                             help='Weight decay (L2 regularization)')
    train_parser.add_argument('--no-batch-norm', action='store_false', dest='batch_norm', 
                             help='Disable batch normalization')
    train_parser.add_argument('--residual', action='store_true', dest='residual_connections', 
                             help='Use residual connections where possible')
    train_parser.add_argument('--layer-norm', action='store_true', 
                             help='Use layer normalization instead of batch normalization')
    train_parser.add_argument('--l1-lambda', type=float, default=0.0, 
                             help='L1 regularization coefficient')
    train_parser.add_argument('--gradient-clip', type=float, default=0.0, 
                             help='Gradient clipping norm (0 to disable)')
    train_parser.add_argument('--augmentation', action='store_true', dest='use_augmentation', 
                             help='Use data augmentation with random noise')
    train_parser.add_argument('--augmentation-noise', type=float, default=0.01, 
                             help='Standard deviation for data augmentation noise')
    
    train_parser.set_defaults(optimize=True, batch_norm=True, residual_connections=False, 
                             layer_norm=False, use_augmentation=False)
    
    predict_parser = subparsers.add_parser('predict', help='Make predictions with a trained model', parents=[common_parser])
    predict_parser.add_argument('--model', type=str, required=True, 
                               help='Path to trained model directory (containing model_metadata.json and model weights)')
    predict_parser.add_argument('--input', type=str, required=True, 
                               help='Path to input CSV file for prediction')
    predict_parser.add_argument('--output', type=str, 
                               help='Output CSV file (default: input_predictions.csv)')
    predict_parser.add_argument('--smiles-col', type=str, default='SMILES', 
                               help='Column name for SMILES strings')
    predict_parser.add_argument('--target-col', type=str, 
                               help='Column name for actual values (if available)')
    predict_parser.add_argument('--use-infile-fp', action='store_true',
                               help='Use fingerprints already present in the input file instead of generating them')
    predict_parser.add_argument('--fp-prefixes', type=str, nargs='+', default=[],
                               help='Column prefixes to identify fingerprint columns in the input file (e.g., morgan_fp_, maccs_)')
    
    args = parser.parse_args()

    if args.mode == 'train':
        os.makedirs(args.output, exist_ok=True)
        
        train_dataset, val_dataset, test_dataset, model_metadata = preprocess_data(
            args.input, 
            target_col=args.target_col,
            smiles_col=args.smiles_col,
            fingerprint_types=args.fingerprints,
            use_input_descriptors=args.use_input_descriptors,
            external_test_file=args.external_test_set,
            n_jobs=args.n_jobs,
            batch_size_fp=args.batch_size_fp,
            use_infile_fp=args.use_infile_fp,
            fp_prefixes=args.fp_prefixes
        )
        
        input_dim = model_metadata['input_dim']
        
        with open(os.path.join(args.output, 'model_metadata.json'), 'w') as f:
            json.dump(model_metadata, f, indent=4)
        
        if args.optimize:
            print(f"\n-- Starting hyperparameter optimization with {args.n_trials} trials")
            
            study_name = "nn_optimization"
            study_db = os.path.join(args.output, "optuna_study.db")
            study = optuna.create_study(
                study_name=study_name,
                storage=f"sqlite:///{study_db}",
                direction="maximize",
                load_if_exists=True
            )
            
            optimization_start = time.time()
            
            # Pass args down to the objective function if needed (e.g., for plot settings)
            objective_func = partial(create_model_and_train, 
                                     train_dataset=train_dataset, 
                                     val_dataset=val_dataset, 
                                     input_dim=input_dim, 
                                     output_dir=args.output, 
                                     args=args) # Pass the full args namespace
            
            study.optimize(objective_func, n_trials=args.n_trials)
            
            optimization_time = time.time() - optimization_start
            
            print(f"\n-- Hyperparameter optimization completed in {optimization_time:.2f} seconds")
            print(f"-- Best trial: {study.best_trial.number}")
            print(f"-- Best validation R²: {study.best_value:.4f}")
            print("-- Best hyperparameters:")
            for key, value in study.best_params.items():
                print(f"   {key}: {value}")
            
            best_params = study.best_params
            hidden_dims = []
            for i in range(best_params["n_layers"]):
                hidden_dims.append(best_params[f"hidden_dim_{i}"])
            
            best_config = {
                'n_layers': best_params["n_layers"],
                'hidden_dims': hidden_dims,
                'dropout': best_params["dropout"],
                'learning_rate': best_params["learning_rate"],
                'weight_decay': best_params["weight_decay"],
                'batch_size': best_params["batch_size"],
                'activation': best_params["activation"],
                'batch_norm': True,
                'residual_connections': False,
                'layer_norm': False,
                'val_r2': study.best_value
            }
            
            with open(os.path.join(args.output, 'best_config.json'), 'w') as f:
                json.dump(best_config, f, indent=4)
            
            # Generate Optuna plots (handle potential errors)
            try:
                 fig1 = optuna.visualization.matplotlib.plot_optimization_history(study)
                 plot_filename = os.path.join(args.output, f'optimization_history.{args.plot_format}')
                 fig1.figure.savefig(plot_filename, dpi=args.plot_dpi, format=args.plot_format, bbox_inches='tight')
                 print(f"-- Optimization history plot saved to {plot_filename}")
                 plt.close(fig1.figure) # Close the figure associated with the axes object
            except Exception as e:
                 print(f"-- Warning: Could not generate or save optimization history plot: {e}")

            try:
                 fig2 = optuna.visualization.matplotlib.plot_param_importances(study)
                 plot_filename = os.path.join(args.output, f'param_importances.{args.plot_format}')
                 fig2.figure.savefig(plot_filename, dpi=args.plot_dpi, format=args.plot_format, bbox_inches='tight')
                 print(f"-- Parameter importances plot saved to {plot_filename}")
                 plt.close(fig2.figure) # Close the figure associated with the axes object
            except Exception as e:
                 print(f"-- Warning: Could not generate or save parameter importances plot: {e}")


            print("\n-- Training final model with best hyperparameters")
            
            best_model = NeuralNetwork(
                input_dim=input_dim, 
                hidden_dims=hidden_dims,
                dropout=best_params["dropout"],
                activation_type=best_params["activation"]
            ).to(device)
            
            train_loader = DataLoader(train_dataset, batch_size=best_params["batch_size"], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=best_params["batch_size"])
            test_loader = DataLoader(test_dataset, batch_size=best_params["batch_size"])
            
            optimizer = torch.optim.Adam(
                best_model.parameters(), 
                lr=best_params["learning_rate"], 
                weight_decay=best_params["weight_decay"]
            )
            criterion = nn.MSELoss()
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
            
            final_model_dir = os.path.join(args.output, "final_model")
            os.makedirs(final_model_dir, exist_ok=True)
            
            best_model, history, _ = train_model(
                best_model, train_loader, val_loader, 
                criterion, optimizer, scheduler, 
                device, epochs=args.epochs,
                patience=args.patience,
                output_dir=final_model_dir,
                l1_lambda=best_params.get("l1_lambda", 0.0), # Use .get for optional params
                gradient_clip=best_params.get("gradient_clip", 0.0),
                use_augmentation=best_params.get("use_augmentation", False),
                augmentation_noise=best_params.get("augmentation_noise", 0.01),
                plot_dpi=args.plot_dpi, # Pass plot args
                plot_format=args.plot_format
            )
            
        else:
            print("\n-- Skipping hyperparameter optimization, using provided settings")
            
            # Use provided hidden dims or default
            if args.hidden_dims:
                hidden_dims = args.hidden_dims
                print(f"-- Using provided hidden dimensions: {hidden_dims}")
            else:
                hidden_dims = [1024, 512, 256, 128, 64]
                print(f"-- Using default hidden dimensions: {hidden_dims}")
            
            # Create model with specified or default architecture
            model = NeuralNetwork(
                input_dim=input_dim, 
                hidden_dims=hidden_dims,
                dropout=args.dropout,
                activation_type=args.activation,
                batch_norm=args.batch_norm,
                residual_connections=args.residual_connections,
                layer_norm=args.layer_norm
            ).to(device)
            
            print("\n-- Model Architecture:")
            print(model)
            print(f"-- Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
            
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
            
            optimizer = torch.optim.Adam(
                model.parameters(), 
                lr=args.learning_rate, 
                weight_decay=args.weight_decay
            )
            criterion = nn.MSELoss()
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
            
            final_model_dir = os.path.join(args.output, "final_model")
            os.makedirs(final_model_dir, exist_ok=True)
            
            best_model, history, _ = train_model(
                model, train_loader, val_loader, 
                criterion, optimizer, scheduler, 
                device, epochs=args.epochs,
                patience=args.patience,
                output_dir=final_model_dir,
                l1_lambda=args.l1_lambda,
                gradient_clip=args.gradient_clip,
                use_augmentation=args.use_augmentation,
                augmentation_noise=args.augmentation_noise,
                plot_dpi=args.plot_dpi,
                plot_format=args.plot_format
            )
            
            best_config = {
                'hidden_dims': hidden_dims,
                'dropout': args.dropout,
                'learning_rate': args.learning_rate,
                'weight_decay': args.weight_decay,
                'batch_size': args.batch_size,
                'activation': args.activation,
                'batch_norm': args.batch_norm,
                'residual_connections': args.residual_connections,
                'layer_norm': args.layer_norm,
                'l1_lambda': args.l1_lambda,
                'gradient_clip': args.gradient_clip,
                'use_augmentation': args.use_augmentation,
                'augmentation_noise': args.augmentation_noise
            }
            
            with open(os.path.join(final_model_dir, 'model_config.json'), 'w') as f:
                json.dump(best_config, f, indent=4)
        
        print("\n-- Evaluating final model on test set")
        # Use consistent batch size for evaluation
        if args.optimize:
            evaluation_batch_size = best_params.get("batch_size", 128)
        else:
            evaluation_batch_size = args.batch_size
        test_loader = DataLoader(test_dataset, batch_size=evaluation_batch_size)
        r2, rmse, mae, results_df = evaluate_model(
            best_model, test_loader, device, 
            output_dir=final_model_dir, 
            plot_dpi=args.plot_dpi, # Pass plot args
            plot_format=args.plot_format,
            target_col_name=args.target_col # Pass target name for plots
        )
        
        torch.save(best_model.state_dict(), os.path.join(final_model_dir, 'final_model.pt'))
        
        best_config['test_metrics'] = {
            'r2': float(r2),
            'rmse': float(rmse),
            'mae': float(mae)
        }
        
        with open(os.path.join(final_model_dir, 'model_config.json'), 'w') as f:
            json.dump(best_config, f, indent=4)
        
        print(f"\n-- Training complete. Model and results saved to {args.output}")
        
        print("\n-- Most Accurate Predictions:")
        print(results_df.nsmallest(5, 'Error')[['SMILES', 'Actual', 'Predicted', 'Error']])
        
        print("\n-- Least Accurate Predictions:")
        print(results_df.nlargest(5, 'Error')[['SMILES', 'Actual', 'Predicted', 'Error']])
    
    elif args.mode == 'predict':
        try:
            model, model_metadata = load_model_with_metadata(args.model)
            
            if not args.output:
                base_name = os.path.splitext(args.input)[0]
                output_file = f"{base_name}_predictions.csv"
            else:
                output_file = args.output
            
            predict_with_model(
                model, args.input, output_file, 
                args.smiles_col, args.target_col, 
                model_metadata, 
                n_jobs=args.n_jobs,
                plot_dpi=args.plot_dpi,
                plot_format=args.plot_format,
                batch_size=args.batch_size,
                batch_size_fp=args.batch_size_fp,
                use_infile_fp=args.use_infile_fp,
                fp_prefixes=args.fp_prefixes
            )
            
        except Exception as e:
            print(f"-- Error during prediction: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

if __name__ == "__main__":
    print(f"-- Using device: {device}")
    main()