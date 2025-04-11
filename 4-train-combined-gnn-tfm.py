#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tqdm import tqdm
import argparse
import json
import sys
from transformers import AutoModel, AutoTokenizer
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, DataStructs
import gc
import multiprocessing
from functools import partial
import re
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else 
                     "mps" if torch.backends.mps.is_available() else 
                     "cpu")

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

class HybridMolecularDataset(Dataset):
    def __init__(self, smiles_list, fingerprints, tokenizer, max_length=128, targets=None):
        self.smiles_list = smiles_list
        self.fingerprints = torch.tensor(fingerprints, dtype=torch.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length
        if targets is not None:
            self.targets = torch.tensor(targets, dtype=torch.float32).view(-1, 1)
        else:
            self.targets = None
        
    def __len__(self):
        return len(self.smiles_list)
    
    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        encoding = self.tokenizer(
            smiles,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        item = {
            'smiles': smiles,
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'fingerprint': self.fingerprints[idx]
        }
        
        if self.targets is not None:
            item['target'] = self.targets[idx]
            
        return item

def _process_morgan_chunk(chunk_data, radius, nBits):
    smiles_chunk, chunk_idx = chunk_data
    fps = []
    
    for smiles in smiles_chunk:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
                arr = np.zeros((1,))
                DataStructs.ConvertToNumpyArray(fp, arr)
                fps.append(arr)
            else:
                fps.append(np.zeros(nBits, dtype=np.int32))
        except Exception as e:
            print(f"-- Error processing SMILES: {smiles[:20]}... - {str(e)}")
            fps.append(np.zeros(nBits, dtype=np.int32))
    
    return (chunk_idx, fps)

def generate_fingerprints(smiles_list, fp_config, n_jobs=None, batch_size_fp=5000):
    radius = fp_config.get('radius', 3)
    nBits = fp_config.get('nBits', 2048)
    
    if n_jobs is None:
        n_jobs = max(1, multiprocessing.cpu_count() - 1)
    else:
        n_jobs = min(n_jobs, multiprocessing.cpu_count())
    
    try_parallel = n_jobs > 1
    
    print(f"-- Using {'parallel' if try_parallel else 'sequential'} processing with {n_jobs if try_parallel else 1} worker(s)")
    
    total_smiles = len(smiles_list)
    num_batches = (total_smiles + batch_size_fp - 1) // batch_size_fp
    print(f"-- Processing {total_smiles} molecules in {num_batches} batches (batch size: {batch_size_fp})")
    
    batch_results = []
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size_fp
        end_idx = min(start_idx + batch_size_fp, total_smiles)
        current_batch = smiles_list[start_idx:end_idx]
        
        chunks = np.array_split(current_batch, min(n_jobs, len(current_batch)))
        chunk_data = [(chunk, i) for i, chunk in enumerate(chunks)]
        
        desc = f"Batch {batch_idx+1}/{num_batches}: Morgan (r={radius})"
        process_func = partial(_process_morgan_chunk, radius=radius, nBits=nBits)
        
        try:
            if try_parallel:
                with multiprocessing.Pool(n_jobs) as pool:
                    results = list(tqdm(
                        pool.imap(process_func, chunk_data),
                        total=len(chunk_data),
                        desc=desc
                    ))
            else:
                results = []
                for i, chunk in enumerate(tqdm(chunk_data, desc=f"{desc} (sequential)")):
                    results.append(process_func(chunk))
        except Exception as e:
            print(f"-- Error in parallel processing: {str(e)}")
            print("-- Falling back to sequential processing")
            results = []
            for i, chunk in enumerate(tqdm(chunk_data, desc=f"{desc} (sequential fallback)")):
                results.append(process_func(chunk))
        
        results.sort(key=lambda x: x[0])
        fp_list = []
        for _, chunk_fps in results:
            fp_list.extend(chunk_fps)
        
        batch_results.append(np.array(fp_list))
        
        gc.collect()
    
    if batch_results:
        final_features = np.vstack(batch_results)
        print(f"-- Generated fingerprints shape: {final_features.shape}")
        return final_features
    else:
        print("-- No valid fingerprints generated.")
        return np.array([])

def load_neural_model(model_dir):
    config_path = os.path.join(model_dir, 'model_config.json')
    if not os.path.exists(config_path):
        config_path = os.path.join(model_dir, 'finetuned_config.json')
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
        raise ValueError("Model metadata file not found. Cannot determine model input structure.")
    
    input_dim = model_metadata.get('input_dim')
    if input_dim is None:
        raise ValueError("Could not determine input dimension from model metadata file.")

    hidden_dims = config.get('hidden_dims')
    if hidden_dims is None:
        hidden_dims = config.get('architecture', {}).get('hidden_dims')
        if hidden_dims is None:
            n_layers = config.get("n_layers")
            if n_layers is not None:
                hidden_dims = []
                for i in range(n_layers):
                    dim = config.get(f"hidden_dim_{i}")
                    if dim is not None:
                        hidden_dims.append(dim)
                    else:
                        raise ValueError(f"Missing 'hidden_dim_{i}' in configuration file needed to reconstruct model.")
            else:
                raise ValueError("Could not find 'hidden_dims' in configuration file to reconstruct model.")

    dropout = config.get('dropout', 0.2)
    if dropout is None:
        dropout = config.get('architecture', {}).get('dropout', 0.2)
    
    activation = config.get('activation', 'relu')
    if activation is None:
        activation = config.get('architecture', {}).get('activation', 'relu')
    
    batch_norm = config.get('batch_norm', True)
    if batch_norm is None:
        batch_norm = config.get('architecture', {}).get('batch_norm', True)
    
    residual_connections = config.get('residual_connections', False)
    if residual_connections is None:
        residual_connections = config.get('architecture', {}).get('residual_connections', False)
    
    layer_norm = config.get('layer_norm', False)
    if layer_norm is None:
        layer_norm = config.get('architecture', {}).get('layer_norm', False)

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
    
    print(f"-- Neural model loaded from {model_dir}")
    print(f"-- Model architecture: {model}")
    print(f"-- Input dimension: {input_dim}")
    
    fp_config = {}
    for fp in model_metadata.get('fingerprint_config', []):
        if fp['name'] == 'morgan':
            fp_config = {'radius': fp['radius'], 'nBits': fp['nBits']}
            break
    
    return model, fp_config, model_metadata

def find_transformer_model_dir(base_dir):
    if os.path.exists(os.path.join(base_dir, 'final_model')):
        return os.path.join(base_dir, 'final_model')
    elif os.path.exists(os.path.join(base_dir, 'model.safetensors')) or os.path.exists(os.path.join(base_dir, 'pytorch_model.bin')):
        return base_dir
    else:
        for root, dirs, files in os.walk(base_dir):
            if 'model.safetensors' in files or 'pytorch_model.bin' in files:
                return root
            if 'final_model' in dirs:
                final_model_dir = os.path.join(root, 'final_model')
                if os.path.exists(os.path.join(final_model_dir, 'model.safetensors')) or os.path.exists(os.path.join(final_model_dir, 'pytorch_model.bin')):
                    return final_model_dir
        
        raise FileNotFoundError(f"Could not find transformer model files in {base_dir}")

class HybridModel(nn.Module):
    def __init__(self, transformer_model, neural_model, transformer_dim, hidden_dim=512, dropout=0.2, freeze_pretrained=False):
        super().__init__()
        
        self.transformer = transformer_model
        self.neural_model = neural_model
        
        if freeze_pretrained:
            for param in self.transformer.parameters():
                param.requires_grad = False
            for param in self.neural_model.parameters():
                param.requires_grad = False
        
        self.neural_encoder = nn.Sequential(*list(neural_model.layers)[:-1])
        
        neural_dim = neural_model.hidden_dims[-1] if neural_model.hidden_dims else neural_model.input_dim
        
        combined_dim = transformer_dim + neural_dim
        
        self.fc_layers = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, input_ids, attention_mask, fingerprint):
        transformer_output = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = transformer_output.last_hidden_state[:, 0, :]
        
        neural_embedding = fingerprint
        for layer in self.neural_encoder:
            neural_embedding = layer(neural_embedding)
        
        combined = torch.cat([pooled_output, neural_embedding], dim=1)
        
        output = self.fc_layers(combined)
        
        return output

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

def train_hybrid_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, 
                        epochs=100, patience=15, output_dir='hybrid_model'):
    
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'best_model.pt')
    early_stopping = EarlyStopping(patience=patience, path=model_path)
    
    train_losses = []
    val_losses = []
    val_r2_scores = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} (Training)"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            fingerprint = batch['fingerprint'].to(device)
            targets = batch['target'].to(device)
            
            outputs = model(input_ids, attention_mask, fingerprint)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} (Validation)"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                fingerprint = batch['fingerprint'].to(device)
                targets = batch['target'].to(device)
                
                outputs = model(input_ids, attention_mask, fingerprint)
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
    
    model.load_state_dict(torch.load(model_path))
    
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'val_r2': val_r2_scores
    }
    
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
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150)
    plt.close()
    
    return model, history

def evaluate_model(model, test_loader, device, output_dir='hybrid_model'):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    test_preds = []
    test_targets = []
    test_smiles = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            fingerprint = batch['fingerprint'].to(device)
            targets = batch['target'].to(device)
            smiles = batch['smiles']
            
            outputs = model(input_ids, attention_mask, fingerprint)
            
            test_preds.extend(outputs.cpu().numpy())
            test_targets.extend(targets.cpu().numpy())
            test_smiles.extend(smiles)
    
    test_targets = np.array(test_targets).flatten()
    test_preds = np.array(test_preds).flatten()
    
    r2 = r2_score(test_targets, test_preds)
    rmse = np.sqrt(mean_squared_error(test_targets, test_preds))
    mae = mean_absolute_error(test_targets, test_preds)
    
    results_df = pd.DataFrame({
        'SMILES': test_smiles,
        'Actual': test_targets,
        'Predicted': test_preds,
        'Error': np.abs(test_targets - test_preds)
    })
    
    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 3, 1)
    plt.scatter(test_targets, test_preds, alpha=0.5)
    min_val = min(np.min(test_targets), np.min(test_preds)) - 0.5
    max_val = max(np.max(test_targets), np.max(test_preds)) + 0.5
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')
    plt.xlabel('Actual logD')
    plt.ylabel('Predicted logD')
    plt.title(f'Predicted vs Actual (R²={r2:.3f})')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    residuals = test_targets - test_preds
    plt.scatter(test_preds, residuals, alpha=0.5)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel('Predicted logD')
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
    plt.savefig(os.path.join(output_dir, 'test_evaluation_plots.png'), dpi=150)
    plt.close()
    
    return r2, rmse, mae, results_df

def validate_inputs(args):
    """
    Validates all input files and options to ensure they have the necessary features
    before proceeding with model training.
    
    Args:
        args: Command line arguments
        
    Returns:
        bool: True if validation passes, False otherwise
    """
    print("-- Validating inputs")
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"-- Error: Input file {args.input} does not exist")
        return False
    
    # Check if external test set exists if provided
    if args.external_test_set and not os.path.exists(args.external_test_set):
        print(f"-- Error: External test set {args.external_test_set} does not exist")
        return False
    
    # Check if neural model directory exists
    if not os.path.exists(args.neural_model):
        print(f"-- Error: Neural model directory {args.neural_model} does not exist")
        return False
    
    # Check if transformer model directory exists
    if os.path.exists(args.transformer_model):
        # Look for model files
        found_transformer = False
        if os.path.exists(os.path.join(args.transformer_model, 'final_model')):
            found_transformer = True
        else:
            for root, dirs, files in os.walk(args.transformer_model):
                if 'model.safetensors' in files or 'pytorch_model.bin' in files:
                    found_transformer = True
                    break
                if 'final_model' in dirs:
                    final_model_dir = os.path.join(root, 'final_model')
                    if os.path.exists(os.path.join(final_model_dir, 'model.safetensors')) or os.path.exists(os.path.join(final_model_dir, 'pytorch_model.bin')):
                        found_transformer = True
                        break
        
        if not found_transformer:
            print(f"-- Error: Could not find transformer model files in {args.transformer_model}")
            return False
    else:
        # Check if it's a HuggingFace model
        try:
            from huggingface_hub import model_info
            try:
                model_info(args.transformer_model)
                print(f"-- Transformer model will be downloaded from HuggingFace: {args.transformer_model}")
            except Exception:
                print(f"-- Error: {args.transformer_model} is not a valid local path or HuggingFace model")
                return False
        except ImportError:
            print(f"-- Error: {args.transformer_model} is not a valid local path and huggingface_hub is not installed")
            return False
    
    # Read the main input file and check for required columns
    try:
        df = pd.read_csv(args.input)
        
        # Check if SMILES column exists or can be auto-detected
        smiles_col = args.smiles_col
        if smiles_col not in df.columns:
            potential_cols = [col for col in df.columns if 'smiles' in col.lower()]
            if potential_cols:
                smiles_col = potential_cols[0]
                print(f"-- SMILES column not found in input file. Will use {smiles_col} instead")
            else:
                print(f"-- Error: SMILES column not found in input file and could not be auto-detected")
                print(f"-- Available columns: {', '.join(df.columns)}")
                return False
        
        # Check if target column exists or can be auto-detected
        target_col = args.target_col
        if target_col not in df.columns:
            potential_cols = [col for col in df.columns if 'logd' in col.lower()]
            if potential_cols:
                target_col = potential_cols[0]
                print(f"-- Target column not found in input file. Will use {target_col} instead")
            else:
                print(f"-- Error: Target column not found in input file and could not be auto-detected")
                print(f"-- Available columns: {', '.join(df.columns)}")
                return False
        
        # Check for missing values
        missing_smiles = df[smiles_col].isna().sum()
        missing_targets = df[target_col].isna().sum()
        
        if missing_smiles > 0:
            print(f"-- Warning: Input file has {missing_smiles} rows with missing SMILES. These will be filtered.")
        
        if missing_targets > 0:
            print(f"-- Warning: Input file has {missing_targets} rows with missing target values. These will be filtered.")
        
        valid_rows = (~df[smiles_col].isna() & ~df[target_col].isna()).sum()
        if valid_rows == 0:
            print(f"-- Error: Input file has no valid rows with both SMILES and target values")
            return False
        
        print(f"-- Input file has {valid_rows} valid rows with both SMILES and target values")
        
        # Check for pre-computed fingerprints if requested
        if args.use_infile_fp and args.fp_prefixes:
            fp_columns = []
            for prefix in args.fp_prefixes:
                prefix_cols = [col for col in df.columns if col.startswith(prefix)]
                if prefix_cols:
                    print(f"-- Found {len(prefix_cols)} columns with prefix '{prefix}' in input file")
                    fp_columns.extend(prefix_cols)
                else:
                    print(f"-- Warning: No columns found with prefix '{prefix}' in input file")
            
            if not fp_columns:
                print(f"-- Error: No fingerprint columns found with specified prefixes in input file")
                print(f"-- Available columns: {', '.join(df.columns)}")
                print(f"-- Will need to generate fingerprints instead")
        
        # Validate external test set if provided
        if args.external_test_set:
            ext_df = pd.read_csv(args.external_test_set)
            
            # Check if SMILES column exists in external test set
            if smiles_col not in ext_df.columns:
                potential_cols = [col for col in ext_df.columns if 'smiles' in col.lower()]
                if potential_cols:
                    ext_smiles_col = potential_cols[0]
                    print(f"-- SMILES column not found in external test set. Will use {ext_smiles_col} instead")
                else:
                    print(f"-- Error: SMILES column not found in external test set and could not be auto-detected")
                    print(f"-- Available columns: {', '.join(ext_df.columns)}")
                    return False
            
            # Check if target column exists in external test set (optional)
            if target_col not in ext_df.columns:
                potential_cols = [col for col in ext_df.columns if 'logd' in col.lower()]
                if potential_cols:
                    ext_target_col = potential_cols[0]
                    print(f"-- Target column not found in external test set. Will use {ext_target_col} instead")
                else:
                    print(f"-- Warning: Target column not found in external test set. Only predictions will be generated.")
            
            # Check for missing SMILES in external test set
            ext_missing_smiles = ext_df[smiles_col if smiles_col in ext_df.columns else ext_smiles_col].isna().sum()
            if ext_missing_smiles > 0:
                print(f"-- Warning: External test set has {ext_missing_smiles} rows with missing SMILES. These will be filtered.")
            
            ext_valid_rows = (~ext_df[smiles_col if smiles_col in ext_df.columns else ext_smiles_col].isna()).sum()
            if ext_valid_rows == 0:
                print(f"-- Error: External test set has no valid rows with SMILES")
                return False
            
            print(f"-- External test set has {ext_valid_rows} valid rows with SMILES")
            
            # Check for pre-computed fingerprints in external test set if requested
            if args.use_infile_fp and args.fp_prefixes:
                ext_fp_columns = []
                for prefix in args.fp_prefixes:
                    prefix_cols = [col for col in ext_df.columns if col.startswith(prefix)]
                    if prefix_cols:
                        print(f"-- Found {len(prefix_cols)} columns with prefix '{prefix}' in external test set")
                        ext_fp_columns.extend(prefix_cols)
                    else:
                        print(f"-- Warning: No columns found with prefix '{prefix}' in external test set")
                
                if not ext_fp_columns:
                    print(f"-- Warning: No fingerprint columns found with specified prefixes in external test set")
                    print(f"-- Will need to generate fingerprints for external test set")
    
    except Exception as e:
        print(f"-- Error validating input files: {str(e)}")
        return False
    
    print("-- Input validation successful")
    return True

def main():
    parser = argparse.ArgumentParser(description='Hybrid Transformer + Neural Model for logD Prediction')
    
    parser.add_argument('--transformer-model', type=str, required=True, 
                        help='Path to pre-trained transformer model directory (containing final_model/ or model files)')
    parser.add_argument('--neural-model', type=str, required=True, 
                        help='Path to pre-trained neural model directory (containing best_model.pt)')
    parser.add_argument('--input', type=str, required=True, 
                        help='Path to input CSV file')
    parser.add_argument('--output', type=str, default='hybrid_model_output', 
                        help='Output directory for model')
    parser.add_argument('--smiles-col', type=str, default='SMILES', 
                        help='Column name for SMILES strings')
    parser.add_argument('--target-col', type=str, default='LOGD', 
                        help='Column name for target values')
    parser.add_argument('--epochs', type=int, default=50, 
                        help='Number of epochs')
    parser.add_argument('--patience', type=int, default=10, 
                        help='Patience for early stopping')
    parser.add_argument('--learning-rate', type=float, default=1e-4, 
                        help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=32, 
                        help='Batch size for training')
    parser.add_argument('--max-length', type=int, default=128, 
                        help='Maximum length for SMILES tokenization')
    parser.add_argument('--n-jobs', type=int, default=None, 
                        help='Number of parallel jobs for fingerprint calculation')
    parser.add_argument('--hidden-dim', type=int, default=512, 
                        help='Hidden dimension size for combined model')
    parser.add_argument('--dropout', type=float, default=0.2, 
                        help='Dropout rate for combined model')
    parser.add_argument('--freeze-pretrained', action='store_true',
                        help='Freeze pre-trained models during training')
    parser.add_argument('--use-infile-fp', action='store_true',
                        help='Use pre-computed fingerprints from input file')
    parser.add_argument('--fp-prefixes', type=str, nargs='+',
                        help='Prefixes of fingerprint columns in input file')
    parser.add_argument('--external-test-set', type=str,
                        help='Path to external test set CSV file for additional evaluation')
    
    args = parser.parse_args()
    
    # Validate all inputs before proceeding
    if not validate_inputs(args):
        print("-- Input validation failed. Exiting.")
        sys.exit(1)
    
    os.makedirs(args.output, exist_ok=True)
    
    print(f"-- Loading data from {args.input}")
    df = pd.read_csv(args.input)
    
    if args.smiles_col not in df.columns:
        potential_cols = [col for col in df.columns if 'smiles' in col.lower()]
        if potential_cols:
            args.smiles_col = potential_cols[0]
            print(f"-- SMILES column not found. Using {args.smiles_col} instead")
        else:
            raise ValueError(f"SMILES column not found in dataset")
    
    if args.target_col not in df.columns:
        potential_cols = [col for col in df.columns if 'logd' in col.lower()]
        if potential_cols:
            args.target_col = potential_cols[0]
            print(f"-- Target column not found. Using {args.target_col} instead")
        else:
            raise ValueError(f"Target column not found in dataset")
    
    df = df.dropna(subset=[args.smiles_col, args.target_col])
    print(f"-- Dataset contains {len(df)} molecules after removing rows with missing data")
    
    # Load neural model
    print(f"-- Loading neural model from {args.neural_model}")
    neural_model, fp_config, model_metadata = load_neural_model(args.neural_model)
    
    # Load transformer model and tokenizer
    print(f"-- Loading transformer model")
    try:
        # Try to find the actual model directory
        transformer_dir = find_transformer_model_dir(args.transformer_model)
        print(f"-- Found transformer model in {transformer_dir}")
        transformer_model = AutoModel.from_pretrained(transformer_dir)
        tokenizer = AutoTokenizer.from_pretrained(transformer_dir)
    except Exception as e:
        print(f"-- Error loading transformer model from directory: {str(e)}")
        print(f"-- Attempting to load from HuggingFace model hub")
        transformer_model = AutoModel.from_pretrained(args.transformer_model)
        tokenizer = AutoTokenizer.from_pretrained(args.transformer_model)
    
    transformer_model.to(device)
    transformer_dim = transformer_model.config.hidden_size
    
    # Generate or load fingerprints for neural model
    fingerprints = None
    
    if args.use_infile_fp and args.fp_prefixes:
        print(f"-- Using pre-computed fingerprints from input file with prefixes: {args.fp_prefixes}")
        fp_columns = []
        
        for prefix in args.fp_prefixes:
            prefix_cols = [col for col in df.columns if col.startswith(prefix)]
            if prefix_cols:
                print(f"-- Found {len(prefix_cols)} columns with prefix '{prefix}'")
                fp_columns.extend(prefix_cols)
            else:
                print(f"-- Warning: No columns found with prefix '{prefix}'")
        
        if fp_columns:
            fingerprints = df[fp_columns].values
            print(f"-- Using {len(fp_columns)} fingerprint columns from input file")
            
            input_dim = model_metadata.get('input_dim')
            if fingerprints.shape[1] != input_dim:
                print(f"-- Warning: Input fingerprint dimension ({fingerprints.shape[1]}) doesn't match model input dimension ({input_dim})")
                if fingerprints.shape[1] > input_dim:
                    print(f"-- Truncating fingerprints to match model input dimension")
                    fingerprints = fingerprints[:, :input_dim]
                else:
                    print(f"-- Fingerprint dimension is smaller than model expects. This may cause issues.")
            
        else:
            print(f"-- Warning: No fingerprint columns found with specified prefixes. Will generate fingerprints.")
    
    if fingerprints is None:
        print(f"-- Generating fingerprints using neural model configuration")
        fingerprints = generate_fingerprints(
            df[args.smiles_col].values, 
            fp_config=fp_config,
            n_jobs=args.n_jobs
        )
    
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=SEED)
    train_df, val_df = train_test_split(train_df, test_size=0.125, random_state=SEED)
    
    print(f"-- Dataset split: {len(train_df)} train, {len(val_df)} validation, {len(test_df)} test")
    
    train_dataset = HybridMolecularDataset(
        train_df[args.smiles_col].values,
        fingerprints[train_df.index],
        tokenizer,
        max_length=args.max_length,
        targets=train_df[args.target_col].values
    )
    
    val_dataset = HybridMolecularDataset(
        val_df[args.smiles_col].values,
        fingerprints[val_df.index],
        tokenizer,
        max_length=args.max_length,
        targets=val_df[args.target_col].values
    )
    
    test_dataset = HybridMolecularDataset(
        test_df[args.smiles_col].values,
        fingerprints[test_df.index],
        tokenizer,
        max_length=args.max_length,
        targets=test_df[args.target_col].values
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    print(f"-- Creating hybrid model")
    hybrid_model = HybridModel(
        transformer_model=transformer_model,
        neural_model=neural_model,
        transformer_dim=transformer_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        freeze_pretrained=args.freeze_pretrained
    )
    hybrid_model.to(device)
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in hybrid_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in hybrid_model.parameters())
    print(f"-- Model has {total_params:,} total parameters, {trainable_params:,} trainable")
    
    optimizer = torch.optim.AdamW(
        [p for p in hybrid_model.parameters() if p.requires_grad], 
        lr=args.learning_rate
    )
    criterion = nn.MSELoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    print(f"-- Starting training for {args.epochs} epochs")
    hybrid_model, history = train_hybrid_model(
        hybrid_model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        epochs=args.epochs,
        patience=args.patience,
        output_dir=args.output
    )
    
    print("-- Evaluating model on test set")
    r2, rmse, mae, results_df = evaluate_model(hybrid_model, test_loader, device, output_dir=args.output)
    
    print(f"\n-- Test Metrics:")
    print(f"   R²: {r2:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAE: {mae:.4f}")
    
    # Save test predictions
    results_csv_path = os.path.join(args.output, 'test_predictions.csv')
    results_df.to_csv(results_csv_path, index=False)
    
    # Create model configuration dictionary here once
    model_config = {
        'transformer_model': args.transformer_model,
        'neural_model': args.neural_model,
        'hidden_dim': args.hidden_dim,
        'dropout': args.dropout,
        'max_length': args.max_length,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'freeze_pretrained': args.freeze_pretrained,
        'use_infile_fp': args.use_infile_fp,
        'fp_prefixes': args.fp_prefixes if args.fp_prefixes else [],
        'test_metrics': {
            'r2': float(r2),
            'rmse': float(rmse),
            'mae': float(mae)
        }
    }
    
    # Save config initially
    with open(os.path.join(args.output, 'model_config.json'), 'w') as f:
        json.dump(model_config, f, indent=4)
    
    # Evaluate on external test set if provided
    if args.external_test_set:
        print(f"\n-- Evaluating model on external test set: {args.external_test_set}")
        try:
            ext_df = pd.read_csv(args.external_test_set)
            
            if args.smiles_col not in ext_df.columns:
                potential_cols = [col for col in ext_df.columns if 'smiles' in col.lower()]
                if potential_cols:
                    ext_smiles_col = potential_cols[0]
                    print(f"-- SMILES column not found in external test set. Using {ext_smiles_col} instead")
                else:
                    raise ValueError(f"SMILES column not found in external test set")
            else:
                ext_smiles_col = args.smiles_col
                
            if args.target_col not in ext_df.columns:
                potential_cols = [col for col in ext_df.columns if 'logd' in col.lower()]
                if potential_cols:
                    ext_target_col = potential_cols[0]
                    print(f"-- Target column not found in external test set. Using {ext_target_col} instead")
                else:
                    print(f"-- Warning: Target column not found in external test set. Only predictions will be generated.")
                    ext_target_col = None
            else:
                ext_target_col = args.target_col
            
            # Check for and filter out missing SMILES
            ext_df = ext_df.dropna(subset=[ext_smiles_col])
            if ext_target_col:
                has_targets = True
                valid_indices = ~ext_df[ext_target_col].isna()
                if (~valid_indices).any():
                    print(f"-- Warning: {(~valid_indices).sum()} rows with missing target values in external test set")
                ext_df = ext_df[valid_indices]
                ext_targets = ext_df[ext_target_col].values
            else:
                has_targets = False
                ext_targets = np.zeros(len(ext_df))
                
            print(f"-- External test set contains {len(ext_df)} molecules after filtering")
            
            # Generate or load fingerprints for external test set
            ext_fingerprints = None
            
            if args.use_infile_fp and args.fp_prefixes:
                print(f"-- Looking for pre-computed fingerprints in external test set")
                ext_fp_columns = []
                
                for prefix in args.fp_prefixes:
                    prefix_cols = [col for col in ext_df.columns if col.startswith(prefix)]
                    if prefix_cols:
                        print(f"-- Found {len(prefix_cols)} columns with prefix '{prefix}' in external test set")
                        ext_fp_columns.extend(prefix_cols)
                    else:
                        print(f"-- Warning: No columns found with prefix '{prefix}' in external test set")
                
                if ext_fp_columns:
                    ext_fingerprints = ext_df[ext_fp_columns].values
                    print(f"-- Using {len(ext_fp_columns)} fingerprint columns from external test set")
                    
                    input_dim = model_metadata.get('input_dim')
                    if ext_fingerprints.shape[1] != input_dim:
                        print(f"-- Warning: External fingerprint dimension ({ext_fingerprints.shape[1]}) doesn't match model input dimension ({input_dim})")
                        if ext_fingerprints.shape[1] > input_dim:
                            print(f"-- Truncating fingerprints to match model input dimension")
                            ext_fingerprints = ext_fingerprints[:, :input_dim]
                        else:
                            print(f"-- Fingerprint dimension is smaller than model expects. This may cause issues.")
                else:
                    print(f"-- Warning: External test set has no valid fingerprint columns. Will generate fingerprints.")
            
            if ext_fingerprints is None:
                print(f"-- Generating fingerprints for external test set")
                ext_fingerprints = generate_fingerprints(
                    ext_df[ext_smiles_col].values, 
                    fp_config=fp_config,
                    n_jobs=args.n_jobs
                )
            
            # Create output directory for external test results first
            ext_test_dir = os.path.join(args.output, 'external_test')
            os.makedirs(ext_test_dir, exist_ok=True)
            
            # Create dataset and dataloader
            ext_test_dataset = HybridMolecularDataset(
                ext_df[ext_smiles_col].values,
                ext_fingerprints,
                tokenizer,
                max_length=args.max_length,
                targets=ext_targets
            )
            
            ext_test_loader = DataLoader(ext_test_dataset, batch_size=args.batch_size)
            
            # Evaluate on external test set
            ext_r2, ext_rmse, ext_mae, ext_results_df = evaluate_model(
                hybrid_model, 
                ext_test_loader, 
                device, 
                output_dir=ext_test_dir
            )
            
            if has_targets:
                print(f"\n-- External Test Metrics:")
                print(f"   R²: {ext_r2:.4f}")
                print(f"   RMSE: {ext_rmse:.4f}")
                print(f"   MAE: {ext_mae:.4f}")
                
                # Add external test metrics to model config
                model_config['external_test_metrics'] = {
                    'r2': float(ext_r2),
                    'rmse': float(ext_rmse),
                    'mae': float(ext_mae)
                }
                
                # Save updated config
                with open(os.path.join(args.output, 'model_config.json'), 'w') as f:
                    json.dump(model_config, f, indent=4)
            
            # Save external test predictions
            ext_results_csv_path = os.path.join(ext_test_dir, 'external_test_predictions.csv')
            ext_results_df.to_csv(ext_results_csv_path, index=False)
            print(f"-- External test predictions saved to {ext_results_csv_path}")
            
        except Exception as e:
            print(f"-- Error evaluating on external test set: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Save final model
    torch.save(hybrid_model.state_dict(), os.path.join(args.output, 'final_model.pt'))
    
    print(f"-- Model saved to {args.output}")
    print(f"-- Test predictions saved to {results_csv_path}")

if __name__ == "__main__":
    print(f"-- Using device: {device}")
    main()