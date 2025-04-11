#!/usr/bin/env python3

import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdMolDescriptors, Descriptors, Crippen, Lipinski
import json
from tqdm import tqdm
import multiprocessing
from functools import partial
import re
import gc
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from torch.utils.data import DataLoader, Dataset # Added for DataLoader

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

device = torch.device("cuda" if torch.cuda.is_available() else
                     "mps" if torch.backends.mps.is_available() else
                     "cpu")

# --- Start: Copied/Adapted from 5-finetune-neural.py ---

class MolecularDataset(Dataset): # Added for DataLoader compatibility
    def __init__(self, features):
        self.features = torch.tensor(features, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]

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
                    # Fallback to default activation if list is shorter
                    layers.append(activations['relu'])
            else:
                # Use the single specified activation
                layers.append(activations.get(activation_type, nn.ReLU())) # Use .get for safety

            layers.append(nn.Dropout(dropout))

            # Store residual connection reference dimensions for forward pass
            # Ensure indices match: check previous layer dimension if i > 0
            if residual_connections and i > 0 and len(hidden_dims) > i-1 and hidden_dims[i-1] == hidden_dim:
                 # Correct calculation for layer index within the block
                 # Linear -> Norm -> Activation -> Dropout (4 layers per block)
                 # We want the output *before* the norm/activation of the *previous* block
                 # The index calculation needs careful review based on exact layer sequence
                 # Let's assume standard block: Linear, Norm, Activation, Dropout
                 # We need the output of the previous Linear layer
                 # The current block starts at index i * 4 (approx)
                 # The previous Linear layer was at index (i-1) * 4
                 # set_attr based on layer index might be fragile.
                 # A simpler approach for residual might be needed if dims match.
                 # For now, keeping the original logic, but it might need adjustment.
                 setattr(self, f'res_point_{i}', (i-1)*4) # Potential index for residual start

            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1)) # Final output layer

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        if not self.residual_connections:
            # Simple sequential forward pass
            for layer in self.layers:
                x = layer(x)
            return x
        else:
            # Forward pass with residual connections where dimensions match
            # This implementation assumes residuals are added after dropout
            # and requires careful index management or a redesign.
            residuals = {}
            layer_block_size = 4 # Assuming Linear, Norm, Activation, Dropout
            current_block_idx = 0
            prev_output = None

            for i, layer in enumerate(self.layers):
                # Identify start of a block (Linear layer)
                if i % layer_block_size == 0 and i < len(self.layers) -1: # Exclude final linear layer
                    current_block_idx = i // layer_block_size
                    # Check if residual connection is possible from the previous block
                    if current_block_idx > 0:
                         # Check if dimensions match based on config
                         if self.hidden_dims[current_block_idx-1] == self.hidden_dims[current_block_idx]:
                              # Store the output *before* the current block's linear layer
                              # This needs the output from the *end* of the previous block
                              residuals[current_block_idx] = x # Store input to the current block

                # Apply the current layer
                x_pre_activation = x # Store before potential activation for residual
                x = layer(x)

                # Check if we are at the end of a block (after Dropout)
                if (i + 1) % layer_block_size == 0 and i < len(self.layers) - 1:
                     block_end_idx = (i + 1) // layer_block_size - 1 # block index just finished
                     # Add residual if available for the *next* block's input index
                     if block_end_idx + 1 in residuals:
                         # Add the stored input from the start of this block
                         x = x + residuals[block_end_idx + 1]

            # Handle the final linear layer separately if residuals applied before it
            if self.layers and isinstance(self.layers[-1], nn.Linear):
                 # If the loop ended before the last layer, apply it now
                 if i < len(self.layers) - 1:
                      x = self.layers[-1](x)

            return x

    def __repr__(self):
        return (f"NeuralNetwork(input_dim={self.input_dim}, "
                f"hidden_dims={self.hidden_dims}, "
                f"dropout={self.dropout_rate}, "
                f"activation={self.activation_type}, "
                f"batch_norm={self.batch_norm}, "
                f"residual_connections={self.residual_connections}, "
                f"layer_norm={self.layer_norm})")


def compute_rdkit_descriptors(mol):
    if mol is None:
        return [np.nan] * 10 # Use NaN for missing values

    descriptors = []
    try: descriptors.append(Descriptors.MolWt(mol))
    except: descriptors.append(np.nan)
    try: descriptors.append(Descriptors.MolLogP(mol))
    except: descriptors.append(np.nan)
    try: descriptors.append(Crippen.MolMR(mol))
    except: descriptors.append(np.nan)
    try: descriptors.append(Descriptors.NumHDonors(mol))
    except: descriptors.append(np.nan)
    try: descriptors.append(Descriptors.NumHAcceptors(mol))
    except: descriptors.append(np.nan)
    try: descriptors.append(Descriptors.TPSA(mol))
    except: descriptors.append(np.nan)
    try: descriptors.append(Lipinski.NumRotatableBonds(mol))
    except: descriptors.append(np.nan)
    try: descriptors.append(Descriptors.FractionCSP3(mol))
    except: descriptors.append(np.nan)
    try: descriptors.append(Descriptors.NumAromaticRings(mol))
    except: descriptors.append(np.nan)
    try: descriptors.append(Descriptors.NumAliphaticRings(mol))
    except: descriptors.append(np.nan)

    return descriptors

def _process_morgan_chunk(chunk_data, radius, nBits):
    smiles_chunk, chunk_idx = chunk_data
    fps = []
    fpgen = rdMolDescriptors.GetMorganFingerprint # Use the function directly

    for smiles in smiles_chunk:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                # Generate fingerprint
                fp = fpgen(mol, radius, nBits=nBits)
                # Convert to numpy array safely
                arr = np.zeros((nBits,), dtype=np.int32) # Correct shape
                DataStructs.ConvertToNumpyArray(fp, arr)
                fps.append(arr)
            else:
                fps.append(np.zeros(nBits, dtype=np.int32))
        except Exception as e:
            # print(f"-- Error processing SMILES: {smiles[:20]}... - {str(e)}") # Optional: for debugging
            fps.append(np.zeros(nBits, dtype=np.int32))

    return (chunk_idx, fps)


def _process_maccs_chunk(chunk_data):
    smiles_chunk, chunk_idx = chunk_data
    fps = []

    for smiles in smiles_chunk:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            try:
                fp = rdMolDescriptors.GetMACCSKeysFingerprint(mol)
                arr = np.zeros((167,), dtype=np.int32) # MACCS keys are 167 bits
                DataStructs.ConvertToNumpyArray(fp, arr)
                fps.append(arr)
            except Exception:
                 fps.append(np.zeros(167, dtype=np.int32))
        else:
            fps.append(np.zeros(167, dtype=np.int32))

    return (chunk_idx, fps)

def _process_rdkit_chunk(chunk_data, nBits):
    smiles_chunk, chunk_idx = chunk_data
    fps = []

    for smiles in smiles_chunk:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            try:
                fp = rdMolDescriptors.RDKFingerprint(mol, fpSize=nBits)
                arr = np.zeros((nBits,), dtype=np.int32)
                DataStructs.ConvertToNumpyArray(fp, arr)
                fps.append(arr)
            except Exception:
                fps.append(np.zeros(nBits, dtype=np.int32))
        else:
            fps.append(np.zeros(nBits, dtype=np.int32))

    return (chunk_idx, fps)

def _process_descriptors_chunk(chunk_data):
    smiles_chunk, chunk_idx = chunk_data
    descs = []

    for smiles in smiles_chunk:
        mol = Chem.MolFromSmiles(smiles)
        descs.append(compute_rdkit_descriptors(mol))

    return (chunk_idx, descs)

def _process_atompair_chunk(chunk_data, nBits=2048): # Allow nBits override
    smiles_chunk, chunk_idx = chunk_data
    fps = []
    fpgen = rdMolDescriptors.GetAtomPairGenerator(fpSize=nBits) # Use fpSize for hashed version

    for smiles in smiles_chunk:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            try:
                # Get hashed Atom Pair fingerprint
                fp_hashed = fpgen.GetHashedFingerprint(mol)
                arr = np.zeros((nBits,), dtype=np.int32)
                DataStructs.ConvertToNumpyArray(fp_hashed, arr)
                fps.append(arr)
            except Exception:
                 fps.append(np.zeros(nBits, dtype=np.int32))
        else:
            fps.append(np.zeros(nBits, dtype=np.int32))
    return (chunk_idx, fps)

def _process_torsion_chunk(chunk_data, nBits=2048): # Allow nBits override
    smiles_chunk, chunk_idx = chunk_data
    fps = []
    fpgen = rdMolDescriptors.GetTopologicalTorsionGenerator(fpSize=nBits) # Use fpSize for hashed version

    for smiles in smiles_chunk:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            try:
                fp_hashed = fpgen.GetHashedFingerprint(mol)
                arr = np.zeros((nBits,), dtype=np.int32)
                DataStructs.ConvertToNumpyArray(fp_hashed, arr)
                fps.append(arr)
            except Exception:
                 fps.append(np.zeros(nBits, dtype=np.int32))
        else:
            fps.append(np.zeros(nBits, dtype=np.int32))
    return (chunk_idx, fps)


def parse_fp_type(fp_type_str):
    fp_type_str = fp_type_str.lower()
    # Morgan: morgan_radius_nbits
    match = re.match(r"morgan_(\d+)_(\d+)", fp_type_str)
    if match:
        return 'morgan', {'radius': int(match.group(1)), 'nBits': int(match.group(2))}

    # RDKit: rdkit_nbits
    match = re.match(r"rdkit_(\d+)", fp_type_str)
    if match:
        return 'rdkit', {'nBits': int(match.group(1))}

    # AtomPair: atompair_nbits
    match = re.match(r"atompair_(\d+)", fp_type_str)
    if match:
        return 'atompair', {'nBits': int(match.group(1))}

    # Torsion: torsion_nbits
    match = re.match(r"torsion_(\d+)", fp_type_str)
    if match:
        return 'torsion', {'nBits': int(match.group(1))}

    # Simple names (use defaults)
    if fp_type_str == 'morgan':
        return 'morgan', {'radius': 3, 'nBits': 2048} # Default from training script
    if fp_type_str == 'rdkit':
        return 'rdkit', {'nBits': 2048}
    if fp_type_str == 'maccs':
        return 'maccs', {}
    if fp_type_str == 'descriptors':
        return 'descriptors', {}
    if fp_type_str == 'atompair':
        return 'atompair', {'nBits': 2048}
    if fp_type_str == 'torsion':
        return 'torsion', {'nBits': 2048}

    print(f"-- Warning: Unrecognized fingerprint format string '{fp_type_str}'. Skipping.")
    return None, None

def generate_features_from_metadata(smiles_list, model_metadata, n_jobs=None, batch_size_fp=5000,
                                    use_infile_fp=False, fp_prefixes=None, input_df=None,
                                    force_sequential=False):
    """Generates features based on model metadata"""
    print_message("Generating features based on model metadata")

    if n_jobs is None:
        n_jobs = max(1, multiprocessing.cpu_count() - 1)
    else:
        n_jobs = min(n_jobs, multiprocessing.cpu_count())

    try_parallel = not force_sequential and n_jobs > 1
    print_message(f"-- Using {'parallel' if try_parallel else 'sequential'} processing with {n_jobs if try_parallel else 1} worker(s)")

    processed_fp_config = model_metadata.get('fingerprint_config', [])
    use_input_descriptors = model_metadata.get('use_input_descriptors', False)
    descriptor_cols = model_metadata.get('descriptor_cols', [])
    scaler_mean = model_metadata.get('scaler_mean')
    scaler_scale = model_metadata.get('scaler_scale')
    expected_input_dim = model_metadata.get('input_dim')
    # Metadata might store fp_prefixes if infile_fp was used during training
    metadata_fp_prefixes = model_metadata.get('fp_prefixes', [])
    metadata_use_infile_fp = model_metadata.get('use_infile_fp', False)

    # --- Fingerprint Generation ---
    fingerprints = np.array([])
    fp_generated = False

    # Prioritize command-line infile FP usage
    if use_infile_fp and fp_prefixes and input_df is not None:
        print_message(f"-- Attempting to use pre-computed fingerprints from input with prefixes: {fp_prefixes}")
        fp_columns = []
        for prefix in fp_prefixes:
            prefix_cols = [col for col in input_df.columns if col.startswith(prefix)]
            if prefix_cols:
                print_message(f"--- Found {len(prefix_cols)} columns with prefix '{prefix}'")
                fp_columns.extend(prefix_cols)
            else:
                 print_message(f"--- Warning: No columns found with prefix '{prefix}'")

        if fp_columns:
            try:
                fingerprints = input_df[fp_columns].values.astype(np.float32) # Ensure numeric
                print_message(f"--- Using {fingerprints.shape[1]} fingerprint columns from input file")
                fp_generated = True # Indicate features were obtained
            except Exception as e:
                 print_message(f"--- Error extracting fingerprint columns: {e}. Will attempt generation.")
                 fingerprints = np.array([])
        else:
            print_message(f"--- No matching fingerprint columns found with specified prefixes.")

    # Fallback to metadata infile FP usage if command-line failed or wasn't specified
    if not fp_generated and metadata_use_infile_fp and metadata_fp_prefixes and input_df is not None:
         print_message(f"-- Attempting to use pre-computed fingerprints based on model metadata with prefixes: {metadata_fp_prefixes}")
         fp_columns = []
         for prefix in metadata_fp_prefixes:
             prefix_cols = [col for col in input_df.columns if col.startswith(prefix)]
             if prefix_cols:
                 print_message(f"--- Found {len(prefix_cols)} columns with prefix '{prefix}'")
                 fp_columns.extend(prefix_cols)
             else:
                 print_message(f"--- Warning: No columns found with prefix '{prefix}' (from metadata)")

         if fp_columns:
             try:
                 fingerprints = input_df[fp_columns].values.astype(np.float32)
                 print_message(f"--- Using {fingerprints.shape[1]} fingerprint columns from input file (metadata spec)")
                 fp_generated = True
             except Exception as e:
                 print_message(f"--- Error extracting fingerprint columns (metadata spec): {e}. Will attempt generation.")
                 fingerprints = np.array([])
         else:
             print_message(f"--- No matching fingerprint columns found with metadata prefixes.")


    # Generate fingerprints if not loaded from file
    if not fp_generated:
        if not processed_fp_config:
            print_message("-- Warning: No fingerprint configuration found in metadata and not using infile FPs.")
            # Cannot proceed without fingerprints unless only descriptors were used
            if not use_input_descriptors:
                 raise ValueError("No fingerprint config in metadata and descriptors not used.")
        else:
            print_message(f"-- Generating fingerprints based on metadata config:")
            fp_type_strings = []
            for fp in processed_fp_config:
                name = fp['name']
                params = {k: v for k, v in fp.items() if k != 'name'}
                print_message(f"--- Type: {name}, Params: {params}")
                # Reconstruct the string representation for processing functions
                if name == 'morgan': fp_type_strings.append(f"morgan_{params['radius']}_{params['nBits']}")
                elif name == 'rdkit': fp_type_strings.append(f"rdkit_{params['nBits']}")
                elif name == 'atompair': fp_type_strings.append(f"atompair_{params.get('nBits', 2048)}") # Use default if missing
                elif name == 'torsion': fp_type_strings.append(f"torsion_{params.get('nBits', 2048)}") # Use default if missing
                else: fp_type_strings.append(name)


            # Process SMILES in batches
            total_smiles = len(smiles_list)
            num_batches = (total_smiles + batch_size_fp - 1) // batch_size_fp
            print_message(f"--- Processing {total_smiles} molecules in {num_batches} batches (batch size: {batch_size_fp})")

            batch_results = []
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size_fp
                end_idx = min(start_idx + batch_size_fp, total_smiles)
                current_batch = smiles_list[start_idx:end_idx]
                current_batch_fps = [] # Store FPs for this batch for each type

                for fp_type_str in fp_type_strings:
                    fp_name, fp_params = parse_fp_type(fp_type_str)
                    if fp_name is None: continue

                    # Create sub-batches for parallel processing
                    chunks = np.array_split(current_batch, min(n_jobs, len(current_batch)))
                    chunk_data = [(chunk, i) for i, chunk in enumerate(chunks)]

                    desc = f"Batch {batch_idx+1}/{num_batches}: {fp_name}"
                    process_func = None
                    if fp_name == 'morgan':
                        desc += f" (r={fp_params['radius']}, n={fp_params['nBits']})"
                        process_func = partial(_process_morgan_chunk, radius=fp_params['radius'], nBits=fp_params['nBits'])
                    elif fp_name == 'rdkit':
                         desc += f" (n={fp_params['nBits']})"
                         process_func = partial(_process_rdkit_chunk, nBits=fp_params['nBits'])
                    elif fp_name == 'maccs':
                        process_func = _process_maccs_chunk
                    elif fp_name == 'descriptors':
                        process_func = _process_descriptors_chunk
                    elif fp_name == 'atompair':
                         desc += f" (n={fp_params['nBits']})"
                         process_func = partial(_process_atompair_chunk, nBits=fp_params['nBits'])
                    elif fp_name == 'torsion':
                         desc += f" (n={fp_params['nBits']})"
                         process_func = partial(_process_torsion_chunk, nBits=fp_params['nBits'])

                    if process_func:
                        results = []
                        try:
                            if try_parallel:
                                with multiprocessing.Pool(n_jobs) as pool:
                                    results = list(tqdm(pool.imap(process_func, chunk_data), total=len(chunk_data), desc=desc))
                            else: # Sequential
                                for chunk in tqdm(chunk_data, desc=f"{desc} (sequential)"):
                                    results.append(process_func(chunk))
                        except Exception as e:
                            print_message(f"\n--- Error during parallel processing for {fp_name}: {e}. Falling back to sequential.")
                            try_parallel = False # Fallback for subsequent types too
                            results = []
                            for chunk in tqdm(chunk_data, desc=f"{desc} (sequential fallback)"):
                                results.append(process_func(chunk))

                        # Combine results from parallel processing
                        results.sort(key=lambda x: x[0])
                        fp_list_for_type = []
                        for _, chunk_fps in results:
                            fp_list_for_type.extend(chunk_fps)
                        current_batch_fps.append(np.array(fp_list_for_type)) # Add results for this FP type

                # Combine different FP types for the current batch
                if current_batch_fps:
                    batch_combined = np.hstack(current_batch_fps)
                    batch_results.append(batch_combined)

                gc.collect()

            # Combine results from all batches
            if batch_results:
                fingerprints = np.vstack(batch_results)
                print_message(f"--- Generated combined fingerprints shape: {fingerprints.shape}")
                fp_generated = True
            else:
                 print_message("--- Warning: No fingerprints were generated.")
                 # Check if descriptors are expected, otherwise raise error
                 if not use_input_descriptors:
                      raise ValueError("Fingerprint generation failed and descriptors not used.")


    # --- Descriptor Handling ---
    scaled_descriptors = np.array([])
    descriptors_generated = False
    if use_input_descriptors:
        if not descriptor_cols:
            print_message("-- Warning: Model metadata indicates use of descriptors, but 'descriptor_cols' is empty.")
        elif input_df is None:
             print_message("-- Warning: Model metadata indicates use of descriptors, but no input dataframe provided to extract them.")
        else:
            print_message(f"-- Using input descriptors based on metadata: {descriptor_cols}")
            # Check if all required descriptor columns exist in the input DataFrame
            missing_cols = [col for col in descriptor_cols if col not in input_df.columns]
            if missing_cols:
                raise ValueError(f"Missing required descriptor columns in input data: {missing_cols}")

            descriptors_raw = input_df[descriptor_cols].values

            # Ensure float type *before* imputation/scaling
            try:
                descriptors_raw = descriptors_raw.astype(float)
            except ValueError as e:
                print_message(f"-- Error converting descriptor columns to float: {e}")
                # Attempt to identify problematic columns/values
                for col in descriptor_cols:
                     try:
                         input_df[col].astype(float)
                     except ValueError:
                         print_message(f"--- Problematic column: {col}")
                         print_message(f"--- Sample non-numeric values: {input_df[pd.to_numeric(input_df[col], errors='coerce').isna()][col].unique()[:5]}")
                raise ValueError("Could not convert descriptor columns to numeric. Check input data.")


            # Impute NaNs
            if np.isnan(descriptors_raw).any():
                print_message("--- Warning: NaNs found in input descriptors. Imputing with mean.")
                # Use training mean for imputation if available
                imputer = SimpleImputer(strategy='mean')
                descriptors_imputed = imputer.fit_transform(descriptors_raw) # Fit on prediction data only if training mean unavailable
            else:
                descriptors_imputed = descriptors_raw

            # Scale descriptors
            if scaler_mean and scaler_scale:
                print_message("--- Scaling descriptors using training data statistics")
                if len(scaler_mean) == descriptors_imputed.shape[1] and len(scaler_scale) == descriptors_imputed.shape[1]:
                    # Ensure scale values are not zero before dividing
                    scale_np = np.array(scaler_scale)
                    scale_np[scale_np == 0] = 1.0 # Avoid division by zero, effectively skipping scaling for that feature
                    scaled_descriptors = (descriptors_imputed - np.array(scaler_mean)) / scale_np
                    descriptors_generated = True
                else:
                    print_message("-- Warning: Scaler dimensions from metadata don't match descriptor count. Skipping scaling.")
            else:
                print_message("-- Warning: Scaler state not found in metadata. Descriptors will not be scaled.")
                scaled_descriptors = descriptors_imputed # Use imputed but unscaled data
                descriptors_generated = True # Mark as generated, even if not scaled

    # --- Combine Features ---
    feature_list = []
    if fp_generated and fingerprints.size > 0:
        feature_list.append(fingerprints)
    if descriptors_generated and scaled_descriptors.size > 0:
        feature_list.append(scaled_descriptors)

    if not feature_list:
        raise ValueError("No features could be generated based on model metadata and input.")

    final_features = np.hstack(feature_list)
    print_message(f"-- Final combined features shape: {final_features.shape}")

    # --- Validate Dimensions ---
    if expected_input_dim is not None and final_features.shape[1] != expected_input_dim:
        raise ValueError(f"Feature dimension mismatch! Model expects {expected_input_dim}, but generated {final_features.shape[1]}. Check metadata and input data alignment.")
    elif expected_input_dim is None:
         print_message("-- Warning: Cannot validate feature dimension as 'input_dim' not found in metadata.")


    return final_features


# --- End: Copied/Adapted from 5-finetune-neural.py ---


def print_message(message, verbose=True):
    """Helper function for conditional printing."""
    # In this script, we generally want all messages
    sys.stdout.write(f"-- {message}\n")
    sys.stdout.flush()

def safe_mol_from_smiles(smiles):
    if not smiles or pd.isna(smiles) or not isinstance(smiles, str):
        return None
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        # Basic check for validity
        if mol is None: return None
        # Optional: Add more checks like atom count > 0
        return mol
    except:
        return None

def canonicalize_smiles(smiles):
    """Canonicalizes SMILES string."""
    if not smiles or pd.isna(smiles) or not isinstance(smiles, str):
        return smiles
    mol = safe_mol_from_smiles(smiles)
    if mol:
        try:
            return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        except:
            return smiles # Return original if canonicalization fails
    return smiles # Return original if mol parsing failed

def load_model_with_metadata(model_dir):
    """Load model and its metadata from the specified directory"""
    config_paths = [
        os.path.join(model_dir, 'model_config.json'),      # From non-optimized training
        os.path.join(model_dir, 'best_config.json'),       # From optimized training
        os.path.join(model_dir, 'finetuned_config.json') # From fine-tuning
    ]

    config_path = next((p for p in config_paths if os.path.exists(p)), None)
    if config_path is None:
        raise FileNotFoundError(f"-- Error: No model configuration file (model_config/best_config/finetuned_config.json) found in {model_dir}")

    print_message(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)

    model_paths = [
        os.path.join(model_dir, 'final_model.pt'), # Saved by train/finetune
        os.path.join(model_dir, 'best_model.pt') # Saved during early stopping
    ]

    model_path = next((p for p in model_paths if os.path.exists(p)), None)
    if model_path is None:
        raise FileNotFoundError(f"-- Error: No model weights file (final_model.pt or best_model.pt) found in {model_dir}")
    print_message(f"Loading model weights from: {model_path}")


    model_metadata_path = os.path.join(model_dir, 'model_metadata.json')
    if not os.path.exists(model_metadata_path):
         raise FileNotFoundError(f"-- Error: Model metadata file (model_metadata.json) is missing in {model_dir}. Cannot determine feature generation.")
    print_message(f"Loading model metadata from: {model_metadata_path}")
    with open(model_metadata_path, 'r') as f:
        model_metadata = json.load(f)

    input_dim = model_metadata.get('input_dim')
    if input_dim is None:
        raise ValueError("-- Error: Could not determine model 'input_dim' from model_metadata.json.")

    # Extract architecture parameters robustly
    arch_params = config.get('architecture', config) # Handle finetuned_config nesting

    hidden_dims = arch_params.get('hidden_dims')
    dropout = arch_params.get('dropout', 0.2) # Default if missing
    activation = arch_params.get('activation', 'relu') # Default if missing
    batch_norm = arch_params.get('batch_norm', True) # Default if missing
    residual_connections = arch_params.get('residual_connections', False) # Default if missing
    layer_norm = arch_params.get('layer_norm', False) # Default if missing

    # Reconstruct hidden_dims if needed (e.g., from Optuna best_config)
    if hidden_dims is None:
        n_layers = arch_params.get("n_layers")
        if n_layers is not None:
            hidden_dims = []
            for i in range(n_layers):
                dim = arch_params.get(f"hidden_dim_{i}")
                if dim is not None:
                    hidden_dims.append(dim)
                else:
                    raise ValueError(f"-- Error: Missing 'hidden_dim_{i}' in configuration file needed to reconstruct model.")
        else:
            # If neither hidden_dims nor n_layers are present
            raise ValueError(f"-- Error: Could not find 'hidden_dims' or 'n_layers' in configuration file to reconstruct model architecture.")

    # Create model instance using the class defined in *this* script
    model = NeuralNetwork(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        activation_type=activation,
        batch_norm=batch_norm,
        residual_connections=residual_connections,
        layer_norm=layer_norm
    )

    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval() # Set to evaluation mode

    print_message(f"Model loaded successfully from {model_dir}")
    print_message(f"Model architecture: {model}") # Print loaded model structure

    return model, model_metadata


def main():
    parser = argparse.ArgumentParser(description='Predict Molecular Properties using a Trained Neural Network')

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input', '-i', help='Input CSV file path')
    input_group.add_argument('--smiles', '-s', help='Direct SMILES string or comma-separated list of SMILES')

    # Output options
    parser.add_argument('--output', '-o', help='Output CSV file path (default: input with _predictions added)')
    parser.add_argument('--model', '-m', required=True, help='Path to trained/finetuned model directory')

    # Processing options
    parser.add_argument('--smiles-col', default='SMILES', help='Column name for SMILES in CSV input (default: SMILES)')
    parser.add_argument('--n-jobs', type=int, default=None, help='Number of parallel jobs for feature generation (default: CPU count - 1)')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size for predictions (default: 256)')
    parser.add_argument('--batch-size-fp', type=int, default=5000, help='Batch size for fingerprint computation (default: 5000)')
    parser.add_argument('--no-canonicalize', action='store_true', help='Skip SMILES canonicalization step')
    parser.add_argument('--force-sequential', action='store_true', help='Force sequential feature generation')
    parser.add_argument('--use-infile-fp', action='store_true',
                       help='Use pre-computed fingerprints from input file (overrides metadata generation)')
    parser.add_argument('--fp-prefixes', type=str, nargs='+', default=[],
                       help='Column prefixes to identify fingerprint columns in the input file (e.g., morgan_fp_, maccs_)')


    # Verbosity
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output messages')

    args = parser.parse_args()

    # --- Setup ---
    print_message(f"Neural Network Prediction Tool")
    print_message(f"Using device: {device}")
    if args.force_sequential:
        print_message("Forcing sequential feature generation.")


    # --- Load Input Data ---
    input_df = None
    if args.input:
        print_message(f"Loading data from {args.input}")
        try:
            input_df = pd.read_csv(args.input)
            # Find SMILES column if not default
            if args.smiles_col not in input_df.columns:
                potential_cols = [col for col in input_df.columns if 'smiles' in col.lower()]
                if potential_cols:
                    args.smiles_col = potential_cols[0]
                    print_message(f"-- Found SMILES column: '{args.smiles_col}'")
                else:
                    raise ValueError(f"SMILES column '{args.smiles_col}' not found in input file and no alternatives detected.")
            smiles_list = input_df[args.smiles_col].astype(str).tolist() # Ensure strings
        except FileNotFoundError:
            print(f"-- Error: Input file not found at {args.input}")
            sys.exit(1)
        except Exception as e:
            print(f"-- Error loading input file: {e}")
            sys.exit(1)
    else:
        # Handle direct SMILES input
        print_message("Processing direct SMILES input")
        if ',' in args.smiles:
            smiles_list = [s.strip() for s in args.smiles.split(',')]
        else:
            smiles_list = [args.smiles.strip()]
        input_df = pd.DataFrame({'SMILES': smiles_list}) # Create a basic DataFrame
        args.smiles_col = 'SMILES' # Set column name for consistency

    print_message(f"Processing {len(smiles_list)} molecules")

    # --- Preprocess SMILES ---
    original_smiles = smiles_list # Keep original for output df
    processed_smiles = []
    if args.no_canonicalize:
        print_message("Skipping SMILES canonicalization")
        processed_smiles = original_smiles
    else:
        print_message("Canonicalizing SMILES")
        processed_smiles = [canonicalize_smiles(s) for s in tqdm(original_smiles, desc="Canonicalizing")]
        # Corrected check for canonicalization changes/failures
        changed_count = sum(1 for os, ps in zip(original_smiles, processed_smiles) if os != ps)
        print_message(f"-- {changed_count} SMILES were canonicalized.")

    # --- Check Molecule Validity ---
    valid_mols = [safe_mol_from_smiles(s) for s in processed_smiles]
    valid_count = sum(1 for mol in valid_mols if mol is not None)
    print_message(f"-- Found {valid_count} valid molecules out of {len(processed_smiles)} processed SMILES.")
    if valid_count == 0:
        print("-- Error: No valid molecules found after processing SMILES. Cannot proceed.")
        sys.exit(1)
    elif valid_count < len(processed_smiles):
        print_message(f"-- Warning: {len(processed_smiles) - valid_count} invalid SMILES detected. Features for these will be zero vectors.")


    # --- Load Model ---
    try:
        model, model_metadata = load_model_with_metadata(args.model)
        target_name = model_metadata.get('target_col', 'Value') # Get target name early
        print_message(f"-- Target property from metadata: {target_name}")
    except Exception as e:
        print(f"{e}") # Error messages from load_model are already formatted
        sys.exit(1)


    # --- Generate Features ---
    try:
        features = generate_features_from_metadata(
            processed_smiles, # Use processed SMILES for feature gen
            model_metadata,
            n_jobs=args.n_jobs,
            batch_size_fp=args.batch_size_fp,
            use_infile_fp=args.use_infile_fp,
            fp_prefixes=args.fp_prefixes,
            input_df=input_df, # Pass the dataframe for descriptor/infile FP extraction
            force_sequential=args.force_sequential
        )
        # --- Feature Sanity Check ---
        if args.verbose:
            print_message("\n-- Feature Matrix Sanity Check:")
            print_message(f"   Shape: {features.shape}")
            try:
                print_message(f"   Min: {np.min(features):.4f}")
                print_message(f"   Max: {np.max(features):.4f}")
                print_message(f"   Mean: {np.mean(features):.4f}")
                zero_rows = np.sum(np.all(features == 0, axis=1))
                print_message(f"   Rows with all zeros: {zero_rows}")
            except Exception as e:
                print_message(f"   Could not calculate feature stats: {e}")
        # --- End Feature Sanity Check ---

    except Exception as e:
        print(f"-- Error generating features: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback for debugging
        sys.exit(1)

    # --- Predict ---
    print_message("Starting predictions")
    prediction_dataset = MolecularDataset(features) # Use the simple Dataset
    # Pin memory if using CUDA
    pin_memory_flag = True if device.type == 'cuda' else False
    prediction_loader = DataLoader(prediction_dataset,
                                   batch_size=args.batch_size,
                                   shuffle=False,
                                   pin_memory=pin_memory_flag,
                                   num_workers=0) # Set num_workers=0 for MPS/CPU unless specifically needed

    predictions = []
    model.eval() # Ensure model is in eval mode
    with torch.no_grad():
        for batch_features in tqdm(prediction_loader, desc="Predicting"):
            batch_features = batch_features.to(device)
            batch_preds = model(batch_features).cpu().numpy().flatten()
            predictions.extend(batch_preds)

    # Convert list of predictions to numpy array for stats
    predictions_np = np.array(predictions)

    # --- Prediction Sanity Check ---
    if args.verbose:
        print_message("\n-- Raw Prediction Sanity Check:")
        print_message(f"   Number of predictions: {len(predictions_np)}")
        try:
            print_message(f"   Min: {np.min(predictions_np):.4f}")
            print_message(f"   Max: {np.max(predictions_np):.4f}")
            print_message(f"   Mean: {np.mean(predictions_np):.4f}")
            print_message(f"   Std Dev: {np.std(predictions_np):.4f}")
            unique_preds = len(np.unique(predictions_np))
            print_message(f"   Unique values: {unique_preds}")
        except Exception as e:
             print_message(f"  Could not calculate prediction stats: {e}")
    # --- End Prediction Sanity Check ---


    # --- Format Output ---
    output_df = input_df.copy() # Start with original input data
    # Add processed SMILES if canonicalization was done
    if not args.no_canonicalize:
        output_df['Processed_SMILES'] = processed_smiles

    # Add predictions using the target name from metadata
    pred_col_name = f"{target_name}_Predicted"
    output_df[pred_col_name] = predictions_np # Use the numpy array

    # Determine output path
    if args.output:
        output_path = args.output
    elif args.input:
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        output_path = f"{base_name}_predictions.csv"
    else:
        output_path = "direct_smiles_predictions.csv"

    # Ensure output directory exists if specified as part of the path
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save results
    try:
        output_df.to_csv(output_path, index=False)
        print_message(f"Predictions saved to {output_path}")
    except Exception as e:
        print(f"-- Error saving output file: {e}")
        sys.exit(1)

    # --- Display Sample Results ---
    print_message("\nSample predictions:")
    display_cols = [args.smiles_col]
    if not args.no_canonicalize:
        display_cols.append('Processed_SMILES')
    display_cols.append(pred_col_name)
    # Ensure columns exist before printing
    display_cols = [col for col in display_cols if col in output_df.columns]
    # Handle potential non-existence of input_df columns if only direct SMILES used
    display_cols = [col for col in display_cols if col in output_df.columns]

    print(output_df[display_cols].head(5).to_string(index=False))

    # Use the already calculated numpy stats if verbose
    print_message("\nPrediction Statistics:")
    if args.verbose and 'predictions_np' in locals() and predictions_np.size > 0 :
         print_message(f"Min predicted {target_name}: {np.min(predictions_np):.4f}")
         print_message(f"Max predicted {target_name}: {np.max(predictions_np):.4f}")
         print_message(f"Mean predicted {target_name}: {np.mean(predictions_np):.4f}")
         print_message(f"Median predicted {target_name}: {np.median(predictions_np):.4f}") # Use median as well
    else:
         # Calculate fresh if not verbose or numpy array not available
         print_message(f"Min predicted {target_name}: {output_df[pred_col_name].min():.4f}")
         print_message(f"Max predicted {target_name}: {output_df[pred_col_name].max():.4f}")
         print_message(f"Mean predicted {target_name}: {output_df[pred_col_name].mean():.4f}")
         print_message(f"Median predicted {target_name}: {output_df[pred_col_name].median():.4f}")


if __name__ == "__main__":
    main()