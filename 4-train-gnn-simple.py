#!/usr/bin/env python
# GNN Training Script from SMILES CSV
# Optimized version with MPS support and multiprocessing

import os
import time
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import multiprocessing
from functools import partial
import warnings

# Suppress specific RDKit deprecation warnings
warnings.filterwarnings("ignore", message="please use MorganGenerator")

# Set seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train a GNN on molecular data')
    parser.add_argument('--csv', type=str, help='Path to input CSV file')
    parser.add_argument('--smiles_col', type=str, default='smiles', help='Column name for SMILES strings')
    parser.add_argument('--target_col', type=str, required=True, help='Column name for target values')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension size')
    parser.add_argument('--output_dir', type=str, default='./gnn_output', help='Output directory')
    parser.add_argument('--n_jobs', type=int, default=None, help='Number of processes for data preparation (default: CPU count-1)')
    parser.add_argument('--use_mps', action='store_true', help='Use Apple Metal Performance Shaders (MPS) if available')
    parser.add_argument('--use_cuda', action='store_true', help='Use CUDA if available')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    return parser.parse_args()

def mol_to_graph(mol):
    """Convert an RDKit molecule to a PyTorch Geometric graph."""
    # Get atom features
    atoms = mol.GetAtoms()
    x = []
    for atom in atoms:
        # Create atom features directly from atom properties
        features = np.zeros(32)

        # Atom type (one-hot encoding of common atoms)
        atom_num = atom.GetAtomicNum()
        if atom_num <= 5:      # H, He, Li, Be, B
            features[0] = 1 if atom_num == 1 else 0  # H
            features[1] = 1 if atom_num == 5 else 0  # B
            features[2] = 1 if atom_num == 6 else 0  # C
            features[3] = 1 if atom_num == 7 else 0  # N
            features[4] = 1 if atom_num == 8 else 0  # O
            features[5] = 1 if atom_num == 9 else 0  # F
            features[6] = 1 if atom_num == 15 else 0  # P
            features[7] = 1 if atom_num == 16 else 0  # S
            features[8] = 1 if atom_num == 17 else 0  # Cl
            features[9] = 1 if atom_num == 35 else 0  # Br
            features[10] = 1 if atom_num == 53 else 0  # I
            features[11] = 1 if atom_num not in [1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53] else 0  # Other

        # Basic properties that are definitely available
        features[12] = atom.GetDegree() / 6.0  # Normalize degree by typical max
        features[13] = atom.GetFormalCharge() / 1.0  # Normalize charge
        features[14] = atom.GetNumExplicitHs() / 4.0  # Normalize explicit H count
        features[15] = atom.GetNumImplicitHs() / 4.0  # Normalize implicit H count
        features[16] = 1.0 if atom.GetIsAromatic() else 0.0  # Aromaticity flag
        features[17] = 1.0 if atom.IsInRing() else 0.0  # Ring membership flag
        features[18] = atom.GetTotalNumHs() / 4.0  # Normalize total H count
        features[19] = atom.GetTotalDegree() / 6.0  # Normalize total degree
        features[20] = atom.GetTotalValence() / 6.0  # Normalize total valence

        # Hybridization state (one-hot encoded)
        hyb = atom.GetHybridization()
        features[21] = 1.0 if hyb == Chem.rdchem.HybridizationType.SP else 0.0
        features[22] = 1.0 if hyb == Chem.rdchem.HybridizationType.SP2 else 0.0
        features[23] = 1.0 if hyb == Chem.rdchem.HybridizationType.SP3 else 0.0

        # Ring size information
        features[24] = 1.0 if atom.IsInRingSize(3) else 0.0
        features[25] = 1.0 if atom.IsInRingSize(4) else 0.0
        features[26] = 1.0 if atom.IsInRingSize(5) else 0.0
        features[27] = 1.0 if atom.IsInRingSize(6) else 0.0

        # Additional features if available
        features[28] = atom.GetNumRadicalElectrons() / 1.0  # Normalize radical electrons

        # Chirality information if available
        try:
            features[29] = 1.0 if atom.GetProp('_CIPCode') == 'R' else 0.0
            features[30] = 1.0 if atom.GetProp('_CIPCode') == 'S' else 0.0
        except:
            features[29] = 0.0
            features[30] = 0.0

        # Last feature for future use
        features[31] = 0.0

        x.append(features)

    x = torch.tensor(np.array(x), dtype=torch.float)

    # Get edge indices
    edge_indices = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        # PyTorch Geometric needs edges in both directions
        edge_indices.append([i, j])
        edge_indices.append([j, i])

    if len(edge_indices) == 0:  # Handle molecules with no bonds
        edge_indices = [[0, 0]]  # Self-loop

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

    return x, edge_index

def process_molecule(row_data, smiles_col, target_col):
    """Process a single molecule row"""
    try:
        smiles = row_data[smiles_col]
        y = row_data[target_col]

        if pd.isna(y):
            return None, f"Skipped {smiles}: Missing target value"

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, f"Skipped {smiles}: Could not parse SMILES"

        # Try to process the molecule
        x, edge_index = mol_to_graph(mol)

        # Create PyTorch Geometric Data object
        data = Data(x=x, edge_index=edge_index, y=torch.tensor([y], dtype=torch.float))
        return data, None
    except Exception as e:
        return None, f"Error processing molecule {row_data[smiles_col]}: {str(e)}"

def prepare_dataset(csv_path, smiles_col, target_col, n_jobs=None):
    """Prepare dataset from CSV file with SMILES and target values."""
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} entries")

    # Process molecules in a single thread if multiprocessing causes issues
    data_list = []
    skipped = 0
    error_count = 0

    # Check if we should use multiprocessing
    use_mp = n_jobs is not None and n_jobs != 1

    if use_mp:
        # Determine the number of processes to use
        if n_jobs is None:
            n_jobs = max(1, multiprocessing.cpu_count() - 1)  # Leave one CPU free
        print(f"Using {n_jobs} processes for data processing")

        # Create a partial function with fixed arguments
        process_func = partial(process_molecule, smiles_col=smiles_col, target_col=target_col)

        # Process molecules in parallel
        data_list = []
        error_messages = []

        # Use a context manager for the pool
        with multiprocessing.Pool(processes=n_jobs) as pool:
            # Process in chunks for better progress reporting
            chunk_size = min(1000, max(100, len(df) // (n_jobs * 10)))
            results = list(tqdm(
                pool.imap(process_func, df.to_dict('records'), chunksize=chunk_size),
                total=len(df),
                desc="Processing molecules"
            ))

        # Process results
        for data, error in results:
            if data is not None:
                data_list.append(data)
            elif error is not None:
                error_messages.append(error)

        # Report errors (limited to avoid console flooding)
        max_errors_to_show = 10
        if error_messages:
            print(f"\nEncountered {len(error_messages)} errors:")
            for i, error in enumerate(error_messages[:max_errors_to_show]):
                print(error)
            if len(error_messages) > max_errors_to_show:
                print(f"...and {len(error_messages) - max_errors_to_show} more errors (not shown)")
    else:
        # Single-threaded processing as fallback
        print("Using single-threaded processing")
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing molecules"):
            try:
                smiles = row[smiles_col]
                y = row[target_col]

                if pd.isna(y):
                    skipped += 1
                    continue

                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    skipped += 1
                    continue

                # Try to process the molecule
                x, edge_index = mol_to_graph(mol)

                # Create PyTorch Geometric Data object
                data = Data(x=x, edge_index=edge_index, y=torch.tensor([y], dtype=torch.float))
                data_list.append(data)
            except Exception as e:
                error_count += 1
                if error_count <= 10:  # Only print the first 10 errors to avoid flooding the console
                    print(f"Error processing molecule {row[smiles_col]}: {str(e)}")
                skipped += 1

        if error_count > 10:
            print(f"...and {error_count - 10} more errors (not shown)")

    print(f"Processed {len(data_list)} molecules successfully, skipped {len(df) - len(data_list)}")
    return data_list

class GNN(nn.Module):
    """Graph Neural Network Model."""
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, output_dim)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm3 = nn.BatchNorm1d(hidden_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # If batch is None (for a single graph), set to zeros of appropriate size
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Graph convolution layers with residual connections
        x1 = F.relu(self.conv1(x, edge_index))
        x1 = self.batch_norm1(x1)

        x2 = F.relu(self.conv2(x1, edge_index))
        x2 = self.batch_norm2(x2)
        x2 = x1 + x2  # Residual connection

        x3 = F.relu(self.conv3(x2, edge_index))
        x3 = self.batch_norm3(x3)
        x3 = x2 + x3  # Residual connection

        # Global pooling and MLP
        x = global_mean_pool(x3, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin2(x)

        return x

def train_model(train_loader, model, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        # Ensure target has same shape as output (add dimension if needed)
        target = data.y.view(-1, 1)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(train_loader.dataset)

def evaluate_model(loader, model, device):
    """Evaluate the model on the given loader."""
    model.eval()
    total_loss = 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            # Ensure target has same shape as output
            target = data.y.view(-1, 1)
            loss = F.mse_loss(output, target)
            total_loss += loss.item() * data.num_graphs

            y_true.extend(data.y.cpu().numpy())
            y_pred.extend(output.cpu().numpy().flatten())  # Flatten predictions to match targets

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return total_loss / len(loader.dataset), rmse, mae, r2

def main():
    args = get_args()

    # Set seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine device based on availability and user preference
    device = torch.device('cpu')  # Default

    if args.use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif args.use_mps and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple MPS (Metal Performance Shaders)")
    else:
        print("Using CPU for training")

    # Prepare dataset with multiprocessing
    data_list = prepare_dataset(args.csv, args.smiles_col, args.target_col, n_jobs=args.n_jobs)

    # Check if data_list is empty
    if not data_list:
        print("Error: No valid molecules were processed. Cannot train the model.")
        return

    # Split data
    train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=args.seed)
    train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=args.seed)

    # Create data loaders - set num_workers=0 for MPS compatibility
    num_workers = 0 if device.type == 'mps' else (4 if args.n_jobs else 0)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, num_workers=num_workers)

    print(f"Train: {len(train_data)}, Validation: {len(val_data)}, Test: {len(test_data)}")

    # Determine input dimension from the first molecule
    input_dim = data_list[0].x.shape[1]

    # Initialize model
    model = GNN(input_dim=input_dim, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Training loop
    best_val_loss = float('inf')
    best_epoch = 0
    patience = 15
    counter = 0

    start_time = time.time()
    print("Starting training...")

    for epoch in range(args.epochs):
        # Train
        train_loss = train_model(train_loader, model, optimizer, device)

        # Evaluate
        val_loss, val_rmse, val_mae, val_r2 = evaluate_model(val_loader, model, device)

        # Update learning rate
        scheduler.step(val_loss)

        # Print progress
        print(f"Epoch: {epoch+1:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}, R²: {val_r2:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            counter = 0
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pt'))
            print(f"New best model saved at epoch {best_epoch}")
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Best model at epoch {best_epoch} with validation loss {best_val_loss:.4f}")

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_model.pt')))
    test_loss, test_rmse, test_mae, test_r2 = evaluate_model(test_loader, model, device)

    print("Test Results:")
    print(f"Loss: {test_loss:.4f}")
    print(f"RMSE: {test_rmse:.4f}")
    print(f"MAE: {test_mae:.4f}")
    print(f"R²: {test_r2:.4f}")

    # Save results to file
    with open(os.path.join(args.output_dir, 'results.txt'), 'w') as f:
        f.write(f"Best epoch: {best_epoch}\n")
        f.write(f"Training time: {training_time:.2f} seconds\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test RMSE: {test_rmse:.4f}\n")
        f.write(f"Test MAE: {test_mae:.4f}\n")
        f.write(f"Test R²: {test_r2:.4f}\n")

    print(f"Results saved to {os.path.join(args.output_dir, 'results.txt')}")

if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # Method already set
        pass
    main()
