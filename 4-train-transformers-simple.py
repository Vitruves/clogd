#!/usr/bin/env python3
# Chemistry Transformer Training Script
# Optimized for M3 MacBook Air with Metal GPU acceleration

import os
import time
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Enable Metal acceleration for M3 chip
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Metal Performance Shaders (MPS) for GPU acceleration")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA for GPU acceleration")
else:
    device = torch.device("cpu")
    print("Using CPU for training (GPU acceleration not available)")

# Define the Transformer model components
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class ChemistryTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=8, num_layers=4, dim_feedforward=512, dropout=0.1):
        super(ChemistryTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src = self.embedding(src) * np.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask, src_key_padding_mask)
        output = self.fc_out(output)
        return output

# Tokenizer for SMILES strings
class SMILESTokenizer:
    def __init__(self):
        # Define vocabulary with common atoms, bonds, and structural tokens
        self.vocab = ["<PAD>", "<SOS>", "<EOS>", "<UNK>",
                      "C", "c", "N", "n", "O", "o", "S", "s", "P", "F", "Cl", "Br", "I",
                      "1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
                      "(", ")", "[", "]", "=", "#", "+", "-", "/", "\\", "@", ".", "%"]
        self.token2idx = {token: idx for idx, token in enumerate(self.vocab)}
        self.idx2token = {idx: token for idx, token in enumerate(self.vocab)}

    def tokenize(self, smiles):
        tokens = []
        i = 0

        while i < len(smiles):
            # Handle two-character tokens (Cl, Br, etc.)
            if i + 1 < len(smiles) and smiles[i:i+2] in self.token2idx:
                tokens.append(smiles[i:i+2])
                i += 2
            elif smiles[i] in self.token2idx:
                tokens.append(smiles[i])
                i += 1
            else:
                tokens.append("<UNK>")
                i += 1

        return tokens

    def encode(self, smiles, max_len=None):
        tokens = ["<SOS>"] + self.tokenize(smiles) + ["<EOS>"]
        if max_len:
            if len(tokens) > max_len:
                tokens = tokens[:max_len-1] + ["<EOS>"]
            else:
                tokens = tokens + ["<PAD>"] * (max_len - len(tokens))
        indices = [self.token2idx.get(token, self.token2idx["<UNK>"]) for token in tokens]
        return indices

    def decode(self, indices):
        tokens = [self.idx2token.get(idx, "<UNK>") for idx in indices]
        # Remove special tokens
        tokens = [t for t in tokens if t not in ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]]
        return "".join(tokens)

    def vocab_size(self):
        return len(self.vocab)

# Dataset class for chemistry data
class MoleculeDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_len=128):
        self.data = pd.read_csv(data_file)
        self.tokenizer = tokenizer
        self.max_len = max_len

        # Data validation and cleaning
        valid_smiles = []
        for i, row in tqdm(self.data.iterrows(), total=len(self.data), desc="Validating SMILES"):
            smiles = row['SMILES']
            if Chem.MolFromSmiles(smiles) is not None:  # Validate with RDKit
                valid_smiles.append(smiles)

        self.valid_smiles = valid_smiles
        print(f"Loaded {len(self.valid_smiles)} valid SMILES strings out of {len(self.data)}")

    def __len__(self):
        return len(self.valid_smiles)

    def __getitem__(self, idx):
        smiles = self.valid_smiles[idx]
        encoded = self.tokenizer.encode(smiles, self.max_len)
        x = torch.tensor(encoded[:-1], dtype=torch.long)  # Input: all tokens except last
        y = torch.tensor(encoded[1:], dtype=torch.long)   # Target: all tokens except first
        return x, y

# Training function with optimizations
def train(model, dataloader, optimizer, criterion, clip_grad=1.0):
    model.train()
    total_loss = 0

    for x, y in tqdm(dataloader, desc="Training"):
        x, y = x.to(device), y.to(device)

        # Create padding mask - for efficiency
        padding_mask = (x == 0)

        optimizer.zero_grad()
        output = model(x, src_key_padding_mask=padding_mask)

        # Reshape for loss calculation
        output = output.reshape(-1, output.shape[-1])
        y = y.reshape(-1)

        loss = criterion(output, y)
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)

# Validation function
def validate(model, dataloader, criterion):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Validating"):
            x, y = x.to(device), y.to(device)
            padding_mask = (x == 0)

            output = model(x, src_key_padding_mask=padding_mask)
            output = output.reshape(-1, output.shape[-1])
            y = y.reshape(-1)

            loss = criterion(output, y)
            total_loss += loss.item()

    return total_loss / len(dataloader)

# Function to generate a molecule from the model
def generate_molecule(model, tokenizer, max_len=100, temperature=1.0):
    model.eval()
    with torch.no_grad():
        # Start with SOS token
        current_tokens = [tokenizer.token2idx["<SOS>"]]

        # Generate tokens one by one
        for _ in range(max_len):
            # Convert current sequence to tensor
            x = torch.tensor([current_tokens], dtype=torch.long).to(device)

            # Get model prediction for next token
            output = model(x)
            next_token_logits = output[0, -1, :] / temperature

            # Apply softmax to get probabilities
            probabilities = F.softmax(next_token_logits, dim=0)

            # Sample from distribution
            next_token = torch.multinomial(probabilities, 1).item()

            # If EOS token is generated, end generation
            if next_token == tokenizer.token2idx["<EOS>"]:
                break

            current_tokens.append(next_token)

        # Decode the generated tokens
        generated_smiles = tokenizer.decode(current_tokens)

        # Validate with RDKit
        mol = Chem.MolFromSmiles(generated_smiles)
        if mol is not None:
            return generated_smiles, True
        else:
            return generated_smiles, False

def main():
    parser = argparse.ArgumentParser(description="Train a chemistry transformer model")
    parser.add_argument("--data_file", type=str, required=True, help="Path to CSV file with SMILES data")
    parser.add_argument("--output_dir", type=str, default="model_output", help="Directory to save model checkpoints")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--max_len", type=int, default=128, help="Maximum length of SMILES strings")
    parser.add_argument("--d_model", type=int, default=128, help="Model dimension")
    parser.add_argument("--nhead", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize tokenizer
    tokenizer = SMILESTokenizer()

    # Load and prepare dataset
    dataset = MoleculeDataset(args.data_file, tokenizer, max_len=args.max_len)

    # Split dataset
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create data loaders with pinned memory for faster data transfer to GPU
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Initialize model
    model = ChemistryTransformer(
        vocab_size=tokenizer.vocab_size(),
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)

    # Initialize optimizer with weight decay for regularization
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # Loss function with label smoothing for better generalization
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        start_time = time.time()

        # Train
        train_loss = train(model, train_loader, optimizer, criterion)

        # Validate
        val_loss = validate(model, val_loader, criterion)

        # Update learning rate
        scheduler.step(val_loss)

        # Save model if it's the best so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(args.output_dir, 'best_model.pt'))

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pt'))

        # Generate sample molecules
        for _ in range(5):
            smiles, valid = generate_molecule(model, tokenizer, temperature=0.8)
            print(f"Generated SMILES: {smiles} | Valid: {valid}")

        # Print epoch statistics
        elapsed_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Time: {elapsed_time:.2f}s | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")

    print("Training complete!")

if __name__ == "__main__":
    main()
