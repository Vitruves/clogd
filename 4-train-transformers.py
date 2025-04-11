#!/usr/bin/env python3
"""
SMILES Transformer training script optimized for large datasets.

Example usage for fixed optimization:
python scripts/4-train-transformers.py --train data/preprocessed/train.csv --eval data/preprocessed/test.csv --output-dir data/models --target-col LogP --batch-size 16 --learning-rate 5e-5 --num-epochs 10 --optimize-memory --fp16

For large datasets and memory optimization:
python scripts/4-train-transformers.py --train data/large/train.csv --eval data/large/test.csv --output-dir data/models --batch-size 8 --gradient-accumulation 4 --optimize-memory --streaming --chunk-size 5000 --tokenizing-batch-size 100 --gradient-checkpointing --dataloader-workers 4

For Apple Silicon and compatibility issues:
python scripts/4-train-transformers.py --train data/preprocessed/train.csv --eval data/preprocessed/test.csv --output-dir data/models --batch-size 32 --use-mps --no-multiprocessing

For troubleshooting pickling issues:
python scripts/4-train-transformers.py --train data/preprocessed/train.csv --eval data/preprocessed/test.csv --output-dir data/models --batch-size 16 --dataloader-workers 0 --no-multiprocessing

For BERT models:
python scripts/4-train-transformers.py --train data/preprocessed/train.csv --eval data/preprocessed/test.csv --output-dir data/models --base-model bert-base-uncased --batch-size 16
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
from datetime import datetime
from datasets import Dataset, concatenate_datasets
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
    PreTrainedModel,
    BertPreTrainedModel,
    RobertaPreTrainedModel,
    AutoConfig,
    TrainerCallback
)
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
import warnings
import shutil
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, LinearLR
import matplotlib.pyplot as plt
import matplotlib
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union
import gc
import traceback
import re
import signal

matplotlib.use('Agg') # Use non-interactive backend

# Attempt RDKit import, handle if not available
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("-- Warning: RDKit not found. --descriptors and --augmentation features will be unavailable.")

warnings.filterwarnings("ignore", message="Some weights of .* were not initialized")
warnings.filterwarnings("ignore", message="You should probably TRAIN this model")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="torch.utils._pytree._register_pytree_node is deprecated") # Ignore MPS specific warning

os.environ["TOKENIZERS_PARALLELISM"] = "false"
TQDM_DISABLE = os.environ.get("TQDM_DISABLE", "0") == "1" # Allow disabling tqdm via env var

# --- Custom Model Definition specifically for fingerprint embeddings ---
class SmilesFingerPrintModel(PreTrainedModel):
    _supports_gradient_checkpointing = True

    def __init__(self, config, num_fingerprint_features, dropout_prob=0.1):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.num_fingerprint_features = num_fingerprint_features
        self.config = config

        # Use only fingerprint features - no transformer
        self.dropout = nn.Dropout(dropout_prob)
        
        # Simple MLP for fingerprint-only prediction
        self.fingerprint_layers = nn.Sequential(
            nn.Linear(self.num_fingerprint_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(128, self.num_labels)
        )
        
        # Initialize weights
        self.post_init()

    def forward(
        self,
        input_ids=None,  # Not used but kept for API compatibility
        attention_mask=None,  # Not used but kept for API compatibility
        descriptors=None,  # This will contain our fingerprints
        labels=None,
        return_dict=None,
        **kwargs  # Accept other unused args
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if descriptors is None:
            raise ValueError("`descriptors` input is required for this fingerprint-only model.")

        # Ensure descriptors are float
        descriptors = descriptors.float()

        # Apply MLP to fingerprints
        logits = self.fingerprint_layers(descriptors)

        loss = None
        if labels is not None:
            # Ensure labels are float and match logits shape
            labels = labels.float().view(-1, self.num_labels)
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )

# --- Custom Model Definition ---
class SmilesTransformerWithDescriptors(PreTrainedModel):
    _supports_gradient_checkpointing = True

    def __init__(self, config, num_descriptors, dropout_prob=0.1):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.num_descriptors = num_descriptors
        self.config = config

        # Load the base transformer model body
        self.transformer = AutoModel.from_config(config)

        # Define the regression head
        classifier_dropout = (
            config.classifier_dropout if hasattr(config, 'classifier_dropout') and config.classifier_dropout is not None
            else config.hidden_dropout_prob # Fallback to general dropout
        )
        self.dropout = nn.Dropout(classifier_dropout if dropout_prob is None else dropout_prob)

        # Calculate input size for the final layer
        transformer_output_dim = config.hidden_size
        combined_features_dim = transformer_output_dim + self.num_descriptors

        # Simple linear head for regression
        self.regressor = nn.Linear(combined_features_dim, self.num_labels)

        # Initialize weights
        self.post_init() # Important for PreTrainedModel


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None, # Needed for some models like BERT
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        descriptors=None, # New input
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if descriptors is None:
            raise ValueError("`descriptors` input is required for this model.")
        if descriptors.shape[1] != self.num_descriptors:
            raise ValueError(f"Expected {self.num_descriptors} descriptor features, but got {descriptors.shape[1]}")


        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Extract sequence embedding - common practice is to use the [CLS] token's output
        # For RoBERTa-like models (ChemBERTa), the first token embedding is usually used.
        sequence_output = transformer_outputs[0] # last_hidden_state
        transformer_embedding = sequence_output[:, 0, :] # Use embedding of the first token ([CLS] or <s>)

        # Ensure descriptors are float
        descriptors = descriptors.float()

        # Concatenate transformer embedding and descriptors
        combined_features = torch.cat((transformer_embedding, descriptors), dim=1)

        # Apply dropout and pass through regressor
        combined_features = self.dropout(combined_features)
        logits = self.regressor(combined_features)

        loss = None
        if labels is not None:
            # Ensure labels are float and match logits shape
            labels = labels.float().view(-1, self.num_labels)
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + transformer_outputs[2:] # Adjust based on base model's output tuple
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

# --- Custom Data Collator ---
@dataclass
class RobustDataCollatorForSmilesAndDescriptors:
    tokenizer: AutoTokenizer
    descriptor_cols: List[str]

    def __call__(self, features):
        # Create a batch dictionary to return
        batch = {}
        
        # Collect all tokenizer inputs
        token_features = {k: [] for k in self.tokenizer.model_input_names}
        labels = []
        desc_features = []

        for f in features:
            # Extract tokenizer features
            for k in self.tokenizer.model_input_names:
                if k in f:
                    # Ensure we have proper int values, not nested structures
                    if isinstance(f[k], list):
                        token_features[k].append(f[k])
                    else:
                        # Handle unexpected data types gracefully
                        print(f"-- Warning: Unexpected data type for {k}: {type(f[k])}")
                        token_features[k].append([0])  # Fallback
            
            # Extract labels 
            if 'labels' in f:
                if isinstance(f['labels'], (int, float)):
                    labels.append(f['labels'])
                else:
                    # Try to convert to float if possible
                    try:
                        labels.append(float(f['labels']))
                    except (ValueError, TypeError):
                        print(f"-- Warning: Could not convert label to float: {f['labels']}")
                        labels.append(0.0)  # Fallback

            # Extract descriptors
            if self.descriptor_cols:
                desc_values = []
                for k in self.descriptor_cols:
                    if k in f:
                        if isinstance(f[k], (int, float)):
                            desc_values.append(float(f[k]))
                        else:
                            # Try to convert to float
                            try:
                                desc_values.append(float(f[k]))
                            except (ValueError, TypeError):
                                print(f"-- Warning: Could not convert descriptor {k} to float: {f[k]}")
                                desc_values.append(0.0)  # Fallback
                    else:
                        # Missing descriptor column
                        desc_values.append(0.0)  # Fallback for missing
                
                desc_features.append(desc_values)

        # Convert token features to tensors
        for k, v in token_features.items():
            if v:  # Only process non-empty lists
                try:
                    batch[k] = torch.tensor(v)
                except Exception as e:
                    print(f"-- Error converting {k} to tensor: {e}")
                    print(f"-- Sample data: {v[:2] if len(v) > 1 else v}")
                    # Create a compatible tensor with zeros as fallback
                    batch[k] = torch.zeros((len(v), 1), dtype=torch.long)

        # Stack descriptor features into a tensor
        if self.descriptor_cols and desc_features:
             try:
                 batch['descriptors'] = torch.tensor(desc_features, dtype=torch.float)
             except Exception as e:
                 print(f"-- Error converting descriptors to tensor: {e}")
                 batch['descriptors'] = torch.zeros((len(desc_features), len(self.descriptor_cols)), dtype=torch.float)

        # Ensure labels are present if they were in the input features
        if labels:
             try:
                 batch['labels'] = torch.tensor(labels, dtype=torch.float).view(-1, 1)
             except Exception as e:
                 print(f"-- Error converting labels to tensor: {e}")
                 batch['labels'] = torch.zeros((len(labels), 1), dtype=torch.float)

        return batch

class StreamingCSVDataset(torch.utils.data.Dataset):
    """Memory-efficient dataset that loads CSV data in chunks."""
    
    def __init__(self, file_path, tokenizer=None, chunksize=10000, 
                smiles_col="SMILES", target_col="LogP", descriptor_cols=None,
                max_length=512, cache_tokenization=True):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.chunksize = chunksize
        self.smiles_col = smiles_col
        self.target_col = target_col
        self.descriptor_cols = descriptor_cols or []
        self.max_length = max_length
        self.cache_tokenization = cache_tokenization
        
        # Get total length and column dtypes
        with pd.read_csv(file_path, chunksize=1) as reader:
            chunk = next(reader)
            self.dtypes = {col: chunk[col].dtype for col in chunk.columns}
        
        self.length = sum(1 for _ in open(file_path)) - 1  # Subtract header
        
        # Cache for chunks
        self.current_chunk = None
        self.current_chunk_start = -1
        self.current_chunk_end = -1
        
        # Cache for tokenized results if enabled
        self.tokenization_cache = {} if cache_tokenization else None
        
        print(f"-- Initialized streaming dataset with {self.length} examples")
        print(f"-- Using columns: SMILES={smiles_col}, target={target_col}, descriptors={descriptor_cols}")
    
    def __len__(self):
        return self.length
    
    def _get_chunk_for_idx(self, idx):
        # Check if we have the chunk containing idx
        if self.current_chunk is None or not (self.current_chunk_start <= idx < self.current_chunk_end):
            # Calculate chunk bounds
            chunk_idx = idx // self.chunksize
            self.current_chunk_start = chunk_idx * self.chunksize
            self.current_chunk_end = min(self.current_chunk_start + self.chunksize, self.length)
            
            # Skip header if first chunk, otherwise skip to appropriate line
            skiprows = 1 if self.current_chunk_start == 0 else self.current_chunk_start + 1
            
            # Read chunk
            self.current_chunk = pd.read_csv(
                self.file_path,
                skiprows=skiprows,
                nrows=self.chunksize,
                dtype=self.dtypes
            )
            
            if self.current_chunk_start > 0:
                # Reset index for chunks after the first one
                self.current_chunk.reset_index(drop=True, inplace=True)
        
        # Return row from current chunk
        return self.current_chunk.iloc[idx - self.current_chunk_start]
    
    def __getitem__(self, idx):
        # Check tokenization cache first if enabled
        if self.tokenization_cache is not None and idx in self.tokenization_cache:
            return self.tokenization_cache[idx]
        
        # Get the row data
        row = self._get_chunk_for_idx(idx)
        
        # Create item dict
        item = {}
        
        # Get SMILES and tokenize if tokenizer is provided
        if self.smiles_col in row:
            smiles = row[self.smiles_col]
            if self.tokenizer is not None:
                # Tokenize SMILES
                tokens = self.tokenizer(
                    smiles,
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors=None
                )
                # Add tokenized data to item
                for k, v in tokens.items():
                    item[k] = v
            else:
                item[self.smiles_col] = smiles
        
        # Add target value
        if self.target_col in row:
            item['labels'] = float(row[self.target_col])
        
        # Add descriptor values
        for col in self.descriptor_cols:
            if col in row:
                item[col] = float(row[col])
        
        # Cache result if enabled
        if self.tokenization_cache is not None:
            self.tokenization_cache[idx] = item
        
        return item

# Very simple custom collator that ensures proper tensor conversion
# Moved to module level so it can be pickled for multiprocessing
class SimpleCustomCollator:
    def __init__(self, descriptor_cols=None):
        self.descriptor_cols = descriptor_cols or []
    
    def __call__(self, features):
        batch = {}
        
        # First, handle tokenizer outputs (input_ids, attention_mask, etc.)
        # These should be lists of integers
        input_feature_keys = ['input_ids', 'attention_mask', 'token_type_ids']
        for key in input_feature_keys:
            if key in features[0]:
                batch[key] = torch.tensor([f[key] for f in features], dtype=torch.long)
        
        # Handle labels
        if "labels" in features[0]:
            batch["labels"] = torch.tensor(
                [float(f["labels"]) for f in features], 
                dtype=torch.float
            ).reshape(-1, 1)  # Ensure proper shape
        
        # Handle descriptors if present
        if self.descriptor_cols:
            desc_values = []
            for f in features:
                row_descriptors = []
                for col in self.descriptor_cols:
                    if col in f:
                        try:
                            row_descriptors.append(float(f[col]))
                        except (ValueError, TypeError):
                            row_descriptors.append(0.0)
                    else:
                        row_descriptors.append(0.0)
                desc_values.append(row_descriptors)
            
            batch['descriptors'] = torch.tensor(desc_values, dtype=torch.float)
        
        return batch

def parse_args():
    parser = argparse.ArgumentParser(description="Train Transformer model on molecular property data with advanced features.")
    # --- I/O ---
    parser.add_argument("--train", required=True, help="Training dataset CSV file")
    parser.add_argument("--eval", required=False, help="External evaluation dataset CSV file (optional)")
    parser.add_argument("--output-dir", required=True, help="Directory to save model, results, and logs")
    # --- Data ---
    parser.add_argument("--smiles-col", default="SMILES", help="Column name for SMILES strings")
    parser.add_argument("--target-col", default="LogP", help="Column name for target property values")
    # --- Feature Source (Mutually Exclusive) ---
    feature_group = parser.add_mutually_exclusive_group()
    feature_group.add_argument("--descriptor-file", default=None, help="Optional CSV file with pre-calculated descriptors (requires --smiles-col for merging)")
    feature_group.add_argument("--descriptors", action="store_true", help="Calculate and use RDKit descriptors as features (requires RDKit)")
    feature_group.add_argument("--use-file-descriptors", action="store_true", help="Use all numerical columns from input files (excluding target) as descriptors")
    feature_group.add_argument("--use-concat", metavar="COLNAME", help="Use only concatenated fingerprint column as features (no transformer embedding)")
    # --- LogP to LogD integration ---
    parser.add_argument("--use-logp-info", action="store_true", help="Use LogP and ionizability information from logp-to-logd output")
    parser.add_argument("--logp-col", default="LogP", help="Column name for LogP values from logp-to-logd")
    parser.add_argument("--ionizable-col", default="is_ionizable", help="Column name for ionizability flag from logp-to-logd")
    # --- Model ---
    parser.add_argument("--base-model", default="DeepChem/ChemBERTa-77M-MTR", help="Hugging Face base transformer model name or path")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability for model classification head")
    # --- Training ---
    parser.add_argument("--num-epochs", type=int, default=10, help="Maximum number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Per-device training batch size")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Peak learning rate")
    parser.add_argument("--optimizer", choices=["adamw"], default="adamw", help="Optimizer (currently only AdamW supported)")
    parser.add_argument("--scheduler", choices=["cosine", "onecycle", "linear"], default="linear", help="Learning rate scheduler type")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Ratio of training steps for linear warmup")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay for AdamW optimizer")
    parser.add_argument("--gradient-accumulation", type=int, default=1, help="Number of steps to accumulate gradients before updating")
    parser.add_argument("--gradient-clipping", type=float, default=1.0, help="Max norm for gradient clipping (0 to disable)")
    parser.add_argument("--fp16", action="store_true", help="Use Automatic Mixed Precision (AMP) training (CUDA only)")
    parser.add_argument("--bf16", action="store_true", help="Use BFloat16 training (newer GPUs/TPUs)")
    # --- Validation & Evaluation ---
    parser.add_argument("--k-folds", type=int, default=0, help="Number of folds for cross-validation (0 to disable CV and train on full data)")
    parser.add_argument("--early-stopping", type=int, default=3, help="Patience for early stopping based on eval loss (requires k-folds = 0 and evaluation_strategy='epoch')")
    # --- Features & Augmentation ---
    parser.add_argument("--augmentation", action="store_true", help="Use SMILES augmentation (random SMILES generation, requires RDKit)")
    parser.add_argument("--augmentation-variants", type=int, default=3, help="Max number of SMILES variants per molecule for augmentation")
    # --- Ensembling ---
    parser.add_argument("--ensemble", type=int, default=1, help="Number of independent models to train for ensembling")
    # --- System & Misc ---
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--use-mps", action="store_true", help="Use MPS acceleration on Apple Silicon (overrides CUDA if available)")
    parser.add_argument("--dataloader-workers", type=int, default=None, help="Number of workers for DataLoader (default: 4, or 0 if MPS)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--save-total-limit", type=int, default=2, help="Limit the total amount of checkpoints")
    # --- Memory Optimization ---
    parser.add_argument("--streaming", action="store_true", help="Process dataset in streaming mode (lower memory usage)")
    parser.add_argument("--chunked-processing", action="store_true", help="Process large datasets in chunks")
    parser.add_argument("--chunk-size", type=int, default=10000, help="Number of examples to process in each chunk")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing to save memory")
    parser.add_argument("--tokenizing-batch-size", type=int, default=1000, help="Batch size for tokenization")
    parser.add_argument("--optimize-memory", action="store_true", help="Apply aggressive memory optimizations")
    # --- Performance Options ---
    parser.add_argument("--torch-compile", action="store_true", help="Use torch.compile for faster training (requires PyTorch 2.0+)")
    parser.add_argument("--tokenizer-parallelism", action="store_true", help="Enable tokenizer parallelism")
    parser.add_argument("--no-multiprocessing", action="store_true", help="Disable multiprocessing")

    args = parser.parse_args()

    if args.descriptors and not RDKIT_AVAILABLE:
        parser.error("--descriptors requires RDKit to be installed.")
    if args.augmentation and not RDKIT_AVAILABLE:
        parser.error("--augmentation requires RDKit to be installed.")
    if args.descriptor_file and not os.path.exists(args.descriptor_file):
        parser.error(f"Descriptor file not found: {args.descriptor_file}")
    if args.fp16 and args.bf16:
        parser.error("Cannot use both --fp16 and --bf16.")
    if args.k_folds > 0:
         warnings.warn("--k-folds > 0 is currently ignored. Training uses the full train set and evaluates on the external set.")
         args.k_folds = 0 # Force disable CV for now

    # Set environment variables based on arguments
    if args.tokenizer_parallelism:
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
    else:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    return args

def generate_smiles_variants(smiles, max_variants=3):
    if not RDKIT_AVAILABLE: return [smiles]
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return [smiles]
    variants = {smiles}
    for _ in range(max_variants * 5): # Try more times to get unique variants
        if len(variants) >= max_variants + 1: break
        try:
            variants.add(Chem.MolToSmiles(mol, doRandom=True))
        except: pass
    return list(variants)

def calculate_molecular_descriptors(smiles):
    if not RDKIT_AVAILABLE: return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return None
        return {
            'Desc_MolWt': Descriptors.MolWt(mol),
            'Desc_LogP_RDKit': Descriptors.MolLogP(mol), # Avoid collision with target
            'Desc_TPSA': Descriptors.TPSA(mol),
            'Desc_NumHDonors': Lipinski.NumHDonors(mol),
            'Desc_NumHAcceptors': Lipinski.NumHAcceptors(mol),
            'Desc_NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
            'Desc_NumAromaticRings': Descriptors.NumAromaticRings(mol),
            'Desc_NumHeavyAtoms': Descriptors.HeavyAtomCount(mol),
            'Desc_FractionCSP3': Descriptors.FractionCSP3(mol),
        }
    except Exception as e:
        print(f"-- Warning: Failed to calculate descriptors for SMILES '{smiles}': {e}", file=sys.stderr)
        return None

def load_and_prepare_data(args):
    descriptor_cols = []
    
    # Handle streaming mode differently
    if args.streaming:
        print(f"-- Using streaming dataset with chunk size {args.chunk_size}")
        
        # Just read a small sample to determine columns and features
        train_sample = pd.read_csv(args.train, nrows=1000)
        
        # Check if eval file is provided
        if args.eval:
            eval_sample = pd.read_csv(args.eval, nrows=1000)
        else:
            print("-- No external evaluation file provided. Will split training data for evaluation.")
            eval_sample = train_sample.copy()
        
        # Determine descriptor columns
        if args.use_logp_info and args.logp_col in train_sample.columns and args.ionizable_col in train_sample.columns:
            descriptor_cols.extend([args.logp_col, args.ionizable_col])
            print(f"-- Using LogP and ionizability as descriptors: {args.logp_col}, {args.ionizable_col}")
            
        elif args.use_file_descriptors:
            # Get numerical columns
            train_num_cols = train_sample.select_dtypes(include=np.number).columns.tolist()
            eval_num_cols = eval_sample.select_dtypes(include=np.number).columns.tolist()
            
            # Remove target column
            try: train_num_cols.remove(args.target_col)
            except ValueError: pass
            try: eval_num_cols.remove(args.target_col)
            except ValueError: pass
            
            # Find common columns
            descriptor_cols = sorted(list(set(train_num_cols) & set(eval_num_cols)))
            if descriptor_cols:
                print(f"-- Using {len(descriptor_cols)} numerical columns as descriptors")
                
        elif args.use_concat:
            # For concatenated fingerprints, we need to inspect the format
            if args.use_concat in train_sample.columns:
                print(f"-- Using concatenated fingerprint column: {args.use_concat}")
                sample = train_sample[args.use_concat].iloc[0]
                
                # Estimate fingerprint length
                if isinstance(sample, str):
                    separator = ',' if ',' in sample else ' '
                    fp_length = len(str(sample).split(separator))
                    print(f"-- Estimated fingerprint length: {fp_length}")
                    descriptor_cols = [f"FP_{i}" for i in range(fp_length)]
                else:
                    print(f"-- Warning: Fingerprint column contains non-string values: {type(sample)}")
            else:
                print(f"-- Warning: Requested fingerprint column {args.use_concat} not found")
                
        elif args.descriptors and RDKIT_AVAILABLE:
            # For RDKit descriptors, compute sample descriptors to get column names
            print("-- Using RDKit molecular descriptors")
            sample_smiles = train_sample[args.smiles_col].iloc[0]
            sample_desc = calculate_molecular_descriptors(sample_smiles)
            if sample_desc:
                descriptor_cols = list(sample_desc.keys())
                print(f"-- Will calculate {len(descriptor_cols)} RDKit descriptors")
                
        # Create streaming datasets
        print("-- Creating streaming datasets...")
        train_dataset = StreamingCSVDataset(
            args.train, 
            tokenizer=None,  # We'll tokenize later
            chunksize=args.chunk_size,
            smiles_col=args.smiles_col,
            target_col=args.target_col,
            descriptor_cols=descriptor_cols
        )
        
        eval_dataset = StreamingCSVDataset(
            args.eval,
            tokenizer=None,
            chunksize=args.chunk_size,
            smiles_col=args.smiles_col,
            target_col=args.target_col,
            descriptor_cols=descriptor_cols
        )
        
        return train_dataset, eval_dataset, descriptor_cols
        
    # Non-streaming mode - process everything in memory
    print(f"-- Loading training data from {args.train}")
    train_df = pd.read_csv(args.train)
    print(f"-- Training dataset: {len(train_df)} entries")
    
    # Check if eval file is provided
    if args.eval:
        print(f"-- Loading evaluation data from {args.eval}")
        eval_df = pd.read_csv(args.eval)
        print(f"-- Evaluation dataset: {len(eval_df)} entries")
    else:
        print("-- No external evaluation file provided. Will use 10% of training data for evaluation.")
        # Split training data for evaluation (80/20 split by default)
        from sklearn.model_selection import train_test_split
        train_df, eval_df = train_test_split(train_df, test_size=0.1, random_state=args.seed)
        print(f"-- Split training dataset: {len(train_df)} train, {len(eval_df)} eval entries")

    # --- Validate Columns ---
    required_cols = [args.smiles_col, args.target_col]
    
    # Add LogP and ionizability columns if specified
    if args.use_logp_info:
        if args.logp_col in train_df.columns and args.ionizable_col in train_df.columns:
            print(f"-- Found LogP-to-LogD output columns: {args.logp_col}, {args.ionizable_col}")
            required_cols.extend([args.logp_col, args.ionizable_col])
        else:
            print(f"-- Warning: Requested LogP-to-LogD columns not found in training data")
    
    # Add concat fingerprint column if specified
    if args.use_concat:
        required_cols.append(args.use_concat)
    
    for df, name in [(train_df, "training"), (eval_df, "evaluation")]:
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in {name} dataset")
        print(f"-- {name.capitalize()} dataset: {len(df)} entries")
        df.dropna(subset=required_cols, inplace=True)
        print(f"-- {name.capitalize()} dataset after dropping NA in required columns: {len(df)} entries")

    # --- Descriptor Handling ---
    descriptor_cols = []
    
    # Handle LogP and ionizability data if requested
    if args.use_logp_info and args.logp_col in train_df.columns and args.ionizable_col in train_df.columns:
        print("-- Using LogP and ionizability information as descriptors")
        
        # Make sure ionizability is numeric
        for df in [train_df, eval_df]:
            if df[args.ionizable_col].dtype != 'float64' and df[args.ionizable_col].dtype != 'int64':
                df[args.ionizable_col] = df[args.ionizable_col].astype(int)
        
        # Add these columns to descriptor list
        logp_ionizable_cols = [args.logp_col, args.ionizable_col]
        descriptor_cols.extend(logp_ionizable_cols)
        
        # Check for missing values
        for df, name in [(train_df, "training"), (eval_df, "evaluation")]:
            nan_count = df[logp_ionizable_cols].isnull().sum().sum()
            if nan_count > 0:
                print(f"-- Warning: Found {nan_count} NaNs in LogP/ionizability data for {name}. Filling with 0.")
                df[logp_ionizable_cols] = df[logp_ionizable_cols].fillna(0)
                
        print(f"-- Added LogP and ionizability as descriptors")
    
    # Handle concatenated fingerprint
    if args.use_concat:
        print(f"-- Using concatenated fingerprint column: {args.use_concat}")
        # Process the concatenated fingerprint into separate features
        def expand_concat_fingerprint(df, col_name):
            # Check the format - expecting comma-separated or space-separated values
            sample = df[col_name].iloc[0]
            if isinstance(sample, str):  # Ensure it's a string
                if ',' in sample:
                    separator = ','
                else:
                    separator = ' '
                
                # Convert concatenated strings to lists of floats
                fingerprints = df[col_name].apply(lambda x: [float(v) for v in str(x).split(separator) if v.strip()])
            else:
                # If not a string, might be already a list or some other format
                print(f"-- Warning: Fingerprint column contains non-string values: {type(sample)}. Attempting to convert.")
                fingerprints = df[col_name].apply(lambda x: [float(v) for v in str(x).split(',')] if isinstance(x, str) else [float(x)])
            
            # Get consistent length
            max_length = fingerprints.apply(len).max()
            print(f"-- Detected fingerprint length: {max_length}")
            
            # Create a new dataframe with expanded fingerprint columns
            fp_cols = [f"FP_{i}" for i in range(max_length)]
            
            # Convert fingerprints to a DataFrame with padded values
            fp_df = pd.DataFrame(
                [fp + [0.0] * (max_length - len(fp)) for fp in fingerprints],
                index=df.index,
                columns=fp_cols
            )
            
            return fp_df, fp_cols
        
        try:
            # Process fingerprints for both datasets
            train_fp_df, fp_cols = expand_concat_fingerprint(train_df, args.use_concat)
            eval_fp_df, _ = expand_concat_fingerprint(eval_df, args.use_concat)
            
            # Combine with original dataframes
            train_df = pd.concat([train_df, train_fp_df], axis=1)
            eval_df = pd.concat([eval_df, eval_fp_df], axis=1)
            
            # Use these columns as descriptors
            descriptor_cols = fp_cols
            print(f"-- Expanded concatenated fingerprint into {len(descriptor_cols)} individual features")
        except Exception as e:
            print(f"-- Error processing concatenated fingerprint: {e}")
            print("-- Proceeding without fingerprint features")
            descriptor_cols = []

    elif args.use_file_descriptors:
        print("-- Using numerical columns from input files as descriptors")
        # Identify numerical columns excluding target and smiles
        train_num_cols = train_df.select_dtypes(include=np.number).columns.tolist()
        eval_num_cols = eval_df.select_dtypes(include=np.number).columns.tolist()

        # Exclude target column
        try: train_num_cols.remove(args.target_col)
        except ValueError: pass
        try: eval_num_cols.remove(args.target_col)
        except ValueError: pass

        # Find common numerical columns
        descriptor_cols = sorted(list(set(train_num_cols) & set(eval_num_cols)))

        if not descriptor_cols:
            print("-- Warning: No common numerical columns found (excluding target) to use as descriptors.")
        else:
            print(f"-- Found {len(descriptor_cols)} common numerical columns to use as descriptors: {', '.join(descriptor_cols)}")
            # Check for missing values in descriptor columns
            for df, name in [(train_df, "training"), (eval_df, "evaluation")]:
                nan_count = df[descriptor_cols].isnull().sum().sum()
                if nan_count > 0:
                    print(f"-- Warning: Found {nan_count} NaNs in descriptor columns for {name} data. Filling with 0.")
                    df[descriptor_cols] = df[descriptor_cols].fillna(0)


    elif args.descriptor_file:
        print(f"-- Loading descriptors from {args.descriptor_file}")
        desc_df = pd.read_csv(args.descriptor_file)
        if args.smiles_col not in desc_df.columns:
             raise ValueError(f"--smiles-col '{args.smiles_col}' not found in descriptor file {args.descriptor_file}")
        original_desc_cols = [c for c in desc_df.columns if c != args.smiles_col]
        desc_df.rename(columns={c: f"DescFile_{c}" for c in original_desc_cols}, inplace=True) # Prefix to avoid name collisions
        descriptor_cols = [c for c in desc_df.columns if c != args.smiles_col]
        print(f"-- Merging {len(descriptor_cols)} descriptors into datasets")
        train_df = pd.merge(train_df, desc_df, on=args.smiles_col, how='left')
        eval_df = pd.merge(eval_df, desc_df, on=args.smiles_col, how='left')
        # Handle NaNs after merge
        for df, name in [(train_df, "training"), (eval_df, "evaluation")]:
             nan_count = df[descriptor_cols].isnull().sum().sum()
             if nan_count > 0:
                 print(f"-- Warning: Found {nan_count} NaNs after merging descriptors for {name} data. Filling with 0.")
                 df[descriptor_cols] = df[descriptor_cols].fillna(0)


    elif args.descriptors:
        if not RDKIT_AVAILABLE:
             raise RuntimeError("--descriptors requires RDKit, but it's not installed.")
        print("-- Calculating RDKit molecular descriptors")
        def add_rdkit_descriptors(df, smiles_col):
            desc_list = df[smiles_col].apply(calculate_molecular_descriptors)
            desc_df = pd.DataFrame(desc_list.tolist(), index=df.index)
            non_null_idx = desc_df.notnull().all(axis=1)
            if (~non_null_idx).sum() > 0:
                 print(f"-- Warning: Failed to compute descriptors for {(~non_null_idx).sum()} SMILES. Rows might be dropped or contain NaNs.")
            # Fill NaNs resulting from calculation errors
            desc_df = desc_df.fillna(0)
            return pd.concat([df, desc_df], axis=1), list(desc_df.columns)

        train_df, descriptor_cols = add_rdkit_descriptors(train_df, args.smiles_col)
        eval_df, _ = add_rdkit_descriptors(eval_df, args.smiles_col) # Use same columns
        print(f"-- Added {len(descriptor_cols)} RDKit descriptors.")
        # Ensure eval_df has the same descriptor columns even if some SMILES failed
        for col in descriptor_cols:
            if col not in eval_df.columns:
                eval_df[col] = 0 # Add missing column filled with 0


    # --- SMILES Augmentation ---
    if args.augmentation:
        if not RDKIT_AVAILABLE:
            raise RuntimeError("--augmentation requires RDKit, but it's not installed.")
        print(f"-- Applying SMILES augmentation (max {args.augmentation_variants} variants)")
        augmented_rows = []
        original_len = len(train_df)
        # Keep track of original index if needed, though augmentation replicates descriptor rows too
        for _, row in train_df.iterrows():
            smiles = row[args.smiles_col]
            variants = generate_smiles_variants(smiles, args.augmentation_variants)
            for variant in variants:
                new_row = row.copy()
                new_row[args.smiles_col] = variant
                augmented_rows.append(new_row)
        train_df = pd.DataFrame(augmented_rows).reset_index(drop=True)
        print(f"-- Augmented training dataset from {original_len} to {len(train_df)} entries")

    # --- Final Dataset Creation ---
    # Ensure descriptor columns are float type before creating Dataset
    for col in descriptor_cols:
         train_df[col] = pd.to_numeric(train_df[col], errors='coerce').fillna(0)
         eval_df[col] = pd.to_numeric(eval_df[col], errors='coerce').fillna(0)

    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)

    return train_dataset, eval_dataset, descriptor_cols


def plot_results(history, eval_results, output_dir):
    epochs = range(1, len(history.get('eval_loss', [])) + 1)
    if not epochs:
        print("-- No evaluation history found, skipping plotting.")
        return

    plt.figure(figsize=(15, 10))

    # --- Loss ---
    plt.subplot(2, 3, 1)
    if 'loss' in history: plt.plot(history['epoch'], history['loss'], 'b-', label='Training Loss')
    if 'eval_loss' in history: plt.plot(epochs, history['eval_loss'], 'r-', label='Validation Loss')
    plt.title('Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # --- Metrics ---
    metrics_to_plot = ['rmse', 'mae', 'r2']
    for i, metric in enumerate(metrics_to_plot):
        plt.subplot(2, 3, i + 2)
        if f'eval_{metric}' in history:
            plt.plot(epochs, history[f'eval_{metric}'], 'g-')
            plt.title(f'Validation {metric.upper()}')
            plt.xlabel("Epoch")
            plt.ylabel(metric.upper())
            plt.grid(True)
            # Add final eval score text
            final_val = eval_results.get(f'eval_{metric}', float('nan'))
            plt.text(0.95, 0.95, f'Final: {final_val:.4f}', transform=plt.gca().transAxes,
                     ha='right', va='top', bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.5))


    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
    plt.suptitle('Training and Validation Metrics', fontsize=16)
    plot_path = os.path.join(output_dir, 'training_metrics.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"-- Training metrics plot saved to {plot_path}")

def plot_predictions(true_values, predicted_values, output_dir, prefix="eval"):
    plt.figure(figsize=(8, 8))
    min_val = min(true_values.min(), predicted_values.min())
    max_val = max(true_values.max(), predicted_values.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')
    plt.scatter(true_values, predicted_values, alpha=0.5, s=10)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title(f"{prefix.capitalize()} Set: True vs Predicted")
    plt.grid(True)
    plt.axis('equal') # Ensure aspect ratio is 1
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plot_path = os.path.join(output_dir, f'{prefix}_predictions.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"-- {prefix.capitalize()} prediction plot saved to {plot_path}")

def set_random_seed(seed):
    """Set random seed for reproducibility across all libraries."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
def validate_dataset(dataset, prefix=""):
    """Validate a dataset before training to catch issues early."""
    print(f"-- Validating {prefix} dataset")
    
    # Check dataset length
    if len(dataset) == 0:
        print(f"-- Warning: Empty {prefix} dataset!")
        return False
        
    # Check for required fields
    first_item = dataset[0]
    print(f"-- First item keys: {list(first_item.keys())}")
    
    # Check for required fields for training
    required_fields = ['input_ids', 'attention_mask', 'labels']
    missing_fields = [field for field in required_fields if field not in first_item]
    if missing_fields:
        print(f"-- Warning: {prefix} dataset missing required fields: {missing_fields}")
        return False
        
    # Check data types
    try:
        if 'input_ids' in first_item:
            if not isinstance(first_item['input_ids'], list) and not torch.is_tensor(first_item['input_ids']):
                print(f"-- Warning: 'input_ids' has unexpected type: {type(first_item['input_ids'])}")
                
        if 'labels' in first_item:
            # Try to convert to float
            try:
                float(first_item['labels'])
            except (ValueError, TypeError):
                print(f"-- Warning: 'labels' cannot be converted to float: {first_item['labels']}")
                
    except Exception as e:
        print(f"-- Error validating dataset: {e}")
        return False
        
    print(f"-- {prefix} dataset passed basic validation")
    return True

# Add signal handler for graceful shutdown
def signal_handler(sig, frame):
    print("-- Gracefully shutting down (Ctrl+C pressed)")
    print("-- Please wait for current batch to complete...")
    # Set global flag to stop training
    global _interrupt_training
    _interrupt_training = True
    # Don't exit immediately, let the trainer handle it

# Global flag to check in trainer
_interrupt_training = False

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)

# Add a custom callback class
class GracefulInterruptCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if _interrupt_training:
            print("-- Stopping training due to interrupt signal")
            control.should_training_stop = True
        return control

def main():
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"{args.target_col}_run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"-- Run directory created: {run_dir}")

    # Add LogP source info to config
    config_dict = vars(args)
    config_dict['descriptor_source'] = 'none'
    if args.use_file_descriptors: config_dict['descriptor_source'] = 'input_file_numerical_columns'
    elif args.descriptor_file: config_dict['descriptor_source'] = 'external_file'
    elif args.descriptors: config_dict['descriptor_source'] = 'rdkit_calculated'
    elif args.use_concat: config_dict['descriptor_source'] = 'concatenated_fingerprint'
    elif args.use_logp_info: config_dict['descriptor_source'] = 'logp_to_logd'
    
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        # Convert Path objects or other non-serializable types if necessary
        json.dump(config_dict, f, indent=2, default=str)

    
    # --- Device Setup ---
    device = "cpu"
    fp16_enabled = args.fp16
    bf16_enabled = args.bf16

    if args.use_mps and torch.backends.mps.is_available():
        print("-- Using MPS acceleration for training (Apple Silicon)")
        device = torch.device("mps")
        if fp16_enabled: print("-- Warning: --fp16 specified with --use-mps, AMP support may vary.")
        if bf16_enabled: print("-- Warning: --bf16 specified with --use-mps, BFloat16 support may vary.")
    elif torch.cuda.is_available():
        print(f"-- Using CUDA acceleration for training (Device: {torch.cuda.get_device_name(0)})")
        device = torch.device("cuda")
        if not fp16_enabled and not bf16_enabled: print("-- Consider using --fp16 or --bf16 for potential speedup on CUDA.")
    else:
        print("-- Using CPU for training")
        if fp16_enabled: print("-- Warning: --fp16 specified but no CUDA device found, disabling AMP.")
        if bf16_enabled: print("-- Warning: --bf16 specified but no CUDA/compatible device found, disabling BFloat16.")
        fp16_enabled = False
        bf16_enabled = False

    # Set number of DataLoader workers
    num_workers = args.dataloader_workers
    if args.no_multiprocessing:
        print("-- Multiprocessing disabled. Using 0 dataloader workers.")
        num_workers = 0
    elif num_workers is None:
        # Default values if not specified
        num_workers = 0 if device.type == 'mps' else 4
        
    print(f"-- Using {num_workers} dataloader workers.")


    # --- Data Loading & Preprocessing ---
    train_dataset, eval_dataset, descriptor_cols = load_and_prepare_data(args)
    use_custom_model = bool(descriptor_cols) # Use custom model if descriptors are present

    # --- Tokenizer ---
    print(f"-- Loading tokenizer for base model: {args.base_model}")
    
    # Check if this is a BERT model
    is_bert_model = bool(re.search(r'bert', args.base_model.lower()))
    if is_bert_model:
        print(f"-- Detected BERT model: {args.base_model}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        
        # Fix for BERT tokenizers
        if is_bert_model and not tokenizer.pad_token:
            print("-- Setting [PAD] token for BERT tokenizer")
            tokenizer.pad_token = tokenizer.eos_token
            
    except Exception as e:
        print(f"-- Error loading tokenizer: {e}")
        print("-- Attempting to load with trust_remote_code=True")
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    
    # MPS-specific optimizations
    if args.use_mps:
        print("-- Applying MPS-specific optimizations")
        # Apple Silicon optimization - reduce batch size if too large
        if args.batch_size > 64 and not args.gradient_accumulation > 1:
            original_batch = args.batch_size
            # Keep total batch size the same by increasing gradient accumulation
            gradient_accum = max(args.gradient_accumulation, original_batch // 64)
            print(f"-- Warning: Large batch size ({original_batch}) on MPS may cause memory issues")
            print(f"-- Consider using --batch-size 64 --gradient-accumulation {gradient_accum}")
    
    # Handle BERT tokenization specifics if needed
    if is_bert_model and not tokenizer.pad_token:
        print("-- Setting [PAD] token for BERT tokenizer")
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        # Very explicitly handle tokenization with clear settings
        tokens = tokenizer(
            examples[args.smiles_col],
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors=None
        )
        result = {}
        for key, value in tokens.items():
            result[key] = value
        
        # Add labels 
        if args.target_col in examples:
            result["labels"] = examples[args.target_col]
        
        # Add descriptor columns if needed
        if descriptor_cols:
            for col in descriptor_cols:
                if col in examples:
                    result[col] = examples[col]
        
        return result

    # Use this simple collator instead of the more complex ones
    print("-- Using simple custom collator")
    data_collator = SimpleCustomCollator(descriptor_cols=descriptor_cols)

    # Make sure to completely remove the SMILES column after tokenization
    print("-- Tokenizing datasets with explicit column removal...")
    columns_to_remove = [args.smiles_col]  # Always remove SMILES column after tokenization

    try:
        tokenized_train = train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=columns_to_remove,
            desc="Tokenizing training dataset"
        )
        tokenized_eval = eval_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=columns_to_remove,
            desc="Tokenizing evaluation dataset"
        )
    except Exception as e:
        print(f"-- Error during batch tokenization: {e}")
        print("-- Falling back to per-example tokenization")
        
        # Process each example individually to avoid batch issues
        def process_individual_examples(dataset):
            processed_data = []
            for i in range(len(dataset)):
                example = dataset[i]
                if isinstance(example, dict):
                    example_dict = example
                else:
                    example_dict = {args.smiles_col: example[args.smiles_col]}
                    if args.target_col in example:
                        example_dict[args.target_col] = example[args.target_col]
                    for col in descriptor_cols:
                        if col in example:
                            example_dict[col] = example[col]
                
                # Apply tokenization
                tokens = tokenizer(
                    example_dict[args.smiles_col],
                    padding='max_length',
                    truncation=True,
                    max_length=512,
                    return_tensors=None
                )
                
                # Create result
                result = {}
                for k, v in tokens.items():
                    result[k] = v
                
                # Add target
                if args.target_col in example_dict:
                    result["labels"] = float(example_dict[args.target_col])
                
                # Add descriptors
                for col in descriptor_cols:
                    if col in example_dict:
                        try:
                            result[col] = float(example_dict[col])
                        except (ValueError, TypeError):
                            result[col] = 0.0
                
                processed_data.append(result)
            
            return processed_data
        
        train_processed = process_individual_examples(train_dataset)
        eval_processed = process_individual_examples(eval_dataset)
        
        # Convert to dataset objects
        from datasets import Dataset
        tokenized_train = Dataset.from_list(train_processed)
        tokenized_eval = Dataset.from_list(eval_processed)

    # --- Metrics ---
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        # If using custom model, predictions might be nested; extract logits if needed
        # Check the output type of the custom model's forward pass
        # Assuming predictions are the logits tensor here
        predictions = predictions.flatten()
        labels = labels.flatten() # Ensure labels are also flat
        mse = mean_squared_error(labels, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(labels, predictions)
        r2 = r2_score(labels, predictions)
        return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}

    # --- Training Setup ---
    ensemble_predictions = []
    all_eval_results = []
    # training_history = {} # Removed, history processing is done per ensemble member

    
    for ensemble_idx in range(args.ensemble):
        ensemble_run_dir = os.path.join(run_dir, f"ensemble_{ensemble_idx}") if args.ensemble > 1 else run_dir
        os.makedirs(ensemble_run_dir, exist_ok=True)

        if args.ensemble > 1:
            print(f"--- Training Ensemble Model {ensemble_idx + 1}/{args.ensemble} ---")
            current_seed = args.seed + ensemble_idx
            np.random.seed(current_seed)
            torch.manual_seed(current_seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(current_seed)
        else:
            current_seed = args.seed

        # --- Model Initialization ---
        if args.use_concat:
            print(f"-- Initializing fingerprint-only model with {len(descriptor_cols)} features")
            config = AutoConfig.from_pretrained(args.base_model, num_labels=1, problem_type="regression")
            model = SmilesFingerPrintModel(
                config=config,
                num_fingerprint_features=len(descriptor_cols),
                dropout_prob=args.dropout
            )
            print("-- Using fingerprint-only model without transformer embeddings")
            use_custom_model = True
            
        elif use_custom_model:
            print(f"-- Initializing hybrid model with {len(descriptor_cols)} descriptors on top of {args.base_model}")
            # First load the config
            config = AutoConfig.from_pretrained(
                args.base_model, 
                num_labels=1,
                problem_type="regression",
            )
            
            # Enable gradient checkpointing for memory efficiency if requested
            if args.gradient_checkpointing:
                config.gradient_checkpointing = True
                print("-- Gradient checkpointing enabled for memory efficiency")
                
            # Initialize our custom model
            model = SmilesTransformerWithDescriptors(
                config=config,
                num_descriptors=len(descriptor_cols),
                dropout_prob=args.dropout
            )
            
            # Try to load base model weights
            try:
                print(f"-- Loading base transformer weights from {args.base_model}")
                base_model = AutoModel.from_pretrained(args.base_model)
                # Copy weights from base model to transformer part of our model
                missing_keys, unexpected_keys = model.transformer.load_state_dict(
                    base_model.state_dict(), strict=False
                )
                if args.verbose:
                    if missing_keys:
                        print(f"-- Missing keys when loading transformer weights: {missing_keys}")
                    if unexpected_keys:
                        print(f"-- Unexpected keys when loading transformer weights: {unexpected_keys}")
                print("-- Base transformer weights loaded successfully")
                # Explicitly delete base model to free memory
                del base_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"-- Warning: Failed to load base model weights: {e}")
                print("-- Training transformer from scratch")

        else:
            print(f"-- Initializing standard sequence classification model from {args.base_model}")
            try:
                # Configure with dropout and gradient checkpointing
                config = AutoConfig.from_pretrained(
                    args.base_model,
                    num_labels=1,
                    problem_type="regression",
                    hidden_dropout_prob=args.dropout,
                    attention_probs_dropout_prob=args.dropout,
                )
                
                if args.gradient_checkpointing:
                    config.gradient_checkpointing = True
                    print("-- Gradient checkpointing enabled for memory efficiency")
                
                # Initialize the model with the config
                model = AutoModelForSequenceClassification.from_pretrained(
                    args.base_model,
                    config=config,
                    ignore_mismatched_sizes=True
                )
                print("-- Model initialized successfully")
            except Exception as e:
                print(f"-- Error initializing model: {e}")
                print("-- Attempting to initialize model with basic config")
                
                # Fallback to just creating from config
                config = AutoConfig.from_pretrained(
                    args.base_model,
                    num_labels=1,
                    problem_type="regression"
                )
                model = AutoModelForSequenceClassification.from_config(config)
                print("-- Model initialized from config only (no pre-trained weights)")

        # Set fixed seed for all operations in this run
        set_seed = args.seed + ensemble_idx if args.ensemble > 1 else args.seed
        set_random_seed(set_seed) 
        print(f"-- Random seed set to {set_seed}")
        
        # Print model size information
        model_size = sum(p.numel() for p in model.parameters())
        print(f"-- Model size: {model_size:,} parameters")
        trainable_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"-- Trainable parameters: {trainable_size:,} ({trainable_size/model_size:.1%})")
        
        # Enable memory optimization by performing operations in-place where possible
        if args.optimize_memory:
            print("-- Enabling memory optimizations")
            for module in model.modules():
                if isinstance(module, nn.ReLU):
                    module.inplace = True

        # During model initialization, add explicit logging for LogP info
        if args.use_logp_info:
            print(f"-- Incorporating LogP ({args.logp_col}) and ionizability ({args.ionizable_col}) information in the model")

        # Validate datasets before training
        print("-- Performing dataset validation checks before training")
        train_valid = validate_dataset(tokenized_train, prefix="training")
        eval_valid = validate_dataset(tokenized_eval, prefix="evaluation")
        
        if not train_valid or not eval_valid:
            print("-- Warning: Dataset validation failed. Training may encounter errors.")
            print("-- You may want to check your data preprocessing or tokenization steps.")
            if args.verbose:
                print("-- Showing first few examples from training dataset:")
                for i in range(min(3, len(tokenized_train))):
                    print(f"-- Example {i}:")
                    example = tokenized_train[i]
                    for k, v in example.items():
                        if k == 'input_ids' and isinstance(v, list):
                            print(f"--   {k}: (list of {len(v)} items)")
                        else:
                            print(f"--   {k}: {v}")
        
        # --- Training Arguments ---
        # Determine evaluation strategy based on early stopping needs
        if args.early_stopping > 0:
            eval_strategy = "epoch"
            load_best = True
            metric_for_best = "rmse"
            print(f"-- Enabling evaluation per epoch for early stopping (patience={args.early_stopping}).")
        else:
            eval_strategy = "no"
            load_best = False
            metric_for_best = None

        # Create simplified training arguments
        training_args = TrainingArguments(
            output_dir=os.path.join(ensemble_run_dir, "checkpoints"),
            overwrite_output_dir=True,
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size * 2,
            gradient_accumulation_steps=args.gradient_accumulation,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            lr_scheduler_type=args.scheduler,
            warmup_ratio=args.warmup_ratio,
            max_grad_norm=args.gradient_clipping if args.gradient_clipping > 0 else None,
            # Use only eval_strategy parameter, not evaluation_strategy
            eval_strategy=eval_strategy,
            save_strategy=eval_strategy,
            # Other parameters
            load_best_model_at_end=load_best,
            metric_for_best_model=metric_for_best,
            greater_is_better=False if metric_for_best == "rmse" else None,
            save_total_limit=args.save_total_limit,
            logging_dir=os.path.join(ensemble_run_dir, "logs"),
            logging_strategy="steps",
            logging_steps=max(10, min(100, int(len(tokenized_train) / args.batch_size / 10))),
            report_to="none",
            remove_unused_columns=False,  # Important for our custom datasets
            fp16=fp16_enabled,
            bf16=bf16_enabled,
            seed=set_seed,
            dataloader_num_workers=num_workers,
            dataloader_pin_memory=True,
            disable_tqdm=TQDM_DISABLE,
            # Added for optimization
            dataloader_prefetch_factor=2 if num_workers > 0 and hasattr(DataLoader, 'prefetch_factor') else None,
            group_by_length=False,  # Set to True for SMILES of similar length if helpful
            label_smoothing_factor=0.0,
            ddp_find_unused_parameters=False,
            optim="adamw_torch_fused" if torch.cuda.is_available() and hasattr(torch.optim, 'AdamW_fused') else "adamw_torch",
        )

        # --- Callbacks ---
        callbacks = []
        if args.early_stopping > 0:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping))
  
        # Add our graceful interrupt callback
        callbacks.append(GracefulInterruptCallback())
  
        # --- Trainer Initialization ---
        print(f"-- Initializing Trainer with batch size={args.batch_size}, grad_accum={args.gradient_accumulation}")
        print(f"-- Effective batch size: {args.batch_size * args.gradient_accumulation} examples")
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=callbacks
        )

        # --- Training ---
        print("-- Starting training...")
        try:
            train_result = trainer.train()
            print("-- Training successful")
            
            # Save the final (or best if load_best_model_at_end=True) model state
            final_model_dir = os.path.join(ensemble_run_dir, "final_model")
            trainer.save_model(final_model_dir)
            
            # Save tokenizer and config with the model if needed later
            if use_custom_model:
                model.config.save_pretrained(final_model_dir)
                model_info = {'descriptor_cols': descriptor_cols, 'base_model_name': args.base_model}
                with open(os.path.join(final_model_dir, "model_info.json"), "w") as f:
                     json.dump(model_info, f)
            tokenizer.save_pretrained(final_model_dir)
            
            print("-- Training finished")
        except KeyboardInterrupt:
            print("-- Training interrupted by user")
            print("-- Saving current model state...")
            # Try to save the current state
            try:
                final_model_dir = os.path.join(ensemble_run_dir, "interrupted_model")
                trainer.save_model(final_model_dir)
                print(f"-- Interrupted model saved to {final_model_dir}")
            except Exception as save_error:
                print(f"-- Could not save interrupted model: {save_error}")
            # Reraise to exit
            raise
        except Exception as e:
            print(f"-- Error during training: {e}")
            print("-- Traceback:")
            traceback.print_exc()
            
            # Try to diagnose the issue
            print("-- Diagnosing issue...")
            print(f"-- Tokenizer type: {type(tokenizer)}")
            
            # Check data shapes
            try:
                batch = next(iter(DataLoader(tokenized_train, batch_size=2, shuffle=False)))
                print(f"-- Sample batch keys: {list(batch.keys())}")
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        print(f"-- {k} shape: {v.shape}, dtype: {v.dtype}")
                    else:
                        print(f"-- {k} type: {type(v)}")
            except Exception as batch_error:
                print(f"-- Could not create sample batch: {batch_error}")
            
            # Try a minimal training run with a tiny subset
            try:
                print("-- Attempting minimal training run with small subset...")
                tiny_train = tokenized_train.select(range(min(10, len(tokenized_train))))
                tiny_eval = tokenized_eval.select(range(min(10, len(tokenized_eval))))
                
                minimal_trainer = Trainer(
                    model=model,
                    args=TrainingArguments(
                        output_dir=os.path.join(ensemble_run_dir, "debug"),
                        num_train_epochs=1,
                        per_device_train_batch_size=2,
                        per_device_eval_batch_size=2,
                        logging_steps=1,
                        save_strategy="no",
                        # Use eval_strategy for older versions of transformers
                        eval_strategy="no", 
                        report_to="none",
                        # Ensure compatibility with older versions
                        remove_unused_columns=False,
                    ),
                    train_dataset=tiny_train,
                    eval_dataset=tiny_eval,
                    tokenizer=tokenizer,
                    data_collator=data_collator,
                )
                minimal_trainer.train()
                print("-- Minimal training succeeded - issue may be with full dataset")
            except Exception as minimal_error:
                print(f"-- Minimal training also failed: {minimal_error}")
            
            print("-- Attempting training with 0 workers (no multiprocessing)...")
            try:
                # Create new training args with no workers
                no_mp_args = TrainingArguments(
                    output_dir=os.path.join(ensemble_run_dir, "checkpoints_no_mp"),
                    overwrite_output_dir=True,
                    num_train_epochs=args.num_epochs,
                    per_device_train_batch_size=args.batch_size,
                    per_device_eval_batch_size=args.batch_size * 2,
                    gradient_accumulation_steps=args.gradient_accumulation,
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay,
                    lr_scheduler_type=args.scheduler,
                    warmup_ratio=args.warmup_ratio,
                    # Use only the eval_strategy parameter
                    eval_strategy=eval_strategy,
                    save_strategy=eval_strategy,
                    load_best_model_at_end=load_best,
                    metric_for_best_model=metric_for_best,
                    logging_dir=os.path.join(ensemble_run_dir, "logs_no_mp"),
                    logging_strategy="steps",
                    logging_steps=max(10, min(100, int(len(tokenized_train) / args.batch_size / 10))),
                    report_to="none",
                    # Critical settings
                    dataloader_num_workers=0,  # No multiprocessing
                    remove_unused_columns=False,
                    fp16=fp16_enabled,
                    bf16=bf16_enabled,
                )
                
                retry_trainer = Trainer(
                    model=model,
                    args=no_mp_args,
                    train_dataset=tokenized_train,
                    eval_dataset=tokenized_eval,
                    tokenizer=tokenizer,
                    data_collator=data_collator,
                    compute_metrics=compute_metrics,
                    callbacks=callbacks
                )
                
                print("-- Restarting training without multiprocessing...")
                train_result = retry_trainer.train()
                print("-- Training successful with no multiprocessing!")
                
                # Save the model
                final_model_dir = os.path.join(ensemble_run_dir, "final_model")
                retry_trainer.save_model(final_model_dir)
                
                # Save additional files
                if use_custom_model:
                    model.config.save_pretrained(final_model_dir)
                    model_info = {'descriptor_cols': descriptor_cols, 'base_model_name': args.base_model}
                    with open(os.path.join(final_model_dir, "model_info.json"), "w") as f:
                        json.dump(model_info, f)
                tokenizer.save_pretrained(final_model_dir)
                
                # Continue with normal flow after saving model
                trainer = retry_trainer  # Use the successful trainer for further evaluation
                
            except Exception as retry_error:
                print(f"-- Retry training also failed: {retry_error}")
                print("-- Troubleshooting suggestions:")
                print("1. Check SMILES formatting in your dataset")
                print("2. Ensure descriptor columns have proper numerical values")
                print("3. Try reducing batch size")
                print("4. Consider using a simpler model configuration for testing")
                print("5. Try running with --dataloader-workers=0 explicitly")
                
                # Continue with evaluation if possible, otherwise skip to next ensemble model
                print("-- Skipping to next ensemble model due to training error")
                continue

        log_history = trainer.state.log_history
        history_df = pd.DataFrame(log_history)
        history_df.to_csv(os.path.join(ensemble_run_dir, "training_log_history.csv"), index=False)
        eval_logs = [log for log in log_history if 'eval_loss' in log]
        train_logs = [log for log in log_history if 'loss' in log and 'eval_loss' not in log]
        processed_history = {'epoch': [e['epoch'] for e in eval_logs]}
        for key in ['eval_loss', 'eval_rmse', 'eval_mae', 'eval_r2']:
             if eval_logs and key in eval_logs[0]:
                 processed_history[key] = [e[key] for e in eval_logs]
        if train_logs:
             train_loss_per_epoch = {}
             for log in train_logs:
                 epoch = int(round(log['epoch']))
                 if epoch not in train_loss_per_epoch: train_loss_per_epoch[epoch] = []
                 train_loss_per_epoch[epoch].append(log['loss'])
             processed_history['loss'] = [np.mean(train_loss_per_epoch.get(int(round(e)), [np.nan])) for e in processed_history['epoch']]

        print("-- Evaluating final model on external evaluation set...")
        final_eval_results = trainer.evaluate(eval_dataset=tokenized_eval)
        print(f"-- External Eval Results: {final_eval_results}")
        all_eval_results.append(final_eval_results)

        with open(os.path.join(ensemble_run_dir, "external_eval_results.json"), "w") as f:
            json.dump(final_eval_results, f, indent=2)

        print("-- Generating predictions on external evaluation set...")
        predictions = trainer.predict(tokenized_eval)
        preds = predictions.predictions.flatten()
        labels = predictions.label_ids.flatten()

        plot_results(processed_history, final_eval_results, ensemble_run_dir)
        plot_predictions(labels, preds, ensemble_run_dir, prefix="external_eval")

        pred_df = pd.DataFrame({
            args.smiles_col: eval_dataset.to_pandas()[args.smiles_col].tolist() if hasattr(eval_dataset, 'to_pandas') else [],
            'true_value': labels,
            'predicted_value': preds
        })
        pred_df.to_csv(os.path.join(ensemble_run_dir, "external_eval_predictions.csv"), index=False)
        
        if args.ensemble > 1:
            ensemble_predictions.append(preds)
            
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.ensemble > 1:
        print(f"--- Evaluating Ensemble ({args.ensemble} models) ---")
        ensemble_pred_array = np.array(ensemble_predictions)
        mean_preds = np.mean(ensemble_pred_array, axis=0)
        true_labels = labels

        ensemble_metrics = {
            "ensemble_mse": mean_squared_error(true_labels, mean_preds),
            "ensemble_rmse": np.sqrt(mean_squared_error(true_labels, mean_preds)),
            "ensemble_mae": mean_absolute_error(true_labels, mean_preds),
            "ensemble_r2": r2_score(true_labels, mean_preds)
        }
        print(f"-- Ensemble Eval Results: {ensemble_metrics}")
        with open(os.path.join(run_dir, "ensemble_eval_results.json"), "w") as f:
            json.dump(ensemble_metrics, f, indent=2)
        plot_predictions(true_labels, mean_preds, run_dir, prefix="ensemble_eval")
        ensemble_pred_df = pd.DataFrame({
            args.smiles_col: eval_dataset.to_pandas()[args.smiles_col].tolist() if hasattr(eval_dataset, 'to_pandas') else [],
            'true_value': true_labels,
            'predicted_value_ensemble_mean': mean_preds,
        })
        for i, preds_i in enumerate(ensemble_predictions):
            ensemble_pred_df[f'predicted_value_model_{i}'] = preds_i
        ensemble_pred_df.to_csv(os.path.join(run_dir, "ensemble_eval_predictions.csv"), index=False)

    print("--- Run Summary ---")
    print(f"-- Configuration saved to: {os.path.join(run_dir, 'config.json')}")
    if args.ensemble == 1:
        final_model_path = os.path.join(run_dir, 'final_model')
        print(f"-- Final model saved to: {final_model_path}")
        if use_custom_model: print(f"--   (Custom model using {len(descriptor_cols)} descriptors)")
        print(f"-- External evaluation results saved to: {os.path.join(run_dir, 'external_eval_results.json')}")
        print(f"-- External evaluation predictions saved to: {os.path.join(run_dir, 'external_eval_predictions.csv')}")
        print(f"-- Plots saved in: {run_dir}")
    else:
        print(f"-- Individual models saved in: {run_dir}/ensemble_*/final_model")
        if use_custom_model: print(f"--   (Custom models using {len(descriptor_cols)} descriptors)")
        print(f"-- Ensemble evaluation results saved to: {os.path.join(run_dir, 'ensemble_eval_results.json')}")
        print(f"-- Ensemble predictions saved to: {os.path.join(run_dir, 'ensemble_eval_predictions.csv')}")
        print(f"-- Individual and ensemble plots saved in: {run_dir}")

    print("-- Training complete")
    print(f"-- All results and models saved in: {run_dir}")


if __name__ == "__main__":
    main()