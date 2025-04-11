#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as GraphDataLoader
from torch_geometric.nn import GCNConv, GATv2Conv, global_mean_pool, global_add_pool, global_max_pool
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import KFold, train_test_split
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from xgboost import callback as xgb_callback
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.ML.Descriptors import MoleculeDescriptors # type: ignore
from rdkit.Chem import rdMolDescriptors
import shap
from transformers import AutoModel, AutoTokenizer, AutoConfig
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import time
import json
import joblib
from tqdm import tqdm
import matplotlib.pyplot as plt
from itertools import chain
import gc
import warnings
import copy
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple, Union, Optional
import logging
import sys

# Configure RDKit logging to suppress aromaticity warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

class LogDPredictionPipeline:
    def __init__(self, args):
        self.args = args
        
        # Set MPS fallback for compatibility with PyTorch Geometric operations
        import os
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        
        self.output_dir = args.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.stage_output_dirs = {
            "stage1": os.path.join(self.output_dir, "stage1_gbm"),
            "stage2": os.path.join(self.output_dir, "stage2_gnn"),
            "stage3": os.path.join(self.output_dir, "stage3_transformer"),
            "stage4": os.path.join(self.output_dir, "stage4_physics"),
            "stage5": os.path.join(self.output_dir, "stage5_meta"),
            "final": os.path.join(self.output_dir, "final_results"),
        }
        
        for dir_path in self.stage_output_dirs.values():
            os.makedirs(dir_path, exist_ok=True)
            
        # Setup logging before any log calls
        self.setup_logging()
        
        # Device selection logic
        if args.force_cpu:
            self.device = torch.device('cpu')
            self.log("Forcing CPU usage as requested")
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.log(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
            self.log("Using Apple MPS (Metal) device with CPU fallback for unsupported operations")
            # Verify MPS is working properly
            try:
                test_tensor = torch.zeros((2, 2)).to('mps')
                self.log("MPS device verification successful")
            except Exception as e:
                self.device = torch.device('cpu')
                self.log(f"MPS device initialization failed, falling back to CPU: {str(e)}")
        else:
            self.device = torch.device('cpu')
            self.log("Using CPU device (no GPU/MPS available)")
        
        self.stage_models = {}
        self.stage_predictions = {}
        self.stage_embeddings = {}

    def setup_logging(self):
        self.logger = logging.getLogger("LogDPipeline")
        self.logger.setLevel(logging.INFO)
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('-- %(message)s')
        console_handler.setFormatter(formatter)
        
        file_handler = logging.FileHandler(os.path.join(self.output_dir, "pipeline.log"))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

    def log(self, message):
        self.logger.info(message)

    def load_data(self):
        self.log("Loading training data from {}".format(self.args.input))
        self.train_df = pd.read_csv(self.args.input)
        
        self.log("Loading external test data from {}".format(self.args.external_test))
        self.test_df = pd.read_csv(self.args.external_test)
        
        # Log column names to help with debugging
        self.log(f"Input columns: {', '.join(self.train_df.columns.tolist())}")
        
        if self.args.smiles_col in self.train_df.columns:
            self.train_smiles = self.train_df[self.args.smiles_col].tolist()
            self.log(f"Extracted {len(self.train_smiles)} SMILES strings from training data")
        else:
            self.train_smiles = None
            raise ValueError(f"SMILES column '{self.args.smiles_col}' not found in training data")
        
        if self.args.smiles_col in self.test_df.columns:
            self.test_smiles = self.test_df[self.args.smiles_col].tolist()
            self.log(f"Extracted {len(self.test_smiles)} SMILES strings from test data")
        else:
            self.test_smiles = None
            raise ValueError(f"SMILES column '{self.args.smiles_col}' not found in test data")
        
        self.target_col = self.args.target
        self.exclude_cols = [self.args.smiles_col, self.target_col]
        
        if self.args.use_infile_descriptors:
            self.train_features = self.train_df.drop(columns=self.exclude_cols)
            self.test_features = self.test_df.drop(columns=self.exclude_cols)
        else:
            self.log("Calculating molecular descriptors and fingerprints")
            self.train_features, self.test_features = self.calculate_molecular_features()
        
        self.y_train = self.train_df[self.target_col].values
        self.y_test = self.test_df[self.target_col].values
        
        self.log(f"Training features shape: {self.train_features.shape}")
        self.log(f"Test features shape: {self.test_features.shape}")
        
        self.preprocess_data()

    def calculate_molecular_features(self):
        train_descriptors = []
        test_descriptors = []
        train_fingerprints = []
        test_fingerprints = []
        
        desc_calculator = self.setup_rdkit_descriptors()
        
        for smiles in tqdm(self.train_smiles, desc="Calculating training descriptors"):
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                train_descriptors.append(self.calculate_descriptors(mol, desc_calculator))
                train_fingerprints.append(self.calculate_fingerprints(mol))
            else:
                self.log(f"Warning: Could not parse SMILES: {smiles}")
                train_descriptors.append([0] * 200)
                train_fingerprints.append([0] * 2048)
        
        for smiles in tqdm(self.test_smiles, desc="Calculating test descriptors"):
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                test_descriptors.append(self.calculate_descriptors(mol, desc_calculator))
                test_fingerprints.append(self.calculate_fingerprints(mol))
            else:
                self.log(f"Warning: Could not parse SMILES: {smiles}")
                test_descriptors.append([0] * 200)
                test_fingerprints.append([0] * 2048)
        
        train_desc_df = pd.DataFrame(train_descriptors, 
                                    columns=[f"desc_{i}" for i in range(len(train_descriptors[0]))])
        test_desc_df = pd.DataFrame(test_descriptors,
                                   columns=[f"desc_{i}" for i in range(len(test_descriptors[0]))])
        
        train_fp_df = pd.DataFrame(train_fingerprints,
                                  columns=[f"fp_{i}" for i in range(len(train_fingerprints[0]))])
        test_fp_df = pd.DataFrame(test_fingerprints,
                                 columns=[f"fp_{i}" for i in range(len(test_fingerprints[0]))])
        
        train_features = pd.concat([train_desc_df, train_fp_df], axis=1)
        test_features = pd.concat([test_desc_df, test_fp_df], axis=1)
        
        return train_features, test_features

    def setup_rdkit_descriptors(self):
        desc_names = [desc_name[0] for desc_name in Descriptors._descList]
        calculator = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)
        return calculator

    def calculate_descriptors(self, mol, calculator):
        try:
            descriptors = list(calculator.CalcDescriptors(mol))
            return [x if np.isfinite(x) else 0.0 for x in descriptors]
        except:
            return [0.0] * len(calculator.GetDescriptorNames())

    def calculate_fingerprints(self, mol, radius=2, nBits=2048):
        try:
            fp = GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
            return list(fp)
        except:
            return [0] * nBits

    def preprocess_data(self):
        self.log("Preprocessing data")
        
        for column in self.train_features.columns:
            # Replace NaN values
            self.train_features[column] = self.train_features[column].fillna(self.args.nan_value)
            self.test_features[column] = self.test_features[column].fillna(self.args.nan_value)
            
            # Replace zero values if specified
            if self.args.zero_value is not None:
                self.train_features[column] = self.train_features[column].replace(0, self.args.zero_value)
                self.test_features[column] = self.test_features[column].replace(0, self.args.zero_value)
        
        # Split descriptors and fingerprints using the provided prefixes
        desc_prefixes = [prefix.strip() for prefix in self.args.descriptors_prefixes.split(',')]
        fp_prefixes = [prefix.strip() for prefix in self.args.fingerprint_prefixes.split(',')]
        
        self.desc_cols = []
        for prefix in desc_prefixes:
            self.desc_cols.extend([col for col in self.train_features.columns if col.startswith(prefix)])
        
        self.fp_cols = []
        for prefix in fp_prefixes:
            self.fp_cols.extend([col for col in self.train_features.columns if col.startswith(prefix)])
        
        self.log(f"Found {len(self.desc_cols)} descriptor columns with prefixes: {desc_prefixes}")
        self.log(f"Found {len(self.fp_cols)} fingerprint columns with prefixes: {fp_prefixes}")
        
        # If no descriptor or fingerprint columns found, try using all features
        if len(self.desc_cols) == 0 and len(self.fp_cols) == 0:
            self.log("Warning: No descriptor or fingerprint columns found with specified prefixes.")
            self.log("Using all columns as general features.")
            # Exclude SMILES and target columns from general features
            self.desc_cols = [col for col in self.train_features.columns 
                               if col not in [self.args.smiles_col, self.args.target]]
            self.log(f"Using {len(self.desc_cols)} general feature columns")
        
        self.train_descriptors = self.train_features[self.desc_cols] if self.desc_cols else pd.DataFrame()
        self.test_descriptors = self.test_features[self.desc_cols] if self.desc_cols else pd.DataFrame()
        
        self.train_fingerprints = self.train_features[self.fp_cols] if self.fp_cols else pd.DataFrame()
        self.test_fingerprints = self.test_features[self.fp_cols] if self.fp_cols else pd.DataFrame()
        
        # Scale descriptor features
        self.train_descriptors_scaled = None
        self.test_descriptors_scaled = None
        
        if not self.train_descriptors.empty:
            self.desc_scaler = RobustScaler()
            self.train_descriptors_scaled = pd.DataFrame(
                self.desc_scaler.fit_transform(self.train_descriptors),
                columns=self.train_descriptors.columns,
                index=self.train_descriptors.index  # Preserve index for later use
            )
            self.test_descriptors_scaled = pd.DataFrame(
                self.desc_scaler.transform(self.test_descriptors),
                columns=self.test_descriptors.columns,
                index=self.test_descriptors.index  # Preserve index for later use
            )
            
            # Save scaler
            joblib.dump(self.desc_scaler, os.path.join(self.output_dir, "descriptor_scaler.pkl"))
        else:
            self.log("Warning: No descriptor columns found with specified prefixes")
        
        # Create X_train_split and X_valid for cross-validation
        X_train_idx, X_valid_idx = train_test_split(
            np.arange(len(self.train_features)), test_size=0.2, random_state=42
        )
        
        self.X_train_split = self.train_features.iloc[X_train_idx]
        self.X_valid = self.train_features.iloc[X_valid_idx]
        self.y_train_split = self.y_train[X_train_idx]
        self.y_valid = self.y_train[X_valid_idx]
        
        self.train_smiles_split = [self.train_smiles[i] for i in X_train_idx]
        self.valid_smiles = [self.train_smiles[i] for i in X_valid_idx]
        
        # Create molecules for GNN
        self.create_molecular_graphs()

    def create_molecular_graphs(self):
        self.log("Creating molecular graphs for GNN models")
        
        # Create molecular graphs for train, valid, and test sets
        self.train_graphs = []
        self.valid_graphs = []
        self.test_graphs = []
        
        for smiles in tqdm(self.train_smiles_split, desc="Processing training molecules"):
            graph = self.smiles_to_graph(smiles)
            if graph is not None:
                self.train_graphs.append(graph)
        
        for smiles in tqdm(self.valid_smiles, desc="Processing validation molecules"):
            graph = self.smiles_to_graph(smiles)
            if graph is not None:
                self.valid_graphs.append(graph)
        
        for smiles in tqdm(self.test_smiles, desc="Processing test molecules"):
            graph = self.smiles_to_graph(smiles)
            if graph is not None:
                self.test_graphs.append(graph)
        
        self.log(f"Created {len(self.train_graphs)} training graphs")
        self.log(f"Created {len(self.valid_graphs)} validation graphs")
        self.log(f"Created {len(self.test_graphs)} test graphs")
                
    def smiles_to_graph(self, smiles):
        try:
            # Use sanitize=False to avoid aromatic atom errors, then try to sanitize with more lenient options
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol is None:
                return None
                
            # Try to sanitize with more specific flags
            try:
                # First try to sanitize without kekulization
                Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ 
                                     Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                
                # Then try to kekulize separately
                Chem.Kekulize(mol, clearAromaticFlags=True)
            except Exception as e:
                # If kekulization fails, continue without it
                self.log(f"Warning: Sanitization/Kekulization issues for SMILES: {smiles}")
                
                # Reset any possibly aromatic flags that could cause issues
                for atom in mol.GetAtoms():
                    if not atom.IsInRing() and atom.GetIsAromatic():
                        atom.SetIsAromatic(False)
                    
                    # Also fix aromatic bonds not in rings
                    for bond in atom.GetBonds():
                        if not bond.IsInRing() and bond.GetIsAromatic():
                            bond.SetIsAromatic(False)
                
                # Make a fresh copy of the molecule to reset internal flags
                try:
                    smi = Chem.MolToSmiles(mol)
                    mol = Chem.MolFromSmiles(smi, sanitize=False)
                    if mol is None:
                        return None
                except:
                    pass
            
            # Get atom features
            atoms = mol.GetAtoms()
            x = []
            for atom in atoms:
                atom_features = [
                    atom.GetAtomicNum(),
                    atom.GetDegree(),
                    atom.GetFormalCharge(),
                    atom.GetChiralTag(),
                    atom.GetIsAromatic() * 1,
                    atom.GetHybridization(),
                    atom.GetTotalNumHs(),
                    atom.GetImplicitValence(),
                    atom.GetExplicitValence(),
                    # Safely calculate LogP contribution for atom
                    self.get_atom_logp_contribution(atom)
                ]
                x.append(atom_features)
            
            x = torch.tensor(x, dtype=torch.float)
            
            # Get edge indices
            edge_indices = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                bond_type = bond.GetBondType()
                edge_indices.append([i, j, float(bond_type)])
                edge_indices.append([j, i, float(bond_type)])
            
            if len(edge_indices) == 0:
                edge_index = torch.zeros((2, 0), dtype=torch.long)
                edge_attr = torch.zeros((0, 1), dtype=torch.float)
            else:
                edge_index = torch.tensor([[x[0], x[1]] for x in edge_indices], dtype=torch.long).t().contiguous()
                edge_attr = torch.tensor([[x[2]] for x in edge_indices], dtype=torch.float)
            
            # Calculate whole-molecule descriptors relevant to LogD
            mol_logp = Crippen.MolLogP(mol, includeHs=True) if mol else 0
            mol_tpsa = rdMolDescriptors.CalcTPSA(mol) if mol else 0
            mol_mr = Crippen.MolMR(mol) if mol else 0
            mol_fractioncsp3 = rdMolDescriptors.CalcFractionCSP3(mol) if mol else 0
            mol_numrings = rdMolDescriptors.CalcNumRings(mol) if mol else 0
            mol_numheteroatoms = rdMolDescriptors.CalcNumHeteroatoms(mol) if mol else 0
            
            # Save as graph-level features
            global_features = torch.tensor([
                mol_logp, mol_tpsa, mol_mr, mol_fractioncsp3, 
                mol_numrings, mol_numheteroatoms
            ], dtype=torch.float)
            
            # Create Data object
            data = Data(
                x=x, 
                edge_index=edge_index, 
                edge_attr=edge_attr,
                global_features=global_features,
                smiles=smiles
            )
            
            return data
            
        except Exception as e:
            self.log(f"Error processing SMILES {smiles}: {str(e)}")
            return None

    def get_atom_logp_contribution(self, atom):
        """Safely calculate LogP contribution for an atom"""
        try:
            smarts = atom.GetSmarts()
            if smarts and len(smarts) > 0:
                atom_mol = Chem.MolFromSmiles(smarts)
                if atom_mol:
                    return Crippen.MolLogP(atom_mol)
            return 0
        except:
            return 0

    def run_pipeline(self):
        self.load_data()
        
        # Stage 1: Feature-based Models Foundation
        self.stage1_gbm_models()
        
        # Stage 2: Graph Neural Network Layer
        self.stage2_gnn_models()
        
        # Stage 3: Molecular Transformer Processing
        self.stage3_transformer_models()
        
        # Stage 4: Physics-informed Neural Networks
        self.stage4_physics_informed()
        
        # Stage 5: Meta-learning Integration
        self.stage5_meta_learning()
        
        # Evaluate final model
        self.evaluate_final_model()
        
        self.log("LogD prediction pipeline completed successfully")

    def stage1_gbm_models(self):
        self.log("[STAGE 1] Training gradient boosting models")
        
        # Define model variants
        all_variants = [
            {"name": "xgb_descriptors", "type": "xgb", "features": "descriptors"},
            {"name": "xgb_fingerprints", "type": "xgb", "features": "fingerprints"},
            {"name": "lgb_descriptors", "type": "lgb", "features": "descriptors"},
            {"name": "lgb_fingerprints", "type": "lgb", "features": "fingerprints"},
            {"name": "cb_descriptors", "type": "cb", "features": "descriptors"},
            {"name": "cb_fingerprints", "type": "cb", "features": "fingerprints"}
        ]
        
        # Filter out model variants if features aren't available
        self.stage1_model_variants = []
        for variant in all_variants:
            if variant["features"] == "descriptors" and (self.train_descriptors_scaled is None or self.train_descriptors_scaled.empty):
                self.log(f"Skipping {variant['name']} - descriptors not available")
                continue
            if variant["features"] == "fingerprints" and (self.train_fingerprints is None or self.train_fingerprints.empty):
                self.log(f"Skipping {variant['name']} - fingerprints not available")
                continue
            self.stage1_model_variants.append(variant)
        
        if not self.stage1_model_variants:
            if len(self.desc_cols) > 0 and not self.train_descriptors.empty and not self.train_descriptors_scaled.empty:
                self.log("No model variants available for training. Using general features as descriptors.")
                # Add general feature models if we have features but they weren't recognized as desc/fp
                self.stage1_model_variants = [
                    {"name": "xgb_general", "type": "xgb", "features": "descriptors"},
                    {"name": "lgb_general", "type": "lgb", "features": "descriptors"},
                    {"name": "cb_general", "type": "cb", "features": "descriptors"}
                ]
            else:
                self.log("Error: No valid feature sets available for Stage 1 models.")
                self.log("Please check your data or specify correct prefixes with --descriptors-prefixes and --fingerprint-prefixes")
                raise ValueError("No valid features available for training")
        
        stage1_models = {}
        stage1_predictions = {}
        stage1_importances = {}
        
        for variant in self.stage1_model_variants:
            self.log(f"Training {variant['name']}")
            
            # Select appropriate features
            if variant["features"] == "descriptors":
                X_train = self.train_descriptors_scaled
                X_valid = self.test_descriptors_scaled
                feature_cols = self.desc_cols
                self.log(f"Using {len(feature_cols)} descriptors/general features")
            else:  # fingerprints
                X_train = self.train_fingerprints
                X_valid = self.test_fingerprints
                feature_cols = self.fp_cols
                self.log(f"Using {len(feature_cols)} fingerprint features")
            
            # Train the model
            if variant["type"] == "xgb":
                model = self.train_xgb(X_train, self.y_train, X_valid, self.y_test)
            elif variant["type"] == "lgb":
                model = self.train_lgb(X_train, self.y_train, X_valid, self.y_test)
            else:  # catboost
                model = self.train_catboost(X_train, self.y_train, X_valid, self.y_test)
            
            # Make predictions
            if variant["type"] == "cb":
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_valid)
            else:
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_valid)
            
            # Calculate feature importance
            feature_importance = self.get_feature_importance(model, variant["type"], X_train, feature_cols)
            
            # Store the model, predictions, and importance
            stage1_models[variant["name"]] = model
            stage1_predictions[variant["name"]] = {
                "train": train_pred,
                "test": test_pred
            }
            stage1_importances[variant["name"]] = feature_importance
            
            # Save the model
            model_path = os.path.join(self.stage_output_dirs["stage1"], f"{variant['name']}.pkl")
            joblib.dump(model, model_path)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(self.y_test, test_pred))
            mae = mean_absolute_error(self.y_test, test_pred)
            r2 = r2_score(self.y_test, test_pred)
            
            self.log(f"  {variant['name']} Test RMSE: {rmse:.4f}")
            self.log(f"  {variant['name']} Test MAE: {mae:.4f}")
            self.log(f"  {variant['name']} Test R²: {r2:.4f}")
            
            # Generate SHAP values for top model
            if variant["type"] == "lgb" and variant["features"] == "descriptors":
                try:
                    self.log(f"Generating SHAP values for {variant['name']}")
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_valid)
                    
                    # Save SHAP values
                    np.save(os.path.join(self.stage_output_dirs["stage1"], f"{variant['name']}_shap_values.npy"), shap_values)
                    
                    # Save SHAP summary plot
                    plt.figure(figsize=(10, 6))
                    shap.summary_plot(shap_values, X_valid, show=False)
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.stage_output_dirs["stage1"], f"{variant['name']}_shap_summary.png"))
                    plt.close()
                except Exception as e:
                    self.log(f"Error generating SHAP values: {str(e)}")
        
        # Ensemble predictions (simple average)
        stage1_ensemble_train = np.mean([
            stage1_predictions[variant["name"]]["train"] 
            for variant in self.stage1_model_variants
        ], axis=0)
        
        stage1_ensemble_test = np.mean([
            stage1_predictions[variant["name"]]["test"] 
            for variant in self.stage1_model_variants
        ], axis=0)
        
        # Calculate ensemble metrics
        ensemble_rmse = np.sqrt(mean_squared_error(self.y_test, stage1_ensemble_test))
        ensemble_mae = mean_absolute_error(self.y_test, stage1_ensemble_test)
        ensemble_r2 = r2_score(self.y_test, stage1_ensemble_test)
        
        self.log(f"Stage 1 Ensemble Test RMSE: {ensemble_rmse:.4f}")
        self.log(f"Stage 1 Ensemble Test MAE: {ensemble_mae:.4f}")
        self.log(f"Stage 1 Ensemble Test R²: {ensemble_r2:.4f}")
        
        # Save predictions for next stage
        stage1_results = pd.DataFrame({
            "SMILES": self.test_smiles,
            "True_LogD": self.y_test,
            "Ensemble_Pred": stage1_ensemble_test
        })
        
        for variant in self.stage1_model_variants:
            stage1_results[f"{variant['name']}_pred"] = stage1_predictions[variant["name"]]["test"]
        
        stage1_results.to_csv(os.path.join(self.stage_output_dirs["stage1"], "stage1_predictions.csv"), index=False)
        
        # Store for the next stage
        self.stage_models["stage1"] = stage1_models
        self.stage_predictions["stage1"] = {
            "ensemble": {
                "train": stage1_ensemble_train,
                "test": stage1_ensemble_test
            },
            "individual": stage1_predictions
        }
        self.stage_importances = stage1_importances

    def train_xgb(self, X_train, y_train, X_valid, y_valid):
        # Log XGBoost version for debugging
        self.log(f"XGBoost version: {xgb.__version__}")
        
        params = {
            'objective': 'reg:squarederror',
            'booster': 'gbtree',
            'learning_rate': 0.05,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'alpha': 1,
            'lambda': 1,
            'n_estimators': 300,  # Increased as we can't use early stopping
            'seed': 42
        }
        
        # Create and fit the model with simplest API
        model = xgb.XGBRegressor(**params)
        self.log("Training XGBoost model with basic configuration")
        model.fit(X_train, y_train)
        
        # Evaluate model manually on validation set
        valid_pred = model.predict(X_valid)
        valid_rmse = np.sqrt(mean_squared_error(y_valid, valid_pred))
        self.log(f"XGBoost validation RMSE: {valid_rmse:.4f}")
        
        return model

    def train_lgb(self, X_train, y_train, X_valid, y_valid):
        # Log LightGBM version for debugging
        self.log(f"LightGBM version: {lgb.__version__}")
        
        params = {
            'objective': 'regression',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'n_estimators': 300,  # Increased as we'll use simpler approach
            'random_state': 42
        }
        
        # Create and fit the model with simplest API
        model = lgb.LGBMRegressor(**params)
        self.log("Training LightGBM model with basic configuration")
        model.fit(X_train, y_train)
        
        # Evaluate model manually on validation set
        valid_pred = model.predict(X_valid)
        valid_rmse = np.sqrt(mean_squared_error(y_valid, valid_pred))
        self.log(f"LightGBM validation RMSE: {valid_rmse:.4f}")
        
        return model

    def train_catboost(self, X_train, y_train, X_valid, y_valid):
        # Log CatBoost version for debugging
        self.log(f"CatBoost version: {cb.__version__}")
        
        params = {
            'loss_function': 'RMSE',
            'iterations': 300,  # Increased as we'll use simpler approach
            'learning_rate': 0.05,
            'depth': 6,
            'l2_leaf_reg': 3,
            'random_seed': 42,
            'verbose': False
        }
        
        # Create and fit the model with simplest API
        model = cb.CatBoostRegressor(**params)
        self.log("Training CatBoost model with basic configuration")
        model.fit(X_train, y_train, verbose=False)
        
        # Evaluate model manually on validation set
        valid_pred = model.predict(X_valid)
        valid_rmse = np.sqrt(mean_squared_error(y_valid, valid_pred))
        self.log(f"CatBoost validation RMSE: {valid_rmse:.4f}")
        
        return model

    def get_feature_importance(self, model, model_type, X, feature_names):
        if model_type == "xgb":
            importance = model.feature_importances_
        elif model_type == "lgb":
            importance = model.feature_importances_
        else:  # catboost
            importance = model.get_feature_importance()
        
        feature_importance = dict(zip(feature_names, importance))
        return feature_importance

    def stage2_gnn_models(self):
        self.log("[STAGE 2] Training graph neural network models")
        
        # Get stage1 predictions for training
        stage1_train_preds = self.stage_predictions["stage1"]["ensemble"]["train"]
        stage1_test_preds = self.stage_predictions["stage1"]["ensemble"]["test"]
        
        # Create PyTorch Geometric datasets from molecular graphs
        from torch_geometric.data import DataLoader, Batch
        from torch_geometric.data import Data
        
        # Transfer stage1 predictions to the graph data objects
        for i, graph in enumerate(self.train_graphs):
            graph.stage1_pred = torch.tensor([stage1_train_preds[i]], dtype=torch.float)
            graph.y = torch.tensor([self.y_train_split[i]], dtype=torch.float)
            
        for i, graph in enumerate(self.valid_graphs):
            if i < len(self.y_valid):
                graph.stage1_pred = torch.tensor([stage1_train_preds[i + len(self.train_graphs)]], dtype=torch.float)
                graph.y = torch.tensor([self.y_valid[i]], dtype=torch.float)
            
        for i, graph in enumerate(self.test_graphs):
            if i < len(stage1_test_preds):
                graph.stage1_pred = torch.tensor([stage1_test_preds[i]], dtype=torch.float)
                graph.y = torch.tensor([self.y_test[i]], dtype=torch.float)
        
        # Set batch size based on available memory
        if torch.cuda.is_available():
            batch_size = 64
        elif torch.backends.mps.is_available():
            # Lower batch size for MPS to avoid memory issues
            batch_size = 32
        else:
            batch_size = 32
        
        # Create data loaders
        train_loader = DataLoader(self.train_graphs, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(self.valid_graphs, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(self.test_graphs, batch_size=batch_size, shuffle=False)
        
        # Train the AttentiveFP model
        self.log("Training AttentiveFP model")
        attentive_fp_model = self.train_attentive_fp(train_loader, test_loader)
        
        # Train the DMPNN model
        self.log("Training DMPNN model")
        dmpnn_model = self.train_dmpnn(train_loader, test_loader)
        
        # Train the GAT model
        self.log("Training GAT model")
        gat_model = self.train_gat(train_loader, test_loader)
        
        # Store the trained models
        self.stage_models["stage2"] = {
            "attentive_fp": attentive_fp_model,
            "dmpnn": dmpnn_model,
            "gat": gat_model
        }
        
        # Generate predictions and embeddings
        self.log("Generating GNN predictions and embeddings")
        
        # Get AttentiveFP predictions
        attentive_fp_train_preds, attentive_fp_train_embeds = self.generate_gnn_outputs(
            attentive_fp_model, train_loader)
        attentive_fp_test_preds, attentive_fp_test_embeds = self.generate_gnn_outputs(
            attentive_fp_model, test_loader)
        
        # Get DMPNN predictions
        dmpnn_train_preds, dmpnn_train_embeds = self.generate_gnn_outputs(
            dmpnn_model, train_loader)
        dmpnn_test_preds, dmpnn_test_embeds = self.generate_gnn_outputs(
            dmpnn_model, test_loader)
        
        # Get GAT predictions
        gat_train_preds, gat_train_embeds = self.generate_gnn_outputs(
            gat_model, train_loader)
        gat_test_preds, gat_test_embeds = self.generate_gnn_outputs(
            gat_model, test_loader)
            
        # Flatten prediction arrays if needed
        attentive_fp_train_preds = attentive_fp_train_preds.flatten()
        attentive_fp_test_preds = attentive_fp_test_preds.flatten()
        dmpnn_train_preds = dmpnn_train_preds.flatten()
        dmpnn_test_preds = dmpnn_test_preds.flatten()
        gat_train_preds = gat_train_preds.flatten()
        gat_test_preds = gat_test_preds.flatten()
        
        # Combine the predictions for stage 2
        self.stage_predictions["stage2"] = {
            "train": (attentive_fp_train_preds + dmpnn_train_preds + gat_train_preds) / 3,
            "test": (attentive_fp_test_preds + dmpnn_test_preds + gat_test_preds) / 3,
            "individual": {
                "attentive_fp": {"train": attentive_fp_train_preds, "test": attentive_fp_test_preds},
                "dmpnn": {"train": dmpnn_train_preds, "test": dmpnn_test_preds},
                "gat": {"train": gat_train_preds, "test": gat_test_preds}
            }
        }
        
        # Store the embeddings for later use
        self.stage_embeddings["stage2"] = {
            "attentive_fp": {
                "train": attentive_fp_train_embeds,
                "test": attentive_fp_test_embeds
            },
            "dmpnn": {
                "train": dmpnn_train_embeds,
                "test": dmpnn_test_embeds
            },
            "gat": {
                "train": gat_train_embeds,
                "test": gat_test_embeds
            }
        }
        
        # Evaluate Stage 2 performance
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        train_pred = self.stage_predictions["stage2"]["train"]
        test_pred = self.stage_predictions["stage2"]["test"]
        
        # Compute metrics
        if len(train_pred) == len(self.y_train_split) and len(test_pred) == len(self.y_test):
            train_rmse = np.sqrt(mean_squared_error(self.y_train_split, train_pred))
            train_mae = mean_absolute_error(self.y_train_split, train_pred)
            train_r2 = r2_score(self.y_train_split, train_pred)
            
            test_rmse = np.sqrt(mean_squared_error(self.y_test, test_pred))
            test_mae = mean_absolute_error(self.y_test, test_pred)
            test_r2 = r2_score(self.y_test, test_pred)
            
            self.log(f"Stage 2 Ensemble Train RMSE: {train_rmse:.4f}")
            self.log(f"Stage 2 Ensemble Train MAE: {train_mae:.4f}")
            self.log(f"Stage 2 Ensemble Train R²: {train_r2:.4f}")
            self.log(f"Stage 2 Ensemble Test RMSE: {test_rmse:.4f}")
            self.log(f"Stage 2 Ensemble Test MAE: {test_mae:.4f}")
            self.log(f"Stage 2 Ensemble Test R²: {test_r2:.4f}")
        else:
            self.log("Warning: Prediction and target length mismatch in Stage 2 evaluation")

    def train_attentive_fp(self, train_loader, test_loader):
        # Force CPU for GNN operations if using MPS
        use_cpu = self.device.type == 'mps'
        if use_cpu:
            self.log("Using CPU for GNN operations due to MPS compatibility issues")
            model_device = torch.device('cpu')
        else:
            model_device = self.device
            
        model = AttentiveFPModel(
            node_feat_dim=10,  # number of atom features
            edge_feat_dim=1,   # number of bond features
            global_feat_dim=6, # number of molecule features
            num_layers=3,
            dropout=0.2,
            hidden_dim=64
        ).to(model_device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
        scheduler = OneCycleLR(
            optimizer, 
            max_lr=0.001, 
            total_steps=50 * len(train_loader),
            pct_start=0.3
        )
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(50):
            model.train()
            train_loss = 0
            
            for batch in train_loader:
                batch = batch.to(model_device)
                optimizer.zero_grad()
                
                out, _ = model(batch)
                loss = criterion(out.view(-1), batch.stage1_pred.view(-1))
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item() * batch.num_graphs
            
            train_loss /= len(train_loader.dataset)
            
            # Evaluate on test set
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(model_device)
                    out, _ = model(batch)
                    loss = criterion(out.view(-1), batch.stage1_pred.view(-1))
                    val_loss += loss.item() * batch.num_graphs
            
            val_loss /= len(test_loader.dataset)
            
            self.log(f"AttentiveFP Epoch {epoch+1}/50, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    self.log(f"Early stopping at epoch {epoch+1}")
                    break
        
        model.load_state_dict(best_model_state)
        # If results need to be moved back to original device
        if use_cpu and self.device.type == 'mps':
            model_cpu = model
            model = AttentiveFPModel(
                node_feat_dim=10,
                edge_feat_dim=1,
                global_feat_dim=6,
                num_layers=3,
                dropout=0.2,
                hidden_dim=64
            ).to(self.device)
            # Copy weights but keep on CPU for inference
            model.load_state_dict(model_cpu.state_dict())
        return model

    def train_dmpnn(self, train_loader, test_loader):
        # Force CPU for GNN operations if using MPS
        use_cpu = self.device.type == 'mps'
        if use_cpu:
            self.log("Using CPU for GNN operations due to MPS compatibility issues")
            model_device = torch.device('cpu')
        else:
            model_device = self.device
            
        model = DMPNNModel(
            node_feat_dim=10,
            edge_feat_dim=1,
            global_feat_dim=6,
            hidden_dim=64,
            num_layers=3,
            dropout=0.2
        ).to(model_device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
        scheduler = OneCycleLR(
            optimizer, 
            max_lr=0.001, 
            total_steps=50 * len(train_loader),
            pct_start=0.3
        )
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(50):
            model.train()
            train_loss = 0
            
            for batch in train_loader:
                batch = batch.to(model_device)
                optimizer.zero_grad()
                
                out, _ = model(batch)
                loss = criterion(out.view(-1), batch.stage1_pred.view(-1))
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item() * batch.num_graphs
            
            train_loss /= len(train_loader.dataset)
            
            # Evaluate on test set
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(model_device)
                    out, _ = model(batch)
                    loss = criterion(out.view(-1), batch.stage1_pred.view(-1))
                    val_loss += loss.item() * batch.num_graphs
            
            val_loss /= len(test_loader.dataset)
            
            self.log(f"DMPNN Epoch {epoch+1}/50, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    self.log(f"Early stopping at epoch {epoch+1}")
                    break
        
        model.load_state_dict(best_model_state)
        # If results need to be moved back to original device
        if use_cpu and self.device.type == 'mps':
            model_cpu = model
            model = DMPNNModel(
                node_feat_dim=10,
                edge_feat_dim=1,
                global_feat_dim=6,
                hidden_dim=64,
                num_layers=3,
                dropout=0.2
            ).to(self.device)
            # Copy weights but keep on CPU for inference
            model.load_state_dict(model_cpu.state_dict())
        return model

    def train_gat(self, train_loader, test_loader):
        # Force CPU for GNN operations if using MPS
        use_cpu = self.device.type == 'mps'
        if use_cpu:
            self.log("Using CPU for GNN operations due to MPS compatibility issues")
            model_device = torch.device('cpu')
        else:
            model_device = self.device
            
        model = GATModel(
            node_feat_dim=10,
            edge_feat_dim=1,
            global_feat_dim=6,
            hidden_dim=64,
            num_layers=3,
            dropout=0.2
        ).to(model_device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
        scheduler = OneCycleLR(
            optimizer, 
            max_lr=0.001, 
            total_steps=50 * len(train_loader),
            pct_start=0.3
        )
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(50):
            model.train()
            train_loss = 0
            
            for batch in train_loader:
                batch = batch.to(model_device)
                optimizer.zero_grad()
                
                out, _ = model(batch)
                loss = criterion(out.view(-1), batch.stage1_pred.view(-1))
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item() * batch.num_graphs
            
            train_loss /= len(train_loader.dataset)
            
            # Evaluate on test set
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(model_device)
                    out, _ = model(batch)
                    loss = criterion(out.view(-1), batch.stage1_pred.view(-1))
                    val_loss += loss.item() * batch.num_graphs
            
            val_loss /= len(test_loader.dataset)
            
            self.log(f"GAT Epoch {epoch+1}/50, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    self.log(f"Early stopping at epoch {epoch+1}")
                    break
        
        model.load_state_dict(best_model_state)
        return model

    def generate_gnn_outputs(self, model, loader):
        # Force CPU for GNN operations if using MPS
        use_cpu = self.device.type == 'mps'
        if use_cpu:
            model_device = torch.device('cpu')
            model = model.to(model_device)
        else:
            model_device = self.device
            
        model.eval()
        all_outputs = []
        all_embeddings = []
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(model_device)
                output, embeddings = model(batch)
                all_outputs.append(output)
                all_embeddings.append(embeddings)
        
        outputs = torch.cat(all_outputs, dim=0).cpu().numpy()
        embeddings = torch.cat(all_embeddings, dim=0).cpu().numpy()
        
        return outputs, embeddings

    def stage3_transformer_models(self):
        self.log("[STAGE 3] Training molecular transformer models")
        
        # Get Stage 1 and Stage 2 predictions
        stage1_train_preds = self.stage_predictions["stage1"]["ensemble"]["train"]
        stage1_test_preds = self.stage_predictions["stage1"]["ensemble"]["test"]
        
        stage2_train_preds = self.stage_predictions["stage2"]["ensemble"]["train"]
        stage2_test_preds = self.stage_predictions["stage2"]["ensemble"]["test"]
        
        # Get GNN embeddings from Stage 2
        gnn_train_embeds = np.concatenate([
            self.stage_embeddings["stage2"]["attentivefp"]["train"],
            self.stage_embeddings["stage2"]["dmpnn"]["train"],
            self.stage_embeddings["stage2"]["gat"]["train"]
        ], axis=1)
        
        gnn_test_embeds = np.concatenate([
            self.stage_embeddings["stage2"]["attentivefp"]["test"],
            self.stage_embeddings["stage2"]["dmpnn"]["test"],
            self.stage_embeddings["stage2"]["gat"]["test"]
        ], axis=1)
        
        # Apply PCA to reduce GNN embedding dimension
        pca = PCA(n_components=min(64, gnn_train_embeds.shape[1]))
        gnn_train_embeds_pca = pca.fit_transform(gnn_train_embeds)
        gnn_test_embeds_pca = pca.transform(gnn_test_embeds)
        
        # Save PCA model
        joblib.dump(pca, os.path.join(self.stage_output_dirs["stage3"], "gnn_embeddings_pca.pkl"))
        
        # Train the augmented transformer model
        transformer_model, tokenizer = self.train_molbert_transformer(
            self.train_smiles,
            self.y_train,
            self.test_smiles,
            self.y_test,
            stage1_train_preds,
            stage1_test_preds,
            stage2_train_preds,
            stage2_test_preds,
            gnn_train_embeds_pca,
            gnn_test_embeds_pca
        )
        
        # Generate transformer predictions and embeddings
        transformer_train_preds, transformer_train_embeds = self.generate_transformer_outputs(
            transformer_model,
            tokenizer,
            self.train_smiles,
            stage1_train_preds,
            stage2_train_preds,
            gnn_train_embeds_pca
        )
        
        transformer_test_preds, transformer_test_embeds = self.generate_transformer_outputs(
            transformer_model,
            tokenizer,
            self.test_smiles,
            stage1_test_preds,
            stage2_test_preds,
            gnn_test_embeds_pca
        )
        
        # Calculate metrics
        transformer_rmse = np.sqrt(mean_squared_error(self.y_test, transformer_test_preds))
        transformer_mae = mean_absolute_error(self.y_test, transformer_test_preds)
        transformer_r2 = r2_score(self.y_test, transformer_test_preds)
        
        self.log(f"Stage 3 Transformer Test RMSE: {transformer_rmse:.4f}")
        self.log(f"Stage 3 Transformer Test MAE: {transformer_mae:.4f}")
        self.log(f"Stage 3 Transformer Test R²: {transformer_r2:.4f}")
        
        # Save predictions for next stage
        stage3_results = pd.DataFrame({
            "SMILES": self.test_smiles,
            "True_LogD": self.y_test,
            "Stage1_Ensemble_Pred": stage1_test_preds,
            "Stage2_Ensemble_Pred": stage2_test_preds,
            "Stage3_Transformer_Pred": transformer_test_preds
        })
        
        stage3_results.to_csv(os.path.join(self.stage_output_dirs["stage3"], "stage3_predictions.csv"), index=False)
        
        # Save model
        transformer_model.save_pretrained(os.path.join(self.stage_output_dirs["stage3"], "transformer_model"))
        tokenizer.save_pretrained(os.path.join(self.stage_output_dirs["stage3"], "transformer_tokenizer"))
        
        # Store for the next stage
        self.stage_models["stage3"] = {
            "transformer": transformer_model,
            "tokenizer": tokenizer
        }
        
        self.stage_predictions["stage3"] = {
            "train": transformer_train_preds,
            "test": transformer_test_preds
        }
        
        self.stage_embeddings["stage3"] = {
            "train": transformer_train_embeds,
            "test": transformer_test_embeds
        }

    def train_molbert_transformer(self, train_smiles, train_y, test_smiles, test_y, 
                                  stage1_train_preds, stage1_test_preds,
                                  stage2_train_preds, stage2_test_preds,
                                  gnn_train_embeds, gnn_test_embeds):
        self.log("Training MolBERT transformer model")
        
        # Load pretrained model or specified model path
        model_path = self.args.transformer_model_path
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            config = AutoConfig.from_pretrained(model_path)
            model = MolecularTransformerModel(model_path, config).to(self.device)
            self.log(f"Loaded transformer model from {model_path}")
        except Exception as e:
            self.log(f"Error loading transformer model: {str(e)}")
            self.log("Using default RobertaModel as fallback")
            model_path = "seyonec/PubChem10M_SMILES_BPE_450k"
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            config = AutoConfig.from_pretrained(model_path)
            model = MolecularTransformerModel(model_path, config).to(self.device)
        
        # Create datasets
        train_dataset = MoleculeDataset(
            smiles=train_smiles,
            labels=train_y,
            stage1_preds=stage1_train_preds,
            stage2_preds=stage2_train_preds,
            gnn_embeds=gnn_train_embeds,
            tokenizer=tokenizer
        )
        
        test_dataset = MoleculeDataset(
            smiles=test_smiles,
            labels=test_y,
            stage1_preds=stage1_test_preds,
            stage2_preds=stage2_test_preds,
            gnn_embeds=gnn_test_embeds,
            tokenizer=tokenizer
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        # Setup optimizer and scheduler
        optimizer = AdamW(model.parameters(), lr=2e-5)
        total_steps = len(train_loader) * 10
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        
        # Training loop
        best_val_loss = float('inf')
        patience = 3
        patience_counter = 0
        
        for epoch in range(10):
            model.train()
            train_loss = 0
            
            for batch in train_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                optimizer.zero_grad()
                
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    gnn_embeddings=batch['gnn_embeddings'],
                    stage1_pred=batch['stage1_pred'],
                    stage2_pred=batch['stage2_pred'],
                    labels=batch['labels']
                )
                
                loss = outputs.loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Evaluate on test set
            model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch in test_loader:
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                    
                    outputs = model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        gnn_embeddings=batch['gnn_embeddings'],
                        stage1_pred=batch['stage1_pred'],
                        stage2_pred=batch['stage2_pred'],
                        labels=batch['labels']
                    )
                    
                    loss = outputs.loss
                    val_loss += loss.item()
            
            val_loss /= len(test_loader)
            
            self.log(f"Transformer Epoch {epoch+1}/10, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    self.log(f"Early stopping at epoch {epoch+1}")
                    break
        
        model.load_state_dict(best_model_state)
        return model, tokenizer

    def generate_transformer_outputs(self, model, tokenizer, smiles_list, stage1_preds, stage2_preds, gnn_embeds):
        model.eval()
        predictions = []
        embeddings = []
        
        dataset = MoleculeDataset(
            smiles=smiles_list,
            labels=np.zeros(len(smiles_list)),  # Dummy labels
            stage1_preds=stage1_preds,
            stage2_preds=stage2_preds,
            gnn_embeds=gnn_embeds,
            tokenizer=tokenizer
        )
        
        loader = DataLoader(dataset, batch_size=16, shuffle=False)
        
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    gnn_embeddings=batch['gnn_embeddings'],
                    stage1_pred=batch['stage1_pred'],
                    stage2_pred=batch['stage2_pred']
                )
                
                preds = outputs.predictions.cpu().numpy()
                embeds = outputs.embeddings.cpu().numpy()
                
                predictions.append(preds)
                embeddings.append(embeds)
        
        predictions = np.concatenate(predictions, axis=0).flatten()
        embeddings = np.concatenate(embeddings, axis=0)
        
        return predictions, embeddings

    def stage4_physics_informed(self):
        self.log("[STAGE 4] Training physics-informed neural network")
        
        # Get predictions from previous stages
        stage1_train_preds = self.stage_predictions["stage1"]["ensemble"]["train"]
        stage1_test_preds = self.stage_predictions["stage1"]["ensemble"]["test"]
        
        stage2_train_preds = self.stage_predictions["stage2"]["ensemble"]["train"]
        stage2_test_preds = self.stage_predictions["stage2"]["ensemble"]["test"]
        
        stage3_train_preds = self.stage_predictions["stage3"]["train"]
        stage3_test_preds = self.stage_predictions["stage3"]["test"]
        
        # Get embeddings from previous stages
        gnn_train_embeds = np.concatenate([
            self.stage_embeddings["stage2"]["attentivefp"]["train"],
            self.stage_embeddings["stage2"]["dmpnn"]["train"],
            self.stage_embeddings["stage2"]["gat"]["train"]
        ], axis=1)
        
        gnn_test_embeds = np.concatenate([
            self.stage_embeddings["stage2"]["attentivefp"]["test"],
            self.stage_embeddings["stage2"]["dmpnn"]["test"],
            self.stage_embeddings["stage2"]["gat"]["test"]
        ], axis=1)
        
        transformer_train_embeds = self.stage_embeddings["stage3"]["train"]
        transformer_test_embeds = self.stage_embeddings["stage3"]["test"]
        
        # Reduce dimensionality for combined embeddings
        pca = PCA(n_components=64)
        combined_train_embeds = np.concatenate([gnn_train_embeds, transformer_train_embeds], axis=1)
        combined_test_embeds = np.concatenate([gnn_test_embeds, transformer_test_embeds], axis=1)
        
        combined_train_embeds_pca = pca.fit_transform(combined_train_embeds)
        combined_test_embeds_pca = pca.transform(combined_test_embeds)
        
        # Save PCA model
        joblib.dump(pca, os.path.join(self.stage_output_dirs["stage4"], "combined_embeddings_pca.pkl"))
        
        # Create feature matrices for physics-informed model
        X_train_physics = np.column_stack([
            stage1_train_preds,
            stage2_train_preds,
            stage3_train_preds,
            combined_train_embeds_pca
        ])
        
        X_test_physics = np.column_stack([
            stage1_test_preds,
            stage2_test_preds,
            stage3_test_preds,
            combined_test_embeds_pca
        ])
        
        # Calculate physics-informed features for each molecule
        physics_train_features = self.calculate_physics_features(self.train_smiles)
        physics_test_features = self.calculate_physics_features(self.test_smiles)
        
        # Combine with other features
        X_train_physics = np.column_stack([X_train_physics, physics_train_features])
        X_test_physics = np.column_stack([X_test_physics, physics_test_features])
        
        # Train physics-informed neural network
        physics_model = self.train_physics_nn(
            X_train_physics, 
            self.y_train, 
            X_test_physics, 
            self.y_test
        )
        
        # Generate predictions
        physics_train_preds = physics_model.predict(X_train_physics)
        physics_test_preds = physics_model.predict(X_test_physics)
        
        # Calculate metrics
        physics_rmse = np.sqrt(mean_squared_error(self.y_test, physics_test_preds))
        physics_mae = mean_absolute_error(self.y_test, physics_test_preds)
        physics_r2 = r2_score(self.y_test, physics_test_preds)
        
        self.log(f"Stage 4 Physics-Informed NN Test RMSE: {physics_rmse:.4f}")
        self.log(f"Stage 4 Physics-Informed NN Test MAE: {physics_mae:.4f}")
        self.log(f"Stage 4 Physics-Informed NN Test R²: {physics_r2:.4f}")
        
        # Save predictions for next stage
        stage4_results = pd.DataFrame({
            "SMILES": self.test_smiles,
            "True_LogD": self.y_test,
            "Stage1_Ensemble_Pred": stage1_test_preds,
            "Stage2_Ensemble_Pred": stage2_test_preds,
            "Stage3_Transformer_Pred": stage3_test_preds,
            "Stage4_Physics_Pred": physics_test_preds
        })
        
        stage4_results.to_csv(os.path.join(self.stage_output_dirs["stage4"], "stage4_predictions.csv"), index=False)
        
        # Save model
        torch.save(physics_model.state_dict(), os.path.join(self.stage_output_dirs["stage4"], "physics_model.pt"))
        
        # Store for the next stage
        self.stage_models["stage4"] = physics_model
        
        self.stage_predictions["stage4"] = {
            "train": physics_train_preds,
            "test": physics_test_preds
        }

    def calculate_physics_features(self, smiles_list):
        self.log("Calculating physics-informed molecular features")
        
        # Features relevant to LogD prediction from physical chemistry perspective
        features = []
        
        for smiles in tqdm(smiles_list, desc="Calculating physics features"):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    features.append(np.zeros(8))
                    continue
                
                # Calculate LogP (octanol-water partition coefficient)
                logp = Crippen.MolLogP(mol)
                
                # Calculate topological polar surface area
                tpsa = rdMolDescriptors.CalcTPSA(mol)
                
                # Calculate molecular weight
                mw = Descriptors.MolWt(mol)
                
                # Calculate number of H-bond donors and acceptors
                hbd = rdMolDescriptors.CalcNumHBD(mol)
                hba = rdMolDescriptors.CalcNumHBA(mol)
                
                # Count rotatable bonds
                rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
                
                # Count aromatic rings
                aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
                
                # Calculate fraction of sp3 carbons
                fsp3 = rdMolDescriptors.CalcFractionCSP3(mol)
                
                features.append([logp, tpsa, mw, hbd, hba, rotatable_bonds, aromatic_rings, fsp3])
                
            except:
                features.append(np.zeros(8))
        
        return np.array(features)

    def train_physics_nn(self, X_train, y_train, X_test, y_test):
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test)
        
        # Create TensorDatasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Create model
        input_dim = X_train.shape[1]
        model = PhysicsInformedNN(input_dim=input_dim).to(self.device)
        
        # Setup optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Loss functions
        mse_loss = nn.MSELoss()
        
        # Training loop
        best_val_loss = float('inf')
        patience = 20
        patience_counter = 0
        
        for epoch in range(100):
            model.train()
            train_loss = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                pred, physics_term = model(batch_X)
                
                # Calculate loss with physics regularization
                loss = mse_loss(pred.squeeze(), batch_y)
                loss += 0.1 * physics_term  # Physics regularization weight
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item() * batch_X.size(0)
            
            train_loss /= len(train_loader.dataset)
            
            # Validation
            model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    
                    pred, physics_term = model(batch_X)
                    loss = mse_loss(pred.squeeze(), batch_y)
                    loss += 0.1 * physics_term
                    
                    val_loss += loss.item() * batch_X.size(0)
            
            val_loss /= len(test_loader.dataset)
            scheduler.step(val_loss)
            
            if (epoch + 1) % 10 == 0:
                self.log(f"Physics NN Epoch {epoch+1}/100, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    self.log(f"Early stopping at epoch {epoch+1}")
                    break
        
        model.load_state_dict(best_model_state)
        return model

    def stage5_meta_learning(self):
        self.log("[STAGE 5] Training Bayesian meta-learner")
        
        # Collect predictions from all previous stages
        stage1_train_preds = self.stage_predictions["stage1"]["ensemble"]["train"]
        stage1_test_preds = self.stage_predictions["stage1"]["ensemble"]["test"]
        
        stage2_train_preds = self.stage_predictions["stage2"]["ensemble"]["train"]
        stage2_test_preds = self.stage_predictions["stage2"]["ensemble"]["test"]
        
        stage3_train_preds = self.stage_predictions["stage3"]["train"]
        stage3_test_preds = self.stage_predictions["stage3"]["test"]
        
        stage4_train_preds = self.stage_predictions["stage4"]["train"]
        stage4_test_preds = self.stage_predictions["stage4"]["test"]
        
        # Combine predictions
        X_train_meta = np.column_stack([
            stage1_train_preds,
            stage2_train_preds,
            stage3_train_preds,
            stage4_train_preds
        ])
        
        X_test_meta = np.column_stack([
            stage1_test_preds,
            stage2_test_preds,
            stage3_test_preds,
            stage4_test_preds
        ])
        
        # Train Bayesian meta-learner
        meta_model = self.train_bayesian_meta_learner(
            X_train_meta, 
            self.y_train, 
            X_test_meta, 
            self.y_test
        )
        
        # Generate predictions with uncertainty
        meta_train_preds, meta_train_uncertainty = meta_model.predict(X_train_meta)
        meta_test_preds, meta_test_uncertainty = meta_model.predict(X_test_meta)
        
        # Calculate metrics
        meta_rmse = np.sqrt(mean_squared_error(self.y_test, meta_test_preds))
        meta_mae = mean_absolute_error(self.y_test, meta_test_preds)
        meta_r2 = r2_score(self.y_test, meta_test_preds)
        
        self.log(f"Stage 5 Meta-Learner Test RMSE: {meta_rmse:.4f}")
        self.log(f"Stage 5 Meta-Learner Test MAE: {meta_mae:.4f}")
        self.log(f"Stage 5 Meta-Learner Test R²: {meta_r2:.4f}")
        
        # Save predictions with uncertainty
        stage5_results = pd.DataFrame({
            "SMILES": self.test_smiles,
            "True_LogD": self.y_test,
            "Stage1_Pred": stage1_test_preds,
            "Stage2_Pred": stage2_test_preds,
            "Stage3_Pred": stage3_test_preds,
            "Stage4_Pred": stage4_test_preds,
            "Final_Pred": meta_test_preds,
            "Uncertainty": meta_test_uncertainty
        })
        
        stage5_results.to_csv(os.path.join(self.stage_output_dirs["stage5"], "stage5_predictions.csv"), index=False)
        
        # Save model
        torch.save(meta_model.state_dict(), os.path.join(self.stage_output_dirs["stage5"], "meta_model.pt"))
        
        # Store final predictions
        self.final_predictions = meta_test_preds
        self.final_uncertainty = meta_test_uncertainty

    def train_bayesian_meta_learner(self, X_train, y_train, X_test, y_test):
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test)
        
        # Create TensorDatasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Create model
        input_dim = X_train.shape[1]
        model = BayesianMetaLearner(input_dim=input_dim).to(self.device)
        
        # Setup optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        best_val_loss = float('inf')
        patience = 20
        patience_counter = 0
        
        for epoch in range(100):
            model.train()
            train_loss = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass with multiple samples for MC Dropout
                outputs = []
                for _ in range(5):  # MC Dropout samples during training
                    output = model(batch_X, True)  # enable_dropout=True
                    outputs.append(output)
                
                # Stack outputs
                outputs = torch.stack(outputs, dim=0)
                mean_output = outputs.mean(dim=0)
                
                # Negative log likelihood loss
                loss = model.nll_loss(mean_output, batch_y)
                
                # Add KL divergence for variational inference
                kl_loss = 0.0
                for module in model.modules():
                    if hasattr(module, 'kl_loss'):
                        kl_loss += module.kl_loss()
                
                total_loss = loss + kl_loss * 0.01  # KL weight
                
                # Backward pass
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += total_loss.item() * batch_X.size(0)
            
            train_loss /= len(train_loader.dataset)
            
            # Validation
            model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    
                    # Multiple forward passes for MC Dropout
                    outputs = []
                    for _ in range(10):  # More samples for validation
                        output = model(batch_X, True)  # enable_dropout=True
                        outputs.append(output)
                    
                    outputs = torch.stack(outputs, dim=0)
                    mean_output = outputs.mean(dim=0)
                    
                    loss = model.nll_loss(mean_output, batch_y)
                    val_loss += loss.item() * batch_X.size(0)
            
            val_loss /= len(test_loader.dataset)
            
            if (epoch + 1) % 10 == 0:
                self.log(f"Meta-Learner Epoch {epoch+1}/100, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    self.log(f"Early stopping at epoch {epoch+1}")
                    break
        
        model.load_state_dict(best_model_state)
        return model

    def evaluate_final_model(self):
        self.log("Evaluating final model performance")
        
        # Create summary of all stages
        all_test_predictions = pd.DataFrame({
            "SMILES": self.test_smiles,
            "True_LogD": self.y_test,
            "Stage1_Ensemble": self.stage_predictions["stage1"]["ensemble"]["test"],
            "Stage2_Ensemble": self.stage_predictions["stage2"]["ensemble"]["test"],
            "Stage3_Transformer": self.stage_predictions["stage3"]["test"],
            "Stage4_Physics": self.stage_predictions["stage4"]["test"],
            "Final_Prediction": self.final_predictions,
            "Prediction_Uncertainty": self.final_uncertainty
        })
        
        # Calculate metrics for all stages
        metrics = {}
        
        # Stage 1
        metrics["Stage1"] = {
            "RMSE": np.sqrt(mean_squared_error(self.y_test, self.stage_predictions["stage1"]["ensemble"]["test"])),
            "MAE": mean_absolute_error(self.y_test, self.stage_predictions["stage1"]["ensemble"]["test"]),
            "R2": r2_score(self.y_test, self.stage_predictions["stage1"]["ensemble"]["test"])
        }
        
        # Stage 2
        metrics["Stage2"] = {
            "RMSE": np.sqrt(mean_squared_error(self.y_test, self.stage_predictions["stage2"]["ensemble"]["test"])),
            "MAE": mean_absolute_error(self.y_test, self.stage_predictions["stage2"]["ensemble"]["test"]),
            "R2": r2_score(self.y_test, self.stage_predictions["stage2"]["ensemble"]["test"])
        }
        
        # Stage 3
        metrics["Stage3"] = {
            "RMSE": np.sqrt(mean_squared_error(self.y_test, self.stage_predictions["stage3"]["test"])),
            "MAE": mean_absolute_error(self.y_test, self.stage_predictions["stage3"]["test"]),
            "R2": r2_score(self.y_test, self.stage_predictions["stage3"]["test"])
        }
        
        # Stage 4
        metrics["Stage4"] = {
            "RMSE": np.sqrt(mean_squared_error(self.y_test, self.stage_predictions["stage4"]["test"])),
            "MAE": mean_absolute_error(self.y_test, self.stage_predictions["stage4"]["test"]),
            "R2": r2_score(self.y_test, self.stage_predictions["stage4"]["test"])
        }
        
        # Final model
        metrics["Final"] = {
            "RMSE": np.sqrt(mean_squared_error(self.y_test, self.final_predictions)),
            "MAE": mean_absolute_error(self.y_test, self.final_predictions),
            "R2": r2_score(self.y_test, self.final_predictions)
        }
        
        # Save metrics
        with open(os.path.join(self.stage_output_dirs["final"], "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)
        
        # Save all predictions
        all_test_predictions.to_csv(os.path.join(self.stage_output_dirs["final"], "all_predictions.csv"), index=False)
        
        # Generate uncertainty correlation plot
        plt.figure(figsize=(10, 6))
        error = np.abs(self.y_test - self.final_predictions)
        plt.scatter(self.final_uncertainty, error, alpha=0.5)
        plt.xlabel("Predicted Uncertainty")
        plt.ylabel("Absolute Error")
        plt.title("Error vs. Uncertainty Correlation")
        plt.savefig(os.path.join(self.stage_output_dirs["final"], "uncertainty_correlation.png"))
        plt.close()
        
        # Generate prediction scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_test, self.final_predictions, alpha=0.5)
        min_val = min(self.y_test.min(), self.final_predictions.min()) - 0.5
        max_val = max(self.y_test.max(), self.final_predictions.max()) + 0.5
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        plt.xlabel("True LogD")
        plt.ylabel("Predicted LogD")
        plt.title(f"LogD Predictions (R² = {metrics['Final']['R2']:.3f}, RMSE = {metrics['Final']['RMSE']:.3f})")
        plt.savefig(os.path.join(self.stage_output_dirs["final"], "prediction_scatter.png"))
        plt.close()
        
        # Print final performance summary
        self.log("\nFinal Performance Summary:")
        for stage, metrics_dict in metrics.items():
            self.log(f"{stage} - RMSE: {metrics_dict['RMSE']:.4f}, MAE: {metrics_dict['MAE']:.4f}, R²: {metrics_dict['R2']:.4f}")

# Model definitions

class AttentiveFPModel(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, global_feat_dim, num_layers=3, dropout=0.2, hidden_dim=64):
        super(AttentiveFPModel, self).__init__()
        
        # Number of features in the model
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.global_feat_dim = global_feat_dim
        self.hidden_dim = hidden_dim
        
        # First embed the node features to hidden_dim
        self.node_embedding = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # GATv2Conv layers with proper dimensions
        self.attn_layers = nn.ModuleList()
        for i in range(num_layers):
            # First layer takes hidden_dim as input
            # Subsequent layers take hidden_dim * 4 from previous GAT layer (4 heads)
            in_channels = hidden_dim if i == 0 else hidden_dim * 4
            self.attn_layers.append(
                GATv2Conv(
                    in_channels=in_channels, 
                    out_channels=hidden_dim, 
                    heads=4,
                    dropout=dropout, 
                    edge_dim=edge_feat_dim
                )
            )
        
        self.global_embedding = nn.Sequential(
            nn.Linear(global_feat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Adjust input dimension for readout
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 4 + hidden_dim + 1, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = data.edge_attr
        
        # Node feature embedding
        x = self.node_embedding(x)
        
        # Apply attention layers with residual connections
        for i, attn_layer in enumerate(self.attn_layers):
            x_new = attn_layer(x, edge_index, edge_attr)
            x = x_new
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)
        
        # Readout (graph-level representation)
        x_g = global_mean_pool(x, batch)
        
        # Reshape global_features before processing
        num_graphs = data.num_graphs
        global_features_reshaped = data.global_features.view(num_graphs, -1)
        
        # Process global features
        global_feat = self.global_embedding(global_features_reshaped)
        
        # Ensure stage1_pred is 2D for concatenation
        stage1_pred = data.stage1_pred.unsqueeze(1) if data.stage1_pred.ndim == 1 else data.stage1_pred
        
        # Combine all features
        combined = torch.cat([x_g, global_feat, stage1_pred], dim=1)
        
        # Final prediction
        embedding = combined
        pred = self.readout(combined)
        
        return pred, embedding

class DMPNNModel(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, global_feat_dim, hidden_dim=64, num_layers=3, dropout=0.2):
        super(DMPNNModel, self).__init__()
        
        # Store dimensions
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.global_feat_dim = global_feat_dim
        self.hidden_dim = hidden_dim
        
        self.node_embedding = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.edge_embedding = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(DMPNNLayer(hidden_dim))
        
        self.global_embedding = nn.Sequential(
            nn.Linear(global_feat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim + 1, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = data.edge_attr
        
        # Node feature embedding
        x = self.node_embedding(x)
        
        # Edge feature embedding - ensure it's not empty
        if edge_attr.shape[0] > 0:
            edge_attr = self.edge_embedding(edge_attr)
        else:
            # Handle molecules with no bonds
            edge_attr = torch.empty((0, self.hidden_dim), device=x.device)
        
        # Apply DMPNN layers
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
        
        # Readout (graph-level representation)
        x_g = global_add_pool(x, batch)
        
        # Reshape global_features before processing
        num_graphs = data.num_graphs
        global_features_reshaped = data.global_features.view(num_graphs, -1)
        
        # Process global features
        global_feat = self.global_embedding(global_features_reshaped)
        
        # Ensure stage1_pred is 2D for concatenation
        stage1_pred = data.stage1_pred.unsqueeze(1) if data.stage1_pred.ndim == 1 else data.stage1_pred
        
        # Combine all features
        combined = torch.cat([x_g, global_feat, stage1_pred], dim=1)
        
        # Final prediction
        embedding = combined
        pred = self.readout(combined)
        
        return pred, embedding

class DMPNNLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(DMPNNLayer, self).__init__()
        
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
    def forward(self, x, edge_index, edge_attr):
        # Check if there are edges
        if edge_index.shape[1] == 0:
            return x
            
        # Aggregate messages
        row, col = edge_index
        
        # Create edge messages
        edge_messages = torch.cat([x[row], x[col], edge_attr], dim=1)
        edge_messages = self.message_mlp(edge_messages)
        
        # Aggregate messages at each node
        out = torch.zeros_like(x)
        for i in range(row.size(0)):
            out[row[i]] += edge_messages[i]
        
        # Update node representations
        out = torch.cat([x, out], dim=1)
        out = self.update_mlp(out)
        
        return out

class GATModel(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, global_feat_dim, hidden_dim=64, num_layers=3, dropout=0.2):
        super(GATModel, self).__init__()
        
        # Store dimensions
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.global_feat_dim = global_feat_dim
        self.hidden_dim = hidden_dim
        
        self.node_embedding = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            # First layer takes hidden_dim as input
            # Subsequent layers take hidden_dim * 4 from previous GAT layer
            in_channels = hidden_dim if i == 0 else hidden_dim * 4
            self.gat_layers.append(
                GATv2Conv(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    heads=4,
                    dropout=dropout,
                    edge_dim=edge_feat_dim
                )
            )
        
        self.global_embedding = nn.Sequential(
            nn.Linear(global_feat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 4 + hidden_dim + 1, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = data.edge_attr
        
        # Node feature embedding
        x = self.node_embedding(x)
        
        # Apply GAT layers
        for gat_layer in self.gat_layers:
            x = gat_layer(x, edge_index, edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)
        
        # Readout (graph-level representation)
        x_g = global_mean_pool(x, batch)
        
        # Reshape global_features before processing
        num_graphs = data.num_graphs
        global_features_reshaped = data.global_features.view(num_graphs, -1)
        
        # Process global features
        global_feat = self.global_embedding(global_features_reshaped)
        
        # Ensure stage1_pred is 2D for concatenation
        stage1_pred = data.stage1_pred.unsqueeze(1) if data.stage1_pred.ndim == 1 else data.stage1_pred
        
        # Combine all features
        combined = torch.cat([x_g, global_feat, stage1_pred], dim=1)
        
        # Final prediction
        embedding = combined
        pred = self.readout(combined)
        
        return pred, embedding

class MoleculeDataset(Dataset):
    def __init__(self, smiles, labels, stage1_preds, stage2_preds, gnn_embeds, tokenizer):
        self.smiles = smiles
        self.labels = labels
        self.stage1_preds = stage1_preds
        self.stage2_preds = stage2_preds
        self.gnn_embeds = gnn_embeds
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.smiles)
    
    def __getitem__(self, idx):
        # Tokenize SMILES
        encoding = self.tokenizer(
            self.smiles[idx],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128
        )
        
        # Remove batch dimension
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(self.labels[idx], dtype=torch.float32),
            "stage1_pred": torch.tensor([self.stage1_preds[idx]], dtype=torch.float32),
            "stage2_pred": torch.tensor([self.stage2_preds[idx]], dtype=torch.float32),
            "gnn_embeddings": torch.tensor(self.gnn_embeds[idx], dtype=torch.float32)
        }

class MolecularTransformerModel(nn.Module):
    def __init__(self, base_model_path, config):
        super(MolecularTransformerModel, self).__init__()
        
        # Load base transformer model
        self.transformer = AutoModel.from_pretrained(base_model_path, config=config)
        
        # Get hidden size from config
        hidden_size = config.hidden_size
        
        # GNN embedding projection
        self.gnn_projection = nn.Linear(64, hidden_size)
        
        # Stage 1 & 2 prediction projections
        self.stage1_projection = nn.Linear(1, hidden_size)
        self.stage2_projection = nn.Linear(1, hidden_size)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Prediction head
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, input_ids, attention_mask, gnn_embeddings, stage1_pred, stage2_pred, labels=None):
        # Process with transformer
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        
        # Get [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # Project GNN embeddings
        gnn_proj = self.gnn_projection(gnn_embeddings)
        
        # Project Stage 1 & 2 predictions
        stage1_proj = self.stage1_projection(stage1_pred)
        stage2_proj = self.stage2_projection(stage2_pred)
        
        # Fuse all representations
        fused = torch.cat([cls_output, gnn_proj, stage1_proj, stage2_proj], dim=1)
        fused = self.fusion(fused)
        
        # Make prediction
        prediction = self.regression_head(fused)
        
        # Return different outputs depending on training or inference
        class Outputs:
            def __init__(self, loss=None, predictions=None, embeddings=None):
                self.loss = loss
                self.predictions = predictions
                self.embeddings = embeddings
                
        if labels is not None:
            # Calculate loss
            loss_fn = nn.MSELoss()
            loss = loss_fn(prediction.view(-1), labels.view(-1))
            return Outputs(loss=loss, predictions=prediction, embeddings=fused)
        else:
            # Just return predictions and embeddings for inference
            return Outputs(predictions=prediction, embeddings=fused)

class PhysicsInformedNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(PhysicsInformedNN, self).__init__()
        
        # Main network
        self.main_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Physics-informed layers
        self.physics_layer1 = nn.Linear(input_dim, hidden_dim)
        self.physics_layer2 = nn.Linear(hidden_dim, 1)
        
        # Parameters for physics constraints
        self.logp_weight = nn.Parameter(torch.tensor([1.0]))
        self.ionization_weight = nn.Parameter(torch.tensor([0.5]))
        self.size_weight = nn.Parameter(torch.tensor([0.3]))
        
    def forward(self, x):
        # Main prediction
        prediction = self.main_network(x)
        
        # Physics constraint term (simplified version of LogD physics)
        # LogD = LogP - log(1 + 10^(pH - pKa)) for bases
        # Here we approximate this relationship with a neural network
        physics_hidden = F.relu(self.physics_layer1(x))
        physics_term = self.physics_layer2(physics_hidden)
        
        # Extract specific features if they exist
        # Let's assume the first few features are [stage1_pred, stage2_pred, stage3_pred]
        # and the rest are embeddings and physics features
        try:
            # Previous LogP estimates (from Stage 1)
            logp_estimate = x[:, 0:1]
            
            # Calculate physics-informed correction
            # The physics term should enforce the relationship between LogP and LogD
            physics_consistency = torch.abs(prediction - logp_estimate - physics_term)
        except:
            # Fallback if feature extraction fails
            physics_consistency = torch.abs(physics_term)
        
        return prediction, physics_consistency

    def predict(self, x):
        x_tensor = torch.FloatTensor(x).to(next(self.parameters()).device)
        self.eval()
        with torch.no_grad():
            prediction, _ = self(x_tensor)
            return prediction.cpu().numpy()

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(BayesianLinear, self).__init__()
        
        # Weight parameters and their prior standard deviation
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(0, 0.1))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features).fill_(0.1))
        
        # Bias parameters and their prior standard deviation
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(0, 0.1))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features).fill_(0.1))
        
        # Prior distributions
        self.weight_prior_mu = torch.zeros_like(self.weight_mu)
        self.weight_prior_sigma = torch.ones_like(self.weight_sigma)
        self.bias_prior_mu = torch.zeros_like(self.bias_mu)
        self.bias_prior_sigma = torch.ones_like(self.bias_sigma)
        
    def forward(self, x, enable_sampling=False):
        if self.training or enable_sampling:
            # Sample weights and biases from their posterior distributions
            weight = self.weight_mu + torch.randn_like(self.weight_sigma) * torch.log1p(torch.exp(self.weight_sigma))
            bias = self.bias_mu + torch.randn_like(self.bias_sigma) * torch.log1p(torch.exp(self.bias_sigma))
        else:
            # Use mean of the posterior during evaluation
            weight = self.weight_mu
            bias = self.bias_mu
            
        return F.linear(x, weight, bias)
    
    def kl_loss(self):
        """Calculate KL divergence between posterior and prior distributions."""
        kl_weight = self._kl_divergence(
            self.weight_mu, self.weight_sigma,
            self.weight_prior_mu, self.weight_prior_sigma
        )
        
        kl_bias = self._kl_divergence(
            self.bias_mu, self.bias_sigma,
            self.bias_prior_mu, self.bias_prior_sigma
        )
        
        return kl_weight + kl_bias
    
    def _kl_divergence(self, mu_q, sigma_q, mu_p, sigma_p):
        """Calculate KL divergence between two Gaussian distributions."""
        var_q = torch.log1p(torch.exp(sigma_q)).pow(2)
        var_p = sigma_p.pow(2)
        
        kl = 0.5 * (
            (var_q / var_p) + 
            (mu_q - mu_p).pow(2) / var_p - 
            1 + 
            torch.log(var_p) - 
            torch.log(var_q)
        )
        
        return kl.sum()

class BayesianMetaLearner(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout_rate=0.2):
        super(BayesianMetaLearner, self).__init__()
        
        # Aleatoric uncertainty (data noise)
        self.aleatoric_net = nn.Sequential(
            BayesianLinear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            BayesianLinear(hidden_dim, 1)
        )
        
        # Mean prediction network
        self.mean_net = nn.Sequential(
            BayesianLinear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            BayesianLinear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            BayesianLinear(hidden_dim // 2, 1)
        )
        
        # Heteroscedastic uncertainty estimator (data-dependent uncertainty)
        self.uncertainty_net = nn.Sequential(
            BayesianLinear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            BayesianLinear(hidden_dim, 1),
            nn.Softplus()  # Ensure positive uncertainty
        )
    
    def forward(self, x, enable_dropout=False):
        # Mean prediction
        mean = self.mean_net(x, enable_dropout)
        
        # Uncertainty prediction (log variance)
        log_var = self.uncertainty_net(x, enable_dropout) + 1e-6  # Add small constant for numerical stability
        
        return torch.cat([mean, log_var], dim=1)
    
    def nll_loss(self, output, target):
        """Negative log-likelihood loss for heteroscedastic uncertainty."""
        mean = output[:, 0]
        log_var = output[:, 1]
        
        # Gaussian negative log-likelihood
        loss = 0.5 * (torch.log(log_var) + (target - mean).pow(2) / log_var)
        
        return loss.mean()
    
    def predict(self, x):
        """Generate predictions with uncertainty."""
        x_tensor = torch.FloatTensor(x).to(next(self.parameters()).device)
        
        self.eval()
        
        # Multiple forward passes for Monte Carlo dropout
        n_samples = 30
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                output = self(x_tensor, enable_dropout=True)
                predictions.append(output[:, 0:1])  # Just the mean predictions
        
        # Stack along a new dimension
        predictions = torch.stack(predictions, dim=0)
        
        # Calculate mean and standard deviation across samples
        pred_mean = predictions.mean(dim=0).cpu().numpy()
        pred_std = predictions.std(dim=0).cpu().numpy()
        
        return pred_mean, pred_std

def parse_arguments():
    parser = argparse.ArgumentParser(description="Advanced LogD Prediction Workflow")
    
    parser.add_argument("--input", required=True, help="Input training data file (CSV)")
    parser.add_argument("--output-dir", required=True, help="Directory to save models and results")
    parser.add_argument("--use-infile-descriptors", action="store_true", help="Use descriptors in input file instead of calculating them")
    parser.add_argument("--smiles-col", default="SMILES", help="Column name containing SMILES strings")
    parser.add_argument("--target", required=True, help="Column name containing LogD values")
    parser.add_argument("--transformer-model-path", help="Path to pretrained transformer model")
    parser.add_argument("--external-test", required=True, help="External test set file (CSV)")
    parser.add_argument("--zero-value", type=float, default=1e-6, help="Value to replace zeros with")
    parser.add_argument("--nan-value", type=float, default=0.0, help="Value to replace NaNs with")
    parser.add_argument("--descriptors-prefixes", type=str, default="desc_", 
                       help="Comma-separated list of prefixes to identify descriptor columns (e.g., 'DESC_,MORDRED_')")
    parser.add_argument("--fingerprint-prefixes", type=str, default="fp_",
                       help="Comma-separated list of prefixes to identify fingerprint columns (e.g., 'FP_,ECFP_')")
    parser.add_argument("--force-cpu", action="store_true", 
                       help="Force CPU usage instead of GPU/MPS for better compatibility")
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Print startup info
    print("-- LogD Prediction Multi-Stage Pipeline")
    print(f"-- Input file: {args.input}")
    print(f"-- External test: {args.external_test}")
    print(f"-- Output directory: {args.output_dir}")
    print(f"-- SMILES column: {args.smiles_col}")
    print(f"-- Target column: {args.target}")
    
    # Create and run pipeline
    pipeline = LogDPredictionPipeline(args)
    pipeline.run_pipeline()

if __name__ == "__main__":
    main()