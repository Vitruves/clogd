#!/usr/bin/env python3

import os
import sys
import argparse
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, QED, Crippen, rdMolDescriptors, AllChem, rdPartialCharges
from rdkit.ML.Descriptors import MoleculeDescriptors # type: ignore
from rdkit.Chem.EState import Fingerprinter as EStateFingerprinter
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator, GetAtomPairGenerator, GetTopologicalTorsionGenerator, GetRDKitFPGenerator
from pathlib import Path
import multiprocessing
import time
import gc
import psutil
import tempfile
import signal
import shutil
import atexit
from concurrent.futures import ProcessPoolExecutor, as_completed

TEMP_DIR = None
TEMP_FILES = []
INTERRUPTED = False
EXECUTOR = None

np.seterr(all='ignore')

def exit_handler():
    global EXECUTOR
    if EXECUTOR:
        EXECUTOR.shutdown(wait=False, cancel_futures=True)
    cleanup_temp_files()

def signal_handler(sig, frame):
    global INTERRUPTED, EXECUTOR
    INTERRUPTED = True
    print_message("\nInterrupted by user. Cleaning up...")
    if EXECUTOR:
        EXECUTOR.shutdown(wait=False, cancel_futures=True)
    cleanup_temp_files()
    sys.exit(1)

def cleanup_temp_files():
    global TEMP_DIR, TEMP_FILES
    for file in TEMP_FILES:
        try:
            if os.path.exists(file):
                os.remove(file)
        except:
            pass
    
    if TEMP_DIR and os.path.exists(TEMP_DIR):
        try:
            shutil.rmtree(TEMP_DIR)
        except:
            pass

def print_message(message):
    sys.stdout.write(f"-- {message}\n")
    sys.stdout.flush()

def update_progress(progress, description="Computing"):
    sys.stdout.write(f"\r-- [ {progress:6.2f}% ] {description}")
    sys.stdout.flush()

def get_available_descriptors():
    descriptors = {
        "2d": {
            "basic": [x[0] for x in Descriptors._descList],
            "lipinski": ["NumHDonors", "NumHAcceptors", "NumRotatableBonds", "HeavyAtomCount", "FractionCSP3"],
            "qed": ["QED", "QED_MW", "QED_ALOGP", "QED_HBA", "QED_HBD", "QED_PSA", "QED_ROTB", "QED_AROM", "QED_ALERTS"],
            "crippen": ["Crippen_LogP", "Crippen_MR"],
            "topology": ["ExactMolWt", "FractionCSP3", "TPSA", "NumRotatableBonds", "NumHBD", "NumHBA", 
                        "NumLipinskiHBD", "NumLipinskiHBA", "NumHeavyAtoms", "NumHeteroatoms", "NumAtoms"],
            "rings": ["NumRings", "NumAromaticRings", "NumSaturatedRings", "NumAliphaticRings", 
                     "NumAromaticHeterocycles", "NumAromaticCarbocycles", "NumSaturatedHeterocycles", 
                     "NumSaturatedCarbocycles", "NumAliphaticHeterocycles", "NumAliphaticCarbocycles"],
            "stereo": ["NumAtomStereoCenters", "NumUnspecifiedAtomStereoCenters"],
            "fragments": ["NumAmideBonds", "NumSpiroAtoms", "NumBridgeheadAtoms"],
            "connectivity": ["HallKierAlpha", "Kappa1", "Kappa2", "Kappa3", 
                           "Chi0v", "Chi0n", "Chi1v", "Chi1n", "Chi2v", "Chi2n", 
                           "Chi3v", "Chi3n", "Chi4v", "Chi4n", "Phi", "LabuteASA"],
            "vsa": ["SlogP_VSA", "SMR_VSA", "PEOE_VSA"],
            "charges": ["GasteigerMin", "GasteigerMax", "GasteigerMean", "GasteigerStd"],
            "fingerprints": ["Morgan2_Count", "Morgan2_Density", "Morgan3_Count", "Morgan3_Density", 
                            "AtomPair_Count", "TopologicalTorsion_Count", "RDKit_Count", "RDKit_Density", 
                            "MACCS_Count", "MACCS_Density"],
            "estate": ["EState_VSA", "EStateMin", "EStateMax", "EStateMean", "EStateStd"],
            "fragments_count": ["amide", "acid", "alcohol", "aldehyde", "ketone", "ether", "ester", 
                               "phenol", "thioether", "sulfonamide", "sulfoxide", "thiocarbonyl", 
                               "sulfonyl", "amine", "nitrile", "nitro", "azide", "halogen", 
                               "aromatic", "heterocycle", "bicyclic", "chiral_center"]
        },
        "3d": {
            "shape": ["NPR1", "NPR2", "PMI1", "PMI2", "PMI3", "RadiusOfGyration", 
                     "InertialShapeFactor", "Eccentricity", "Asphericity", "SpherocityIndex"],
            "autocorr3d": ["AUTOCORR3D"],
            "rdf": ["RDF"],
            "morse": ["MORSE"],
            "whim": ["WHIM"],
            "getaway": ["GETAWAY"],
            "coulomb": ["CoulombMatSize", "CoulombMatMax", "CoulombMatMin", "CoulombMatMean"],
            "charges3d": ["EEM_Min", "EEM_Max", "EEM_Mean", "EEM_Std"],
            "volume": ["MolVolume"],
            "usr": ["USR", "USRCAT"],
            "oxidation": ["OxidationNum_Min", "OxidationNum_Max", "OxidationNum_Mean", "OxidationNum_Std"],
            "surface": ["SurfaceArea"]
        }
    }
    return descriptors

def print_available_descriptors():
    descriptors = get_available_descriptors()
    
    print_message("Available RDKit Descriptors:")
    print_message("============================")
    
    print_message("\n2D Descriptors:")
    for category, desc_list in descriptors["2d"].items():
        print_message(f"  {category} ({len(desc_list)})")
        for i, desc in enumerate(desc_list):
            if i < 10 or i > len(desc_list) - 3:
                print_message(f"    - {desc}")
            elif i == 10:
                print_message(f"    - ... ({len(desc_list) - 13} more)")
    
    print_message("\n3D Descriptors:")
    for category, desc_list in descriptors["3d"].items():
        print_message(f"  {category} ({len(desc_list)})")
        for desc in desc_list:
            print_message(f"    - {desc}")
    
    print_message("\nUsage Examples:")
    print_message("  --compute 2d                # All 2D descriptors")
    print_message("  --compute 3d                # All 3D descriptors")
    print_message("  --compute all               # All descriptors")
    print_message("  --compute basic,fingerprints # Specific categories")
    print_message("  --compute NumHDonors,TPSA,Morgan2_Count # Specific descriptors")

class ProgressTracker:
    def __init__(self, total_steps, description="Computing"):
        self.total_steps = total_steps
        self.completed_steps = multiprocessing.Value('i', 0)
        self.description = description
        update_progress(0.0, self.description)
    
    def increment(self, steps=1):
        with self.completed_steps.get_lock():
            self.completed_steps.value += steps
            completed = self.completed_steps.value
            if self.total_steps > 0:
                current_progress = min(100, (completed / self.total_steps * 100))
                update_progress(current_progress, self.description)
    
    def finalize(self):
        update_progress(100.0, self.description)
        sys.stdout.write("\n")
        sys.stdout.flush()

def check_memory_usage(threshold=0.7):
    try:
        vm = psutil.virtual_memory()
        return vm.percent >= (threshold * 100)
    except:
        return False

def save_to_disk(data, file_path):
    import pickle
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(data, f, protocol=4)
        return True
    except:
        return False

def load_from_disk(file_path):
    import pickle
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except:
        return None

def prepare_3d_molecule(mol):
    if mol is None:
        return None
    
    try:
        if mol.GetNumConformers() > 0:
            conf = mol.GetConformer()
            if conf.Is3D():
                return mol
        
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.UFFOptimizeMolecule(mol)
        return mol
    except:
        return None

def compute_rdkit_descriptors(mol_data):
    smiles, compute_3d, prefix, idx, compute_descriptors = mol_data
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return idx, {}, smiles
        
        result_dict = {}
        available_descriptors = get_available_descriptors()
        descriptors_to_compute = set()
        
        if compute_descriptors == "all":
            for dimension in ["2d", "3d"]:
                for category in available_descriptors[dimension]:
                    descriptors_to_compute.update(available_descriptors[dimension][category])
        elif compute_descriptors == "2d":
            for category in available_descriptors["2d"]:
                descriptors_to_compute.update(available_descriptors["2d"][category])
        elif compute_descriptors == "3d":
            for category in available_descriptors["3d"]:
                descriptors_to_compute.update(available_descriptors["3d"][category])
        elif compute_descriptors:
            categories = compute_descriptors.split(",")
            for category in categories:
                category = category.strip()
                found = False
                
                for dimension in ["2d", "3d"]:
                    if category in available_descriptors[dimension]:
                        descriptors_to_compute.update(available_descriptors[dimension][category])
                        found = True
                
                if not found:
                    descriptors_to_compute.add(category)
        
        compute_all = not compute_descriptors or compute_descriptors == "all" or compute_descriptors == "2d"
        compute_all_3d = not compute_descriptors or compute_descriptors == "all" or compute_descriptors == "3d"
        
        # 1. Basic molecular properties from standard RDKit descriptors
        if compute_all or any(desc in descriptors_to_compute for desc in available_descriptors["2d"]["basic"]):
            calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
            basic_descriptors = calc.CalcDescriptors(mol)
            for i, name in enumerate([x[0] for x in Descriptors._descList]):
                if compute_all or name in descriptors_to_compute:
                    try:
                        value = basic_descriptors[i]
                        if value is not None and not np.isnan(value):
                            result_dict[f"{prefix}_{name}"] = float(value)
                    except:
                        pass
        
        # 2. Lipinski properties
        if compute_all or any(desc in descriptors_to_compute for desc in available_descriptors["2d"]["lipinski"]):
            try:
                if compute_all or "NumHDonors" in descriptors_to_compute:
                    result_dict[f"{prefix}_NumHDonors"] = Lipinski.NumHDonors(mol)
                if compute_all or "NumHAcceptors" in descriptors_to_compute:
                    result_dict[f"{prefix}_NumHAcceptors"] = Lipinski.NumHAcceptors(mol)
                if compute_all or "NumRotatableBonds" in descriptors_to_compute:
                    result_dict[f"{prefix}_NumRotatableBonds"] = Lipinski.NumRotatableBonds(mol)
                if compute_all or "HeavyAtomCount" in descriptors_to_compute:
                    result_dict[f"{prefix}_HeavyAtomCount"] = Lipinski.HeavyAtomCount(mol)
                if compute_all or "FractionCSP3" in descriptors_to_compute:
                    result_dict[f"{prefix}_FractionCSP3"] = Lipinski.FractionCSP3(mol)
            except:
                pass
        
        # 3. QED (Quantitative Estimate of Drug-likeness) properties
        if compute_all or any(desc in descriptors_to_compute for desc in available_descriptors["2d"]["qed"]):
            try:
                qed_props = QED.properties(mol)
                if compute_all or "QED" in descriptors_to_compute:
                    result_dict[f"{prefix}_QED"] = QED.qed(mol)
                if compute_all or "QED_MW" in descriptors_to_compute:
                    result_dict[f"{prefix}_QED_MW"] = qed_props.MW
                if compute_all or "QED_ALOGP" in descriptors_to_compute:
                    result_dict[f"{prefix}_QED_ALOGP"] = qed_props.ALOGP
                if compute_all or "QED_HBA" in descriptors_to_compute:
                    result_dict[f"{prefix}_QED_HBA"] = qed_props.HBA
                if compute_all or "QED_HBD" in descriptors_to_compute:
                    result_dict[f"{prefix}_QED_HBD"] = qed_props.HBD
                if compute_all or "QED_PSA" in descriptors_to_compute:
                    result_dict[f"{prefix}_QED_PSA"] = qed_props.PSA
                if compute_all or "QED_ROTB" in descriptors_to_compute:
                    result_dict[f"{prefix}_QED_ROTB"] = qed_props.ROTB
                if compute_all or "QED_AROM" in descriptors_to_compute:
                    result_dict[f"{prefix}_QED_AROM"] = qed_props.AROM
                if compute_all or "QED_ALERTS" in descriptors_to_compute:
                    result_dict[f"{prefix}_QED_ALERTS"] = qed_props.ALERTS
            except:
                pass
        
        # 4. Crippen descriptors
        if compute_all or any(desc in descriptors_to_compute for desc in available_descriptors["2d"]["crippen"]):
            try:
                if compute_all or "Crippen_LogP" in descriptors_to_compute:
                    result_dict[f"{prefix}_Crippen_LogP"] = Crippen.MolLogP(mol)
                if compute_all or "Crippen_MR" in descriptors_to_compute:
                    result_dict[f"{prefix}_Crippen_MR"] = Crippen.MolMR(mol)
            except:
                pass
        
        # 5. rdMolDescriptors - topological and functional group counts
        if compute_all:
            try:
                # Basic molecular properties
                result_dict[f"{prefix}_ExactMolWt"] = rdMolDescriptors.CalcExactMolWt(mol)
                result_dict[f"{prefix}_FractionCSP3"] = rdMolDescriptors.CalcFractionCSP3(mol)
                result_dict[f"{prefix}_TPSA"] = rdMolDescriptors.CalcTPSA(mol)
                result_dict[f"{prefix}_NumRotatableBonds"] = rdMolDescriptors.CalcNumRotatableBonds(mol)
                result_dict[f"{prefix}_NumHBD"] = rdMolDescriptors.CalcNumHBD(mol)
                result_dict[f"{prefix}_NumHBA"] = rdMolDescriptors.CalcNumHBA(mol)
                result_dict[f"{prefix}_NumLipinskiHBD"] = rdMolDescriptors.CalcNumLipinskiHBD(mol)
                result_dict[f"{prefix}_NumLipinskiHBA"] = rdMolDescriptors.CalcNumLipinskiHBA(mol)
                result_dict[f"{prefix}_NumHeavyAtoms"] = rdMolDescriptors.CalcNumHeavyAtoms(mol)
                result_dict[f"{prefix}_NumHeteroatoms"] = rdMolDescriptors.CalcNumHeteroatoms(mol)
                result_dict[f"{prefix}_NumAtoms"] = rdMolDescriptors.CalcNumAtoms(mol)
                
                # Ring properties
                result_dict[f"{prefix}_NumRings"] = rdMolDescriptors.CalcNumRings(mol)
                result_dict[f"{prefix}_NumAromaticRings"] = rdMolDescriptors.CalcNumAromaticRings(mol)
                result_dict[f"{prefix}_NumSaturatedRings"] = rdMolDescriptors.CalcNumSaturatedRings(mol)
                result_dict[f"{prefix}_NumAliphaticRings"] = rdMolDescriptors.CalcNumAliphaticRings(mol)
                result_dict[f"{prefix}_NumAromaticHeterocycles"] = rdMolDescriptors.CalcNumAromaticHeterocycles(mol)
                result_dict[f"{prefix}_NumAromaticCarbocycles"] = rdMolDescriptors.CalcNumAromaticCarbocycles(mol)
                result_dict[f"{prefix}_NumSaturatedHeterocycles"] = rdMolDescriptors.CalcNumSaturatedHeterocycles(mol)
                result_dict[f"{prefix}_NumSaturatedCarbocycles"] = rdMolDescriptors.CalcNumSaturatedCarbocycles(mol)
                result_dict[f"{prefix}_NumAliphaticHeterocycles"] = rdMolDescriptors.CalcNumAliphaticHeterocycles(mol)
                result_dict[f"{prefix}_NumAliphaticCarbocycles"] = rdMolDescriptors.CalcNumAliphaticCarbocycles(mol)
                
                # Stereo properties
                result_dict[f"{prefix}_NumAtomStereoCenters"] = rdMolDescriptors.CalcNumAtomStereoCenters(mol)
                result_dict[f"{prefix}_NumUnspecifiedAtomStereoCenters"] = rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters(mol)
                
                # Special atom counts
                result_dict[f"{prefix}_NumAmideBonds"] = rdMolDescriptors.CalcNumAmideBonds(mol)
                result_dict[f"{prefix}_NumSpiroAtoms"] = rdMolDescriptors.CalcNumSpiroAtoms(mol)
                result_dict[f"{prefix}_NumBridgeheadAtoms"] = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
                
                # Hall-Kier Alpha value
                result_dict[f"{prefix}_HallKierAlpha"] = rdMolDescriptors.CalcHallKierAlpha(mol)
                
                # Kappa descriptors
                result_dict[f"{prefix}_Kappa1"] = rdMolDescriptors.CalcKappa1(mol)
                result_dict[f"{prefix}_Kappa2"] = rdMolDescriptors.CalcKappa2(mol)
                result_dict[f"{prefix}_Kappa3"] = rdMolDescriptors.CalcKappa3(mol)
                
                # Chi connectivity descriptors
                result_dict[f"{prefix}_Chi0v"] = rdMolDescriptors.CalcChi0v(mol)
                result_dict[f"{prefix}_Chi0n"] = rdMolDescriptors.CalcChi0n(mol)
                result_dict[f"{prefix}_Chi1v"] = rdMolDescriptors.CalcChi1v(mol)
                result_dict[f"{prefix}_Chi1n"] = rdMolDescriptors.CalcChi1n(mol)
                result_dict[f"{prefix}_Chi2v"] = rdMolDescriptors.CalcChi2v(mol)
                result_dict[f"{prefix}_Chi2n"] = rdMolDescriptors.CalcChi2n(mol)
                result_dict[f"{prefix}_Chi3v"] = rdMolDescriptors.CalcChi3v(mol)
                result_dict[f"{prefix}_Chi3n"] = rdMolDescriptors.CalcChi3n(mol)
                result_dict[f"{prefix}_Chi4v"] = rdMolDescriptors.CalcChi4v(mol)
                result_dict[f"{prefix}_Chi4n"] = rdMolDescriptors.CalcChi4n(mol)
                
                # Phi
                result_dict[f"{prefix}_Phi"] = rdMolDescriptors.CalcPhi(mol)
                
                # Labute ASA
                result_dict[f"{prefix}_LabuteASA"] = rdMolDescriptors.CalcLabuteASA(mol)
                
                # Molecular formula
                result_dict[f"{prefix}_MolFormula"] = rdMolDescriptors.CalcMolFormula(mol)
                
                # BCUT2D descriptors
                bcuts = rdMolDescriptors.BCUT2D(mol)
                if len(bcuts) >= 6:
                    result_dict[f"{prefix}_BCUT2D_MWLOW"] = bcuts[0]
                    result_dict[f"{prefix}_BCUT2D_MWHIGH"] = bcuts[1]
                    result_dict[f"{prefix}_BCUT2D_CHGLOW"] = bcuts[2]
                    result_dict[f"{prefix}_BCUT2D_CHGHIGH"] = bcuts[3]
                    result_dict[f"{prefix}_BCUT2D_LOGPLOW"] = bcuts[4]
                    result_dict[f"{prefix}_BCUT2D_LOGPHIGH"] = bcuts[5]
                
                # VSA descriptors
                slogp_vsa = rdMolDescriptors.SlogP_VSA_(mol)
                smr_vsa = rdMolDescriptors.SMR_VSA_(mol)
                peoe_vsa = rdMolDescriptors.PEOE_VSA_(mol)
                
                for i, value in enumerate(slogp_vsa):
                    result_dict[f"{prefix}_SlogP_VSA{i+1}"] = value
                
                for i, value in enumerate(smr_vsa):
                    result_dict[f"{prefix}_SMR_VSA{i+1}"] = value
                
                for i, value in enumerate(peoe_vsa):
                    result_dict[f"{prefix}_PEOE_VSA{i+1}"] = value
                
                # MQNs (Molecular Quantum Numbers) descriptors
                mqns = rdMolDescriptors.MQNs_(mol)
                for i, value in enumerate(mqns):
                    result_dict[f"{prefix}_MQN{i+1}"] = value
                
                # Autocorrelation descriptors
                autocorr2d = rdMolDescriptors.CalcAUTOCORR2D(mol)
                for i, value in enumerate(autocorr2d):
                    result_dict[f"{prefix}_AUTOCORR2D_{i+1}"] = value
            except:
                pass
        
        # 6. Partial charges
        if compute_all or any(desc in descriptors_to_compute for desc in available_descriptors["2d"]["charges"]):
            try:
                Chem.rdPartialCharges.ComputeGasteigerCharges(mol)
                charges = [atom.GetDoubleProp("_GasteigerCharge") for atom in mol.GetAtoms()]
                if charges:
                    if compute_all or "GasteigerMin" in descriptors_to_compute:
                        result_dict[f"{prefix}_GasteigerMin"] = min(charges)
                    if compute_all or "GasteigerMax" in descriptors_to_compute:
                        result_dict[f"{prefix}_GasteigerMax"] = max(charges)
                    if compute_all or "GasteigerMean" in descriptors_to_compute:
                        result_dict[f"{prefix}_GasteigerMean"] = sum(charges) / len(charges)
                    if compute_all or "GasteigerStd" in descriptors_to_compute:
                        result_dict[f"{prefix}_GasteigerStd"] = np.std(charges)
            except:
                pass
        
        # 7. Fingerprints using the new recommended generators (fixes deprecation warnings)
        if compute_all or any(desc in descriptors_to_compute for desc in available_descriptors["2d"]["fingerprints"]):
            try:
                # Morgan fingerprints
                morgan_gen = GetMorganGenerator(radius=2, fpSize=2048)
                morgan_fp = morgan_gen.GetFingerprint(mol)
                if compute_all or "Morgan2_Count" in descriptors_to_compute:
                    result_dict[f"{prefix}_Morgan2_Count"] = morgan_fp.GetNumOnBits()
                if compute_all or "Morgan2_Density" in descriptors_to_compute:
                    result_dict[f"{prefix}_Morgan2_Density"] = morgan_fp.GetNumOnBits() / 2048.0
                
                # Morgan fingerprints radius 3
                morgan_gen3 = GetMorganGenerator(radius=3, fpSize=2048)
                morgan_fp3 = morgan_gen3.GetFingerprint(mol)
                if compute_all or "Morgan3_Count" in descriptors_to_compute:
                    result_dict[f"{prefix}_Morgan3_Count"] = morgan_fp3.GetNumOnBits()
                if compute_all or "Morgan3_Density" in descriptors_to_compute:
                    result_dict[f"{prefix}_Morgan3_Density"] = morgan_fp3.GetNumOnBits() / 2048.0
                
                # Atom pair fingerprints
                ap_gen = GetAtomPairGenerator()
                ap_fp = ap_gen.GetSparseCountFingerprint(mol)
                if compute_all or "AtomPair_Count" in descriptors_to_compute:
                    result_dict[f"{prefix}_AtomPair_Count"] = len(ap_fp.GetNonzeroElements())
                
                # Topological torsion fingerprints
                tt_gen = GetTopologicalTorsionGenerator()
                tt_fp = tt_gen.GetSparseCountFingerprint(mol)
                if compute_all or "TopologicalTorsion_Count" in descriptors_to_compute:
                    result_dict[f"{prefix}_TopologicalTorsion_Count"] = len(tt_fp.GetNonzeroElements())
                
                # RDKit fingerprints
                rdkit_gen = GetRDKitFPGenerator(fpSize=2048)
                rdkit_fp = rdkit_gen.GetFingerprint(mol)
                if compute_all or "RDKit_Count" in descriptors_to_compute:
                    result_dict[f"{prefix}_RDKit_Count"] = rdkit_fp.GetNumOnBits()
                if compute_all or "RDKit_Density" in descriptors_to_compute:
                    result_dict[f"{prefix}_RDKit_Density"] = rdkit_fp.GetNumOnBits() / 2048.0
                
                # MACCS Keys
                maccs_fp = Chem.rdMolDescriptors.GetMACCSKeysFingerprint(mol)
                if compute_all or "MACCS_Count" in descriptors_to_compute:
                    result_dict[f"{prefix}_MACCS_Count"] = maccs_fp.GetNumOnBits()
                if compute_all or "MACCS_Density" in descriptors_to_compute:
                    result_dict[f"{prefix}_MACCS_Density"] = maccs_fp.GetNumOnBits() / 166.0
            except:
                pass
        
        # 8. EState values and indices
        if compute_all or any(desc in descriptors_to_compute for desc in available_descriptors["2d"]["estate"]):
            try:
                from rdkit.Chem.EState import EState_VSA
                estate_vsa = EState_VSA.EState_VSA_(mol)
                for i, value in enumerate(estate_vsa):
                    result_dict[f"{prefix}_EState_VSA{i+1}"] = value
                
                from rdkit.Chem.EState import EState
                estate_indices = EState.EStateIndices(mol)
                if estate_indices:
                    if compute_all or "EStateMin" in descriptors_to_compute:
                        result_dict[f"{prefix}_EStateMin"] = min(estate_indices)
                    if compute_all or "EStateMax" in descriptors_to_compute:
                        result_dict[f"{prefix}_EStateMax"] = max(estate_indices)
                    if compute_all or "EStateMean" in descriptors_to_compute:
                        result_dict[f"{prefix}_EStateMean"] = sum(estate_indices) / len(estate_indices)
                    if compute_all or "EStateStd" in descriptors_to_compute:
                        result_dict[f"{prefix}_EStateStd"] = np.std(estate_indices)
            except:
                pass
        
        # 9. Additional structural information
        if compute_all or any(desc in descriptors_to_compute for desc in available_descriptors["2d"]["fragments_count"]):
            try:
                # Count fragment types
                fragments = {
                    "amide": "[NX3,NX4+][CX3](=[OX1])[#6]",
                    "acid": "[OX2H][CX3](=[OX1])[#6]",
                    "alcohol": "[OX2H][CX4;!$(C([OX2H])[O,S,#7,#15])]",
                    "aldehyde": "[CX3H]=[OX1]",
                    "ketone": "[#6][CX3](=[OX1])[#6]",
                    "ether": "[OD2]([#6])[#6]",
                    "ester": "[#6][CX3](=[OX1])[OX2][#6]",
                    "phenol": "[OX2H][cX3]:[c]",
                    "thioether": "[#16X2]([#6])[#6]",
                    "sulfonamide": "[#16X4]([OX1])([OX1])[NX3]",
                    "sulfoxide": "[#16X3][#6]",
                    "thiocarbonyl": "[#6X3]=[#16X1]",
                    "sulfonyl": "[#6][SX4](=[OX1])(=[OX1])[#6]",
                    "amine": "[NX3;H2,H1,H0;!$(NC=[!#6]);!$(NC#[!#6])]",
                    "nitrile": "[NX1]#[CX2]",
                    "nitro": "[$([NX3](=O)=O),$([NX3+](=O)[O-])]",
                    "azide": "[$(*-[NX2-]-[NX2+]#[NX1]),$(*-[NX2]=[NX2+]=[NX1-])]",
                    "halogen": "[F,Cl,Br,I]",
                    "aromatic": "a",
                    "heterocycle": "[!#6;r]",
                    "bicyclic": "[$([*R2]([*R])([*R]))]",
                    "chiral_center": "[*;X4;v4](*)(*)(*)(*)"
                }
                
                for name, smarts in fragments.items():
                    if compute_all or name in descriptors_to_compute:
                        try:
                            pattern = Chem.MolFromSmarts(smarts)
                            if pattern:
                                matches = mol.GetSubstructMatches(pattern)
                                result_dict[f"{prefix}_Has{name.capitalize()}"] = 1 if matches else 0
                                result_dict[f"{prefix}_Num{name.capitalize()}"] = len(matches)
                        except:
                            pass
                
                # Atom type counts
                atom_types = {}
                for atom in mol.GetAtoms():
                    symbol = atom.GetSymbol()
                    if symbol in atom_types:
                        atom_types[symbol] += 1
                    else:
                        atom_types[symbol] = 1
                
                for symbol, count in atom_types.items():
                    result_dict[f"{prefix}_Num{symbol}"] = count
            except:
                pass
        
        # 10. 3D descriptors if requested
        if compute_3d and (compute_all_3d or any(desc in descriptors_to_compute for dimension in ["3d"] for desc in available_descriptors[dimension].values())):
            try:
                mol3d = prepare_3d_molecule(mol)
                if mol3d:
                    # PMI and shape descriptors
                    if compute_all_3d or any(desc in descriptors_to_compute for desc in available_descriptors["3d"]["shape"]):
                        result_dict[f"{prefix}_NPR1"] = rdMolDescriptors.CalcNPR1(mol3d)
                        result_dict[f"{prefix}_NPR2"] = rdMolDescriptors.CalcNPR2(mol3d)
                        result_dict[f"{prefix}_PMI1"] = rdMolDescriptors.CalcPMI1(mol3d)
                        result_dict[f"{prefix}_PMI2"] = rdMolDescriptors.CalcPMI2(mol3d)
                        result_dict[f"{prefix}_PMI3"] = rdMolDescriptors.CalcPMI3(mol3d)
                        result_dict[f"{prefix}_RadiusOfGyration"] = rdMolDescriptors.CalcRadiusOfGyration(mol3d)
                        result_dict[f"{prefix}_InertialShapeFactor"] = rdMolDescriptors.CalcInertialShapeFactor(mol3d)
                        result_dict[f"{prefix}_Eccentricity"] = rdMolDescriptors.CalcEccentricity(mol3d)
                        result_dict[f"{prefix}_Asphericity"] = rdMolDescriptors.CalcAsphericity(mol3d)
                        result_dict[f"{prefix}_SpherocityIndex"] = rdMolDescriptors.CalcSpherocityIndex(mol3d)
                    
                    # 3D AUTOCORR
                    if compute_all_3d or "AUTOCORR3D" in descriptors_to_compute:
                        autocorr3d = rdMolDescriptors.CalcAUTOCORR3D(mol3d)
                        for i, value in enumerate(autocorr3d):
                            result_dict[f"{prefix}_AUTOCORR3D_{i+1}"] = value
                    
                    # RDF descriptors
                    if compute_all_3d or "RDF" in descriptors_to_compute:
                        rdf = rdMolDescriptors.CalcRDF(mol3d)
                        for i, value in enumerate(rdf):
                            result_dict[f"{prefix}_RDF_{i+1}"] = value
                    
                    # MORSE descriptors
                    if compute_all_3d or "MORSE" in descriptors_to_compute:
                        morse = rdMolDescriptors.CalcMORSE(mol3d)
                        for i, value in enumerate(morse):
                            result_dict[f"{prefix}_MORSE_{i+1}"] = value
                    
                    # WHIM descriptors
                    if compute_all_3d or "WHIM" in descriptors_to_compute:
                        whim = rdMolDescriptors.CalcWHIM(mol3d)
                        for i, value in enumerate(whim):
                            result_dict[f"{prefix}_WHIM_{i+1}"] = value
                    
                    # GETAWAY descriptors
                    if compute_all_3d or "GETAWAY" in descriptors_to_compute:
                        getaway = rdMolDescriptors.CalcGETAWAY(mol3d)
                        for i, value in enumerate(getaway):
                            result_dict[f"{prefix}_GETAWAY_{i+1}"] = value
                    
                    # Volume and surface area
                    if compute_all_3d or any(desc in descriptors_to_compute for desc in available_descriptors["3d"]["coulomb"]):
                        vol = rdMolDescriptors.CalcCoulombMat(mol3d)
                        if len(vol) > 0:
                            result_dict[f"{prefix}_CoulombMatSize"] = len(vol)
                            result_dict[f"{prefix}_CoulombMatMax"] = np.max(vol)
                            result_dict[f"{prefix}_CoulombMatMin"] = np.min(vol)
                            result_dict[f"{prefix}_CoulombMatMean"] = np.mean(vol)
                    
                    # EEM charges
                    if compute_all_3d or any(desc in descriptors_to_compute for desc in available_descriptors["3d"]["charges3d"]):
                        try:
                            eem_charges = rdMolDescriptors.CalcEEMcharges(mol3d)
                            if eem_charges:
                                result_dict[f"{prefix}_EEM_Min"] = min(eem_charges)
                                result_dict[f"{prefix}_EEM_Max"] = max(eem_charges)
                                result_dict[f"{prefix}_EEM_Mean"] = sum(eem_charges) / len(eem_charges)
                                result_dict[f"{prefix}_EEM_Std"] = np.std(eem_charges)
                        except:
                            pass
                    
                    # Volume with AllChem
                    if compute_all_3d or "MolVolume" in descriptors_to_compute:
                        try:
                            result_dict[f"{prefix}_MolVolume"] = AllChem.ComputeMolVolume(mol3d)
                        except:
                            pass
                    
                    # USR descriptors
                    if compute_all_3d or "USR" in descriptors_to_compute:
                        try:
                            usr = rdMolDescriptors.GetUSR(mol3d)
                            for i, value in enumerate(usr):
                                result_dict[f"{prefix}_USR_{i+1}"] = value
                        except:
                            pass
                            
                    if compute_all_3d or "USRCAT" in descriptors_to_compute:
                        try:
                            usrcat = rdMolDescriptors.GetUSRCAT(mol3d)
                            for i, value in enumerate(usrcat):
                                result_dict[f"{prefix}_USRCAT_{i+1}"] = value
                        except:
                            pass
                    
                    # Oxidation numbers
                    if compute_all_3d or any(desc in descriptors_to_compute for desc in available_descriptors["3d"]["oxidation"]):
                        try:
                            ox_nums = rdMolDescriptors.CalcOxidationNumbers(mol3d)
                            if ox_nums:
                                result_dict[f"{prefix}_OxidationNum_Min"] = min(ox_nums)
                                result_dict[f"{prefix}_OxidationNum_Max"] = max(ox_nums)
                                result_dict[f"{prefix}_OxidationNum_Mean"] = sum(ox_nums) / len(ox_nums)
                                result_dict[f"{prefix}_OxidationNum_Std"] = np.std(ox_nums)
                        except:
                            pass
                    
                    # Surface properties
                    if compute_all_3d or "SurfaceArea" in descriptors_to_compute:
                        try:
                            from rdkit.Chem.rdFreeSASA import CalcSASA
                            result_dict[f"{prefix}_SurfaceArea"] = CalcSASA(mol3d)
                        except:
                            pass
            except:
                pass

        return idx, result_dict, smiles
    except:
        return idx, {}, smiles

def get_optimal_worker_count():
    try:
        cpus = multiprocessing.cpu_count()
        mem = psutil.virtual_memory()
        mem_gb = mem.available / (1024**3)
        mem_workers = max(1, int(mem_gb))
        return min(max(1, cpus - 1), mem_workers)
    except:
        return max(1, multiprocessing.cpu_count() - 1)

def main():
    global TEMP_DIR, TEMP_FILES, EXECUTOR, INTERRUPTED
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    atexit.register(exit_handler)
    
    parser = argparse.ArgumentParser(description='Compute RDKit descriptors for molecules')
    parser.add_argument('--input', '-i', required=True, help='Input file path (CSV, TSV, SDF)')
    parser.add_argument('--output', '-o', required=True, help='Output file path')
    parser.add_argument('--smiles-col', '-s', default='SMILES', help='Name of column containing SMILES (default: SMILES)')
    parser.add_argument('--compute-3d', '-3d', action='store_true', help='Compute 3D descriptors')
    parser.add_argument('--append', '-a', action='store_true', help='Append descriptors to input file')
    parser.add_argument('--prefix', '-p', default='rdkit', help='Prefix for descriptor column names')
    parser.add_argument('--n-jobs', '-j', type=int, default=None, help='Number of parallel jobs')
    parser.add_argument('--ram-offloading', action='store_true', help='Enable RAM offloading to disk when needed')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--list-descriptors', action='store_true', help='List all available descriptors and exit')
    parser.add_argument('--compute', default=None, help='Which descriptors to compute (2d, 3d, all, or comma-separated list)')
    
    args = parser.parse_args()
    
    print_message("RDKit Descriptor Calculator")
    
    if args.list_descriptors:
        print_available_descriptors()
        return 0
    
    if args.ram_offloading:
        print_message("RAM offloading enabled")
        try:
            TEMP_DIR = tempfile.mkdtemp(prefix="rdkit_")
        except:
            TEMP_DIR = os.path.join(os.path.dirname(args.output), "temp_rdkit")
            os.makedirs(TEMP_DIR, exist_ok=True)
    
    input_path = args.input
    output_path = args.output
    smiles_col = args.smiles_col
    compute_3d = args.compute_3d
    n_jobs = args.n_jobs if args.n_jobs else get_optimal_worker_count()
    
    print_message(f"Processing input file: {input_path}")
    
    file_ext = os.path.splitext(input_path)[1].lower()
    
    try:
        if file_ext == '.sdf':
            from rdkit.Chem import PandasTools
            df = PandasTools.LoadSDF(input_path)
            if 'ROMol' in df.columns:
                df[smiles_col] = [Chem.MolToSmiles(mol) if mol is not None else None for mol in df['ROMol']]
            print_message(f"Loaded SDF with {len(df)} molecules")
        elif file_ext == '.csv':
            df = pd.read_csv(input_path)
            print_message(f"Loaded CSV with {len(df)} rows")
        elif file_ext == '.tsv':
            df = pd.read_csv(input_path, sep='\t')
            print_message(f"Loaded TSV with {len(df)} rows")
        elif file_ext in ['.xlsx', '.xls']:
            df = pd.read_excel(input_path)
            print_message(f"Loaded Excel file with {len(df)} rows")
        else:
            try:
                df = pd.read_csv(input_path)
                print_message(f"Loaded file as CSV with {len(df)} rows")
            except:
                df = pd.read_csv(input_path, sep='\t')
                print_message(f"Loaded file as TSV with {len(df)} rows")
    except Exception as e:
        print_message(f"Error loading input file: {str(e)}")
        sys.exit(1)
    
    if smiles_col not in df.columns:
        print_message(f"SMILES column '{smiles_col}' not found in input file")
        available_columns = ', '.join(df.columns)
        print_message(f"Available columns: {available_columns}")
        sys.exit(1)
    
    print_message(f"Computing RDKit descriptors for {len(df)} molecules")
    print_message(f"3D descriptors: {'enabled' if compute_3d else 'disabled'}")
    if args.compute:
        print_message(f"Computing descriptors: {args.compute}")
    print_message(f"Using {n_jobs} parallel processes")
    
    smiles_list = df[smiles_col].tolist()
    mol_data = [(smiles, compute_3d, args.prefix, i, args.compute) for i, smiles in enumerate(smiles_list)]
    
    descriptor_columns = {}
    memory_check_interval = max(1, min(100, len(smiles_list) // 20))
    offload_count = 0
    
    progress_tracker = ProgressTracker(len(smiles_list), "Computing descriptors")
    
    try:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            EXECUTOR = executor
            futures = [executor.submit(compute_rdkit_descriptors, data) for data in mol_data]
            
            results_buffer = {}
            
            for future_idx, future in enumerate(as_completed(futures)):
                if INTERRUPTED:
                    break
                
                try:
                    idx, result_dict, smiles = future.result()
                    
                    if result_dict:
                        results_buffer[idx] = result_dict
                        
                        for col_name in result_dict:
                            if col_name not in descriptor_columns:
                                descriptor_columns[col_name] = np.full(len(df), np.nan)
                            
                            descriptor_columns[col_name][idx] = result_dict[col_name]
                    
                    if args.ram_offloading and future_idx % memory_check_interval == 0:
                        if check_memory_usage(0.75):
                            temp_file = os.path.join(TEMP_DIR, f"rdkit_{offload_count}.pkl")
                            save_to_disk(descriptor_columns, temp_file)
                            TEMP_FILES.append(temp_file)
                            
                            offload_count += 1
                            descriptor_columns = {}
                            gc.collect()
                    
                    progress_tracker.increment()
                    
                except Exception:
                    progress_tracker.increment()
                
                if INTERRUPTED:
                    break
            
            EXECUTOR = None
        
        if not INTERRUPTED:
            progress_tracker.finalize()
            
            print_message("Creating result DataFrame")
            
            if args.ram_offloading and TEMP_FILES:
                if descriptor_columns:
                    temp_file = os.path.join(TEMP_DIR, f"rdkit_{offload_count}.pkl")
                    save_to_disk(descriptor_columns, temp_file)
                    TEMP_FILES.append(temp_file)
                    descriptor_columns = {}
                
                all_columns = {}
                
                for temp_file in TEMP_FILES:
                    if INTERRUPTED:
                        break
                        
                    temp_data = load_from_disk(temp_file)
                    if temp_data:
                        for col_name, values in temp_data.items():
                            if col_name not in all_columns:
                                all_columns[col_name] = values
                            else:
                                mask = ~np.isnan(values)
                                all_columns[col_name][mask] = values[mask]
                    
                    try:
                        os.remove(temp_file)
                        TEMP_FILES.remove(temp_file)
                    except:
                        pass
                
                if args.append:
                    new_df = pd.DataFrame(all_columns)
                    output_df = pd.concat([df, new_df], axis=1)
                    print_message(f"Appended {len(all_columns)} descriptors to input data")
                else:
                    result_data = {smiles_col: smiles_list}
                    result_data.update(all_columns)
                    output_df = pd.DataFrame(result_data)
                    print_message(f"Created new dataframe with {len(all_columns)} descriptors")
            else:
                if args.append:
                    new_df = pd.DataFrame(descriptor_columns)
                    output_df = pd.concat([df, new_df], axis=1)
                    print_message(f"Appended {len(descriptor_columns)} descriptors to input data")
                else:
                    result_data = {smiles_col: smiles_list}
                    result_data.update(descriptor_columns)
                    output_df = pd.DataFrame(result_data)
                    print_message(f"Created new dataframe with {len(descriptor_columns)} descriptors")
            
            if args.ram_offloading:
                cleanup_temp_files()
            
            if INTERRUPTED:
                print_message("Operation interrupted by user before saving")
                sys.exit(1)
            
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            print_message(f"Saving results to {output_path}")
            
            try:
                file_ext = os.path.splitext(output_path)[1].lower()
                
                if file_ext == '.csv':
                    output_df.to_csv(output_path, index=False)
                elif file_ext == '.tsv':
                    output_df.to_csv(output_path, sep='\t', index=False)
                elif file_ext in ['.xlsx', '.xls']:
                    output_df.to_excel(output_path, index=False)
                else:
                    output_df.to_csv(output_path, index=False)
                    
                print_message(f"Results saved to {output_path}")
                descriptor_count = len(output_df.columns) - (1 if not args.append else len(df.columns))
                print_message(f"Computed {descriptor_count} RDKit descriptors for {len(df)} molecules")
            except Exception as e:
                print_message(f"Error saving file: {str(e)}")
    
    except Exception as e:
        print_message(f"Error: {str(e)}")
    finally:
        if args.ram_offloading:
            cleanup_temp_files()

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
