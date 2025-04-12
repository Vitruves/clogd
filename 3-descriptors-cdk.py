#!/usr/bin/env python
"""
CDK 2.11 Molecular Descriptors Calculator - Computes all available CDK descriptors
"""

import os
import sys
import time
import argparse
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import tempfile
import csv
import pandas as pd
import signal
import traceback
import json

# JPype setup for Java/CDK access
import jpype
import jpype.imports
from jpype.types import *

# Import Java classes needed throughout the script
def init_java_imports():
    """Initialize Java imports when JVM is running"""
    if jpype.isJVMStarted():
        try:
            from java.lang import Class
            return True
        except:
            return False
    return False

def get_descriptor_engine(descriptor_type):
    """Get descriptor engine for the specified type"""
    from org.openscience.cdk.qsar import DescriptorEngine
    from org.openscience.cdk.silent import SilentChemObjectBuilder
    from java.lang import Class
    
    # Get the builder
    builder = SilentChemObjectBuilder.getInstance()
    
    # Get class based on descriptor type
    if descriptor_type.lower() == "molecular":
        interface_class = Class.forName("org.openscience.cdk.qsar.IMolecularDescriptor")
    elif descriptor_type.lower() == "atomic":
        interface_class = Class.forName("org.openscience.cdk.qsar.IAtomicDescriptor")
    elif descriptor_type.lower() == "bond":
        interface_class = Class.forName("org.openscience.cdk.qsar.IBondDescriptor")
    else:
        print(f"-- Warning: Unknown descriptor type {descriptor_type}, using molecular")
        interface_class = Class.forName("org.openscience.cdk.qsar.IMolecularDescriptor")
    
    # Create the engine with the interface class and builder
    engine = DescriptorEngine(interface_class, builder)
    
    return engine

def get_descriptor_list():
    """Get a list of all available molecular descriptors in CDK with detailed information"""
    if not jpype.isJVMStarted():
        return []
        
    from org.openscience.cdk.qsar import DescriptorEngine
    from org.openscience.cdk.silent import SilentChemObjectBuilder
    from java.lang import Class
    
    # Get the builder
    builder = SilentChemObjectBuilder.getInstance()
    
    # Get the interface class
    interface_class = Class.forName("org.openscience.cdk.qsar.IMolecularDescriptor")
    
    # Create the engine with the interface class and builder
    engine = DescriptorEngine(interface_class, builder)
    
    descriptors = engine.getDescriptorInstances()
    
    # Get detailed descriptor information
    desc_list = []
    for desc in descriptors:
        # Get basic info
        desc_info = {
            'name': desc.getSpecification().getSpecificationReference().split('#')[-1],
            'class': desc.getClass().getName(),
            'implementation': desc.getSpecification().getImplementationTitle(),
            'vendor': desc.getSpecification().getImplementationVendor(),
            'identifier': desc.getSpecification().getImplementationIdentifier()
        }
        
        # Get parameter information
        try:
            param_names = desc.getParameterNames()
            if param_names and len(param_names) > 0:
                params = []
                for param_name in param_names:
                    param_type = desc.getParameterType(param_name)
                    param_value = None
                    try:
                        all_params = desc.getParameters()
                        if all_params and len(all_params) > 0:
                            # Find the index of this parameter
                            param_idx = [i for i, name in enumerate(param_names) if name == param_name]
                            if param_idx and all_params[param_idx[0]]:
                                param_value = str(all_params[param_idx[0]])
                    except:
                        pass
                        
                    params.append({
                        'name': param_name,
                        'type': str(param_type).replace('class ', '') if param_type else 'Unknown',
                        'default': param_value
                    })
                desc_info['parameters'] = params
        except:
            desc_info['parameters'] = []
            
        # Get descriptor value names
        try:
            value_names = desc.getDescriptorNames()
            if value_names and len(value_names) > 0:
                desc_info['values'] = value_names
            else:
                desc_info['values'] = [desc_info['name']]
        except:
            desc_info['values'] = [desc_info['name']]
            
        desc_list.append(desc_info)
    
    return desc_list

def extract_descriptor_value(result):
    """Extract value(s) from descriptor result"""
    # Handle different return types
    if hasattr(result.getValue(), 'doubleValue'):
        # Single value result
        return [result.getValue().doubleValue()]
    elif hasattr(result.getValue(), 'length'):
        # IntegerArrayResult
        values = []
        for i in range(result.getValue().length):
            values.append(result.getValue().get(i))
        return values
    else:
        # Try to handle DoubleArrayResult or other array types
        try:
            values = []
            array_length = result.getValue().length()
            for i in range(array_length):
                values.append(result.getValue().get(i))
            return values
        except:
            # Last resort, try to convert to string and parse
            try:
                value_str = str(result.getValue()).replace('[', '').replace(']', '')
                return [float(x.strip()) for x in value_str.split(',') if x.strip()]
            except:
                # If all else fails, return the string representation
                return [str(result.getValue())]

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Fast CDK Molecular Descriptor Calculator for millions of molecules")
    parser.add_argument("--input", required=True, help="Input file (CSV or SMILES)")
    parser.add_argument("--output", required=True, help="Output file")
    parser.add_argument("--smiles-col", default=0, help="SMILES column name or index (for CSV)")
    parser.add_argument("--keep-original-cols", action="store_true",
                        help="Keep original columns in output (for CSV)")
    parser.add_argument("--no-header", action="store_true",
                        help="Input CSV has no header row")
    parser.add_argument("--delimiter", default=",", help="CSV delimiter")
    parser.add_argument("--chunk-size", type=int, default=10000,
                        help="Number of molecules per chunk")
    parser.add_argument("--processes", type=int, default=0,
                        help="Number of processes (0 for auto)")
    parser.add_argument("--output-delimiter", default=None,
                        help="Output file delimiter (defaults to same as input)")
    parser.add_argument("--skip-errors", action="store_true",
                        help="Skip rows with calculation errors instead of marking them")
    parser.add_argument("--verbose", action="store_true",
                        help="Print verbose output")
    parser.add_argument("--list-descriptors", action="store_true",
                        help="List all available descriptors and exit")
    parser.add_argument("--descriptor-filter", nargs="+", default=None,
                        help="Only compute descriptors containing these terms in name (case-insensitive)")
    parser.add_argument("--descriptor-type", choices=["molecular", "atomic", "bond"],
                        default="molecular", help="Type of descriptors to calculate")
    parser.add_argument("--descriptor-param", nargs="+", action="append", default=None,
                        help="Set descriptor parameters in format: DescriptorName:paramName:value")
    parser.add_argument("--list-descriptor-types", action="store_true",
                        help="List available descriptor types and exit")
    parser.add_argument("--debug", action="store_true",
                        help="Print additional debug information")
    parser.add_argument("--reliable-only", action="store_true",
                        help="Use only reliable descriptors known to work well with most molecules")
    parser.add_argument("--minimal", action="store_true",
                        help="Use only a minimal set of the most commonly used descriptors")
    parser.add_argument("--descriptor-timeout", type=int, default=5,
                        help="Maximum time (seconds) allowed for a single descriptor calculation")
    return parser.parse_args()

def set_descriptor_parameters(descriptors, param_settings):
    """Set parameters for specific descriptors"""
    if not param_settings:
        return descriptors
        
    modified_descriptors = []
    
    # Process each descriptor
    for descriptor in descriptors:
        # Check if this descriptor has any parameters to set
        desc_name = descriptor.getSpecification().getSpecificationReference().split('#')[-1]
        matching_params = [p for p in param_settings if p[0].lower() == desc_name.lower()]
        
        if not matching_params:
            # No parameters to set for this descriptor
            modified_descriptors.append(descriptor)
            continue
            
        try:
            # Get current parameters
            param_names = descriptor.getParameterNames()
            current_params = descriptor.getParameters() if param_names else []
            
            # Create new parameters array (copy of current)
            new_params = list(current_params) if current_params else []
            
            # Update parameters
            for desc_param in matching_params:
                # Format is [descriptor_name, param_name, value]
                if len(desc_param) != 3:
                    print(f"-- Warning: Invalid parameter format for {desc_param}")
                    continue
                    
                # Find parameter index by name
                _, param_name, param_value = desc_param
                try:
                    param_idx = -1
                    for i, name in enumerate(param_names):
                        if name.lower() == param_name.lower():
                            param_idx = i
                            break
                            
                    if param_idx == -1:
                        print(f"-- Warning: Parameter {param_name} not found for descriptor {desc_name}")
                        continue
                        
                    # Get parameter type
                    param_type = descriptor.getParameterType(param_name)
                    if str(param_type).endswith("Integer"):
                        new_params[param_idx] = JInt(int(param_value))
                    elif str(param_type).endswith("Double"):
                        new_params[param_idx] = JDouble(float(param_value))
                    elif str(param_type).endswith("Boolean"):
                        new_params[param_idx] = JBoolean(param_value.lower() in ["true", "yes", "1"])
                    else:
                        new_params[param_idx] = param_value
                        
                    print(f"-- Set {desc_name}.{param_name} = {param_value}")
                        
                except Exception as e:
                    print(f"-- Warning: Error setting parameter {param_name} for {desc_name}: {e}")
            
            # Set new parameters
            if new_params:
                descriptor.setParameters(new_params)
            
            modified_descriptors.append(descriptor)
            
        except Exception as e:
            print(f"-- Warning: Error processing parameters for {desc_name}: {e}")
            modified_descriptors.append(descriptor)
    
    return modified_descriptors

def get_reliable_descriptors():
    """Returns a list of descriptors known to be reliable for most molecules"""
    return [
        "ALOGPDescriptor",        # ALogP, ALogp2, AMR
        "XLogPDescriptor",        # XLogP
        "TPSADescriptor",         # TopoPSA
        "WeightDescriptor",       # MW (molecular weight)
        "ApolDescriptor",         # apol (atomic polarizability)
        "BPolDescriptor",         # bpol (bond polarizability)
        "FractionalCSP3Descriptor", # Fsp3
        "FractionalPSADescriptor", # tpsaEfficiency
        "ZagrebIndexDescriptor",  # Zagreb
        "MannholdLogPDescriptor", # MLogP
        "HybridizationRatioDescriptor", # HybRatio
        "FMFDescriptor",          # FMF (fragment complexity)
        "AtomCountDescriptor",    # nAtom
        "BondCountDescriptor",    # nB
        "RotatableBondsCountDescriptor", # nRotB
        "HBondDonorCountDescriptor", # nHBDon
        "HBondAcceptorCountDescriptor", # nHBAcc
        "VABCDescriptor",         # VABC (volume descriptor)
        "VAdjMaDescriptor",       # VAdjMat
        "PetitjeanNumberDescriptor", # PetitjeanNumber
        "KierHallSmartsDescriptor", # khs.*
        "BCUTDescriptor",         # BCUT.*
        "SmallRingDescriptor",    # nRings*
        "AromaticAtomsCountDescriptor", # naAromAtom
        "AromaticBondsCountDescriptor", # nAromBond
        "WienerNumbersDescriptor", # WPATH, WPOL
        "CarbonTypesDescriptor"   # C*SP*
    ]

def get_minimal_descriptors():
    """Returns a minimal set of descriptors that are most commonly used"""
    return [
        "ALOGPDescriptor",        # ALogP, ALogp2, AMR
        "XLogPDescriptor",        # XLogP
        "TPSADescriptor",         # TopoPSA
        "WeightDescriptor",       # MW (molecular weight)
        "RotatableBondsCountDescriptor", # nRotB
        "HBondDonorCountDescriptor", # nHBDon
        "HBondAcceptorCountDescriptor", # nHBAcc
        "FractionalCSP3Descriptor", # Fsp3
    ]

def process_chunk(chunk_data):
    """Process a chunk of molecules in a separate process"""
    try:
        chunk_id, data, output_path, is_csv, smiles_col_idx, delimiter, descriptor_filter, descriptor_type, param_settings, reliable_only, minimal_only, descriptor_timeout = chunk_data

        # Each process needs its own JVM
        if not jpype.isJVMStarted():
            try:
                print(f"-- Worker {chunk_id}: Starting JVM")
                jpype.startJVM(classpath=[os.path.abspath("cdk-2.11.jar")], convertStrings=True)
                from java.lang import Class
                print(f"-- Worker {chunk_id}: JVM started successfully")
            except Exception as e:
                print(f"-- Worker {chunk_id}: Error starting JVM: {e}")
                return None

        # Import CDK classes
        try:
            print(f"-- Worker {chunk_id}: Importing CDK classes")
            from org.openscience.cdk.silent import SilentChemObjectBuilder
            from org.openscience.cdk.smiles import SmilesParser
            from org.openscience.cdk.tools import CDKHydrogenAdder
            from org.openscience.cdk.qsar import DescriptorEngine
            from org.openscience.cdk.geometry import GeometryUtil
            from org.openscience.cdk.modeling.builder3d import ModelBuilder3D
            from java.lang import Class, System
            print(f"-- Worker {chunk_id}: CDK classes imported successfully")
        except Exception as e:
            print(f"-- Worker {chunk_id}: Error importing CDK classes: {e}")
            return None

        try:
            # Initialize CDK objects
            print(f"-- Worker {chunk_id}: Initializing CDK objects")
            builder = SilentChemObjectBuilder.getInstance()
            parser = SmilesParser(builder)
            h_adder = CDKHydrogenAdder.getInstance(builder)
            
            # Set up aromaticity detection
            print(f"-- Worker {chunk_id}: Setting up aromaticity model")
            from org.openscience.cdk.aromaticity import Aromaticity
            from org.openscience.cdk.graph import Cycles
            from org.openscience.cdk.tools.manipulator import AtomContainerManipulator

            # Try both import paths for ElectronDonation
            try:
                from org.openscience.cdk.aromaticity.ElectronDonation import Daylight
                daylight_donator = Daylight()
            except ImportError:
                try:
                    from org.openscience.cdk.aromaticity import ElectronDonation
                    daylight_donator = ElectronDonation.daylight()
                except ImportError:
                    from org.openscience.cdk.aromaticity import ElectronDonation
                    daylight_donator = ElectronDonation.cdk()
                    
            # Create aromaticity model with cycle finder
            cycles = Cycles.all()
            aromaticity_model = Aromaticity(daylight_donator, cycles)
            
            # Get descriptor engine based on type
            interface_class = None
            if descriptor_type.lower() == "molecular":
                interface_class = Class.forName("org.openscience.cdk.qsar.IMolecularDescriptor")
            elif descriptor_type.lower() == "atomic":
                interface_class = Class.forName("org.openscience.cdk.qsar.IAtomicDescriptor")
            elif descriptor_type.lower() == "bond":
                interface_class = Class.forName("org.openscience.cdk.qsar.IBondDescriptor")
            else:
                interface_class = Class.forName("org.openscience.cdk.qsar.IMolecularDescriptor")
                
            print(f"-- Worker {chunk_id}: Creating descriptor engine")
            engine = DescriptorEngine(interface_class, builder)
            print(f"-- Worker {chunk_id}: Getting descriptor instances")
            all_descriptors = engine.getDescriptorInstances()
            print(f"-- Worker {chunk_id}: Found {len(all_descriptors)} descriptors")
        except Exception as e:
            print(f"-- Worker {chunk_id}: Error initializing descriptor engine: {e}")
            traceback.print_exc()
            return None

        # Filter descriptors if specified
        if minimal_only:
            print(f"-- Worker {chunk_id}: Using minimal descriptors only")
            minimal_list = get_minimal_descriptors()
            descriptors = [d for d in all_descriptors if any(rel.lower() in d.getClass().getName().lower() for rel in minimal_list)]
            print(f"-- Worker {chunk_id}: Selected {len(descriptors)} minimal descriptors")
            
            # Print which descriptors were selected
            if len(descriptors) > 0:
                desc_names = [d.getClass().getName().split('.')[-1] for d in descriptors]
                print(f"-- Worker {chunk_id}: Selected descriptors: {', '.join(desc_names)}")
        elif reliable_only:
            print(f"-- Worker {chunk_id}: Using reliable descriptors only")
            reliable_list = get_reliable_descriptors()
            descriptors = [d for d in all_descriptors if any(rel.lower() in d.getClass().getName().lower() for rel in reliable_list)]
            print(f"-- Worker {chunk_id}: Selected {len(descriptors)} reliable descriptors")
            
            # Print which descriptors were selected
            if len(descriptors) > 0:
                desc_names = [d.getClass().getName().split('.')[-1] for d in descriptors]
                print(f"-- Worker {chunk_id}: Selected descriptors: {', '.join(desc_names)}")
        elif descriptor_filter:
            print(f"-- Worker {chunk_id}: Filtering descriptors")
            descriptors = [d for d in all_descriptors if any(filter_term.lower() in d.getSpecification().getSpecificationReference().lower() for filter_term in descriptor_filter)]
            print(f"-- Worker {chunk_id}: Filtered to {len(descriptors)} descriptors")
        else:
            descriptors = all_descriptors
            
        # Fewer than 5 descriptors could indicate filtering issues
        if len(descriptors) < 5:
            print(f"-- Worker {chunk_id}: WARNING - Very few descriptors selected: {len(descriptors)}")
            
        # Set custom parameters if provided
        if param_settings:
            try:
                print(f"-- Worker {chunk_id}: Setting descriptor parameters")
                descriptors = set_descriptor_parameters(descriptors, param_settings)
            except Exception as param_error:
                print(f"-- Worker {chunk_id}: Error setting descriptor parameters: {param_error}")
                traceback.print_exc()
        
        # Generate descriptor names for header
        print(f"-- Worker {chunk_id}: Generating descriptor names")
        descriptor_names = []
        descriptor_meta = {}
        
        for descriptor in descriptors:
            try:
                desc_class = descriptor.getClass().getName().split('.')[-1]
                desc_name = descriptor.getSpecification().getSpecificationReference().split('#')[-1]
                
                # Get parameter names for this descriptor
                param_names = descriptor.getDescriptorNames()
                if not param_names or len(param_names) == 0:
                    col_name = desc_name 
                    descriptor_names.append(col_name)
                    descriptor_meta[col_name] = {'class': desc_class, 'index': len(descriptor_names)-1}
                else:
                    for param in param_names:
                        col_name = f"{desc_name}.{param}"
                        descriptor_names.append(col_name)
                        descriptor_meta[col_name] = {'class': desc_class, 'index': len(descriptor_names)-1}
            except Exception as desc_name_error:
                print(f"-- Worker {chunk_id}: Error getting descriptor names: {desc_name_error}")
                unique_id = f"Unknown.{len(descriptor_names)}"
                descriptor_names.append(unique_id)
                descriptor_meta[unique_id] = {'class': 'Unknown', 'index': len(descriptor_names)-1}
        
        # Save descriptor metadata for debugging
        with open(f"{output_path}.meta", 'w') as mf:
            json.dump(descriptor_meta, mf, indent=2)
            
        print(f"-- Worker {chunk_id}: Created {len(descriptor_names)} descriptor columns")
        
        # Also save descriptor names to a separate file for this chunk
        with open(f"{output_path}.header", 'w') as hf:
            hf.write(delimiter.join(descriptor_names))
            
        # Track which descriptors fail the most
        descriptor_errors = {desc.getClass().getName().split('.')[-1]: 0 for desc in descriptors}
            
        with open(output_path, 'w', newline='') as f:
            if is_csv:
                # CSV processing - keep original columns if present
                writer = csv.writer(f, delimiter=delimiter)
                
                # Write chunk data with descriptor values
                for idx, row in enumerate(data):
                    try:
                        if idx % 100 == 0 or idx == len(data) - 1:
                            print(f"-- Worker {chunk_id}: Processed {idx}/{len(data)} molecules")
                            
                        smiles = row[smiles_col_idx]

                        if not smiles or pd.isna(smiles) or smiles == "":
                            # Handle empty SMILES
                            row_data = ["ERROR"] * len(descriptor_names)
                            writer.writerow(row + row_data)
                            continue

                        # Parse SMILES and prepare molecule properly
                        try:
                            mol = parser.parseSmiles(smiles)
                            
                            # Configure atoms
                            AtomContainerManipulator.percieveAtomTypesAndConfigureAtoms(mol)
                            
                            # Add explicit and implicit hydrogens
                            h_adder.addImplicitHydrogens(mol)
                            
                            # Detect aromaticity
                            aromaticity_model.apply(mol)
                            
                            try:
                                # Try to generate 3D coordinates for 3D descriptors
                                if not GeometryUtil.has3DCoordinates(mol):
                                    try:
                                        mb3d = ModelBuilder3D.getInstance()
                                        mol = mb3d.generate3DCoordinates(mol, True)
                                    except Exception as e_3d:
                                        # Continue without 3D coordinates if generation fails
                                        print(f"-- Worker {chunk_id}: Warning - Failed to generate 3D coords for {smiles}: {e_3d}")
                            except Exception as e_geo:
                                # Continue if GeometryUtil or 3D generation fails
                                pass
                                
                        except Exception as prep_error:
                            print(f"-- Worker {chunk_id}: Error preparing molecule '{smiles}': {prep_error}")
                            row_data = ["ERROR"] * len(descriptor_names)
                            writer.writerow(row + row_data)
                            continue

                        # Calculate all descriptors with timeout safety
                        all_values = []
                        desc_errors = 0
                        
                        for desc_idx, descriptor in enumerate(descriptors):
                            try:
                                # Calculate descriptor with timeout protection
                                start_time = time.time()
                                max_time = descriptor_timeout  # seconds per descriptor
                                
                                result = None
                                while time.time() - start_time < max_time:
                                    try:
                                        result = descriptor.calculate(mol)
                                        break
                                    except Exception as timeout_err:
                                        if time.time() - start_time >= max_time:
                                            raise Exception(f"Descriptor calculation timed out after {max_time}s")
                                        time.sleep(0.1)  # Small delay before retry
                                
                                if result is None:
                                    raise Exception("Descriptor calculation timed out")
                                    
                                values = extract_descriptor_value(result)
                                all_values.extend(values)
                            except Exception as desc_error:
                                # If descriptor calculation fails, add ERROR for each value
                                desc_errors += 1
                                desc_name = descriptor.getClass().getName().split('.')[-1]
                                descriptor_errors[desc_name] = descriptor_errors.get(desc_name, 0) + 1
                                
                                param_names = descriptor.getDescriptorNames()
                                param_count = max(1, len(param_names) if param_names else 0)
                                all_values.extend(["ERROR"] * param_count)

                        # Write result with original row
                        writer.writerow(row + all_values)
                        if desc_errors > len(descriptors) / 2:
                            print(f"-- Worker {chunk_id}: {desc_errors} descriptor errors for molecule {idx}")
                    except Exception as e:
                        # Handle any errors during calculation
                        print(f"-- Worker {chunk_id}: Error processing molecule {idx}: {e}")
                        if len(row) > 0:  # Make sure row exists
                            row_data = ["ERROR"] * len(descriptor_names)
                            writer.writerow(row + row_data)
            else:
                # Simple SMILES list processing
                writer = csv.writer(f, delimiter=delimiter)
                
                # Write chunk data with descriptor values
                for idx, smiles in enumerate(data):
                    try:
                        if idx % 100 == 0 or idx == len(data) - 1:
                            print(f"-- Worker {chunk_id}: Processed {idx}/{len(data)} molecules")
                            
                        if not smiles or pd.isna(smiles) or smiles == "":
                            # Handle empty SMILES
                            row_data = ["ERROR"] * len(descriptor_names)
                            writer.writerow([smiles] + row_data)
                            continue

                        # Parse SMILES
                        try:
                            mol = parser.parseSmiles(smiles)
                            
                            # Configure atoms
                            AtomContainerManipulator.percieveAtomTypesAndConfigureAtoms(mol)
                            
                            # Add explicit and implicit hydrogens
                            h_adder.addImplicitHydrogens(mol)
                            
                            # Detect aromaticity 
                            aromaticity_model.apply(mol)
                            
                            try:
                                # Try to generate 3D coordinates for 3D descriptors
                                if not GeometryUtil.has3DCoordinates(mol):
                                    try:
                                        mb3d = ModelBuilder3D.getInstance()
                                        mol = mb3d.generate3DCoordinates(mol, True)
                                    except Exception as e_3d:
                                        # Continue without 3D coordinates if generation fails
                                        print(f"-- Worker {chunk_id}: Warning - Failed to generate 3D coords for {smiles}: {e_3d}")
                            except Exception as e_geo:
                                # Continue if GeometryUtil or 3D generation fails
                                pass
                                
                        except Exception as prep_error:
                            print(f"-- Worker {chunk_id}: Error preparing molecule '{smiles}': {prep_error}")
                            row_data = ["ERROR"] * len(descriptor_names)
                            writer.writerow([smiles] + row_data)
                            continue

                        # Calculate all descriptors with timeout safety
                        all_values = []
                        desc_errors = 0
                        
                        for desc_idx, descriptor in enumerate(descriptors):
                            try:
                                # Calculate descriptor with timeout protection
                                start_time = time.time()
                                max_time = descriptor_timeout  # seconds per descriptor
                                
                                result = None
                                while time.time() - start_time < max_time:
                                    try:
                                        result = descriptor.calculate(mol)
                                        break
                                    except Exception as timeout_err:
                                        if time.time() - start_time >= max_time:
                                            raise Exception(f"Descriptor calculation timed out after {max_time}s")
                                        time.sleep(0.1)  # Small delay before retry
                                
                                if result is None:
                                    raise Exception("Descriptor calculation timed out")
                                    
                                values = extract_descriptor_value(result)
                                all_values.extend(values)
                            except Exception as desc_error:
                                # If descriptor calculation fails, add ERROR for each value
                                desc_errors += 1
                                desc_name = descriptor.getClass().getName().split('.')[-1]
                                descriptor_errors[desc_name] = descriptor_errors.get(desc_name, 0) + 1
                                
                                param_names = descriptor.getDescriptorNames()
                                param_count = max(1, len(param_names) if param_names else 0)
                                all_values.extend(["ERROR"] * param_count)

                        # Write result immediately
                        writer.writerow([smiles] + all_values)
                    except Exception as e:
                        row_data = ["ERROR"] * len(descriptor_names)
                        writer.writerow([smiles] + row_data)

        # Report descriptor error statistics
        print(f"-- Worker {chunk_id}: Descriptor error statistics:")
        sorted_errors = sorted(descriptor_errors.items(), key=lambda x: x[1], reverse=True)
        for desc_name, error_count in sorted_errors:
            if error_count > 0:
                error_rate = (error_count / len(data)) * 100
                print(f"--   {desc_name}: {error_count} errors ({error_rate:.1f}%)")

        print(f"-- Worker {chunk_id}: Completed - total: {len(data)}, processed: {len(data)}")
        return output_path
    except Exception as e:
        print(f"-- Error in process_chunk: {e}")
        traceback.print_exc()
        return None

def main():
    # Parse command line arguments
    args = parse_args()

    # Initialize JVM in main process
    if not jpype.isJVMStarted():
        try:
            print("-- Starting JVM with cdk-2.11.jar...")
            jpype.startJVM(classpath=[os.path.abspath("cdk-2.11.jar")], convertStrings=True)
            init_java_imports()
        except Exception as e:
            print(f"-- Error starting JVM - failed")
            print(f"-- {e}")
            traceback.print_exc()
            sys.exit(1)
            
    # Handle list descriptor types
    if args.list_descriptor_types:
        print("-- Available descriptor types:")
        print("--   molecular: Calculates properties of entire molecules")
        print("--   atomic: Calculates properties of atoms")
        print("--   bond: Calculates properties of bonds")
        sys.exit(0)
    
    # Debug mode info
    if args.debug:
        print("-- DEBUG: JVM Version Info:")
        try:
            from java.lang import System
            print(f"--   Java Version: {System.getProperty('java.version')}")
            print(f"--   Java Home: {System.getProperty('java.home')}")
            
            # Try various ways to get CDK version
            try:
                from org.openscience.cdk import CDKConstants
                if hasattr(CDKConstants, 'CURRENT_MODULE_VERSION'):
                    print(f"--   CDK Version: {CDKConstants.CURRENT_MODULE_VERSION}")
                else:
                    # Try other version constants in CDK
                    print(f"--   CDK Class Info: {dir(CDKConstants)}")
                    
                    # Get manifest info as alternative
                    try:
                        from java.lang import Package
                        cdk_class = CDKConstants.class_
                        if hasattr(cdk_class, 'getPackage'):
                            cdk_package = cdk_class.getPackage()
                            if cdk_package:
                                print(f"--   CDK Implementation: {cdk_package.getImplementationTitle()} {cdk_package.getImplementationVersion()}")
                    except Exception as pkg_err:
                        print(f"--   Error getting package info: {pkg_err}")
            except Exception as cdk_err:
                print(f"--   Error accessing CDK constants: {cdk_err}")
                
            # List descriptors engine details
            try:
                from org.openscience.cdk.qsar import DescriptorEngine
                from org.openscience.cdk.silent import SilentChemObjectBuilder
                from java.lang import Class
                builder = SilentChemObjectBuilder.getInstance()
                interface_class = Class.forName("org.openscience.cdk.qsar.IMolecularDescriptor")
                engine = DescriptorEngine(interface_class, builder)
                descriptors = engine.getDescriptorInstances()
                print(f"--   Found {len(descriptors)} molecular descriptors")
                
                # Print descriptor set info
                if args.minimal:
                    minimal_list = get_minimal_descriptors()
                    print(f"--   Using minimal descriptors: {', '.join(minimal_list)}")
                elif args.reliable_only:
                    reliable_list = get_reliable_descriptors()
                    print(f"--   Using reliable descriptors: {', '.join(reliable_list)}")
            except Exception as eng_err:
                print(f"--   Error initializing descriptor engine: {eng_err}")
                
        except Exception as e:
            print(f"--   Error getting Java info: {e}")
    
    # Handle list-descriptors command
    if args.list_descriptors:
        try:
            from org.openscience.cdk.qsar import DescriptorEngine
            from org.openscience.cdk.silent import SilentChemObjectBuilder
            from java.lang import Class
            
            # Get the builder
            builder = SilentChemObjectBuilder.getInstance()
            
            # Determine interface class based on descriptor type
            if args.descriptor_type.lower() == "molecular":
                interface_class = Class.forName("org.openscience.cdk.qsar.IMolecularDescriptor")
            elif args.descriptor_type.lower() == "atomic":
                interface_class = Class.forName("org.openscience.cdk.qsar.IAtomicDescriptor")
            elif args.descriptor_type.lower() == "bond":
                interface_class = Class.forName("org.openscience.cdk.qsar.IBondDescriptor")
            else:
                interface_class = Class.forName("org.openscience.cdk.qsar.IMolecularDescriptor")
                
            # Create engine with interface class and builder
            engine = DescriptorEngine(interface_class, builder)
            
            print(f"-- Available CDK {args.descriptor_type} descriptors:")
            print("-- --------------------------")
            
            # Get descriptors directly from engine
            all_descriptors = engine.getDescriptorInstances()
            
            if args.debug:
                print(f"-- DEBUG: Found {len(all_descriptors)} descriptors")
                
            # Process descriptors into our format
            descriptors = []
            for desc in all_descriptors:
                desc_info = {
                    'name': desc.getSpecification().getSpecificationReference().split('#')[-1],
                    'class': desc.getClass().getName(),
                    'implementation': desc.getSpecification().getImplementationTitle(),
                    'vendor': desc.getSpecification().getImplementationVendor(),
                    'identifier': desc.getSpecification().getImplementationIdentifier()
                }
                
                # Get parameter information
                try:
                    param_names = desc.getParameterNames()
                    if param_names and len(param_names) > 0:
                        params = []
                        for param_name in param_names:
                            param_type = desc.getParameterType(param_name)
                            param_value = None
                            try:
                                all_params = desc.getParameters()
                                if all_params and len(all_params) > 0:
                                    # Find the index of this parameter
                                    param_idx = [i for i, name in enumerate(param_names) if name == param_name]
                                    if param_idx and all_params[param_idx[0]]:
                                        param_value = str(all_params[param_idx[0]])
                            except:
                                pass
                                
                            params.append({
                                'name': param_name,
                                'type': str(param_type).replace('class ', '') if param_type else 'Unknown',
                                'default': param_value
                            })
                        desc_info['parameters'] = params
                except Exception as param_error:
                    if args.debug:
                        print(f"-- DEBUG: Error getting parameters for {desc_info['name']}: {param_error}")
                    desc_info['parameters'] = []
                    
                # Get descriptor value names
                try:
                    value_names = desc.getDescriptorNames()
                    if value_names and len(value_names) > 0:
                        desc_info['values'] = value_names
                    else:
                        desc_info['values'] = [desc_info['name']]
                except Exception as value_error:
                    if args.debug:
                        print(f"-- DEBUG: Error getting value names for {desc_info['name']}: {value_error}")
                    desc_info['values'] = [desc_info['name']]
                    
                descriptors.append(desc_info)
                
            # Apply name filtering if specified
            if args.descriptor_filter:
                filtered_descriptors = [d for d in descriptors 
                                        if any(filter_term.lower() in d['name'].lower() 
                                            for filter_term in args.descriptor_filter)]
            else:
                filtered_descriptors = descriptors
                
            # Mark reliable descriptors
            reliable_list = get_reliable_descriptors()
            
            for i, desc in enumerate(filtered_descriptors):
                is_reliable = any(rel.lower() in desc['class'].lower() for rel in reliable_list)
                reliability = "RELIABLE" if is_reliable else ""
                
                print(f"-- [{i+1}] {desc['name']} ({desc['class']}) {reliability}")
                print(f"--     Implementation: {desc['implementation']} by {desc['vendor']}")
                print(f"--     Descriptor Values: {', '.join(desc['values'])}")
                
                if 'parameters' in desc and desc['parameters']:
                    print(f"--     Parameters:")
                    for param in desc['parameters']:
                        default_value = f" (default: {param['default']})" if param['default'] else ""
                        print(f"--       - {param['name']} : {param['type']}{default_value}")
                print("-- ")
        except Exception as e:
            print(f"-- Error getting descriptor list - failed")
            print(f"-- {e}")
            traceback.print_exc()
        sys.exit(0)

    # Process descriptor parameters
    param_settings = []
    if args.descriptor_param:
        for param_setting in args.descriptor_param:
            if len(param_setting) >= 3:
                descriptor_name = param_setting[0]
                param_name = param_setting[1]
                param_value = param_setting[2]
                param_settings.append([descriptor_name, param_name, param_value])
            else:
                print(f"-- Warning: Invalid parameter format: {param_setting}")
                print("-- Expected format: DescriptorName:paramName:value")
    
    # Set output delimiter
    output_delimiter = args.output_delimiter if args.output_delimiter else args.delimiter

    # Determine optimal chunk size and processes
    num_cores = cpu_count()
    num_processes = args.processes if args.processes > 0 else max(1, num_cores - 1)
    molecules_per_chunk = args.chunk_size

    print(f"-- Using {num_processes} processes with {molecules_per_chunk} molecules per chunk")

    # Detect if input is CSV based on file extension and arguments
    is_csv = args.input.lower().endswith('.csv') or args.smiles_col != 0

    # Create temp directory for chunk results
    temp_dir = tempfile.mkdtemp(prefix="cdk_descriptors_")
    if args.verbose:
        print(f"-- Using temporary directory: {temp_dir}")

    # Parse input file
    try:
        if is_csv:
            print(f"-- Reading CSV file: {args.input}...")
            # Read CSV using pandas
            df = pd.read_csv(args.input, delimiter=args.delimiter,
                             header=None if args.no_header else 0,
                             on_bad_lines='warn')

            # Get SMILES column INDEX (not name)
            smiles_col_idx = None
            if isinstance(args.smiles_col, str) and not args.no_header:
                # Find column index by name
                if args.smiles_col not in df.columns:
                    print(f"-- Error: Column '{args.smiles_col}' not found in CSV - failed")
                    print(f"-- Available columns: {', '.join(str(c) for c in df.columns)}")
                    sys.exit(1)
                # Convert column name to index position
                smiles_col_idx = df.columns.get_loc(args.smiles_col)
            else:
                # Use numeric index
                smiles_col_idx = int(args.smiles_col)
                if smiles_col_idx >= len(df.columns):
                    print(f"-- Error: Column index {smiles_col_idx} out of range (max: {len(df.columns)-1}) - failed")
                    sys.exit(1)

            # Display which column we're using
            if args.verbose:
                if not args.no_header:
                    col_name = df.columns[smiles_col_idx]
                    print(f"-- Using SMILES from column '{col_name}' (index {smiles_col_idx})")
                else:
                    print(f"-- Using SMILES from column index {smiles_col_idx}")

            # Convert DataFrame to list of lists for processing
            if args.no_header:
                data = df.values.tolist()
                header = None
            else:
                data = df.values.tolist()
                header = df.columns.tolist()

            total_mols = len(data)
        else:
            print(f"-- Reading SMILES from {args.input}...")
            # Read simple SMILES file
            with open(args.input, 'r') as f:
                data = [line.strip() for line in f if line.strip()]
            smiles_col_idx = 0  # In simple mode, each row is just a SMILES string
            header = ["SMILES"]
            total_mols = len(data)
    except Exception as e:
        print(f"-- Error reading input file - failed")
        print(f"-- {e}")
        import shutil
        shutil.rmtree(temp_dir)
        sys.exit(1)

    print(f"-- Found {total_mols} molecules")
    
    # Filter descriptors if specified
    if args.descriptor_filter:
        print(f"-- Filtering descriptors with terms: {', '.join(args.descriptor_filter)}")

    # Split into chunks
    chunks = []
    for i, chunk_data in enumerate(
            [data[i:i+molecules_per_chunk]
             for i in range(0, len(data), molecules_per_chunk)]):
        output_path = os.path.join(temp_dir, f"chunk_{i}.csv")
        chunks.append((i, chunk_data, output_path, is_csv, smiles_col_idx, output_delimiter, 
                      args.descriptor_filter, args.descriptor_type, param_settings, 
                      args.reliable_only, args.minimal, args.descriptor_timeout))

    print(f"-- Processing {total_mols} molecules in {len(chunks)} chunks...")

    # Process chunks in parallel
    start_time = time.time()
    chunk_results = []

    try:
        with Pool(processes=num_processes) as pool:
            for result_path in tqdm(
                    pool.imap_unordered(process_chunk, chunks),
                    total=len(chunks),
                    desc="-- Processing chunks"):
                if result_path:  # Only add successful results
                    chunk_results.append(result_path)

        # Check if all chunks were processed successfully
        if len(chunk_results) < len(chunks):
            print(f"-- Warning: {len(chunks) - len(chunk_results)} chunks failed to process")

        # Combine results
        print("-- Combining results...")

        with open(args.output, 'w', newline='') as outf:
            writer = csv.writer(outf, delimiter=output_delimiter)
            
            # Construct final header
            header_row = []
            
            # Add original columns if keeping them
            if is_csv and args.keep_original_cols and header:
                header_row.extend(header)
            elif not is_csv:
                header_row.append("SMILES")
            
            # Find and process descriptor header
            descriptor_header = None
            if chunk_results:
                # First look for a header file
                for result_path in chunk_results:
                    header_path = f"{result_path}.header"
                    if os.path.exists(header_path):
                        with open(header_path, 'r') as f:
                            header_content = f.read().strip()
                            descriptor_header = header_content.split(output_delimiter)
                        break
                        
                # If no separate header file, try to read from the first chunk
                if descriptor_header is None and len(chunk_results) > 0:
                    try:
                        with open(chunk_results[0], 'r', newline='') as inf:
                            reader = csv.reader(inf, delimiter=output_delimiter)
                            try:
                                # Skip original columns in first chunk if needed
                                first_row = next(reader)
                                if is_csv:
                                    descriptor_header = first_row[len(data[0]):] if data and len(data[0]) > 0 else first_row
                                else:
                                    descriptor_header = first_row[1:] if len(first_row) > 1 else []
                            except StopIteration:
                                print("-- Warning: First chunk file is empty")
                    except Exception as e:
                        print(f"-- Error reading descriptor header: {e}")
            
            if descriptor_header:
                header_row.extend(descriptor_header)
                writer.writerow(header_row)
            else:
                print("-- Error: Could not determine descriptor names - failed")
                sys.exit(1)
            
            # Write data rows from chunks
            rows_written = 0
            for result_path in chunk_results:
                try:
                    with open(result_path, 'r', newline='') as inf:
                        reader = csv.reader(inf, delimiter=output_delimiter)
                        
                        # Skip header row of chunk if present
                        if not is_csv and os.path.basename(result_path).startswith("chunk_0"):
                            try:
                                next(reader)
                            except StopIteration:
                                continue
                            
                        for row in reader:
                            if not args.skip_errors or "ERROR" not in row:
                                writer.writerow(row)
                                rows_written += 1
                except Exception as e:
                    print(f"-- Error reading chunk {result_path}: {e}")

        elapsed_time = time.time() - start_time
        molecules_per_second = total_mols / elapsed_time if elapsed_time > 0 else 0

        print(f"-- Finished processing {total_mols} molecules in {elapsed_time:.2f} seconds")
        print(f"-- Processing rate: {molecules_per_second:.2f} molecules per second")
        print(f"-- Wrote {rows_written} rows to output file")
        print(f"-- Results written to {args.output} - success")

    except KeyboardInterrupt:
        print("\n-- Process interrupted by user - cleaning up...")
    except Exception as e:
        print(f"-- Error during processing - failed")
        print(f"-- {e}")
        traceback.print_exc()
    finally:
        # Clean up temp files
        import shutil
        try:
            shutil.rmtree(temp_dir)
        except:
            print(f"-- Warning: Could not remove temporary directory: {temp_dir}")

        # Shut down JVM
        if jpype.isJVMStarted():
            jpype.shutdownJVM()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n-- Process terminated by user")
        sys.exit(1)
    except Exception as e:
        print(f"-- Unhandled error - failed")
        print(f"-- {e}")
        traceback.print_exc()
        sys.exit(1)