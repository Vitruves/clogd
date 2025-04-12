# Molecular Descriptor Calculator

A fast JavaScript-based tool for calculating molecular descriptors from SMILES strings. This tool was created as an alternative to the Python/CDK-based descriptor calculator.

## Features

- Process CSV files containing SMILES strings
- Calculate common molecular descriptors
- Process in chunks for efficient memory usage
- Parallel processing support
- Configurable descriptor types
- CSV output with original columns preserved (optional)

## Installation

1. Make sure you have Node.js installed (v14 or higher)
2. Clone this repository
3. Install dependencies:

```bash
npm install
```

## Usage

Basic usage:

```bash
node process-molecules.js --input data/raw/test.csv --output data/raw/output-test.csv
```

Full options:

```bash
node process-molecules.js --input data/raw/test.csv --output data/raw/output-test.csv --smiles-col SMILES --keep-original-cols --chunk-size 1000 --processes 4 --descriptor-type all --verbose
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `--input FILE` | Input file (CSV) |
| `--output FILE` | Output file |
| `--smiles-col NAME` | SMILES column name or index (for CSV) |
| `--keep-original-cols` | Keep original columns in output (for CSV) |
| `--no-header` | Input CSV has no header row |
| `--delimiter CHAR` | CSV delimiter (default: ,) |
| `--chunk-size NUM` | Number of molecules per chunk (default: 1000) |
| `--processes NUM` | Number of processes (default: number of CPU cores - 1) |
| `--skip-errors` | Skip rows with calculation errors instead of marking them |
| `--verbose` | Print verbose output |
| `--descriptor-type TYPE` | Type of descriptors to calculate (all, basic, extended) |
| `--help` | Show help message |

## Available Descriptors

### Basic Descriptors
- logP
- molWt
- numAtoms
- numHeavyAtoms
- numBonds
- numRotatableBonds
- TPSA
- numHBondDonors
- numHBondAcceptors

### Extended Descriptors (all)
- All basic descriptors plus:
- exactMolWt
- numHeteroatoms
- numRings
- numAromaticRings
- Fsp3 (fraction of sp3 hybridized carbon atoms)
- lipinskiRuleOfFive (count of violations)

## Requirements

- Node.js v14+
- NPM or Yarn for dependency management

## License

MIT 