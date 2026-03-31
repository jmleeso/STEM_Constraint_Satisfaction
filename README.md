# STEM_Constraint_Satisfaction

# Paper Title: Error-Guaranteed Compression with Preservation of Downstream Quantities for Electron Microscopy

This repository contains the official implementation of the Constraint Satisfaction (CS) solver presented in our paper: **"Error-Guaranteed Compression with Preservation of Downstream Quantities for Electron Microscopy"**. 

Our method applies a post-processing step using the Dykstra projection algorithm to decompressed 4D-STEM (Four-Dimensional Scanning Transmission Electron Microscopy) data. It guarantees the preservation of specific physical constraints (e.g., moments) and significantly improves the physical fidelity of the reconstructed data, such as Virtual Bright Field (VBF) and Differential Phase Contrast (DPC) images.

## Repository Structure

To keep the repository simple and easy to run, all core scripts are located in the root directory:

* `CS_solver.py`: The main script that runs the Dykstra projection algorithm (supports both CPU and PyTorch-based GPU execution) to correct the decompressed data.
* `Error_metric.py`: A script to evaluate the reconstruction quality by computing the NRMSE and intensity fluctuation baseline (Poisson resampling) before and after applying the CS solver.
* `utils.py`: Utility functions for parsing and loading raw `.npy` and Swift `.ndata1` 4D-STEM formats into Py4DSTEM objects.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/jmleeso/STEM_Constraint_Satisfaction.git
   cd STEM_Constraint_Satisfaction
   ```
2. Install the required dependencies. Note that `py4DSTEM` version `0.14.9` is strictly required for compatibility:
   ```bash
   pip install -r requirements.txt
   ```

## Data Preparation

Place your sample 4D-STEM data and calibration files in the `data/` directory. By default, the scripts expect the following file to exist:
* Calibration file: `./data/U100_calibratoins.hdf5`
* Metadata to load the original data: `./data/metadata.json`

*Note: For a quick start, we provide cropped/sampled 4D-STEM data in the `data/` folder. The original D1 dataset and its sample decompressed data by MGARD used in the paper can be downloaded from [Link].*

## Data Preparation

Place your sample 4D-STEM data and calibration files in the `data/` directory. By default, the scripts expect the following files to exist:
* Calibration file: `./data/U100_calibratoins.hdf5`
* Metadata to load the original data: `./data/metadata.json`

### Downloading the Full Dataset

The original D1 dataset (approx. 16GB) and its sample decompressed data (compressed by MGARD with a compression ratio of 27.8) used in the paper are hosted securely on Dropbox.
We provide a Python script to download the sample datasets into the `./data` directory. Run the following command:

```bash
python download_sample_data.py
```

## Usage

### 1. Running the CS Solver
Run the `CS_solver.py` to apply the constraint satisfaction algorithm to your decompressed data. It supports GPU acceleration for fast processing.

```bash
python CS_solver.py <path_to_original_data_dir> <path_to_decompressed_file.npy> \
    --output ./output/CS_output.npy \
    --device cuda \
    --precision double
```

**Arguments:**
* `input_path`: Directory containing the original `.npy` and metadata `.json`.
* `decompressed_file`: The decompressed data file before correction.
* `-o`, `--output`: Path to save the CS-corrected data.
* `-d`, `--device`: Choose `cuda` (default) or `cpu`.
* `-p`, `--precision`: Choose `single` or `double` (default) precision for the solver.

The script will automatically compute the NRMSE for VBF and DPC to verify the correction performance and save the visualization results in the output directory.

### 2. Evaluating Error Metrics
To compute the fluctuation baseline and validate the statistical error metrics (beta targets) of decompressed data before CS and after CS:

```bash
python Error_metric.py <path_to_original_data_dir> <path_to_decompressed_file.npy> <path_to_CS_output.npy> \
    --output ./output/baseline_b_flat.npy
```
