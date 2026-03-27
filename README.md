# STEM_Constraint_Satisfaction

# [Paper Title: Error-Guaranteed Compression with Preservation of Downstream Quantities for Electron Microscopy]

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
   git clone [https://github.com/](https://github.com/)[jmleeso]/[STEM_Constraint_Satisfaction].git
   cd [STEM_Constraint_Satisfaction]
