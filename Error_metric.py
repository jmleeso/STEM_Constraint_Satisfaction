import sys
import os
import time
import argparse

import numpy as np
from pathlib import Path
from tqdm import tqdm
from utils import load_swift_to_py4DSTEM

import h5py
import py4DSTEM
from py4DSTEM import __version__, read, save, show
from py4DSTEM.process.phase import DPC, Parallax
print(f'py4DSTEM v{py4DSTEM.__version__}')

## Estimate fluctuation based on Poisson resampling
def beta_from_delta(delta: float) -> float:
    return np.sqrt((1.0 + delta)**2 - 1.0)

def compute_e_flat_chunked(orig, recon, q_k=0.95, chunk=32, shift=False):
    # orig,recon: (Nx,Ny,Nkx,Nky)
    Nx, Ny, Nkx, Nky = orig.shape
    Nxy = Nx * Ny

    o = orig.reshape(Nxy, Nkx, Nky)
    r = recon.reshape(Nxy, Nkx, Nky)

    e = np.empty(Nxy, dtype=np.float64)

    for s in tqdm(range(0, Nxy, chunk), desc="Computing e_flat"):
        rr = o[s:s+chunk] - r[s:s+chunk]                       # (B,Nkx,Nky)
        F = np.fft.fft2(rr, axes=(-2, -1))
        if shift:
            F = np.fft.fftshift(F, axes=(-2, -1))
        mag = np.abs(F).reshape(F.shape[0], -1)               # (B, Nkx*Nky)
        e[s:s+chunk] = np.quantile(mag, q_k, axis=1)
    return e  # length Nxy

def compute_b_flat_poisson_chunked(orig, q_k=0.95, scale=1.0, seed=0, repeats=3, chunk=16, shift=False):
    Nx, Ny, Nkx, Nky = orig.shape
    Nxy = Nx * Ny
    o = orig.reshape(Nxy, Nkx, Nky)

    rng = np.random.default_rng(seed)
    b = np.empty(Nxy, dtype=np.float64)

    for s in tqdm(range(0, Nxy, chunk), desc="Computing b_flat"):
        pat = o[s:s+chunk]
        lam = np.clip(pat * scale / 2.0, 0, None)

        vals = []
        for _ in range(repeats):
            p1 = rng.poisson(lam).astype(np.float32) / scale
            p2 = rng.poisson(lam).astype(np.float32) / scale
            rr = p1 - p2
            F = np.fft.fft2(rr, axes=(-2, -1))
            if shift:
                F = np.fft.fftshift(F, axes=(-2, -1))
            mag = np.abs(F).reshape(F.shape[0], -1)
            vals.append(np.quantile(mag, q_k, axis=1))  # (B,)

        b[s:s+chunk] = np.median(np.stack(vals, axis=0), axis=0)
    return b

def compute_R_from_e_b(e_flat, b_flat, q_r=0.95, eps=1e-12):
    ratio = e_flat / (b_flat + eps)
    R = np.quantile(ratio, q_r)
    return R, ratio

def compute_nrmse(x, y):
    """
    root mean square error: square-root of sum of all (x_i-y_i)**2
    """
    assert(x.shape == y.shape)
    mse = np.mean((x-y)**2)
    maxv = np.max(x)
    minv = np.min(x)
    return np.sqrt(mse)/(maxv - minv)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str, help="original data directory containing .npy and .json")
    parser.add_argument("decompressed_file", type=str, help="decompressed data file")
    parser.add_argument("CS_output_file", type=str, help="CS-corrected data file")
    parser.add_argument('-o', '--output', type=str, default='./output/baseline_b_flat.npy', help='path to save the computed b_flat values')   

    args = parser.parse_args()
    file_data = args.input_path
    decompressed_file = args.decompressed_file
    CS_output_file = args.CS_output_file
    out_path = Path(args.output)
    if out_path.suffix != '.npy':
        out_path = out_path.with_suffix('.npy')
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print('##### Data Load #####\n')

    file_cal = r'./data/U100_calibratoins.hdf5'
    cal_file = h5py.File(file_cal, "a")
    
    q_cal_val = cal_file['q-calibratoins']['20mm']['2024-02-29'][()]
    q_cal_units = cal_file['q-calibratoins']['20mm']['2024-02-29'].attrs['units']
    cal_file.close()

    nominal_defocus = -20 * 10
    energy = 100E3
    probe_semiangle_mrad = 32.
    
    # Load original dataset
    dataset = load_swift_to_py4DSTEM(file_data, verbose=False)
    Qbin = int(512/dataset.shape[-1])
    dataset.calibration.set_Q_pixel_size(q_cal_val*Qbin)
    dataset.calibration.set_Q_pixel_units(q_cal_units)
    if dataset.calibration.R_pixel_units == 'nm':
        dataset.calibration.set_R_pixel_units('A')
        dataset.calibration.set_R_pixel_size(dataset.calibration.R_pixel_size*10)

    # Load decompressed dataset
    recon = np.fromfile(decompressed_file, dtype=np.float32).reshape(256,256,256,256)
    recon = np.ascontiguousarray(np.transpose(recon, (2,3,0,1)))

    recon_correct = np.load(CS_output_file)

    recon_nrmse = compute_nrmse(dataset.data, recon)
    recon_correct_nrmse = compute_nrmse(dataset.data, recon_correct)
    print(f"Decompressed NRMSE: {recon_nrmse:.4f}")
    print(f"CS-corrected NRMSE: {recon_correct_nrmse:.4f}")

    print('###### Compute flucation baseline of the original data ######\n')
    # This is required to compute just once for the original data, and can be reused for different reconstructions

    # Check if there is pre-computed b_flat for the original data
    if out_path.exists():
        print(f"Loading pre-computed b_flat from {out_path}")
        b_flat = np.load(out_path)
    else:
        print(f"Computing b_flat for the original data and saving to {out_path}")
        b_flat = compute_b_flat_poisson_chunked(dataset.data, q_k=0.95, scale=1.0, seed=0, repeats=3, chunk=32, shift=False)
        np.save(out_path, b_flat)
    
    delta = 0.05
    beta = beta_from_delta(delta)

    print('###### Compute score for the reconstructed and CS-corrected reconstruction data ######\n')
    e_flat_recon = compute_e_flat_chunked(dataset.data, recon, q_k=0.95, chunk=32, shift=False)
    R_recon, _ = compute_R_from_e_b(e_flat_recon, b_flat, q_r=0.95)
    e_flat_recon_corrected = compute_e_flat_chunked(dataset.data, recon_correct, q_k=0.95, chunk=32, shift=False)
    R_recon_corrected, _ = compute_R_from_e_b(e_flat_recon_corrected, b_flat, q_r=0.95)
    print('###### End ######\n')

    print(f"Recon befre CS | NRMSE {recon_nrmse:.3e} | R {R_recon:.4f} | beta (target) {beta:.4f} | pass {R_recon <= beta}")
    print(f"Recon after CS | NRMSE {recon_correct_nrmse:.3e} | R {R_recon_corrected:.4f} | beta (target) {beta:.4f} | pass {R_recon_corrected <= beta}")

if __name__ == "__main__":
    main()