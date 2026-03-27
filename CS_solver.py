import sys
import os
import time
import argparse

import numpy as np
from pathlib import Path
import torch
from tqdm import tqdm
from emdfile import tqdmnd
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from utils import load_swift_to_py4DSTEM

import h5py
import py4DSTEM
from py4DSTEM import __version__, read, save, show
from py4DSTEM.process.phase import DPC, Parallax
print(f'py4DSTEM v{py4DSTEM.__version__}')

def compute_nrmse(x, y):
    """
    root mean square error: square-root of sum of all (x_i-y_i)**2
    """
    assert(x.shape == y.shape)
    mse = np.mean((x-y)**2)
    maxv = np.max(x)
    minv = np.min(x)
    return np.sqrt(mse)/(maxv - minv)

def construct_constraint_matrix(dataset, mask):
    intensity = dataset.data[125,125]
    kx = np.arange(intensity.shape[-2], dtype=np.float32)
    ky = np.arange(intensity.shape[-1], dtype=np.float32)
    kya, kxa = np.meshgrid(ky, kx)
    kya_vec = kya.reshape(-1)
    kxa_vec = kxa.reshape(-1)

    B = [np.ones(256*256), mask.reshape(-1), kxa_vec, kya_vec]
    B = np.asarray(B)
    B_inv = np.linalg.inv(B @ B.T)

    return B, B_inv

# Constraint Satisfaction Post-processing: CPU ver.
def proj_eq_affine(x, B, Ginv, c):
    """
    Project x onto {f: B f = c}, where B is a constraint matrix has shape (k, D). k is number of constraints, D is feature dim
    Ginv = inv(BB^T) shape (k, k)
    """
    r = B @ x - c
    t = Ginv @ r
    return x - (t[:, None] * B).sum(axis=0)

def dykstra_project(y, B, Ginv, c, max_iter=30, tol=1e-6):
    """
    Solve: min 0.5||f - y||^2  s.t. Bf=c, f>=0
    using Dykstra. y is f_R flattened (N,).
    """
    x = y.copy()
    p = np.zeros_like(x)
    q = np.zeros_like(x)

    for k in range(max_iter):
        # 1) equality projection
        y1 = proj_eq_affine(x + p, B, Ginv, c)
        p = x + p - y1

        # 2) nonneg projection
        x_new = np.maximum(y1 + q, 0.0)
        q = y1 + q - x_new

        # stopping
        denom = np.linalg.norm(x) + 1e-12
        if np.linalg.norm(x_new - x) / denom < tol:
            x = x_new
            break
        x = x_new

    return x

# Constraint Satisfaction Post-processing: Torch ver.
def prepare_dykstra_torch(B_rows_np, Ginv_np, device="cuda", dtype=torch.float32):
    """
    B_rows_np: (k, N)
    Ginv_np:   (k, k)

    returns:
        BT: (N, k) = B^T
        E : (k, N) = Ginv^T @ B
    """
    device = torch.device(device)

    B = torch.as_tensor(B_rows_np, dtype=dtype, device=device).contiguous()
    G = torch.as_tensor(Ginv_np, dtype=dtype, device=device).contiguous()

    BT = B.t().contiguous()          # (N, k)
    E = torch.mm(G.t(), B).contiguous()  # (k, N)

    return BT, E

@torch.inference_mode()
def dykstra_project_batch_torch_workspace_(
    X, BT, E, C, P, Q, U, R, D=None,
    max_iter=30, check_every=0, tol=1e-6
):
    """
    In-place / workspace Dykstra on GPU.

    Inputs:
        X : (batch, N)
        BT: (N, k)
        E : (k, N) = Ginv^T @ B
        C : (batch, k)

        P, Q, U: (batch, N) workspace
        R      : (batch, k) workspace
        D      : (batch, N) workspace for stopping check, or None

    Returns:
        final solution tensor view on GPU
    """
    P.zero_()
    Q.zero_()

    X_cur = X
    U_cur = U

    if check_every:
        tol2 = tol * tol

    for it in range(max_iter):
        # U = X + P_old
        torch.add(X_cur, P, out=U_cur)

        # R = B(U) - C
        torch.mm(U_cur, BT, out=R)
        R.sub_(C)

        # P = P_new = B^T Ginv (B U - C)
        torch.mm(R, E, out=P)

        # y1 = U - P
        U_cur.sub_(P)

        # tmp = y1 + Q_old
        U_cur.add_(Q)

        # Q_new = min(tmp, 0)
        torch.clamp(U_cur, max=0.0, out=Q)

        # X_new = max(tmp, 0)
        U_cur.clamp_min_(0.0)

        # stopping check
        if check_every and ((it + 1) % check_every == 0):
            torch.sub(U_cur, X_cur, out=D)
            num2 = (D * D).sum(dim=1)
            den2 = (X_cur * X_cur).sum(dim=1)
            if torch.all(num2 < tol2 * (den2 + 1e-24)).item():
                return U_cur

        # swap buffers
        X_cur, U_cur = U_cur, X_cur

    return X_cur


@torch.inference_mode()
def run_all_dykstra_torch(
    recon,
    true_moments,
    B_rows,
    Ginv,
    chunk_size=128,
    max_iter=30,
    tol=1e-6,
    check_every=0,
    device="cuda",
    dtype=torch.float32,
    out=None,
):
    """
    GPU-optimized batched Dykstra runner.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("dtype must be torch.float32 or torch.float64")

    H, W, h2, w2 = recon.shape
    if true_moments.shape[:2] != (H, W):
        raise ValueError("true_moments spatial shape must match recon")

    N = h2 * w2
    k = true_moments.shape[-1]

    if np.asarray(B_rows).shape != (k, N):
        raise ValueError(f"B_rows must have shape ({k}, {N})")
    if np.asarray(Ginv).shape != (k, k):
        raise ValueError(f"Ginv must have shape ({k}, {k})")
    if not recon.flags["C_CONTIGUOUS"]:
        raise ValueError("recon must be C-contiguous")

    np_dtype = np.float32 if dtype == torch.float32 else np.float64

    if out is None:
        recon_correct = np.empty(recon.shape, dtype=np_dtype)
    else:
        if out.shape != recon.shape:
            raise ValueError("out.shape must match recon.shape")
        if out.dtype != np_dtype:
            raise ValueError(f"out.dtype must be {np_dtype}")
        if not out.flags["C_CONTIGUOUS"]:
            raise ValueError("out must be C-contiguous")
        recon_correct = out

    Y_all_np = recon.reshape(H * W, N)
    Out_all_np = recon_correct.reshape(H * W, N)

    Y_all_t_cpu = torch.from_numpy(Y_all_np)
    Out_all_t_cpu = torch.from_numpy(Out_all_np)

    C_all_t = torch.as_tensor(
        true_moments.reshape(H * W, k),
        dtype=dtype,
        device=device,
    ).contiguous()

    # constant matrices on GPU
    BT_t, E_t = prepare_dykstra_torch(B_rows, Ginv, device=device, dtype=dtype)

    # reusable GPU workspaces
    X_buf = torch.empty((chunk_size, N), device=device, dtype=dtype)
    P_buf = torch.empty_like(X_buf)
    Q_buf = torch.empty_like(X_buf)
    U_buf = torch.empty_like(X_buf)
    R_buf = torch.empty((chunk_size, k), device=device, dtype=dtype)
    D_buf = torch.empty_like(X_buf) if check_every else None

    device = torch.device(device)
    torch.cuda.synchronize(device)
    start = time.perf_counter()

    total = H * W
    for s in tqdm(range(0, total, chunk_size), desc=f"Dykstra torch batched ({device})"):
        #e = min(s + chunk_size, total)
        e = s + chunk_size
        if e > total:
            e = total
        m = e - s

        Xv = X_buf[:m]
        Pv = P_buf[:m]
        Qv = Q_buf[:m]
        Uv = U_buf[:m]
        Rv = R_buf[:m]
        Dv = D_buf[:m] if D_buf is not None else None

        # CPU -> GPU
        Xv.copy_(Y_all_t_cpu[s:e], non_blocking=False)

        # solve
        X_final = dykstra_project_batch_torch_workspace_(
            X=Xv,
            BT=BT_t,
            E=E_t,
            C=C_all_t[s:e],
            P=Pv,
            Q=Qv,
            U=Uv,
            R=Rv,
            D=Dv,
            max_iter=max_iter,
            check_every=check_every,
            tol=tol,
        )

        Out_all_t_cpu[s:e].copy_(X_final, non_blocking=False)

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    print(f"CS solver execution time (GPU): {elapsed:.4f} seconds")

    return recon_correct

# Analysis fucntion
def run_vbf(dataset, probe_qx0, probe_qy0, probe_radius_pixels, out_path, filename='original'):
    dataset.get_virtual_image(
            mode = 'circle',
            geometry = ((probe_qx0, probe_qy0),probe_radius_pixels),
            return_mask=False,
            name = 'bright_field',
    )
    plt.figure()
    plt.imshow(dataset.tree('bright_field').data, cmap='viridis')
    plt.colorbar()
    fig = plt.gcf()
    fig.savefig(f'{out_path.parent}/vbf_{filename}.png', dpi=300, bbox_inches='tight')
    plt.clf()
    plt.close(fig)
    return dataset.tree('bright_field').data

def run_dpc(dataset, energy, out_path, filename='original', device='cpu'):
    dpc = DPC(
            datacube=dataset, 
            energy = energy,
            verbose=True,
            device=device,
    )
    dpc = dpc.preprocess(
            vectorized_com_calculation=False,
    )
    fig = plt.gcf()
    fig.savefig(f'{out_path.parent}/dpc_preprocess_{filename}.png', dpi=300, bbox_inches='tight')
    plt.clf()
    plt.close('all')
    
    dpc =dpc.reconstruct().visualize()
    fig = plt.gcf()
    fig.savefig(f'{out_path.parent}/dpc_object_phase_{filename}.png', dpi=300, bbox_inches='tight')
    plt.clf()
    plt.close('all')

    return dpc._com_x, dpc._com_y, dpc.object_phase
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str, help="original data directory containing .npy and .json")
    parser.add_argument("decompressed_file", type=str, help="decompressed data file")
    parser.add_argument('-o', '--output', type=str, default='./output/CS_output.npy', help='output store path')    
    parser.add_argument('-d', '--device', type=str, default='cuda',
                                            choices=['cpu', 'cuda'],
                                            help='device to run the solver on ("cpu" or "cuda")')
    parser.add_argument('-p', '--precision', type=str, default='double',
                                            choices=['single', 'double'],
                                            help='precision for the CS solver ("single" or "double")')

    args = parser.parse_args()
    file_data = args.input_path
    decompressed_file = args.decompressed_file
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU.")
        device = 'cpu'
    if args.precision == 'single':
        torch_dtype = torch.float32
    else:
        torch_dtype = torch.float64        

    out_path = Path(args.output)
    if out_path.suffix != '.npy':
        out_path = out_path.with_suffix('.npy')
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    print('##### Data Load and Preparation for CS Solver #####\n')

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
    recon_nrmse = compute_nrmse(dataset.data, recon)

    # Get Mask for VBF Preservation
    dp_mean = dataset.get_dp_mean()
    probe_radius_pixels, probe_qx0, probe_qy0 = dataset.get_probe_size(dataset.tree('dp_mean').data, plot=False)
    # overlay selected detector position over mean dp
    dataset.position_detector(
            mode = 'circle',
            geometry = (
                        (probe_qx0, probe_qy0),
                        probe_radius_pixels
            )
    )
    
    mask = dataset.get_virtual_image(
            mode = 'circle',
            geometry = ((probe_qx0, probe_qy0),probe_radius_pixels),
            return_mask=True,
            name = 'bright_field',       # the output will be stored in `dataset`'s tree with this name
    )

    # Prepare necessary inputs for Constraint Satisfaction
    # In real senario, this computation is required just once and metadata is stored along with compressed data
    # Matric for moment computation
    B, B_inv = construct_constraint_matrix(dataset, mask)
    # Ground-truth moments
    true_moments = np.zeros((256,256,4))
    for rx, ry in tqdmnd(256, 256, desc="True moments computation", disable=True):
        x_orig = dataset.data[rx][ry].reshape(-1)        
        c = B @ x_orig
        true_moments[rx][ry] = c

    # Run CS solver
    print('###### Start CS Solver ######')
    if device == 'cpu':
        recon_correct = np.zeros_like(recon)
        
        start_time = time.perf_counter()
        
        for rx, ry in tqdmnd(256, 256, desc="Constraint Satisfaction (Dykstra)"):
            y = recon[rx][ry].reshape(-1)
            c = true_moments[rx][ry]            
            y_new = dykstra_project(y, B, B_inv, c)            
            recon_correct[rx][ry] = y_new.reshape(256, 256)
            
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time        
        print(f"CS solver execution time (CPU): {elapsed_time:.4f} seconds")

    else:
        # check_every: 0 means no intermediate stopping check, which is recommended for better performance.
        #              Can set it to a positive integer (e.g., 5 or 10) to enable extra stopping checks for input tolerance.
        recon_correct = run_all_dykstra_torch(
                recon=recon,
                true_moments=true_moments,
                B_rows=B,
                Ginv=B_inv,
                chunk_size=128,
                max_iter=30,
                tol=1e-6,
                check_every=0,
                device="cuda",
                dtype=torch_dtype 
        )

    recon_correct_nrmse = compute_nrmse(dataset.data, recon_correct)
    print('###### End CS Solver ######\n')
    # Store corrected recon
    np.save(out_path, recon_correct)

    print('###### Compute VBF and DPC ######\n')
    # Orignal - VBF and DPC
    vbf_orig = run_vbf(dataset, probe_qx0, probe_qy0, probe_radius_pixels, out_path, 'original')
    dpc_com_x_orig, dpc_com_y_orig, dpc_object_phase_orig = run_dpc(dataset, energy, out_path=out_path, filename='original', device='cpu')

    # Decompressed data before CS - VBF and DPC
    dataset.data = recon
    vbf_recon = run_vbf(dataset, probe_qx0, probe_qy0, probe_radius_pixels, out_path, 'recon_before_cs')
    dpc_com_x_recon, dpc_com_y_recon, dpc_object_phase_recon = run_dpc(dataset, energy, out_path=out_path, filename='recon_before_cs', device='cpu')

    # Decompressed data after CS - VBF and DPC
    dataset.data = recon_correct
    vbf_recon_correct = run_vbf(dataset, probe_qx0, probe_qy0, probe_radius_pixels, out_path, 'recon_after_cs')
    dpc_com_x_recon_correct, dpc_com_y_recon_correct, dpc_object_phase_recon_correct = run_dpc(dataset, energy, out_path=out_path, filename='recon_after_cs', device='cpu')

    # Compute NRMSEs
    vbf_recon_nrmse = compute_nrmse(vbf_orig, vbf_recon)
    dpc_com_x_recon_nrmse = compute_nrmse(dpc_com_x_orig, dpc_com_x_recon)
    dpc_com_y_recon_nrmse = compute_nrmse(dpc_com_y_orig, dpc_com_y_recon)
    dpc_object_phase_recon_nrmse = compute_nrmse(dpc_object_phase_orig, dpc_object_phase_recon)
    vbf_recon_correct_nrmse = compute_nrmse(vbf_orig, vbf_recon_correct)
    dpc_com_x_recon_correct_nrmse = compute_nrmse(dpc_com_x_orig, dpc_com_x_recon_correct)
    dpc_com_y_recon_correct_nrmse = compute_nrmse(dpc_com_y_orig, dpc_com_y_recon_correct)
    dpc_object_phase_recon_correct_nrmse = compute_nrmse(dpc_object_phase_orig, dpc_object_phase_recon_correct)
    print(f'###### Output  ######')
    print('###### End VBF and DPC Computation ######\n')

    print('======== Final NRMSE Results =======')
    print('Raw Data NRMSE before CS:', recon_nrmse)
    print('Raw Data NRMSE after CS:', recon_correct_nrmse)
    print('VBF NRMSE before CS:', vbf_recon_nrmse)
    print('VBF NRMSE after CS:', vbf_recon_correct_nrmse)
    print('DPC Com X NRMSE before CS:', dpc_com_x_recon_nrmse)
    print('DPC Com X NRMSE after CS:', dpc_com_x_recon_correct_nrmse)
    print('DPC Com Y NRMSE before CS:', dpc_com_y_recon_nrmse)
    print('DPC Com Y NRMSE after CS:', dpc_com_y_recon_correct_nrmse)
    print('DPC Object Phase NRMSE before CS:', dpc_object_phase_recon_nrmse)
    print('DPC Object Phase NRMSE after CS:', dpc_object_phase_recon_correct_nrmse)

if __name__ == "__main__":
    main()