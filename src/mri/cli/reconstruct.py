from hydra_zen import store, zen
from hydra.conf import HydraConf, JobConf

from mri.cli.utils import save_data
from mri.cli.base_configs import raw_config, traj_config
from mri.operators.fourier.utils import discard_frequency_outliers
from mrinufft.io.utils import add_phase_to_kspace_with_shifts
from pymrt.recipes.coils import compress_svd
from mri.reconstructors import SelfCalibrationReconstructor

import numpy as np
import pickle as pkl
import logging
import os
from functools import partial

log = logging.getLogger(__name__)



def dc_adjoint(obs_file: str, traj_file: str, coil_compress: str|int, debug: int,
               obs_reader, traj_reader, fourier, output_filename: str = "dc_adjoint.pkl"):
    """
    Reconstructs an image using the adjoint operator.

    Parameters
    ----------
    obs_file : str
        Path to the observed kspace data file.
    traj_file : str
        Path to the trajectory file.
    obs_reader : callable
        A function that reads the observed data file and returns
        the raw data and data header.
    traj_reader : callable
        A function that reads the trajectory file and returns the trajectory
        data and parameters.
    fourier: Callable
        A Callable returning a Fourier Operator
    coil_compress : str|int, optional default -1
        The number of singular values to keep in the coil compression.
        If -1, coil compression is not applied 
    output_filename: str, optional default 'dc_adjoint.pkl'
        The output file name with the right extension.
        It can be:
        1) *.pkl / *.mat: Holds the reconstructed results saved in dictionary as `recon`.
        2) *.nii : NIFTI file holding the reconstructed images.
        
    Returns
    -------
    None
        The reconstructed image is saved as 'dc_adjoint.pkl' file.
    """
    raw_data, data_header = obs_reader(obs_file)
    log.debug(f"Data Header: {data_header}")
    try:
        if data_header["trajectory_name"] != os.path.basename(traj_file):
            log.warn("Trajectory file does not match the trajectory in the data file")
    except KeyError:
        log.warn("Trajectory name not found in data header, Skipped Validation")
    shots, traj_params = traj_reader(
        traj_file,
        dwell_time=traj_reader.keywords['raster_time'] / data_header["oversampling_factor"],
    )
    log.debug(f"Trajectory Parameters: {traj_params}")
    kspace_loc = shots.reshape(-1, traj_params["dimension"])
    normalized_shifts = (
        np.array(data_header["shifts"])
        / np.array(traj_params["FOV"])
        * np.array(traj_params["img_size"])
        / 1000
    )
    if kspace_loc.max() > 0.5 or kspace_loc.min() < 0.5:
        log.debug(f"K-space locations are above the unity range, discarding the outlier data")
        kspace_loc, kspace_data = discard_frequency_outliers(kspace_loc, np.squeeze(raw_data))
    kspace_data = kspace_data.astype(np.complex64)
    kspace_loc = kspace_loc.astype(np.float32)
    log.debug(f"Phase shifting raw data for Normalized shifts: {normalized_shifts}")
    kspace_data = add_phase_to_kspace_with_shifts(
        kspace_data, kspace_loc, normalized_shifts
    )
    if coil_compress != -1:
        kspace_data = np.ascontiguousarray(compress_svd(
            kspace_data,
            k_svd=coil_compress,
            coil_axis=0
        ))
    fourier.keywords['smaps'] = partial(
        fourier.keywords['smaps'],
        kspace_data=kspace_data,
    )
    fourier_op = fourier(
        kspace_loc,
        traj_params["img_size"],
        n_coils=data_header["n_coils"] if coil_compress == -1 else coil_compress,
    )
    if debug > 0:
        intermediate = {
            'density_comp': fourier_op.impl.density,
            'smaps': fourier_op.impl.smaps,
            'traj_params': traj_params,
            'data_header': data_header,
        }
        if coil_compress != -1:
            intermediate['kspace_data'] = kspace_data
        log.debug('Saving Smaps and denisty_comp')
        pkl.dump(intermediate, open('intermediate.pkl', 'wb'))
    dc_adjoint = fourier_op.adj_op(kspace_data)
    if not fourier_op.impl.uses_sense:
        dc_adjoint = np.linalg.norm(dc_adjoint, axis=-1)
    save_data(output_filename, dc_adjoint, data_header)
    return dc_adjoint, (fourier_op, kspace_data, traj_params, data_header)
    
    
    
def recon(obs_file: str, traj_file: str, mu: float, num_iterations: int, coil_compress: str|int, 
          algorithm: str, debug: int, obs_reader, traj_reader, fourier, linear, sparsity,
          output_filename: str = "recon.pkl"):
    """Reconstructs an MRI image using the given parameters.

    Parameters
    ----------
    obs_file : str
        Path to the file containing the observed k-space data.
    traj_file : str
        Path to the file containing the trajectory data.
    mu : float
        Regularization parameter for the sparsity constraint.
    num_iterations : int
        Number of iterations for the reconstruction algorithm.
    coil_compress : str | int
        Method or factor for coil compression.
    algorithm : str
        Optimization algorithm to use for reconstruction.
    debug : int
        Debug level for printing debug information.
    obs_reader : callable
        Object for reading the observed k-space data.
    traj_reader : callable
        Object for reading the trajectory data.
    fourier : callable
        Object representing the Fourier operator.
    linear : callable
        Object representing the linear operator.
    sparsity : callable
        Object representing the sparsity operator.
    output_filename : str, optional
        Path to save the reconstructed image, by default "recon.pkl"
    """
    recon_adjoint, additional_data = dc_adjoint(
        obs_file,
        traj_file,
        coil_compress,
        debug,
        obs_reader,
        traj_reader,
        fourier,
        output_filename='dc_adjoint.pkl',
    )
    fourier_op, kspace_data, traj_params, data_header = additional_data
    linear_op = linear(shape=tuple(traj_params["img_size"]), dim=traj_params['dimension'])
    linear_op.op(recon_adjoint)
    sparse_op = sparsity(coeffs_shape=linear_op.coeffs_shape, weights=mu)
    reconstructor = SelfCalibrationReconstructor(
        fourier_op=fourier_op,
        linear_op=linear_op,
        regularizer_op=sparse_op,
        verbose=1,
        lipschitz_cst=fourier_op.impl.get_lipschitz_cst(),
    )
    recon, costs, metrics = reconstructor.reconstruct(
        kspace_data=kspace_data,
        optimization_alg=algorithm,
        x_init=recon_adjoint, # gain back the first step by initializing with DC Adjoint
        num_iterations=num_iterations,
    )
    data_header['costs'] = costs
    data_header['metrics'] = metrics
    save_data(output_filename, recon, data_header)

store(HydraConf(job=JobConf(chdir=True)))
store(
    dc_adjoint,
    obs_reader=raw_config,
    traj_reader=traj_config,
    coil_compress=10,
    debug=1,
    hydra_defaults=[
        "_self_",
        {"fourier": "gpu"},
        {"fourier/density_comp": "pipe"},
        {"fourier/smaps": "low_frequency"},
    ],
    name="dc_adjoint",
)
store(
    recon,
    obs_reader=raw_config,
    traj_reader=traj_config,
    algorithm="pogm",
    num_iterations=30,
    coil_compress=10,
    debug=1,
    hydra_defaults=[
        "_self_",
        {"fourier": "gpu"},
        {"fourier/density_comp": "pipe"},
        {"fourier/smaps": "low_frequency"},
        {"linear": "gpu"},
        {"sparsity": "weighted_sparse"},
    ],
    name="recon",
)

store(
    recon,
    obs_reader=raw_config,
    traj_reader=traj_config,
    algorithm="pogm",
    num_iterations=30,
    coil_compress=5,
    debug=1,
    hydra_defaults=[
        "_self_",
        {"fourier": "gpu_lowmem"},
        {"fourier/density_comp": "pipe_lowmem"},
        {"fourier/smaps": "low_frequency"},
    ],
    name="recon_lowmem",
)

store.add_to_hydra_store()
zen(recon).hydra_main(
    config_name="recon",
    config_path=None,
    version_base="1.3",
)
