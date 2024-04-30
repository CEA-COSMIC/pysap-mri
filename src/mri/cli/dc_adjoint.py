from hydra_zen import store, zen

from mri.cli.utils import save_data
from mri.cli.base_configs import raw_config, traj_config
from mri.operators.fourier.utils import discard_frequency_outliers
from mrinufft.io.utils import add_phase_to_kspace_with_shifts
from pymrt.recipes.coils import compress_svd

import numpy as np
import logging
import os
from functools import partial

log = logging.getLogger(__name__)


def recon(obs_file: str, traj_file: str, obs_reader, traj_reader, fourier, coil_compress: str|int = -1,
          output_filename: str = "dc_adjoint.pkl"):
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
            #TODO: Add scope for debug by saving intermediate results also in output.
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
        n_coils=data_header["n_coils"],
    )
    recon = fourier_op.adj_op(kspace_data)
    if not fourier_op.uses_sense:
        recon = np.linalg.norm(recon, axis=-1)
    save_data(output_filename, recon, data_header)

store(
    recon,
    obs_reader=raw_config,
    traj_reader=traj_config,
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
    hydra_defaults=[
        "_self_",
        {"fourier": "gpu_lowmem"},
        {"fourier/density_comp": "pipe_lowmem"},
        {"fourier/smaps": "low_frequency"},
    ],
    name="dc_adjoint_lowmem",
)



store.add_to_hydra_store()
zen(recon).hydra_main(
    config_name="dc_adjoint",
    config_path=None,
    version_base="1.3",
)
