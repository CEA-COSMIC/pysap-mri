from hydra_zen import store, builds, zen

from mrinufft.io import read_trajectory
from mri.operators import NonCartesianFFT
from mri.operators.fourier.utils import estimate_density_compensation
from mri.cli.utils import save_data
from mrinufft.io.nsp import read_arbgrad_rawdat
from mrinufft.io.utils import add_phase_to_kspace_with_shifts
from mrinufft.extras.utils import get_smaps
from pymrt.recipes.coils import compress_svd

import numpy as np
import logging
import os

log = logging.getLogger(__name__)
raw_config = builds(read_arbgrad_rawdat, populate_full_signature=True, zen_partial=True)

traj_config = builds(
    read_trajectory,
    populate_full_signature=True,
    zen_exclude=["dwell_time"],
    zen_partial=True,
)

density_est_config = builds(
    estimate_density_compensation,
    populate_full_signature=True,
    zen_exclude=['kspace_loc', 'volume_shape'],
    zen_partial=True,
)
smaps_config = builds(
    get_smaps("low_frequency"),
    populate_full_signature=True,
    # We estimate density, with separate args. It is passed by compute_smaps in mri-nufft
    zen_exclude=["density"],
    zen_partial=True,
)
fourier_op_config = builds(
    NonCartesianFFT,
    populate_full_signature=True,
    zen_exclude=["n_coils"],
    zen_partial=True,
)


fourier_store = store(group="fourier")
fourier_store(fourier_op_config, name="cpu")
fourier_store(
    fourier_op_config,
    implementation="gpuNUFFT",
    name="gpu",
)
fourier_store(
    fourier_op_config,
    upsampfac=1,
    implementation="gpuNUFFT",
    name="gpu_lowmem",
)

smaps_store = store(group="fourier/smaps")
smaps_store(smaps_config, name="low_frequency")
density_store = store(group="fourier/density_comp")
density_store(density_est_config, implementation="pipe", name="pipe")
density_store(density_est_config, implementation="pipe", osf=1, name="pipe_lowmem")



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
    try:
        if data_header["trajectory_name"] != os.path.basename(traj_file):
            log.warn("Trajectory file does not match the trajectory in the data file")
    except KeyError:
        log.warn("Trajectory name not found in data header, Skipped Validation")
    shots, traj_params = traj_reader(
        traj_file,
        dwell_time=traj_reader.keywords['raster_time'] / data_header["oversampling_factor"],
    )
    kspace_loc = shots.reshape(-1, traj_params["dimension"])
    normalized_shifts = (
        np.array(data_header["shifts"])
        / np.array(traj_params["FOV"])
        * np.array(traj_params["img_size"])
        / 1000
    )
    log.debug(f"Normalized shifts: {normalized_shifts}")
    kspace_data = add_phase_to_kspace_with_shifts(
        np.squeeze(raw_data), kspace_loc, normalized_shifts
    )
    if coil_compress != -1:
        kspace_data = np.ascontiguousarray(compress_svd(
            kspace_data,
            k_svd=coil_compress,
            coil_axis=0
        ))
    fourier_op = fourier(
        kspace_loc, traj_params["img_size"], n_coils=data_header["n_coils"]
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
