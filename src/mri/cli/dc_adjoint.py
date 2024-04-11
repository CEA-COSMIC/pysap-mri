from hydra_zen import store, builds, zen

from mrinufft.io import read_trajectory
from mri.operators import NonCartesianFFT
from mrinufft.io.nsp import read_siemens_rawdat
from mrinufft.io.utils import add_phase_to_kspace_with_shifts

from mrinufft.trajectories.utils import DEFAULT_RASTER_TIME

import numpy as np
import pickle as pkl
import logging
import os

log = logging.getLogger(__name__)
raw_config = builds(read_siemens_rawdat, populate_full_signature=True, zen_partial=True)
traj_config = builds(
    read_trajectory,
    populate_full_signature=True,
    zen_exclude=["dwell_time"],
    zen_partial=True
)

fourier_op_config = builds(
    NonCartesianFFT,
    populate_full_signature=True,
    zen_partial=True,
    zen_exclude=["n_coils"]
)
fourier_store = store(group="fourier")
fourier_store(fourier_op_config, name="cpu")
fourier_store(
    fourier_op_config,
    implementation="gpuNUFFT",
    density_comp="pipe",
    name="gpu"
)


def recon(obs_file: str, traj_file: str, obs_reader, traj_reader, fourier):
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
    fourier : callable
        A function that performs Fourier transform operations.

    Returns
    -------
    None
        The reconstructed image is saved as 'dc_adjoint.pkl' file.
    """
    raw_data, data_header = obs_reader(obs_file)
    if "trajectory_name" in data_header.keys():
        if data_header["trajectory_name"] != os.path.basename(traj_file):
            log.warn("Trajectory file does not match the trajectory in the data file")
    else:
        log.warn("Trajectory name not found in data header, validation step not done")
    shots, traj_params = traj_reader(
        traj_file,
        dwell_time=DEFAULT_RASTER_TIME/data_header["oversampling_factor"]
    )
    kspace_loc = shots.reshape(-1, traj_params["dimension"])
    normalized_shifts =  (
        np.array(data_header["shifts"]) / np.array(traj_params["FOV"]) 
        * np.array(traj_params["img_size"]) / 1000
    )
    log.debug(f"Normalized shifts: {normalized_shifts}")
    kspace_data = add_phase_to_kspace_with_shifts(
        np.squeeze(raw_data),
        kspace_loc,
        normalized_shifts
    )
    fourier_op = fourier(
        kspace_loc,
        traj_params["img_size"],
        n_coils=data_header["n_coils"]
    )
    per_ch_image = fourier_op.adj_op(kspace_data)
    combined_image = np.linalg.norm(per_ch_image, axis=-1)
    pkl.dump(combined_image, open("dc_adjoint.pkl", "wb"))
    
store(
    recon,
    obs_reader=raw_config,
    traj_reader=traj_config,
    hydra_defaults=[
        "_self_",
        {"fourier": "gpu"},
    ],
    name="dc_adjoint"
)


if __name__ == "__main__":
    store.add_to_hydra_store()
    zen(recon).hydra_main(
        config_name="dc_adjoint",
        config_path=None,
        version_base="1.3",
    )