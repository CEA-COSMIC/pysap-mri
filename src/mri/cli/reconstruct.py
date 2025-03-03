from hydra_zen import store, zen

from mri.io.output import save_data
from mri.cli.utils import raw_config, traj_config, setup_hydra_config, get_outdir_path
from mri.operators.fourier.utils import discard_frequency_outliers
from mrinufft.io.utils import add_phase_to_kspace_with_shifts, remove_extra_kspace_samples
from pymrt.recipes.coils import compress_svd
from mri.reconstructors import SelfCalibrationReconstructor
from mri.reconstructors.ggrappa import do_grappa_and_append_data, GRAPPA_RECON_AVAILABLE

import json
import numpy as np
import pickle as pkl
import logging, os, glob
from functools import partial


log = logging.getLogger(__name__)

save_data_hydra = lambda x, *args, **kwargs: save_data(get_outdir_path(x), *args, **kwargs)


def dc_adjoint(obs_file: str|np.ndarray, traj_file: str, coil_compress: str|int, debug: int,
               obs_reader, traj_reader, fourier, grappa_recon=None, output_filename: str = "dc_adjoint.nii",
               return_data=False):
    """
    Reconstructs an image using the adjoint operator.

    Parameters
    ----------
    obs_file : str or np.ndarray
        Path to the observed kspace data file.
    traj_file : str
        Path to the trajectory file or the folder holding trajectory file.
        If folder is provided, the trajectory name is picked up and the data header 
        and the trajectory is obtained by searching recursively.
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
    grappa_af: Union[list[int], tuple[int, ...]], optional default 1
        The acceleration factor for the GRAPPA reconstruction.
        
    Returns
    -------
    None
        The reconstructed image is saved as 'dc_adjoint.pkl' file.
    """
    raw_data, data_header = obs_reader(obs_file)
    if obs_reader.keywords['slice_num'] is not None:
        data_header['slice_num'] = obs_reader.keywords['slice_num']
    log.info(f"Data Header: {data_header}")
    try:
        if not os.path.isdir(traj_file) and data_header["trajectory_name"] != os.path.basename(traj_file):
            log.warn("Trajectory file does not match the trajectory in the data file")
    except KeyError:
        log.warn("Trajectory name not found in data header, Skipped Validation")
    if os.path.isdir(traj_file):
        search_folder = traj_file
        found_trajs = glob.glob(os.path.join(search_folder, "**", data_header['trajectory_name']), recursive=True)
        if len(found_trajs) == 0:
            log.error(f"Trajectory {traj_file} from data_header not found in {search_folder}")
            exit(1)
        if len(found_trajs) > 1:
            log.warn("More than one file found, choosing first one")
        traj_file = found_trajs[0]
    elif not os.path.exists(traj_file):
        log.error("Trajectory not found! exiting")
        exit(1)
    log.debug(f"Loading trajectory from {traj_file}")
    shots, traj_params = traj_reader(
        traj_file,
        dwell_time=traj_reader.keywords['raster_time'] / data_header["oversampling_factor"],
    )
    # Need to have image sizes as even to ensure no issues
    traj_params['img_size'] = np.asarray([
        size + 1 if size % 2 else size 
        for size in traj_params['img_size']
    ])
    log.info(f"Trajectory Parameters: {traj_params}")
    data_header["shifts"] = data_header['shifts'][:traj_params["dimension"]]
    normalized_shifts = (
        np.array(data_header["shifts"])
        / np.array(traj_params["FOV"])
        * np.array(traj_params["img_size"])
        / 1000
    )
    kspace_data = np.squeeze(raw_data).astype(np.complex64)
    kspace_loc = shots.reshape(-1, traj_params["dimension"]).astype(np.float32)
    kspace_data = remove_extra_kspace_samples(kspace_data, shots.shape[1])
    kspace_data = kspace_data.reshape(kspace_data.shape[0], -1)
    log.info(f"Phase shifting raw data for Normalized shifts: {normalized_shifts}")
    kspace_data = add_phase_to_kspace_with_shifts(
        kspace_data, kspace_loc.reshape(-1, traj_params["dimension"]), normalized_shifts
    )
    try:
        af_string = data_header['trajectory_name'].split('_G')[1].split('_')[0].split('x')
        if len(af_string) > 1 and 'd' in af_string[1]:
            af_caipi = af_string[1].split('d')
            af_string[1] = af_caipi[0]
            grappa_recon.keywords['delta'] = int(af_caipi[1])
        grappa_recon.keywords['af'] = tuple([int(float(af)) for af in af_string])
    except:
        grappa_recon.keywords['af'] = (1, )
        grappa_recon.keywords['delta'] = 0
    if grappa_recon is not None and np.prod(grappa_recon.keywords['af'])>1:
        log.info("Performing GRAPPA Reconstruction: AF: %s", af_string)
        log.info("GRAPPA AF: %s", grappa_recon.keywords['af'])
        kspace_loc, kspace_data = do_grappa_and_append_data(
            kspace_loc,
            kspace_data,
            traj_params,
            grappa_recon,
            acs=data_header["acs"], # Pass ACS if read in data (external)
        )
    if coil_compress != -1:
        log.info("Compressing coils")
        kspace_data = np.ascontiguousarray(compress_svd(
            kspace_data,
            k_svd=coil_compress,
            coil_axis=0
        )).astype(np.complex64)
    if kspace_loc.max() > 0.5 or kspace_loc.min() < 0.5:
        log.warn(f"K-space locations are above the unity range, discarding the outlier data")
        if data_header["type"] == "retro_recon":
            kspace_loc = discard_frequency_outliers(kspace_loc)
            kspace_data = np.squeeze(raw_data)
        else:
            kspace_loc, kspace_data = discard_frequency_outliers(kspace_loc, kspace_data)
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
            'traj_params': traj_params,
            'data_header': data_header,
            'kspace_loc': kspace_loc,
        }
        save_data_hydra('smaps.nii', fourier_op.impl.smaps)
        if coil_compress != -1:
            intermediate['kspace_data'] = kspace_data
        log.info("Saving Smaps and denisty_comp as intermediates")
        pkl.dump(intermediate, open(get_outdir_path('intermediate.pkl'), 'wb'))
    log.info("Getting the DC Adjoint")
    dc_adjoint = fourier_op.adj_op(kspace_data)
    if not fourier_op.impl.uses_sense:
        dc_adjoint = np.linalg.norm(dc_adjoint, axis=0)
    log.info("Saving DC Adjoint")
    data_header['traj_params'] = traj_params
    save_data_hydra(output_filename, dc_adjoint, data_header)
    if return_data:
        return dc_adjoint, (fourier_op, kspace_data, traj_params, data_header)
    
    
    
def recon(obs_file: str, traj_file: str, mu: float, num_iterations: int, coil_compress: str|int, 
          algorithm: str, debug: int, obs_reader, traj_reader, fourier, linear, sparsity,
          output_filename: str = "recon.nii", remove_dc_for_recon: bool = True, validation_recon: np.ndarray = None, metrics: dict = None, 
          grappa_recon=None):
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
    remove_dc_for_recon: bool, optional
        Whether to remove the density compensation for reconstruction, by default True
        Note that it will still be used to estimate x_init
    validation_recon: np.ndarray, optional
        The validation reconstruction to compare the results with, by default None
    metrics: dict, optional
        List of metrics to evaluate the reconstruction, by default None
    """
    recon_adjoint, additional_data = dc_adjoint(
        obs_file,
        traj_file,
        coil_compress,
        debug,
        obs_reader,
        traj_reader,
        fourier,
        grappa_recon=grappa_recon,
        output_filename='dc_adj_' + output_filename,
        return_data=True,
    )
    fourier_op, kspace_data, traj_params, data_header = additional_data
    if remove_dc_for_recon:
        fourier_op.impl.density = None
    K = fourier_op.op(recon_adjoint)
    alpha = np.mean(np.linalg.norm(kspace_data, axis=0)) / np.mean(np.linalg.norm(K, axis=0))
    recon_adjoint *= alpha
    linear_op = linear(shape=tuple(traj_params["img_size"]), dim=traj_params['dimension'])
    linear_op.op(recon_adjoint)
    sparse_op = sparsity(coeffs_shape=linear_op.coeffs_shape, weights=mu)
    log.info("Setting up reconstructor")
    reconstructor = SelfCalibrationReconstructor(
        fourier_op=fourier_op,
        linear_op=linear_op,
        regularizer_op=sparse_op,
        verbose=1,
        lipschitz_cst=fourier_op.impl.get_lipschitz_cst(),
    )
    log.info("Starting reconstruction")
    recon, costs, metrics_iter = reconstructor.reconstruct(
        kspace_data=kspace_data,
        optimization_alg=algorithm,
        x_init=recon_adjoint, # gain back the first step by initializing with DC Adjoint
        num_iterations=num_iterations,
    )
    if validation_recon is not None:
        log.info("getting metrics of the reconstruction")
        final_metrics = {}
        for metric, function in metrics.items():
            final_metrics[metric] = function(recon, validation_recon)
            final_metrics[f"dc_{metric}"] = function(recon_adjoint, validation_recon)
        log.info(f"Final Metrics: {final_metrics}")
        with open(get_outdir_path('metrics.json'), 'w') as f:
            final_metrics["traj"] = data_header["trajectory_name"]
            f.write(json.dumps(final_metrics, indent=4))
        data_header['metrics'] = final_metrics
    data_header['costs'] = costs
    data_header['metrics_iter'] = metrics_iter
    log.info("Saving reconstruction results")
    save_data_hydra(output_filename, recon, data_header)

setup_hydra_config()
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
        {"grappa_recon": "disable"} if GRAPPA_RECON_AVAILABLE else {},
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
    mu=1e-7,
    debug=1,
    hydra_defaults=[
        "_self_",
        {"fourier": "gpu"},
        {"fourier/density_comp": "pipe"},
        {"grappa_recon": "disable"} if GRAPPA_RECON_AVAILABLE else {},
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
    mu=1e-7,
    debug=1,
    hydra_defaults=[
        "_self_",
        {"fourier": "gpu_lowmem"},
        {"grappa_recon": "disable"} if GRAPPA_RECON_AVAILABLE else {},
        {"fourier/density_comp": "pipe_lowmem"},
        {"fourier/smaps": "low_frequency"},
    ],
    name="recon_lowmem",
)

# Setup the Hydra Config and callbacks.
store.add_to_hydra_store()


def run_recon():
    zen(recon).hydra_main(
        config_name="recon",
        config_path=None,
        version_base="1.3",
    )

def run_adjoint():
    zen(dc_adjoint).hydra_main(
        config_name="dc_adjoint",
        config_path=None,
        version_base="1.3",
    )
