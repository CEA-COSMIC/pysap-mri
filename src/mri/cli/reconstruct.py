from hydra_zen import store, zen

from mri.io.output import save_data
from mri.cli.utils import raw_config, traj_config, setup_hydra_config, get_outdir_path
from mri.operators.fourier.utils import discard_frequency_outliers
from mrinufft.io.utils import add_phase_to_kspace_with_shifts, remove_extra_kspace_samples
from pymrt.recipes.coils import compress_svd
from mri.reconstructors import SelfCalibrationReconstructor
from mri.reconstructors.ggrappa import do_grappa_and_append_data, GRAPPA_RECON_AVAILABLE
from mrinufft.operators import FourierOperatorBase
import json
import numpy as np
import pickle as pkl
import logging
import os
import glob
from functools import partial
import pickle

import nibabel as nib
from scipy.ndimage import zoom
from mrinufft.operators.off_resonance import MRIFourierCorrected
from mrinufft import get_operator
from mrinufft.extras.smaps import low_frequency
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
log = logging.getLogger(__name__)

save_data_hydra = lambda x, * \
    args, **kwargs: save_data(get_outdir_path(x), *args, **kwargs)


def visualize_map(B0map, slice_index=None, vmin=-100, vmax=100, subtitle=None):
    """
    Function to visualize B0map in three views: Axial, Sagittal, and Coronal, with fixed scaling.

    Parameters:
        B0map (numpy.ndarray): 3D array representing the B0 field map.
        slice_index (int, optional): Index of the slice to display. Defaults to the middle slice.
        vmin (float): Minimum value for colormap normalization (fixed scale for comparison).
        vmax (float): Maximum value for colormap normalization (fixed scale for comparison).
        subtitle (str, optional): Title for the entire figure.
    """
    if slice_index is None:
        slice_index = B0map.shape[2] // 2  # Default to middle slice

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), gridspec_kw={
                             'width_ratios': [1, 1, 1]})

    # Define colormap and normalization for the colorbar
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap('viridis')

    # **Axial View**
    ax = axes[0]
    axial = B0map[:, :, slice_index]
    ax.imshow(np.rot90(axial, 2, (1, 0)), origin='lower',
              vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_title('Axial')

    # **Sagittal View**
    ax = axes[1]
    sagittal = B0map[:, slice_index, :]
    rotated_sagittal = np.fliplr(np.rot90(sagittal, 1, (1, 0)))
    ax.imshow(rotated_sagittal, origin='lower',
              vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_title('Sagittal')

    # **Coronal View**
    ax = axes[2]
    coronal = B0map[slice_index, :, :]
    rotated_coronal = np.fliplr(np.rot90(coronal, 1, (1, 0)))
    ax.imshow(rotated_coronal, origin='lower', vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_title('Coronal')

    # Adjust layout to avoid overlap
    plt.subplots_adjust(right=0.85)

    # **Colorbar**
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
                 cax=cbar_ax).set_label("")

    # Title for the entire figure
    plt.suptitle(subtitle)
    plt.show()


def show_3_planes(volume, save_path, title="Volume", cmap='gray'):
    import matplotlib.pyplot as plt
    import os

    x, y, z = volume.shape
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(volume[:, :, z // 2], cmap=cmap)
    axes[0].set_title("Axial")

    axes[1].imshow(volume[:, y // 2, :], cmap=cmap)
    axes[1].set_title("Coronal")

    axes[2].imshow(volume[x // 2, :, :], cmap=cmap)
    axes[2].set_title("Sagittal")

    for ax in axes:
        ax.axis("off")
    fig.suptitle(title)
    plt.tight_layout()

    # Save before showing
    plt.savefig(save_path)
    print(f"✅ Saved: {os.path.abspath(save_path)}")


def resample_b0_map(b0_map, target_shape):
    """Resample B0 map to match target image shape."""
    current_shape = b0_map.shape
    zoom_factors = [t / c for t, c in zip(target_shape, current_shape)]
    return zoom(b0_map, zoom_factors, order=1)  # Linear interpolation


def dc_adjoint(obs_file: str | np.ndarray, traj_file: str, coil_compress: str | int, debug: int,
               obs_reader, traj_reader, fourier, grappa_recon=None, output_filename: str = "dc_adjoint.nii",
               return_data=False, orc: bool = False, b0_map_file: str = None):
    """
    Reconstructs an image using the adjoint operator with optional B0 off-resonance correction.
    """
    preprocessed_file = 'preprocessed_data.pkl'
    smaps_file = 'smaps.pkl'

    if os.path.exists(preprocessed_file):
        with open(preprocessed_file, 'rb') as f:
            preprocessed_data = pickle.load(f)
        kspace_data = preprocessed_data['kspace_data']
        kspace_loc = preprocessed_data['kspace_loc']
        data_header = preprocessed_data['data_header']
        traj_params = preprocessed_data['traj_params']
        shots = preprocessed_data['shots']
        log.info("Loaded preprocessed data from preprocessed_data.pkl")
    else:
        # Step 1: Read Raw K-Space Data and Data Header
        raw_data, data_header = obs_reader(obs_file)

        # Step 2: Handle Slice Number If Present
        if obs_reader.keywords['slice_num'] is not None:
            data_header['slice_num'] = obs_reader.keywords['slice_num']

        log.info(f"Data Header: {data_header}")

        # Step 3: Validate Trajectory File (Make Sure It Matches the Data)
        try:
            if not os.path.isdir(traj_file) and data_header["trajectory_name"] != os.path.basename(traj_file):
                log.warn(
                    "Trajectory file does not match the trajectory in the data file")
        except KeyError:
            log.warn("Trajectory name not found in data header, Skipped Validation")

        # Step 4: Find the Correct Trajectory File If a Folder is Given
        if os.path.isdir(traj_file):
            search_folder = traj_file
            found_trajs = glob.glob(os.path.join(
                search_folder, "**", data_header['trajectory_name']), recursive=True)
            if len(found_trajs) == 0:
                log.error(
                    f"Trajectory {traj_file} from data_header not found in {search_folder}")
                exit(1)
            if len(found_trajs) > 1:
                log.warn("More than one file found, choosing first one")
            traj_file = found_trajs[0]
        elif not os.path.exists(traj_file):
            log.error("Trajectory not found! Exiting")
            exit(1)

        log.debug(f"Loading trajectory from {traj_file}")

        # Step 5: Load Trajectory Data and Parameters
        shots, traj_params = traj_reader(
            traj_file,
            dwell_time=traj_reader.keywords['raster_time'] /
            data_header["oversampling_factor"],
        )

        # Step 6: Ensure Image Sizes are Even
        traj_params['img_size'] = np.asarray([
            size + 1 if size % 2 else size for size in traj_params['img_size']
        ])
        log.info(f"Trajectory Parameters: {traj_params}")

        # Step 7: Normalize Phase Shifts
        data_header["shifts"] = data_header['shifts'][:traj_params["dimension"]]
        normalized_shifts = (
            np.array(data_header["shifts"])
            / np.array(traj_params["FOV"])
            * np.array(traj_params["img_size"])
            / 1000
        )

        # Step 8: Convert Raw Data into k-Space Format
        kspace_data = np.squeeze(raw_data).astype(np.complex64)
        kspace_loc = shots.reshape(-1,
                                   traj_params["dimension"]).astype(np.float32)

        # Step 9: Remove Extra k-Space Samples
        kspace_data = remove_extra_kspace_samples(kspace_data, shots.shape[1])
        kspace_data = kspace_data.reshape(kspace_data.shape[0], -1)

        # Step 10: Apply Phase Shift Corrections
        log.info(
            f"Phase shifting raw data for Normalized shifts: {normalized_shifts}")
        kspace_data = add_phase_to_kspace_with_shifts(
            kspace_data, kspace_loc.reshape(-1,
                                            traj_params["dimension"]), normalized_shifts
        )

        # Step 11: GRAPPA Reconstruction (If Enabled)
        try:
            af_string = data_header['trajectory_name'].split('_G')[1].split('_')[
                0].split('x')
            if len(af_string) > 1 and 'd' in af_string[1]:
                af_caipi = af_string[1].split('d')
                af_string[1] = af_caipi[0]
                grappa_recon.keywords['delta'] = int(af_caipi[1])
            grappa_recon.keywords['af'] = tuple(
                [int(float(af)) for af in af_string])
        except:
            grappa_recon.keywords['af'] = (1, )
            grappa_recon.keywords['delta'] = 0
        if grappa_recon is not None and np.prod(grappa_recon.keywords['af']) > 1:
            log.info("Performing GRAPPA Reconstruction: AF: %s", af_string)
            log.info("GRAPPA AF: %s", grappa_recon.keywords['af'])
            kspace_loc, kspace_data = do_grappa_and_append_data(
                kspace_loc,
                kspace_data,
                traj_params,
                grappa_recon,
                acs=data_header["acs"],  # Pass ACS if read in data (external)
            )
        # Step 12: Save Processed Data to Pickle for Future Use
        preprocessed_data = {
            'kspace_data': kspace_data,
            'kspace_loc': kspace_loc,
            'data_header': data_header,
            'traj_params': traj_params,
            'shots': shots
        }
        with open(preprocessed_file, 'wb') as f:
            pickle.dump(preprocessed_data, f)
        log.info("Saved preprocessed data to preprocessed_data.pkl")

    required_vars = {'kspace_data', 'kspace_loc',
                     'data_header', 'traj_params', 'shots'}
    if not all(var in preprocessed_data for var in required_vars):
        log.error(
            f"Missing required variables in preprocessed_data: {required_vars - set(preprocessed_data.keys())}")
        raise ValueError("Incomplete preprocessed data")

    smaps = None
    if os.path.exists(smaps_file):
        with open(smaps_file, 'rb') as f:
            smaps = pickle.load(f)
        log.info("Loaded smaps from smaps.pkl")
    else:
        log.warning("WARNING: smaps is None, computing manually")

        if coil_compress != -1:
            log.info("Compressing coils")
            kspace_data = np.ascontiguousarray(compress_svd(
                kspace_data,
                k_svd=coil_compress,
                coil_axis=0
            )).astype(np.complex64)
        if kspace_loc.max() > 0.5 or kspace_loc.min() < 0.5:
            log.warn(
                f"K-space locations are above the unity range, discarding the outlier data")
            if data_header["type"] == "retro_recon":
                kspace_loc = discard_frequency_outliers(kspace_loc)
                kspace_data = np.squeeze(raw_data)
            else:
                kspace_loc, kspace_data = discard_frequency_outliers(
                    kspace_loc, kspace_data)

        fourier.keywords['smaps'] = partial(
            fourier.keywords['smaps'],
            kspace_data=kspace_data,
        )

        log.info(" Computing smaps using low_frequency method")
        smaps, sos = low_frequency(
            traj=kspace_loc,
            shape=traj_params["img_size"],
            kspace_data=kspace_data,
            backend="gpunufft"
        )

        if smaps is None:
            log.error("Failed to compute smaps")
        with open(smaps_file, 'wb') as f:
            pickle.dump(smaps, f)
        log.info("Saved computed smaps to smaps.pkl")

        fourier_op = fourier(
            kspace_loc,
            traj_params["img_size"],
            n_coils=data_header["n_coils"] if coil_compress == -
            1 else coil_compress,
            smaps=smaps,
        )
        """
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
            pkl.dump(intermediate, open(
                get_outdir_path('intermediate.pkl'), 'wb'))
        
        """

    target_shape = (32, 64, 64, 36)
    zoom_factors = (
        1,
        target_shape[1]/smaps.shape[1],
        target_shape[2]/smaps.shape[2],
        target_shape[3]/smaps.shape[3]
    )

    log.info("New smaps shape: {smaps.shape}")
    if orc and b0_map_file:
        log.info("Applying B0 off-resonance correction with MRI Fourier Operator")
        b0_map_nii = nib.load(b0_map_file)
        b0_map = b0_map_nii.get_fdata().astype(np.float32)
        current_shape = b0_map.shape
        target_shape = (384, 384, 208)
        zoom_factors = tuple(t/c for t, c in zip(target_shape, current_shape))
        b0_map_interp = zoom(b0_map, zoom_factors, order=1)
        b0_map_aligned = np.flip(b0_map_interp, (0, 2))
        dwell_time = traj_reader.keywords['raster_time'] / \
            data_header["oversampling_factor"]
        # Convert milliseconds to seconds
        TE = 20e-3
        obs_time = 20.48e-3
        # kspace points acquired per shot  : 40642560 : kspace_loc.shape[0]and we have 3969 shots so : kspace_loc.shape[0]/n_shots
        n_pts = 10240
        n_shots = 3969  # 40642560/2048/5
        start_time = TE - (obs_time / 2)
        end_time = TE + (obs_time / 2)

        readout_time_single = np.linspace(start_time, end_time, num=n_pts)
        readout_time = np.tile(readout_time_single, (n_shots, 1))
        readout_time = readout_time.reshape(-1, 1)
        readout_time = readout_time[:kspace_loc.shape[0]]
        readout_time = readout_time.squeeze()

        if kspace_data.shape[0] == 32 and kspace_data.shape[1] == 40642525:
            kspace_data = kspace_data.T
        nufft = get_operator("gpunufft")(
            samples=2 * np.pi * kspace_loc,
            shape=(384, 384, 208),
            n_coils=32,
            density=True,
            smaps=smaps,
        )

        orc_nufft = MRIFourierCorrected(
            nufft, b0_map=b0_map_aligned, readout_time=readout_time, backend='cpu'
        )

        log.info("Getting the DC adjoint")
        dc_adjoint = orc_nufft.adj_op(kspace_data)
        dc_adjoint = np.squeeze(abs(dc_adjoint))
        if len(dc_adjoint.shape) > 3:
            log.info(
                f"Reshaping dc_adjoint from {dc_adjoint.shape} to {traj_params['img_size']}")
            dc_adjoint = dc_adjoint.reshape(traj_params["img_size"])
        log.info("Saving DC Adjoint")

        save_data_hydra(output_filename, dc_adjoint, data_header)
        if return_data:
            return dc_adjoint, (fourier_op, kspace_data, traj_params, data_header)
    else:

        log.info("Getting the DC Adjoint")
        fourier.keywords['smaps'] = partial(
            fourier.keywords['smaps'],
            kspace_data=kspace_data,
        )
        fourier_op = fourier(
            kspace_loc,
            traj_params["img_size"],
            n_coils=data_header["n_coils"] if coil_compress == -
            1 else coil_compress,
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
            pkl.dump(intermediate, open(
                get_outdir_path('intermediate.pkl'), 'wb'))
        dc_adjoint = fourier_op.adj_op(kspace_data)
        if not getattr(fourier_op, 'uses_sense', False):
            dc_adjoint = np.linalg.norm(dc_adjoint, axis=0)

        log.info("Saving DC Adjoint")
        save_data_hydra(output_filename, dc_adjoint, data_header)
        if return_data:
            return dc_adjoint, (fourier_op, kspace_data, traj_params, data_header)


def recon(obs_file: str, traj_file: str, mu: float, num_iterations: int, coil_compress: str | int,
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
    alpha = np.mean(np.linalg.norm(kspace_data, axis=0)) / \
        np.mean(np.linalg.norm(K, axis=0))
    recon_adjoint *= alpha
    linear_op = linear(shape=tuple(
        traj_params["img_size"]), dim=traj_params['dimension'])
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
        x_init=recon_adjoint,  # gain back the first step by initializing with DC Adjoint
        num_iterations=num_iterations,
    )
    if validation_recon is not None:
        log.info("getting metrics of the reconstruction")
        final_metrics = {}
        for metric, function in metrics.items():
            final_metrics[metric] = function(recon, validation_recon)
            final_metrics[f"dc_{metric}"] = function(
                recon_adjoint, validation_recon)
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
        {"grappa_recon": "disable"},
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
        {"grappa_recon": "disable"},
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
        {"grappa_recon": "disable"},
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
