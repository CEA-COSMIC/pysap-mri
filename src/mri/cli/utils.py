from hydra_zen import store, builds
from hydra.conf import HydraConf, JobConf, SweepDir
import hydra

from mrinufft.io import read_trajectory
from mri.operators import NonCartesianFFT, WaveletN
from mri.optimizers.utils.cost import GenericCost
from mri.operators.fourier.utils import estimate_density_compensation
from mrinufft.io.nsp import read_arbgrad_rawdat
from mrinufft.extras.utils import get_smaps
from mri.operators import NonCartesianFFT, WeightedSparseThreshold
from modopt.opt.linear import Identity
import os


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
linear_config = builds(
    WaveletN,
    populate_full_signature=True,
    zen_partial=True,
    wavelet_name="sym8",
    nb_scale=4,
    zen_exclude=["shape"]
)
sparsity_config = builds(
    WeightedSparseThreshold,
    populate_full_signature=True,
    zen_partial=True,
    linear=Identity(),
    use_gpu=True,
    zen_exclude=["coeffs_shape", "linear", "weights", "use_gpu"]
)
cost_config = builds(
    GenericCost,
    cost_interval=None,
    test_range=4,
    verbose=0,
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

linear_store = store(group="linear")
linear_store(linear_config, name="gpu")

sparsity_store = store(group="sparsity")
sparsity_store(sparsity_config, name="weighted_sparse")

def setup_hydra_config():
    """
    Set up the configuration for Hydra.

    Returns
    -------
    None
        This function does not return anything.
    """
    outdir = os.environ.get('RECON_OUTDIR', 'recon')
    store(
        HydraConf(
            job=JobConf(name="recon"),
            sweep=SweepDir(dir=os.path.join(outdir, "${hydra.job.name}") + "/${now:%Y-%m-%d-%H-%M-%S}"),
            callbacks={
                'git_infos': {
                    '_target_': "hydra_callbacks.GitInfo",
                    'clean': True
                },
                'resource_monitor': {
                    '_target_': "hydra_callbacks.ResourceMonitor",
                    'sample_interval': 1,
                    'gpu_monit': True,
                },
                'runtime_perf': {
                    '_target_': "hydra_callbacks.RuntimePerformance"
                },
            },
            verbose=True,
        )
    )

def get_outdir_path(filename=''):
    """Get the output directory path.

    This function returns the path of the output directory where the files will be saved.

    Parameters
    ----------
    filename : str, optional
        The name of the file to be appended to the output directory path, by default ''

    Returns
    -------
    str
        The path of the output directory.

    """
    out = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    if filename != '':
        out = os.path.join(out, filename)
    return out