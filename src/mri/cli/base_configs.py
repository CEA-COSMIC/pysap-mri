from hydra_zen import store, builds

from mrinufft.io import read_trajectory
from mri.operators import NonCartesianFFT, WaveletN
from mri.operators.fourier.utils import estimate_density_compensation
from mrinufft.io.nsp import read_arbgrad_rawdat
from mrinufft.extras.utils import get_smaps
from mri.operators import NonCartesianFFT, WeightedSparseThreshold
from modopt.opt.linear import Identity
from modopt.opt.linear.wavelet import CupyWaveletTransform


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
    nb_scale=3,
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


