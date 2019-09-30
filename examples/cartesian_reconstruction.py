"""
Neuroimaging cartesian reconstruction
=====================================

Credit: A Grigis, L Elgueddari, H Carrie

In this tutorial we will reconstruct an MRI image from the sparse kspace
measurments.

Import neuroimaging data
------------------------

We use the toy datasets available in pysap, more specifically a 2D brain slice
and the acquistion cartesian scheme.
We also add some gaussian noise in the image space.
"""

# Package import
import pysap
from pysap.data import get_sample_data
from mri.numerics.reconstruct import sparse_rec_fista
from mri.numerics.reconstruct import sparse_rec_condatvu
from mri.numerics.fourier import FFT2
from mri.numerics.utils import generate_operators
from mri.numerics.utils import convert_mask_to_locations, \
    convert_locations_to_mask

# Third party import
import numpy as np
from scipy.io import loadmat

# Loading input data
image_name = '../../../Data/meas_MID41_CSGRE_ref_OS1_FID14687.mat'
k_space_ref = loadmat(image_name)['ref']
k_space_ref /= np.linalg.norm(k_space_ref)
Sl = np.zeros((32, 512, 512), dtype='complex128')
for channel in range(k_space_ref.shape[-1]):
    Sl[channel] = np.fft.fftshift(np.fft.ifft2(np.reshape(
            k_space_ref[:, channel], (512, 512))))
image = pysap.Image(data=np.sqrt(np.sum(np.abs(Sl)**2, 0)))

# Add Noise to input MRI Image

# Obtain MRI Mask
radial_mask = get_sample_data("mri-radial-samples")
kspace_loc = radial_mask.data
mask = pysap.Image(data=convert_locations_to_mask(kspace_loc, image.shape))

# View Input
image.show()
mask.show()


#############################################################################
# Generate the kspace
# -------------------
#
# From the 2D brain slice and the acquistion mask, we generate the acquisition
# measurments, the observed kspace.
# We then reconstruct the zero order solution.


# Get the locations of the kspace samples
kspace_loc = convert_mask_to_locations(np.fft.fftshift(mask.data))
# Generate the subsampled kspace
fourier_op = FFT2(samples=kspace_loc, shape=image.shape)
kspace_data = fourier_op.op(image)

# Zero order solution
image_rec0 = pysap.Image(data=fourier_op.adj_op(kspace_data),
                         metadata=image.metadata)
image_rec0.show()

#############################################################################
# FISTA optimization
# ------------------
#
# We now want to refine the zero order solution using a FISTA optimization.
# Here no cost function is set, and the optimization will reach the
# maximum number of iterations. Fill free to play with this parameter.

# Generate operators
gradient_op, linear_op, prox_op, cost_op = generate_operators(
    data=kspace_data,
    wavelet_name="sym8",
    samples=kspace_loc,
    nb_scales=4,
    mu=5e-6,
    non_cartesian=False,
    uniform_data_shape=None,
    gradient_space="synthesis",
    padding_mode="periodization")

# Start the FISTA reconstruction
max_iter = 200
x_final, transform, costs, metrics = sparse_rec_fista(
    gradient_op,
    linear_op,
    prox_op,
    cost_op,
    lambda_init=10,
    max_nb_of_iter=max_iter,
    atol=1e-4,
    verbose=1)
image_rec = pysap.Image(data=np.abs(x_final))
image_rec.show()

#############################################################################
# Condata-Vu optimization
# -----------------------
#
# We now want to refine the zero order solution using a Condata-Vu
# optimization.
# Here no cost function is set, and the optimization will reach the
# maximum number of iterations. Fill free to play with this parameter.

# Generate operators
gradient_op, linear_op, prox_op, cost_op = generate_operators(
    data=kspace_data,
    wavelet_name="BsplineWaveletTransformATrousAlgorithm",
    samples=kspace_loc,
    nb_scales=4,
    mu=1e-9,
    non_cartesian=False,
    uniform_data_shape=None,
    gradient_space="analysis",
    padding_mode="symmetric")

# Start the CONDAT-VU reconstruction
max_iter = 20
x_final, transform, costs, metrics = sparse_rec_condatvu(
    gradient_op,
    linear_op,
    prox_op,
    cost_op,
    std_est=None,
    std_est_method="dual",
    std_thr=2.,
    tau=None,
    sigma=None,
    relaxation_factor=1.0,
    max_nb_of_iter=max_iter,
    add_positivity=False,
    atol=1e-4,
    verbose=1)
image_rec = pysap.Image(data=np.abs(x_final))
image_rec.show()
cost_rec = pysap.Image(data=costs)
cost_rec.show()
