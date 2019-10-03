"""
Neuroimaging non-cartesian reconstruction
=========================================

Author: Chaithya G R

In this tutorial we will reconstruct an MRI image from non-cartesian kspace
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
from mri.numerics.fourier import NFFT, FFT2
from mri.numerics.reconstruct import sparse_rec_fista
from mri.numerics.utils import generate_operators
from mri.numerics.utils import convert_locations_to_mask, \
    convert_mask_to_locations

# Third party import
from modopt.math.metrics import ssim
import numpy as np

# Loading input data
image = pysap.utils.load_image('../../../Data/Pysap_examples/base_image.npy')

# Obtain MRI non-cartesian mask
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

# Get the locations of the kspace samples and the associated observations
fourier_op = NFFT(samples=kspace_loc, shape=image.shape)
kspace_obs = fourier_op.op(image.data)

# Zero order solution
mask_gridded = convert_locations_to_mask(kspace_loc, (image.shape))
kspace_gridded_loc = convert_mask_to_locations(np.fft.fftshift(mask_gridded))
fourier_op_gridded = FFT2(samples=kspace_gridded_loc, shape=image.shape)
kspace_gridded_obs = fourier_op_gridded.op(image.data)
image_rec0 = pysap.Image(data=fourier_op_gridded.adj_op(kspace_gridded_obs))
image_rec0.show()
base_ssim = ssim(image_rec0, image)
print('The Base SSIM is : ' + str(base_ssim))

#############################################################################
# FISTA optimization
# ------------------
#
# We now want to refine the zero order solution using a FISTA optimization.
# Here no cost function is set, and the optimization will reach the
# maximum number of iterations.

# Generate operators
gradient_op, linear_op, prox_op, cost_op = generate_operators(
    data=kspace_obs,
    wavelet_name="UndecimatedBiOrthogonalTransform",
    samples=kspace_loc,
    mu=6.45e-07,
    nb_scales=4,
    non_cartesian=True,
    uniform_data_shape=image.shape,
    gradient_space="synthesis")

# Start the FISTA reconstruction
max_iter = 200
x_final, transform, costs, metrics = sparse_rec_fista(
    gradient_op=gradient_op,
    linear_op=linear_op,
    prox_op=prox_op,
    cost_op=cost_op,
    lambda_init=1.0,
    max_nb_of_iter=max_iter,
    atol=1e-4,
    verbose=1)
image_rec = pysap.Image(data=np.abs(x_final))
image_rec.show()
recon_ssim = ssim(image_rec, image)
print('The Reconstruction SSIM is : ' + str(recon_ssim))
