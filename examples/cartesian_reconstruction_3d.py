"""
Neuroimaging cartesian reconstruction
=====================================

Author: LElgueddari

In this tutorial we will reconstruct an MRI image from the sparse 3D kspace
measurements.

Import neuroimaging data
------------------------

We use the toy datasets available in pysap, more specifically the 3D orange
and the cartesian acquisition scheme.
"""

# Package import
from modopt.math.metrics import ssim
from mri.numerics.fourier import FFT
from mri.numerics.reconstruct import sparse_rec_fista
from mri.numerics.utils import generate_operators
from mri.numerics.utils import convert_mask_to_locations
from pysap.data import get_sample_data
import pysap

# Third party import
import numpy as np

# Loading input data and convert it into a single channel using Sum-Of-Squares
image = get_sample_data('3d-pmri')
image.data = np.sqrt(np.sum(np.abs(image.data)**2, axis=0))

# Obtain K-Space Cartesian Mask
mask = get_sample_data("2d-cartesian-poisson-disk")
mask.data = np.repeat(np.expand_dims(mask.data, axis=-1), 160, axis=-1)

# View Input
# image.show()
# mask.show()

#############################################################################
# Generate the kspace
# -------------------
#
# From the 3D Orange volume and the acquisition mask, we retrospectively
# undersample the k-space using a cartesian acquisition mask
# We then reconstruct the zero order solution as a baseline


# Get the locations of the kspace samples
kspace_loc = convert_mask_to_locations(mask.data)
# Generate the subsampled kspace
fourier_op = FFT(samples=kspace_loc, shape=image.shape)
kspace_data = fourier_op.op(image)

# Zero order solution
image_rec0 = pysap.Image(data=fourier_op.adj_op(kspace_data),
                         metadata=image.metadata)
# image_rec0.show()

# Calculate SSIM
base_ssim = ssim(image_rec0, image)
print(base_ssim)

#############################################################################
# FISTA optimization
# ------------------
#
# We now want to refine the zero order solution using a FISTA optimization.
# The cost function is set to Proximity Cost + Gradient Cost

# Generate operators
gradient_op, linear_op, prox_op, cost_op = generate_operators(
    data=kspace_data,
    wavelet_name="sym8",
    samples=kspace_loc,
    nb_scales=2,
    mu=2*1e-11,
    fourier_type='cartesian',
    uniform_data_shape=None,
    gradient_space="synthesis",
    padding_mode="periodization")

# Start the FISTA reconstruction
max_iter = 200
x_final, costs, metrics = sparse_rec_fista(
    gradient_op,
    linear_op,
    prox_op,
    cost_op,
    lambda_init=1,
    max_nb_of_iter=max_iter,
    atol=1e-4,
    verbose=1)
image_rec = pysap.Image(data=np.abs(x_final))
# image_rec.show()
# Calculate SSIM
recon_ssim = ssim(image_rec, image)
print('The Reconstruction SSIM is : ' + str(recon_ssim))
