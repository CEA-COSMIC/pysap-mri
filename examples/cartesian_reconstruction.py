"""
Neuroimaging cartesian reconstruction
=====================================

Author: Chaithya G R

In this tutorial we will reconstruct an MRI image from the sparse kspace
measurements.

Import neuroimaging data
------------------------

We use the toy datasets available in pysap, more specifically a 2D brain slice
and the cartesian acquisition scheme.
"""

# Package import
from modopt.math.metrics import ssim
from mri.operators import FFT
from mri.operators.utils import convert_mask_to_locations
from mri.reconstructors import SingleChannelReconstructor
import pysap
from pysap.data import get_sample_data

# Third party import
import numpy as np

# Loading input data
image = get_sample_data('2d-mri')

# Obtain K-Space Cartesian Mask
mask = get_sample_data("cartesian-mri-mask")

# View Input
# image.show()
# mask.show()

#############################################################################
# Generate the kspace
# -------------------
#
# From the 2D brain slice and the acquisition mask, we retrospectively
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

# Setup the reconstructor
reconstructor = SingleChannelReconstructor(
    kspace_data=kspace_data,
    kspace_loc=kspace_loc,
    uniform_data_shape=fourier_op.shape,
    wavelet_name="sym8",
    mu=2 * 1e-7,
    nb_scale=4,
    fourier_type='cartesian',
    gradient_method='synthesis',
    optimization_alg='fista',
    verbose=1
)
# Start Reconstruction
x_final, costs, metrics = reconstructor.reconstruct(num_iterations=200)
image_rec = pysap.Image(data=np.abs(x_final))
# image_rec.show()
# Calculate SSIM
recon_ssim = ssim(image_rec, image)
print('The Reconstruction SSIM is : ' + str(recon_ssim))
