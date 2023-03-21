"""
Neuroimaging Cartesian reconstruction
=====================================

Author: Pierre-Antoine Comby / Chaithya G R

In this tutorial we will reconstruct an MR image from the sparse k-space
measurements.

Moreover we will see the benefit of automating the  tuning of the regularisation parameters.

Import neuroimaging data
------------------------

We use the toy datasets available in pysap, more specifically a 2D brain slice
and the Cartesian acquisition scheme.
"""

# Package import
from mri.operators import FFT, WaveletN
from mri.operators.utils import convert_mask_to_locations
from mri.reconstructors import SingleChannelReconstructor
from mri.operators.proximity.weighted import AutoWeightedSparseThreshold
import pysap
from pysap.data import get_sample_data

# Third party import
from modopt.opt.proximity import SparseThreshold
from modopt.opt.linear import Identity
from modopt.math.metrics import ssim
import numpy as np

# Loading input data
image = get_sample_data('2d-mri')

# Obtain k-space Cartesian Mask
mask = get_sample_data("cartesian-mri-mask")

# View Input
# image.show()
# mask.show()

#%%
# Generate the kspace
# -------------------
#
# From the 2D brain slice and the sampling mask, we retrospectively
# undersample the k-space using a Cartesian acquisition mask
# We then reconstruct the zero order solution as a baseline


# Get the locations of the kspace samples
kspace_loc = convert_mask_to_locations(mask.data)
# Generate the subsampled kspace
fourier_op = FFT(samples=kspace_loc, shape=image.shape)
kspace_data = fourier_op.op(image)

# Zero filled solution
image_rec0 = pysap.Image(data=fourier_op.adj_op(kspace_data),
                         metadata=image.metadata)
# image_rec0.show()

# Calculate SSIM
base_ssim = ssim(image_rec0, image)
print(base_ssim)

#%%
# FISTA optimization
# ------------------
#
# We now want to refine the zero order solution by computing the Compressed sensing one,
# using FISTA optimization.
# The cost function is set to Proximity Cost + Gradient Cost

# Setup the operators
linear_op = WaveletN(wavelet_name="sym8", nb_scales=4)
coeffs = linear_op.op(image_rec0)

#%%
# the auto estimation of the threshold uses the methods of :cite:`donoho1994`.
# The noise standard deviation is estimated on the largest scale using the detail (HH) band.
# A single threshold is then also estimated for each scale.

regularizer_op = AutoWeightedSparseThreshold(
    coeffs_shape=coeffs.shape,
    linear=Identity(),
    update_period=5,
    sigma_range="global",
    tresh_range="scale",
    threshold_estimation="sure",
    thresh_type="soft"
)
#%% The rest of the setup is similar to classical example.

# Setup Reconstructor
reconstructor = SingleChannelReconstructor(
    fourier_op=fourier_op,
    linear_op=linear_op,
    regularizer_op=regularizer_op,
    gradient_formulation='synthesis',
    verbose=1,
)

#%%
#  With everythiing setup we can start Reconstruction
x_final, costs, metrics = reconstructor.reconstruct(
    kspace_data=kspace_data,
    optimization_alg='fista',
    num_iterations=200,
)
image_rec = pysap.Image(data=np.abs(x_final))
# image_rec.show()
# Calculate SSIM
recon_ssim = ssim(image_rec, image)
print('The Reconstruction SSIM is : ' + str(recon_ssim))
