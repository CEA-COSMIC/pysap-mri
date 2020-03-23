"""
Non Cartesian Self Calibrating Reconstruction
====----=====================================

Author: Chaithya G R

In this tutorial we will reconstruct an MRI image from non cartesian kspace
measurements.

Import neuroimaging data
------------------------

We use the toy datasets available in pysap, more specifically a 2D parallel MRI
brain slice on 32 channels and the acquisition non cartesian scheme.
"""

# Package import
from mri.operators import NonCartesianFFT, WaveletN
from mri.reconstructors import SelfCalibrationReconstructor
from mri.reconstructors.utils.extract_sensitivity_maps import get_Smaps
import pysap
from pysap.data import get_sample_data

# Third party import
from modopt.math.metrics import ssim
from modopt.opt.linear import Identity
from modopt.opt.proximity import SparseThreshold
import numpy as np

# Loading input data
cartesian_ref_image = get_sample_data('2d-pmri')
image = pysap.Image(data=np.sqrt(np.sum(cartesian_ref_image.data**2, axis=0)))

# Obtain MRI cartesian mask
mask = get_sample_data("mri-radial-samples")
kspace_loc = mask.data

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

# Get the locations of the kspace samples and the associated observations
fourier_op = NonCartesianFFT(samples=kspace_loc, shape=image.shape,
                             n_coils=cartesian_ref_image.shape[0], implementation='gpuNUFFT')
kspace_obs = fourier_op.op(cartesian_ref_image)

# Obtain SMaps
Smaps, SOS = get_Smaps(
    k_space=kspace_obs,
    img_shape=fourier_op.shape,
    samples=kspace_loc,
    thresh=(0.01, 0.01),
    min_samples=kspace_loc.min(axis=0),
    max_samples=kspace_loc.max(axis=0),
    mode='gridding',
    method='linear',
    n_cpu=-1,
)

# Setup Fourier Operator with SENSE
fourier_op_sense = NonCartesianFFT(
    samples=kspace_loc,
    shape=image.shape,
    n_coils=cartesian_ref_image.shape[0],
    smaps=Smaps,
    implementation='gpuNUFFT',
)

#############################################################################
# FISTA optimization
# ------------------
#
# We now want to refine the zero order solution using a FISTA optimization.
# The cost function is set to Proximity Cost + Gradient Cost

# Setup the operators
linear_op = WaveletN(
    wavelet_name='sym8',
    nb_scale=4,
)
regularizer_op = SparseThreshold(Identity(), 0, thresh_type="soft")
# Setup Reconstructor
reconstructor = SelfCalibrationReconstructor(
    fourier_op=fourier_op_sense,
    linear_op=linear_op,
    regularizer_op=regularizer_op,
    gradient_formulation='synthesis',
    verbose=1,
)

x_final, costs, metrics = reconstructor.reconstruct(
    kspace_data=kspace_obs,
    optimization_alg='fista',
    num_iterations=100,
)
image_rec = pysap.Image(data=x_final)
recon_ssim = ssim(image_rec, image)
print('The Reconstruction SSIM is : ' + str(recon_ssim))
