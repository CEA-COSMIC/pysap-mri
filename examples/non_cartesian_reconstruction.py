"""
Neuroimaging non-cartesian reconstruction
=========================================

Credit: L Elgueddari,

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
from mri.numerics.reconstruct import sparse_rec_fista, sparse_rec_pogm
from mri.numerics.utils import generate_operators
from mri.numerics.utils import convert_locations_to_mask, convert_mask_to_locations

# Third party import
import numpy as np


# Loading input data
image = get_sample_data("mri-slice-nifti")
# Add Noise to input MRI Image
image.data += np.random.randn(*image.shape) * 20.

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

# Get the locations of the kspace samples and the associated observations
fourier_op = NFFT(samples=kspace_loc, shape=image.shape)
kspace_obs = fourier_op.op(image.data)

# Zero order solution
image_rec0 = pysap.Image(data=fourier_op.adj_op(kspace_obs))
image_rec0.show()

kspace_gridded_loc = convert_mask_to_locations(convert_locations_to_mask(kspace_loc, (512,512)))
fourier_op_gridded = NFFT(samples=kspace_gridded_loc, shape=image.shape)
image_rec1 = pysap.Image(data=fourier_op_gridded.adj_op(fourier_op_gridded.op(image.data)))
image_rec1.show()

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
    mu=1e-5,
    nb_scales=4,
    non_cartesian=True,
    uniform_data_shape=image.shape,
    gradient_space="synthesis")

# Start the FISTA reconstruction
max_iter = 20
x_final, transform, costs, metrics = sparse_rec_fista(
    gradient_op=gradient_op,
    linear_op=linear_op,
    prox_op=prox_op,
    cost_op=cost_op,
    lambda_init=1,
    max_nb_of_iter=max_iter,
    atol=1e-4,
    verbose=1)
image_rec = pysap.Image(data=np.abs(x_final))
image_rec.show()
cost_rec = pysap.Image(data=costs)
cost_rec.show()
