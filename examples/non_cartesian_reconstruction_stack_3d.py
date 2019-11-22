"""
Neuroimaging non-cartesian reconstruction
=========================================

Author: Chaithya G R

In this tutorial we will reconstruct an MRI image from non-cartesian kspace
measurements, using Stacked3D NonCartesianFFT.

Import neuroimaging data
------------------------

We use the toy datasets available in pysap, more specifically a 3D Orange.
"""

# Package import
from mri.operators import Stacked3DNFFT, WaveletN
from mri.operators.utils import convert_locations_to_mask, \
    gridded_inverse_fourier_transform_stack, get_stacks_fourier
from mri.reconstructors import SingleChannelReconstructor
import pysap
from pysap.data import get_sample_data

# Third party import
from modopt.math.metrics import ssim
import numpy as np

# Loading input data
image = get_sample_data('3d-pmri')
image = pysap.Image(data=np.sqrt(np.sum(np.abs(image.data)**2, axis=0)))

# Obtain MRI non-cartesian mask
radial_mask = get_sample_data("mri-radial-samples")
z_locations = np.repeat(np.linspace(-0.5, 0.5, image.shape[2], endpoint=False),
                        radial_mask.shape[0])
z_locations = z_locations[:, np.newaxis]
kspace_loc = np.hstack([np.tile(radial_mask.data, (image.shape[2], 1)),
                        z_locations])
mask = pysap.Image(data=convert_locations_to_mask(kspace_loc, image.shape))

# View Input
# image.show()
# mask.show()

#############################################################################
# Generate the kspace
# -------------------
#
# From the 2D brain slice and the acquisition mask, we retrospectively
# undersample the k-space using a radial acquisition mask
# We then reconstruct the zero order solution as a baseline

# Get the locations of the kspace samples and the associated observations
fourier_op = Stacked3DNFFT(kspace_loc=kspace_loc,
                           shape=image.shape,
                           implementation='cpu',
                           n_coils=1)
kspace_obs = fourier_op.op(image.data)

# Gridded solution
grid_space = [np.linspace(-0.5, 0.5, num=image.shape[i])
              for i in range(len(image.shape) - 1)]
grid = np.meshgrid(*tuple(grid_space))
kspace_plane_loc, z_sample_loc, sort_pos = get_stacks_fourier(kspace_loc)
grid_soln = gridded_inverse_fourier_transform_stack(kspace_plane_loc,
                                                    z_sample_loc,
                                                    kspace_obs,
                                                    tuple(grid),
                                                    'linear')
image_rec0 = pysap.Image(data=grid_soln)
# image_rec0.show()
base_ssim = ssim(image_rec0, image)
print('The Base SSIM is : ' + str(base_ssim))

#############################################################################
# FISTA optimization
# ------------------
#
# We now want to refine the zero order solution using a FISTA optimization.
# The cost function is set to Proximity Cost + Gradient Cost

# TODO get the right mu operator
# Setup the operators
linear_op = WaveletN(wavelet_name="sym8", nb_scales=4)
# Setup Reconstructor
reconstructor = SingleChannelReconstructor(
    fourier_op=fourier_op,
    linear_op=linear_op,
    mu=6 * 1e-9,
    gradient_method='synthesis',
    verbose=1
)
# Start Reconstruction
x_final, costs, metrics = reconstructor.reconstruct(
    kspace_data=kspace_obs,
    optimization_alg='fista',
    num_iterations=10,
)
image_rec = pysap.Image(data=np.abs(x_final))
# image_rec.show()
recon_ssim = ssim(image_rec, image)
print('The Reconstruction SSIM is : ' + str(recon_ssim))
