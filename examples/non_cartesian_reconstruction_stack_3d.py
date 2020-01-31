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
    gridded_inverse_fourier_transform_stack, get_stacks_fourier, \
    convert_mask_to_locations
from mri.reconstructors import SingleChannelReconstructor
import pysap
from pysap.data import get_sample_data

# Third party import
from modopt.math.metrics import ssim
from modopt.opt.linear import Identity
from modopt.opt.proximity import SparseThreshold
import numpy as np

# Loading input data
image = get_sample_data('3d-pmri')
image = pysap.Image(data=np.sqrt(np.sum(np.abs(image.data)**2, axis=0)))

# Reducing the size of the volume for faster computation
image.data = image.data[:, :, 48: -48]

# Obtain MRI non-cartesian sampling plane
mask_radial = get_sample_data("mri-radial-samples")

# Tiling the plane on the z-direction
# sampling_z = np.ones(image.shape[2])  # no sampling
sampling_z = np.random.randint(2, size=image.shape[2])  # random sampling
sampling_z[22: 42] = 1
Nz = sampling_z.sum()  # Number of acquired plane

z_locations = np.repeat(convert_mask_to_locations(sampling_z),
                        mask_radial.shape[0])
z_locations = z_locations[:, np.newaxis]
kspace_loc = np.hstack([np.tile(mask_radial.data, (Nz, 1)),
                        z_locations])
mask = pysap.Image(data=np.moveaxis(
    convert_locations_to_mask(kspace_loc, image.shape), -1, 0))

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
grid_space = [np.linspace(-0.5, 0.5, num=img_shape)
              for img_shape in image.shape[:-1]]
grid = np.meshgrid(*tuple(grid_space))
kspace_plane_loc, z_sample_loc, sort_pos, idx_mask_z = get_stacks_fourier(
    kspace_loc,
    image.shape)
grid_soln = gridded_inverse_fourier_transform_stack(
    kspace_data_sorted=kspace_obs[sort_pos],
    kspace_plane_loc=kspace_plane_loc,
    idx_mask_z=idx_mask_z,
    grid=tuple(grid),
    volume_shape=image.shape,
    method='linear')

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
linear_op = WaveletN(wavelet_name="sym8", nb_scales=4, dim=3)
regularizer_op = SparseThreshold(Identity(), 6 * 1e-9, thresh_type="soft")
# Setup Reconstructor
reconstructor = SingleChannelReconstructor(
    fourier_op=fourier_op,
    linear_op=linear_op,
    regularizer_op=regularizer_op,
    gradient_formulation='synthesis',
    verbose=1,
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
