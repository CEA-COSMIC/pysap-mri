"""
Neuroimaging cartesian reconstruction
=====================================

Credit: L Elgueddari

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
import mri.reconstruct.linear as linear_operators
from modopt.opt.cost import costObj
from mri.reconstruct.fourier import FFT2
from mri.reconstruct.fourier import NFFT
from mri.parallel_mri_online.proximity import OWL
from mri.parallel_mri_online.gradient import Grad2D_pMRI
from mri.reconstruct.utils import convert_mask_to_locations
from mri.numerics.reconstruct import sparse_rec_fista

# Third party import
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

# Loading input data
Sl = get_sample_data("2d-pmri")
SOS = np.sqrt(np.sum(np.abs(Sl)**2, 0))

cartesian_reconstruction = True

mask = get_sample_data("mri-mask")
mask.data = mask.data[::4, ::4]
# mask.show()
image = pysap.Image(data=np.abs(SOS), metadata=mask.metadata)
# image.show()

#############################################################################
# Generate the kspace
# -------------------
#
# From the 2D brain slice and the acquistion mask, we generate the acquisition
# measurments, the observed kspace.
# We then reconstruct the zero order solution.

# Generate the subsampled kspace
if cartesian_reconstruction:
    mask.data = np.fft.fftshift(mask.data)
    kspace_loc = convert_mask_to_locations(mask.data)
    kspace_data = []
    [kspace_data.append(mask.data * np.fft.fft2(Sl[channel]))
        for channel in range(Sl.shape[0])]
else:
    kspace_loc = convert_mask_to_locations(mask.data)
    fourier_op_1 = NFFT(samples=kspace_loc, shape=image.shape)
    kspace_data = []
    for channel in range(Sl.shape[0]):
        kspace_data.append(fourier_op_1.op(Sl[channel]))

kspace_data = np.asarray(kspace_data)
#############################################################################
# FISTA optimization
# ------------------
#
# We now want to refine the zero order solution using a FISTA optimization.
# Here no cost function is set, and the optimization will reach the
# maximum number of iterations. Fill free to play with this parameter.

# Start the FISTA reconstruction
# import ipdb; ipdb.set_trace()
max_iter = 150

linear_op = linear_operators.WaveletN(wavelet_name='db4', nb_scale=4,
                                      num_channels=kspace_data.shape[0],
                                      n_cpu=8)

if cartesian_reconstruction:
    fourier_op = FFT2(samples=kspace_loc, shape=(512, 512))
else:
    fourier_op = NFFT(samples=kspace_loc, shape=(512, 512))

gradient_op_cd = Grad2D_pMRI(data=kspace_data,
                             fourier_op=fourier_op,
                             linear_op=linear_op)

mu_value = 1e-5
beta = 1e-15
prox_op = OWL(mu_value,
              beta,
              mode='band_based',
              bands_shape=linear_op.coeffs_shape,
              n_channel=32)

x_final, y_final, cost, metrics = sparse_rec_fista(
    gradient_op=gradient_op_cd,
    linear_op=linear_op,
    prox_op=prox_op,
    cost_op=costObj([gradient_op_cd, prox_op]),
    lambda_init=0.0,
    max_nb_of_iter=max_iter,
    atol=0e-4,
    verbose=0)
image_rec = pysap.Image(data=np.sqrt(np.sum(np.abs(x_final)**1, axis=0)))
image_rec.show()
plt.plot(cost)
plt.show()
