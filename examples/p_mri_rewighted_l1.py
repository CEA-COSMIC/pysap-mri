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
from mri.numerics.reconstruct import sparse_rec_condatvu
from modopt.opt.proximity import SparseThreshold

# Third party import
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

# Loading input data
image_name = '../../../Data/meas_MID41_CSGRE_ref_OS1_FID14687.mat'
k_space_ref = loadmat(image_name)['ref']
k_space_ref /= np.linalg.norm(k_space_ref)

cartesian_reconstruction = False
decimated = True

if cartesian_reconstruction:
    Sl = np.zeros((32, 512, 512), dtype='complex128')
    for channel in range(k_space_ref.shape[-1]):
        Sl[channel] = np.fft.fftshift(np.fft.ifft2(np.reshape(
            k_space_ref[:, channel], (512, 512))))
    SOS = np.sqrt(np.sum(np.abs(Sl)**2, 0))
else:
    full_samples_loc = convert_mask_to_locations(np.ones((512, 512)))
    gen_image_op = NFFT(samples=full_samples_loc, shape=(512,512))
    Sl = np.zeros((32, 512, 512), dtype='complex128')
    for channel in range(k_space_ref.shape[-1]):
        Sl[channel] = gen_image_op.adj_op(np.reshape(k_space_ref[:, channel], (512, 512)))
    SOS = np.sqrt(np.sum(np.abs(Sl)**2, 0))

mask = get_sample_data("mri-mask")
mask.show()
image = pysap.Image(data=np.abs(SOS), metadata=mask.metadata)
image.show()

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

if decimated:
    linear_op = linear_operators.Wavelet2(wavelet_name='db4', nb_scale=4, multichannel=True)
else:
    linear_op = linear_operators.WaveletUD2(wavelet_id=24, nb_scale=4, multichannel=True)


if cartesian_reconstruction:
    fourier_op = FFT2(samples=kspace_loc, shape=(512, 512))
else:
    fourier_op = NFFT(samples=kspace_loc, shape=(512, 512))

mu_value = 1e-5
beta = 1e-15
prox_op = SparseThreshold(linear_op, mu_value, thresh_type="soft")

gradient_op_cd = Grad2D_pMRI(data=kspace_data,
                             fourier_op=fourier_op,
                             linear_op=None)

x_final, y_final, costs, metrics = sparse_rec_condatvu(
    gradient_op=gradient_op_cd,
    linear_op=linear_op,
    cost_op=None,
    prox_dual_op=prox_op,
    std_est=None,
    tau=None,
    sigma=None,
    relaxation_factor=1.0,
    nb_of_reweights=3,
    std_est_method='primal',
    max_nb_of_iter=3,
    add_positivity=False,
    atol=1e-4,
    verbose=1)

image_rec_y = pysap.Image(data=np.sqrt(np.sum(np.abs(y_final)**2, axis=0)))
image_rec_y.show()

image_rec = pysap.Image(data=np.sqrt(np.sum(np.abs(x_final)**2, axis=0)))
image_rec.show()
plt.imsave("OSCAR_Decimated_filter_CVu.png", image_rec)
