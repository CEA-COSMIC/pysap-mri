"""
Neuroimaging cartesian reconstruction
=====================================

Credit: Chaithya G R

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
from pysap.extensions.transform import PyWaveletTransformBase
from modopt.opt.cost import costObj
from mri.reconstruct.fourier import NFFT
from mri.parallel_mri_online.proximity import OWL
from mri.parallel_mri_online.gradient import Grad2D_pMRI
from mri.reconstruct.utils import convert_locations_to_mask
from mri.reconstruct.utils import convert_mask_to_locations
from mri.numerics.reconstruct import sparse_rec_fista
from mri.numerics.reconstruct import sparse_rec_condatvu
from skimage.measure import compare_ssim

# Third party import
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import twixreader

def get_raw_data(filename):
    #Function that reads a SIEMENS .dat file and returns a k space data
    file = twixreader.read_twix(filename)
    measure = file.read_measurement(1)
    buffer = measure.get_meas_buffer(0)
    data = np.asarray(buffer[:])
    data = np.swapaxes(data,1,2)
    data = np.reshape(data, (data.shape[0]*data.shape[1], data.shape[2]))
    return data.T

def get_samples(filename):
    sample_locations = loadmat(filename)['samples']
    norm_factor = 2*np.max(np.abs(sample_locations), axis=0)
    sample_locations = sample_locations / norm_factor * np.pi
    return sample_locations

# Loading input data
N = 256
Nz = 100
image_name = '/neurospin/optimed/Chaithya/20190802_benchmark_3D_v5/raw/' \
            'meas_MID00457_FID13121_nsCSGRE3D_N384_FOV192_Nz64_nc32_ns2049_OS2.dat'
mask_name = '/neurospin/optimed/Chaithya/20190802_benchmark_3D_v5/samples_9.mat'
kspace_data = get_raw_data(image_name)
kspace_loc = get_samples(mask_name)

decimated = True
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

if decimated:
    linear_op = PyWaveletTransformBase(nb_scale=4, dim=2, multichannel=True, )
else:
    linear_op = linear_operators.WaveletUD2(wavelet_id=24, nb_scale=4, multichannel=True)


fourier_op = NFFT(samples=kspace_loc, shape=(64, 64, 30))

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
plt.imsave("OSCAR_Undecimated_filter_SRF.png", image_rec)
