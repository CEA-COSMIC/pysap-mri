"""
Neuroimaging cartesian reconstruction
=====================================

Credit: Chaithya G R

In this tutorial we will reconstruct a 3D-MRI image from the sparse kspace
measurments.
"""

# Package import
import pysap
from mri.reconstruct.fourier import NFFT, NUFFT
from mri.parallel_mri.gradient import Gradient_pMRI
from mri.numerics.linear import WaveletN
from mri.parallel_mri.cost import GenericCost
from mri.reconstruct.utils import imshow3D
from mri.numerics.proximity import Threshold
from mri.parallel_mri.extract_sensitivity_maps \
    import extract_k_space_center_and_locations, get_Smaps
from mri.numerics.reconstruct import sparse_rec_fista

# Third party import
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import twixreader
import datetime


def get_raw_data(filename):
    # Function that reads a SIEMENS .dat file and returns a k space data
    file = twixreader.read_twix(filename)
    measure = file.read_measurement(1)
    buffer = measure.get_meas_buffer(0)
    data = np.asarray(buffer[:])
    data = np.swapaxes(data, 1, 2)
    data = np.reshape(data, (data.shape[0] * data.shape[1], data.shape[2]))
    return data.T


def get_samples(filename):
    sample_locations = loadmat(filename)['samples']
    norm_factor = 2 * np.max(np.abs(sample_locations), axis=0)
    sample_locations = sample_locations / norm_factor
    return sample_locations


# Loading input data
N = 64
Nz = 30
thresh = 0.01
mu = 5e-6
max_iter = 10
kspace_loc = []
kspace_data = []

try:
    (kspace_loc, kspace_data) = np.load(
        "/neurospin/optimed/Chaithya/Temp_data.npy",
        allow_pickle=True)
except:
    print("Could not find temp file, loading data!")
    image_name = \
        '/neurospin/optimed/Chaithya/20190802_benchmark_3D_v5/raw/' \
        'meas_MID00457_FID13121_nsCSGRE3D_N384_FOV192_Nz64_nc32_ns2049_OS2.dat'
    mask_name = \
        '/neurospin/optimed/Chaithya/20190802_benchmark_3D_v5/samples_9.mat'
    kspace_data = get_raw_data(image_name)
    kspace_loc = get_samples(mask_name)
    np.save("/neurospin/optimed/Chaithya/Temp_data.npy",
            (kspace_loc, kspace_data))

try:
    Smaps = np.load(
        "/neurospin/optimed/Chaithya/Temp_data_full.npy",
        allow_pickle=True)
except:
    data_thresholded, samples_thresholded = \
        extract_k_space_center_and_locations(
            data_values=kspace_data,
            samples_locations=kspace_loc,
            thr=(thresh, thresh, thresh),
            img_shape=(N, N, Nz))
    Smaps, SOS_Smaps = get_Smaps(
        k_space=data_thresholded,
        img_shape=(N, N, Nz),
        samples=samples_thresholded,
        mode='NFFT',
        min_samples=np.min(samples_thresholded, axis=0),
        max_samples=np.max(samples_thresholded, axis=0), n_cpu=1)
    np.save("/neurospin/optimed/Chaithya/Temp_data_full.npy", Smaps)

imshow3D(np.abs(Smaps[32]))
linear_op = WaveletN(wavelet_name="sym8",
                     nb_scale=4, dim=3)
try:
    fourier_op = NUFFT(samples=kspace_loc,
                       shape=(N, N, Nz), platform='gpu')
    fourier = "NUFFT"
except:
    print("GPU Version of NUFFT could not be loaded, using NFFT")
    fourier_op = NFFT(samples=kspace_loc, shape=(N, N, Nz))
    fourier = "NFFT"

gradient_op = Gradient_pMRI(data=kspace_data,
                            fourier_op=fourier_op,
                            linear_op=linear_op, S=Smaps)
prox_op = Threshold(mu)
cost_synthesis = GenericCost(
    gradient_op=gradient_op,
    prox_op=prox_op,
    linear_op=None,
    initial_cost=1e6,
    tolerance=1e-4,
    cost_interval=1,
    test_range=4,
    verbose=True,
    plot_output=None)

#############################################################################
# FISTA optimization
# ------------------
#
# We now want to refine the zero order solution using a FISTA optimization.
# Here no cost function is set, and the optimization will reach the
# maximum number of iterations. Fill free to play with this parameter.

# Start the FISTA reconstruction
x_final, y_final, cost, metrics = sparse_rec_fista(
    gradient_op=gradient_op,
    linear_op=linear_op,
    prox_op=prox_op,
    cost_op=cost_synthesis,
    lambda_init=0.0,
    max_nb_of_iter=max_iter,
    atol=1e-4,
    verbose=1)

currentDT = datetime.datetime.now()
np.save("/neurospin/optimed/Chaithya/Results/MANIAC/"
        + "Thresh_" + str(thresh) + "-FFT_" + fourier
        + "-Opt_sparseRecFista" + "-NIt_" + str(max_iter)
        + "-N_" + str(N) + "-Nz" + str(Nz) + "-D" + str(currentDT.day) + "M"
        + str(currentDT.month) + "Y" + str(currentDT.year)
        + str(currentDT.hour) + ":" + str(currentDT.minute)
        + "_runtest.npy", (x_final, cost))
