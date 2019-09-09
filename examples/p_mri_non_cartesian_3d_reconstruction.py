"""
Neuroimaging cartesian reconstruction
=====================================

Credit: L. El Gueddari

In this tutorial we will reconstruct an MRI image from the sparse kspace
measurments.

"""

# Package import
from pysap.data import get_sample_data
from mri.numerics.proximity import Threshold
from mri.numerics.gradient import Gradient_pMRI
from mri.numerics.reconstruct import sparse_rec_fista
from mri.numerics.reconstruct import sparse_rec_condatvu
from mri.reconstruct.fourier import NUFFT, NFFT
from mri.reconstruct.utils import imshow3D
from mri.parallel_mri.cost import GenericCost
from mri.reconstruct.linear import WaveletN
from mri.reconstruct.utils import normalize_frequency_locations
from mri.parallel_mri.extract_sensitivity_maps import (
    extract_k_space_center_and_locations,
    get_Smaps)

# Third party import
import numpy as np
import matplotlib.pyplot as plt

# Loading input data
Il = get_sample_data("3d-pmri")
Iref = np.squeeze(np.sqrt(np.sum(np.abs(Il)**2, axis=0)))
Smaps = np.asarray([Il[channel]/Iref for channel in range(Il.shape[0])])

imshow3D(Iref, display=True)

samples = get_sample_data("mri-radial-3d-samples").data

samples = normalize_frequency_locations(samples)
#############################################################################
# Generate the kspace
# -------------------
#
# From the 3D brain data and the acquistion mask, we generate the acquisition
# measurments, the observed kspace.
# We then reconstruct the zero order solution.

# Generate the subsampled kspace

gen_fourier_op = NFFT(samples=samples,
                      shape=(128, 128, 160))

print('Generate the k-space')
kspace_data = np.asarray([gen_fourier_op.op(Il[channel]) for channel
                          in range(Il.shape[0])])

print("K-space locations  shape", samples.shape)

min_samples = [np.min(samples[:, idx]) for idx in range(samples.shape[1])]
max_samples = [np.max(samples[:, idx]) for idx in range(samples.shape[1])]

print('After normalization, k-space samples min', min_samples)
print('After normalization, k-space samples max', max_samples)

kspace_center, samples_center = extract_k_space_center_and_locations(
    data_values=kspace_data,
    samples_locations=samples,
    thr=(0.05, 0.05, 0.05),
    img_shape=Iref.shape)

print("Center of k-space extracted above the threshold ", kspace_center.shape)
print("Center of k-space locations extracted above the threshold ",
      samples_center.shape)

Smaps, I_SOS = get_Smaps(
    k_space=kspace_center,
    img_shape=Iref.shape,
    samples=samples_center,
    mode='gridding',
    min_samples=min_samples,
    max_samples=max_samples)

print("Smaps' shape ", Smaps.shape)

imshow3D(np.abs(I_SOS), display=True)

#############################################################################
# FISTA optimization
# ------------------
#
# We now want to refine the zero order solution using a FISTA optimization.
# Here no cost function is set, and the optimization will reach the
# maximum number of iterations. Fill free to play with this parameter.

# Start the FISTA reconstruction
max_iter = 10
linear_op = WaveletN(wavelet_name="sym4",
                     nb_scale=4, dim=3)

fourier_op = NFFT(samples=samples, shape=Iref.shape)

print('Generate the zero order solution')

rec_0 = np.asarray([fourier_op.adj_op(kspace_data[l]) for l in range(32)])
imshow3D(np.squeeze(np.sqrt(np.sum(np.abs(rec_0)**2, axis=0))),
         display=True)

gradient_op = Gradient_pMRI(data=kspace_data,
                            fourier_op=fourier_op,
                            linear_op=linear_op,
                            S=Smaps)

prox_op = Threshold(0)

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


x_final, transform, cost, metrics = sparse_rec_fista(
    gradient_op=gradient_op,
    linear_op=linear_op,
    prox_op=prox_op,
    cost_op=cost_synthesis,
    lambda_init=1.0,
    max_nb_of_iter=max_iter,
    atol=1e-4,
    verbose=1)
imshow3D(np.abs(x_final), display=True)


plt.figure()
plt.plot(cost)
plt.show()
#############################################################################
# Condata-Vu optimization
# -----------------------
#
# We now want to refine the zero order solution using a Condata-Vu
# optimization.
# Here no cost function is set, and the optimization will reach the
# maximum number of iterations. Fill free to play with this parameter.

# Start the CONDAT-VU reconstruction
max_iter = 1
gradient_op_cd = Gradient_pMRI(data=kspace_data,
                               fourier_op=fourier_op,
                               S=Smaps)

cost_analysis = GenericCost(
    gradient_op=gradient_op_cd,
    prox_op=prox_op,
    linear_op=linear_op,
    initial_cost=1e6,
    tolerance=1e-4,
    cost_interval=1,
    test_range=4,
    verbose=True,
    plot_output=None)

x_final, transform = sparse_rec_condatvu(
    gradient_op=gradient_op_cd,
    linear_op=linear_op,
    prox_dual_op=prox_op,
    cost_op=cost_analysis,
    std_est=None,
    std_est_method="dual",
    std_thr=2.,
    tau=None,
    sigma=None,
    relaxation_factor=1.0,
    nb_of_reweights=0,
    max_nb_of_iter=max_iter,
    add_positivity=False,
    atol=1e-4,
    verbose=1)

imshow3D(np.abs(x_final), display=True)
