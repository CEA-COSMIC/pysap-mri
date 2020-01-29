#!/usr/bin/env python
# coding: utf-8
# In[1]:
# Package import
from mri.operators.fourier.non_cartesian import Stacked3DNFFT
from mri.operators.fourier.utils import normalize_frequency_locations
from mri.operators import FFT, WaveletN
from mri.operators.utils import convert_mask_to_locations
from mri.reconstructors import SelfCalibrationReconstructor
from pysap.data import get_sample_data
from sparkling.utils.gradient import get_kspace_loc_from_gradfile
from sparkling.utils.shots import convert_NCxNSxD_to_NCNSxD
import pysap
# Third party import
from modopt.math.metrics import ssim
from modopt.opt.linear import Identity
from modopt.opt.proximity import SparseThreshold
import numpy as np
import numpy.linalg as nl
import matplotlib.pyplot as plt
import twixreader as tw
(kspace_locations, kspace_data) = np.load('temp.npy', allow_pickle=True)
print(kspace_locations.shape)
# In[4]:
N = 384
Nz = 32
thresh = 0.05
threshz = 0.5
mu = 5e-6
max_iter = 15
num_channels = 34
fourier_op = Stacked3DNFFT(kspace_loc=kspace_locations, shape=(N, N, Nz), n_coils=num_channels)
#fourier_op.samples = kspace_locations
linear_op = WaveletN(
    wavelet_name='sym8',
    nb_scale=4,
    dim=3,
)
regularizer_op = SparseThreshold(Identity(), 1.5e-8, thresh_type="soft")
reconstructor = SelfCalibrationReconstructor(
    fourier_op=fourier_op,
    linear_op=linear_op,
    regularizer_op=regularizer_op,
    gradient_formulation='synthesis',
    kspace_portion=(0.05, 0.05, 0.5),
    smaps_extraction_mode='Stack',
    verbose=1,
)
x_final, costs, metrics = reconstructor.reconstruct(
    kspace_data=kspace_data,
    optimization_alg='fista',
    num_iterations=10,
)