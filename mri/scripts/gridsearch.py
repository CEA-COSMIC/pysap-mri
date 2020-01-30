# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################


import itertools
import copy

from joblib import Parallel, delayed


def gridsearch(reconstructor, kspace_data, linear_kwargs={}, regularizer_kwargs={},
               reconstructor_kwargs={}, optimizer_kwargs={}, n_jobs=1):

    cross_product_list = list(itertools.product(
        *tuple(linear_kwargs.values()),
        *tuple(regularizer_kwargs.values()),
        *tuple(reconstructor_kwargs.values()),
        *tuple(optimizer_kwargs.values()),
    ))
    key_names = sum(
        list([
            list(linear_kwargs.keys()),
            list(regularizer_kwargs.keys()),
            list(reconstructor_kwargs.keys()),
            list(optimizer_kwargs.keys()),
        ]), []
    )
    all_reconstructors = []
    for case in cross_product_list:
        iterator = 0
        recon = copy.deepcopy(reconstructor)
        for _ in range(len(linear_kwargs)):
            recon.linear_op.__dict__[key_names[iterator]] = case[iterator]
            iterator = iterator + 1
        recon.linear_op.reinitialize_operator()
        for _ in range(len(reconstructor_kwargs)):
            recon.prox_op.__dict__[key_names[iterator]] = case[iterator]
            iterator = iterator + 1
        for _ in range(len(reconstructor_kwargs)):
            recon.__dict__[key_names[iterator]] = case[iterator]
            iterator = iterator + 1
        all_reconstructors.append(recon)
        recon.reconstruct(kspace_data=kspace_data, optimization_alg='fista', num_iterations=2)
    All_results = Parallel(n_jobs=n_jobs)(
        delayed(all_reconstructors[i].reconstruct)
        (kspace_data=kspace_data, optimization_alg='fista', num_iterations=2)
        for i in range(len(cross_product_list))
    )
    return  All_results






from mri.operators import FFT, WaveletN
from mri.operators.utils import convert_mask_to_locations
from mri.reconstructors import SingleChannelReconstructor
import pysap
from pysap.data import get_sample_data
from modopt.opt.proximity import SparseThreshold
from modopt.opt.linear import Identity
import numpy as np
image = get_sample_data('2d-mri')
mask = get_sample_data("cartesian-mri-mask")
kspace_loc = convert_mask_to_locations(mask.data)
fourier_op = FFT(samples=kspace_loc, shape=image.shape)
kspace_data = fourier_op.op(image)
linear_op = WaveletN(wavelet_name="sym8", nb_scales=4)
regularizer_op = SparseThreshold(Identity(), 2 * 1e-7, thresh_type="soft")
# Setup Reconstructor
reconstructor = SingleChannelReconstructor(
    fourier_op=fourier_op,
    linear_op=linear_op,
    regularizer_op=regularizer_op,
    gradient_formulation='synthesis',
    verbose=1,
)
reconstructor.reconstruct(kspace_data=kspace_data, optimization_alg='fista', num_iterations=2)
linear_kwargs = {
    'wavelet_name': ['sym8', 'sym12'],
    'nb_scale': [3, 4]
}
regularizer_kwargs = {
    'weights': np.geomspace(1e-8, 1e-6, 20)
}
results = gridsearch(
    reconstructor=reconstructor,
    kspace_data=kspace_data,
    linear_kwargs=linear_kwargs,
    regularizer_kwargs=regularizer_kwargs,
    n_jobs=1,
)