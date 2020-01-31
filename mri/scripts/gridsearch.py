# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

from mri.operators import WaveletN

from modopt.opt.proximity import SparseThreshold
import itertools

from joblib import Parallel, delayed


def reconstruct_case(case, key_names, init_classes,
                     kspace_data, fourier_kwargs):
    fourier_op = fourier_kwargs['init_class'](**fourier_kwargs['args'])
    linear_op = init_classes[0](
        **dict(zip(key_names[0], case[0])))
    regularizer_op = init_classes[1](
        **dict(zip(key_names[1], case[1]))
    )
    reconstructor = init_classes[2](
        fourier_op=fourier_op,
        linear_op=linear_op,
        regularizer_op=regularizer_op,
        **dict(zip(key_names[2], case[2]))
    )
    raw_results = reconstructor.reconstruct(
        kspace_data=kspace_data,
        **dict(zip(key_names[3], case[3]))
    )
    return raw_results, case


def gridsearch(linear_kwargs, regularizer_kwargs, reconstructor_kwargs,
               optimizer_kwargs, n_jobs=1, **kwargs):
    cross_product_list = list(itertools.product(
        *tuple(linear_kwargs['args'].values()),
        *tuple(regularizer_kwargs['args'].values()),
        *tuple(reconstructor_kwargs['args'].values()),
        *tuple(optimizer_kwargs['args'].values()),
    ))
    key_names = list([
        list(linear_kwargs['args'].keys()),
        list(regularizer_kwargs['args'].keys()),
        list(reconstructor_kwargs['args'].keys()),
        list(optimizer_kwargs['args'].keys()),
    ])
    reshaped_cross_product = []
    for i in range(len(cross_product_list)):
        iterator = iter(cross_product_list[i])
        reshaped_cross_product.append([[next(iter(iterator)) for _ in sublist] for sublist in key_names])
    init_classes = [
        linear_kwargs['init_class'],
        regularizer_kwargs['init_class'],
        reconstructor_kwargs['init_class'],
    ]
    results, test_cases = zip(*Parallel(n_jobs=n_jobs)(
        delayed(reconstruct_case)
        (reshaped_cross_product[i], key_names, init_classes, **kwargs)
        for i in range(len(cross_product_list))
    ))
    return results, test_cases, key_names


from mri.operators import FFT, WaveletN, NonCartesianFFT
from mri.operators.utils import convert_mask_to_locations
from mri.reconstructors import SingleChannelReconstructor
from pysap.data import get_sample_data
from modopt.opt.proximity import SparseThreshold
from modopt.opt.linear import Identity
import numpy as np

image = get_sample_data('2d-mri')
mask = get_sample_data("cartesian-mri-mask")
kspace_loc = convert_mask_to_locations(mask.data)
fourier_op = NonCartesianFFT(samples=kspace_loc, shape=image.shape)
kspace_data = fourier_op.op(image.data)
fourier_kwargs = {
    'init_class': NonCartesianFFT,
    'args':
        {
            'samples': kspace_loc,
            'shape': image.shape,
        }
}
linear_kwargs = {
    'init_class': WaveletN,
    'args':
        {
            'wavelet_name': ['sym8', 'sym12'],
            'nb_scale': [3, 4]
        }
}
regularizer_kwargs = {
    'init_class': SparseThreshold,
    'args':
        {
            'linear': [Identity()],
            'weights': np.geomspace(1e-8, 1e-6, 2),
        }
}
optimizer_kwargs = {
    # Just following convention
    'args':
        {
            'optimization_alg': ['fista', 'pogm'],
            'num_iterations': [10, 20],
        }
}
reconstructor_kwargs = {
    'init_class': SingleChannelReconstructor,
    'args':
        {
            'gradient_formulation': ['synthesis'],
        }
}
results = gridsearch(
    kspace_data=kspace_data,
    fourier_kwargs=fourier_kwargs,
    linear_kwargs=linear_kwargs,
    regularizer_kwargs=regularizer_kwargs,
    optimizer_kwargs=optimizer_kwargs,
    reconstructor_kwargs=reconstructor_kwargs,
    n_jobs=1,
)
