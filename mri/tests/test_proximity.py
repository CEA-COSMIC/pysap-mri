# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import
import unittest
import numpy as np

# Package import
from mri.operators import WaveletN
from mri.operators.proximity.weighted import WeightedSparseThreshold


class TestProximity(unittest.TestCase):
    # Test proximity operators
    def test_weighted_sparse_threshold(self):
        # Test the weighted sparse threshold operator
        num_scales = 3
        linear_op = WaveletN('sym8', nb_scales=num_scales)
        coeff = linear_op.op(np.zeros((128, 128)))
        coeffs_shape = linear_op.coeffs_shape
        scales_shape = np.unique(coeffs_shape, axis=0)
        constant_weights = WeightedSparseThreshold(
            weights=1e-10,
            coeffs_shape=coeffs_shape,
        )
        assert np.all(constant_weights.weights[:np.prod(coeffs_shape[0])] == 0)
        assert np.all(
            constant_weights.weights[np.prod(coeffs_shape[0]):] == 1e-10
        )

        # Scale weights
        custom_scale_weights = np.arange(num_scales + 1)
        scale_based = WeightedSparseThreshold(
            weights=custom_scale_weights,
            coeffs_shape=coeffs_shape,
            weight_type='scale_based',
            zero_weight_coarse=False,
        )
        start = 0
        for i, scale_shape in enumerate(scales_shape):
            scale_sz = np.prod(scale_shape)
            stop = start + scale_sz * np.sum(scale_shape == coeffs_shape)
            np.testing.assert_equal(
                scale_based.weights[start:stop],
                custom_scale_weights[i],
            )
            start = stop

        # Custom Weights
        custom_weights = np.random.random(coeff.shape)
        custom = WeightedSparseThreshold(
            weights=custom_weights,
            coeffs_shape=coeffs_shape,
            weight_type='custom',
        )
        assert np.all(custom.weights[:np.prod(coeffs_shape[0])] == 0)
        np.testing.assert_equal(
            custom.weights[np.prod(coeffs_shape[0]):],
            custom_weights[np.prod(coeffs_shape[0]):],
        )
