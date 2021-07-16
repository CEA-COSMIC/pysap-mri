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
from itertools import product

# Package import
from mri.operators import WaveletN
from mri.operators.proximity.weighted import WeightedSparseThreshold


class TestProximity(unittest.TestCase):
    def test_weighted_sparse_threshold(self):
        num_scales = 3
        linear_op = WaveletN('sym8', nb_scales=num_scales)
        coeff = linear_op.op(np.zeros((128, 128)))
        coeffs_shape = linear_op.coeffs_shape
        constant_weights = WeightedSparseThreshold(
            weights=1e-10,
            coeffs_shape=coeffs_shape,
        )
        assert np.all(constant_weights.weights[:np.prod(coeffs_shape[0])] == 0)
        assert np.all(constant_weights.weights[np.prod(coeffs_shape[0]):] == 1e-10)
        custom_scale_weights = np.arange(num_scales)
        custom_scale_weights = WeightedSparseThreshold(
            weights=custom_scale_weights,
            coeffs_shape=coeffs_shape,
            weight_type='custom_scale',
            zero_weights_coarse=False,
        )