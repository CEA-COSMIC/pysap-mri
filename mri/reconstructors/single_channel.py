# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

from ._base import ReconstructorWaveletBase
from .utils import GenericCost
from mri.operators import GradSynthesis, GradAnalysis

from modopt.opt.proximity import SparseThreshold
from modopt.opt.linear import Identity


class SingleChannelReconstructor(ReconstructorWaveletBase):
    def __init__(self, kspace_data, kspace_loc, uniform_data_shape,
                 wavelet_name, mu, padding_mode="zero", nb_scale=4,
                 fourier_type='non-cartesian', gradient_method="synthesis",
                 nfft_implementation='cpu', lips_calc_max_iter=10,
                 num_check_lips=10, optimization_alg='pogm',
                 lipschitz_cst=None, verbose=0):
        # Initialize the Fourier and Linear Operator
        super(SingleChannelReconstructor, self).__init__(
            kspace_loc=kspace_loc,
            uniform_data_shape=uniform_data_shape,
            wavelet_name=wavelet_name,
            padding_mode=padding_mode,
            nb_scale=nb_scale,
            n_coils=1,
            fourier_type=fourier_type,
            wavelet_op_per_channel=False,
            nfft_implementation=nfft_implementation,
            verbose=verbose)
        # Initialize gradient operator and proximity operators
        if gradient_method == "synthesis":
            self.gradient_op = GradSynthesis(
                data=kspace_data,
                linear_op=self.linear_op,
                fourier_op=self.fourier_op,
                max_iter_spec_rad=lips_calc_max_iter,
                lipschitz_cst=lipschitz_cst,
                num_check_lips=num_check_lips)
            self.prox_op = SparseThreshold(Identity(), mu, thresh_type="soft")
        elif gradient_method == "analysis":
            self.gradient_op = GradAnalysis(
                data=kspace_data,
                fourier_op=self.fourier_op,
                max_iter_spec_rad=lips_calc_max_iter,
                lipschitz_cst=lipschitz_cst,
                num_check_lips=num_check_lips)
            self.prox_op = SparseThreshold(self.linear_op, mu, thresh_type="soft")
        else:
            raise ValueError("gradient_method must be either "
                             "'synthesis' or 'analysis'")
        self.cost_op = GenericCost(gradient_op=self.gradient_op,
                                   prox_op=self.prox_op,
                                   verbose=verbose)
