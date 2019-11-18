# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
This implements calibrationless reconstruction with different proximities
"""

from .base import ReconstructorWaveletBase
from ..optimizers.utils.cost import GenericCost
from mri.operators import GradSynthesis, GradAnalysis

from modopt.opt.proximity import SparseThreshold
from modopt.opt.linear import Identity


class SparseCalibrationlessReconstructor(ReconstructorWaveletBase):
    """ This class implements a calibrationless reconstruction based on the
    L1-norm regularization.
    For the Analysis case finds the solution for x of:
        (1/2) * sum(||F x_l - yl||^2_2, n_coils) +
                    mu * sum(||Wt x_l||_1, n_coils)
    For the Synthesis case finds the solution of:
        (1/2) * sum(||F Wt alpha_l - yl||^2_2, n_coils) +
                    mu * sum(||alpha_l||_1, n_coils)
    Parameters
    ----------
    kspace_loc: np.ndarray
        the mask samples in the Fourier domain.
    image_shape: tuple (optional, default None)
        the shape of the matrix containing the image data.
    wavelet_name: str | int
        if implementation is with waveletN the wavelet name to be used during
        the decomposition, else implementation with waveletUD2 where the
        wavelet name is wavelet_id Refer to help of mr_transform under option
        '-t' to choose the right wavelet_id. This specifies Wt in above
        equation.
    mu: float
        The regularization parameter value
    padding_mode: str, default "zero"
        ways to extend the signal when computing the decomposition.
    nb_scales: int, default 4
        the number of scales in the wavelet decomposition.
    fourier_type: str (optional, default 'non-cartesian')
        type of fourier operator : 'cartesian' | 'non-cartesian' | 'stack'
    gradient_method: str (optional, default 'synthesis')
        the space where the gradient operator is defined: 'analysis' or
        'synthesis'
    nfft_implementation: str, default 'cpu'
        way to implement NFFT : 'cpu' | 'cuda' | 'opencl'
    lips_calc_max_iter: int, default 10
        Defines the maximum number of iterations to calculate the lipchitz
        constant
    num_check_lips: int, default 10
        Number of iterations to check if the lipchitz constant is correct
    optimization_alg: str, default 'pogm'
        Type of optimization algorithm to use, 'pogm' | 'fista' | 'condatvu'
    lipschitz_cst: int, default None
        The user specified lipschitz constant. If this is not specified,
        it is calculated using PowerMethod
    n_jobs: int, default 1
        Number of parallel jobs for linear operator
    verbose: int, default 0
        Verbosity level.
            1 => Print basic debug information
            5 => Print all initialization information
            20 => Calculate cost at the end of each iteration.
                NOTE : This is computationally intensive.
    """

    def __init__(self, kspace_loc, image_shape, n_coils,
                 wavelet_name, mu, padding_mode="zero", nb_scale=4,
                 fourier_type='non-cartesian', gradient_method="synthesis",
                 nfft_implementation='cpu', lips_calc_max_iter=10,
                 num_check_lips=10, optimization_alg='pogm',
                 lipschitz_cst=None, n_jobs=1, verbose=0):
        self.optimization_alg = optimization_alg
        self.gradient_method = gradient_method
        self.GradSynthesis = GradSynthesis
        self.GradAnalysis = GradAnalysis
        self.verbose = verbose
        # Initialize the Fourier and Linear Operator
        super(SparseCalibrationlessReconstructor, self).__init__(
            kspace_loc=kspace_loc,
            uniform_data_shape=image_shape,
            wavelet_name=wavelet_name,
            padding_mode=padding_mode,
            nb_scale=nb_scale,
            n_coils=n_coils,
            fourier_type=fourier_type,
            wavelet_op_per_channel=True,
            nfft_implementation=nfft_implementation,
            lips_calc_max_iter=lips_calc_max_iter,
            num_check_lips=num_check_lips,
            lipschitz_cst=lipschitz_cst,
            verbose=verbose)
        # Initialize gradient operator and proximity operators
        if self.gradient_method == "synthesis":
            self.prox_op = SparseThreshold(
                Identity(),
                mu,
                thresh_type="soft",
            )
        elif self.gradient_method == "analysis":
            self.prox_op = SparseThreshold(
                self.linear_op,
                mu,
                thresh_type="soft",
            )
        self.cost_op = GenericCost(
            gradient_op=self.gradient_op,
            prox_op=self.prox_op,
            verbose=self.verbose >= 20,
        )
