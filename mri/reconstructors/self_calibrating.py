# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
This implements the self-calibrating reconstruction for the multi-channel case.
"""

from .base import ReconstructorWaveletBase
from mri.optimizers.utils.cost import GenericCost
from mri.operators import GradSelfCalibrationSynthesis, \
    GradSelfCalibrationAnalysis
from .utils.extract_sensitivity_maps import get_Smaps

from modopt.opt.proximity import SparseThreshold
from modopt.opt.linear import Identity


class SelfCalibrationReconstructor(ReconstructorWaveletBase):
    """ Self Calibrating reconstruction for multi-channel acquisition.
    The coil sensitivity is estimated from a small portion of the  k-space
    center and used to reconstruct the complex image.

    For the Analysis case finds the solution for x of:
        (1/2) * sum(||F Sl x - yl||^2_2, n_coils) + mu * ||Wt x||_1

    For the Synthesise case finds the solution of:
        (1/2) * sum(||F Sl Wt alpha - yl||^2_2, n_coils) + mu * ||alpha||_1

    The sensitivity information is taken to be the low-resolution of the image
    extractes from the k-space portion given in the parameter

    Attributes
    ----------
    kspace_loc: np.ndarray
        the k-space samples locations of shape [M, d] where d is the dimension
    image_shape: tuple of int
        shape of the image (not necessarily a square matrix).
    n_coils: int
        Number of coils used to acquire the signal in case of multiarray
        receiver coils acquisition. If n_coils > 1, data shape must be
        [n_coils, *data_shape]
    wavelet_name: str | int
        if implementation is with waveletN the wavelet name to be used during
        the decomposition, else implementation with waveletUD2 where the
        wavelet name is wavelet_id Refer to help of mr_transform under option
        '-t' to choose the right wavelet_id. This define Wt in above equation.
    mu: float
        The regularization parameter value
    padding_mode: str (optional, default zero)
        The padding mode used in the Wavelet transform,
        'zero' | 'periodization'
    nb_scale: int (optional default is 4)
        the number of scale in the used by the multi-scale wavelet
        decomposition
    kspace_portion: int or tuple (default is 0.1 in all dimension)
        int or tuple indicating the k-space portion used to estimate the coil
        sensitivity information.
        if int, will be evaluated to (0.1,)*nb_dim of the image
    smaps_extraction_mode: string 'FFT' | 'NFFT' | 'Stack' | 'gridding' default
        Defines the mode in which we would want to interpolate to extract the
        sensitivity information,
        NOTE: FFT should be considered only if the input has
        been sampled on the grid
    smaps_gridding_method: string 'linear' (default) | 'cubic' | 'nearest'
        For gridding mode, it defines the way interpolation must be done used
        by the sensitivity extraction method.
    fourier_type: str (optional, default 'cartesian')
        type of fourier operator : 'cartesian' | 'non-cartesian' | 'stack'
    gradient_method: str (optional, default 'synthesis')
        the space where the gradient operator is defined: 'analysis' or
        'synthesis'
    nfft_implementation: str, default 'cpu'
        way to implement NFFT : 'cpu' | 'cuda' | 'opencl'
    lips_calc_max_iter: int, default 10
        Defines the maximum number of iterations to calculate the lipchitz
        constant
    optimization_alg: str, default 'pogm'
        Type of optimization algorithm to use, 'pogm' | 'fista' | 'condatvu'
    lipschitz_cst: int, default None
        The user specified lipschitz constant
    n_jobs: int, default 1
        The number of CPUs used to accelerate the reconstruction
    verbose: int, default 0
        Verbosity level.
            1 => Print basic debug information
            5 => Print all initialization information
            20 => Calculate cost at the end of each iteration.
                NOTE : This is computationally intensive.
    """

    def __init__(self, kspace_loc, image_shape, n_coils,
                 wavelet_name, mu, padding_mode="zero", nb_scale=4,
                 kspace_portion=0.1, smaps_extraction_mode='gridding',
                 smaps_gridding_method='linear', fourier_type='non-cartesian',
                 gradient_method="synthesis", nfft_implementation='cpu',
                 lips_calc_max_iter=10, num_check_lips=10,
                 optimization_alg='pogm', lipschitz_cst=None, n_jobs=1,
                 verbose=0):
        self.verbose = verbose
        self.kspace_loc = kspace_loc
        self.n_jobs = n_jobs
        self.n_coils = n_coils
        self.image_shape = image_shape
        self.optimization_alg = optimization_alg
        self.mu = mu
        self.smaps_gridding_method = smaps_gridding_method
        self.smaps_extraction_mode = smaps_extraction_mode
        self.gradient_method = gradient_method
        self.lipschitz_cst = lipschitz_cst
        self.lips_calc_max_iter = lips_calc_max_iter
        self.num_check_lips = num_check_lips
        if type(kspace_portion) == float:
            self.kspace_portion = (kspace_portion,) * kspace_loc.shape[-1]
        else:
            self.kspace_portion = kspace_portion
        if len(self.kspace_portion) != kspace_loc.shape[-1]:
            raise ValueError("The k-space portion size used to estimate the" +
                             "sensitivity information is not aligned with" +
                             " the input dimension")
        super(SelfCalibrationReconstructor, self).__init__(
            kspace_loc=kspace_loc,
            uniform_data_shape=self.image_shape,
            wavelet_name=wavelet_name,
            padding_mode=padding_mode,
            nb_scale=nb_scale,
            n_coils=n_coils,
            fourier_type=fourier_type,
            wavelet_op_per_channel=False,
            nfft_implementation=nfft_implementation,
            n_jobs=self.n_jobs,
            verbose=verbose,
        )



    def reconstruct(self, kspace_data, x_init=None, num_iterations=100,
                    **kwargs):
        """ This method calculates operator transform.
        Parameters
        ----------
        kspace_data: np.ndarray
            the acquired value in the Fourier domain.
            this is y in above equation.
        x_init: np.ndarray (optional, default None)
            input initial guess image for reconstruction
        num_iterations: int (optional, default 100)
            number of iterations of algorithm
        """
        if self.n_coils != kspace_data.shape[0]:
            raise ValueError("The provided number of coil (n_coils) do not " +
                             "match the data itself")

        Smaps, _ = get_Smaps(k_space=kspace_data,
                             img_shape=self.image_shape,
                             samples=self.kspace_loc,
                             thresh=self.kspace_portion,
                             min_samples=self.kspace_loc.min(axis=0),
                             max_samples=self.kspace_loc.max(axis=0),
                             mode=self.smaps_extraction_mode,
                             method=self.smaps_gridding_method,
                             n_cpu=self.n_jobs)
        # Initialize gradient operator and proximity operators
        if self.gradient_method == "synthesis":
            self.gradient_op = GradSelfCalibrationSynthesis(
                linear_op=self.linear_op,
                fourier_op=self.fourier_op,
                Smaps=Smaps,
                max_iter_spec_rad=self.lips_calc_max_iter,
                lipschitz_cst=self.lipschitz_cst,
                num_check_lips=self.num_check_lips,
            )
            self.prox_op = SparseThreshold(
                Identity(),
                self.mu,
                thresh_type="soft",
            )
        elif self.gradient_method == "analysis":
            self.gradient_op = GradSelfCalibrationAnalysis(
                fourier_op=self.fourier_op,
                Smaps=Smaps,
                max_iter_spec_rad=self.lips_calc_max_iter,
                lipschitz_cst=self.lipschitz_cst,
                num_check_lips=self.num_check_lips,
            )
            self.prox_op = SparseThreshold(
                self.linear_op,
                self.mu,
                thresh_type="soft",
            )
        else:
            raise ValueError("gradient_method must be either "
                             "'synthesis' or 'analysis'")
        self.cost_op = GenericCost(
            gradient_op=self.gradient_op,
            prox_op=self.prox_op,
            verbose=self.verbose >= 20,
        )
        super(SelfCalibrationReconstructor, self).reconstruct(
            kspace_data,
            x_init,
            num_iterations,
            **kwargs,
        )
        return self.x_final, self.costs, self.metrics