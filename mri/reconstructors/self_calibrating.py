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
# System import
import warnings

# Module import
from .base import ReconstructorBase
from ..operators import GradSelfCalibrationSynthesis, \
    GradSelfCalibrationAnalysis, WaveletN
from .utils.extract_sensitivity_maps import get_Smaps


class SelfCalibrationReconstructor(ReconstructorBase):
    """ Self Calibrating reconstruction for multi-channel acquisition.
    The coil sensitivity is estimated from a small portion of the  k-space
    center and used to reconstruct the complex image.

    For the Analysis case finds the solution for x of:
        (1/2) * sum(||F Sl x - yl||^2_2, n_coils) + mu * H( Wt x )

    For the Synthesise case finds the solution of:
        (1/2) * sum(||F Sl Wt alpha - yl||^2_2, n_coils) + mu * H (alpha)

    The sensitivity information is taken to be the low-resolution of the image
    extractes from the k-space portion given in the parameter

    Parameters
    ----------
    fourier_op: object of class FFT, NonCartesianFFT or Stacked3DNFFT in
                mri.operators
        Defines the fourier operator F in the above equation.
    linear_op: object, (optional, default None)
        Defines the linear sparsifying operator Wt. This must operate on x and
        have 2 functions, op(x) and adj_op(coeff) which implements the
        operator and adjoint opertaor. For wavelets, this can be object of
        class WaveletN or WaveletUD2 from mri.operators .
        If None, sym8 wavelet with nb_scales=3 is chosen.
    regularizer_op: operator, (optional default None)
        Defines the regularization operator for the regularization function H.
        If None, the  regularization chosen is Identity and the optimization
        turns to gradient descent.
    gradient_formulation: str between 'analysis' or 'synthesis',
        default 'synthesis'
        defines the formulation of the image model which defines the gradient.
    lips_calc_max_iter: int, default 10
        Defines the maximum number of iterations to calculate the lipchitz
        constant
    num_check_lips: int, default 10
        Number of iterations to check if the lipchitz constant is correct
    lipschitz_cst: int, default None
        The user specified lipschitz constant. If this is not specified,
        it is calculated using PowerMethod
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
    n_jobs: int, default 1
        The number of CPUs used to accelerate the reconstruction
    verbose: int, default 0
        Verbosity level.
            1 => Print basic debug information
            5 => Print all initialization information
            20 => Calculate cost at the end of each iteration.
                NOTE : This is computationally intensive.
            30 => Print the debug information of operators if defined by class
    Note:
    -----
    The user is expected to specify the either prox_op or mu to obtain
    reconstructions, else the above equations lose the regularization terms
    resulting in inverse transform as solution.
    The reconstruction in this case proceeds with a warning.
    """

    def __init__(self, fourier_op, linear_op=None, regularizer_op=None,
                 gradient_formulation="synthesis", lips_calc_max_iter=10,
                 num_check_lips=10, lipschitz_cst=None, kspace_portion=0.1,
                 smaps_extraction_mode='gridding',
                 smaps_gridding_method='linear', n_jobs=1, verbose=0):
        if linear_op is None:
            linear_op = WaveletN(
                # TODO change nb_scales to max_nb_scale - 1
                wavelet_name="sym8",
                nb_scale=3,
                dim=len(fourier_op.shape),
                n_coils=1,
                verbose=bool(verbose >= 30),
            )
        # Ensure that we are in right multichannel config
        if linear_op.n_coils != 1:
            raise ValueError("The value of n_coils for linear operation must "
                             "be 1 for Self-Calibrating reconstruction!")
        if gradient_formulation == 'analysis':
            grad_class = GradSelfCalibrationAnalysis
        elif gradient_formulation == 'synthesis':
            grad_class = GradSelfCalibrationSynthesis
        super(SelfCalibrationReconstructor, self).__init__(
            fourier_op=fourier_op,
            linear_op=linear_op,
            regularizer_op=regularizer_op,
            gradient_formulation=gradient_formulation,
            grad_class=grad_class,
            lipschitz_cst=lipschitz_cst,
            lips_calc_max_iter=lips_calc_max_iter,
            num_check_lips=num_check_lips,
            init_gradient_op=False,
            verbose=verbose,
        )
        self.smaps_gridding_method = smaps_gridding_method
        self.smaps_extraction_mode = smaps_extraction_mode
        if type(kspace_portion) == float:
            self.kspace_portion = (kspace_portion,) * \
                                  self.fourier_op.samples.shape[-1]
        else:
            self.kspace_portion = kspace_portion
        if len(self.kspace_portion) != self.fourier_op.samples.shape[-1]:
            raise ValueError("The k-space portion size used to estimate the" +
                             "sensitivity information is not aligned with" +
                             " the input dimension")
        self.n_jobs = n_jobs

    def reconstruct(self, kspace_data, optimization_alg='pogm', x_init=None,
                    num_iterations=100, reinit_grad_op=True, **kwargs):
        """ This method calculates operator transform.
        Parameters
        ----------
        kspace_data: np.ndarray
            the acquired value in the Fourier domain.
            this is y in above equation.
        optimization_alg: str (optional, default 'pogm')
            Type of optimization algorithm to use, 'pogm' | 'fista' |
            'condatvu'
        x_init: np.ndarray (optional, default None)
            input initial guess image for reconstruction
        num_iterations: int (optional, default 100)
            number of iterations of algorithm
        reinit_grad_op: bool (optional, default False)
            A boolean value to check if the gradient operator must be
            re-initialzied.
            Note that this would recompute the lipchitz constant.
            This must be set to True if you want the Smaps to be updated
            in this reconstruction. The first reconstruction would need
            this to be True.
        """
        if self.fourier_op.n_coils != kspace_data.shape[0]:
            raise ValueError("The provided number of coil (n_coils) do not "
                             "match the data itself")
        if reinit_grad_op is False and 'Smaps' not in self.extra_grad_args:
            warnings.warn("reinit_grad_op was set to False and Smaps was "
                          "not found, re-calculating Smaps and "
                          "initializing gradient anyway!")
            reinit_grad_op = True
        if reinit_grad_op:
            # Extract Sensitivity maps and initialize gradient
            Smaps, _ = get_Smaps(
                k_space=kspace_data,
                img_shape=self.fourier_op.shape,
                samples=self.fourier_op.samples,
                thresh=self.kspace_portion,
                min_samples=self.fourier_op.samples.min(axis=0),
                max_samples=self.fourier_op.samples.max(axis=0),
                mode=self.smaps_extraction_mode,
                method=self.smaps_gridding_method,
                n_cpu=self.n_jobs
            )
            self.extra_grad_args['Smaps'] = Smaps
            self.initialize_gradient_op(**self.extra_grad_args)
        # Start Reconstruction
        super(SelfCalibrationReconstructor, self).reconstruct(
            kspace_data,
            optimization_alg,
            x_init,
            num_iterations,
            **kwargs,
        )
        return self.x_final, self.costs, self.metrics
