# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
This implements the single channel reconstruction.
"""
# System import
import warnings

from .base import ReconstructorWaveletBase
from ..optimizers.utils.cost import GenericCost
from ..operators import GradSynthesis, GradAnalysis
from ..optimizers import pogm, condatvu, fista

from modopt.opt.proximity import SparseThreshold
from modopt.opt.linear import Identity


class SingleChannelReconstructor(ReconstructorWaveletBase):
    """ This class implements the Single channel MR image Reconstruction.
    For the Analysis case finds the solution  for x of:
        (1/2) * sum(||F x - y||^2_2, 1) + mu * ||Wt x||_1

    For the Synthesise case finds the solution of:
        (1/2) * sum(||F Wt alpha - yl||^2_2, 1) + mu * ||alpha||_1
    Parameters
    ----------
    kspace_loc: np.ndarray
        the k-space samples locations of shape [M, d] where d is the dimension
    uniform_data_shape: tuple of int
        shape of the image (not necessarly a square matrix).
    wavelet_name: str | int
        if implementation is with waveletN the wavelet name to be used during
        the decomposition, else implementation with waveletUD2 where the
        wavelet name is wavelet_id Refer to help of mr_transform under option
        '-t' to choose the right wavelet_id. This define Wt in above equation.
    padding_mode: str (optional, default zero)
        The padding mode used in the Wavelet transform,
        'zero' | 'periodization'
    nb_scale: int (optional default is 4)
        the number of scale in the used by the multi-scale wavelet
        decomposition
    fourier_type: str (optional, default 'cartesian')
        type of fourier operator : 'cartesian' | 'non-cartesian' | 'stack'
    nfft_implementation: str, default 'cpu'
        way to implement NFFT : 'cpu' | 'cuda' | 'opencl'
    verbose: int, default 0
        Verbosity level.
            1 => Print basic debug information
            5 => Print all initialization information
            20 => Calculate cost at the end of each iteration.
                NOTE : This is computationally intensive.
            30 => Print the debug information of wavelet transform from pysap.
    """

    def __init__(self, kspace_loc, uniform_data_shape,
                 wavelet_name, padding_mode="zero", nb_scale=4,
                 fourier_type='non-cartesian', nfft_implementation='cpu',
                 verbose=0):
        self.verbose = verbose
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


    def reconstruct(self, kspace_data, mu=0, gradient_method="synthesis",
                    recalculate_lipchitz_cst=True, lips_calc_max_iter=10,
                    lipschitz_cst=None, num_check_lips=5,
                    optimization_alg='pogm', x_init=None, num_iterations=100,
                    **kwargs):
        """ This method calculates operator transform.
        For reference, this is the cosdt function being minimized:
            For the Analysis case finds the solution  for x of:
                (1/2) * sum(||F x - y||^2_2, 1) + mu * ||Wt x||_1
            For the Synthesise case finds the solution of:
                (1/2) * sum(||F Wt alpha - y||^2_2, 1) + mu * ||alpha||_1
        Parameters
        ----------
        kspace_data: np.ndarray
            the acquired value in the Fourier domain.
            This is y in above equation.
        mu: float, optional default 0 (no regularization)
            The regularization parameter value
        gradient_method: str (optional, default 'synthesis')
            the space where the gradient operator is defined: 'analysis' or
            'synthesis'
        recalculate_lipchitz_cst: bool, (optional, default True)
            if this is set to False, the old lipchitz constant is picked up if
            it exists. If it doesnt exist, the lipchitz constant is
            recalculated with a warning message that old lipschitz_cst was
            not found.
        lips_calc_max_iter: int, default 10
            Defines the maximum number of iterations to calculate the lipchitz
            constant
        num_check_lips: int, default 10
            Number of iterations to check if the lipchitz constant is correct
        lipschitz_cst: int, default None
            The user specified lipschitz constant. If this is not specified,
            it is calculated using PowerMethod
        optimization_alg: str, default 'pogm'
            Type of optimization algorithm to use, 'pogm' | 'fista' |
            'condatvu'
        x_init: np.ndarray (optional, default None)
            input initial guess image for reconstruction
        num_iterations: int (optional, default 100)
            number of iterations of algorithm
        """
        # Initialize gradient operator and proximity operators
        if recalculate_lipchitz_cst == False and lipschitz_cst is None:
            if hasattr(self, 'spec_rad'):
                lipschitz_cst = self.spec_rad
            else:
                warnings.warn('`recalculate_lipchitz_cst` was set to False, '
                              'while lipchitz constant was not initialized, '
                              'recalculating lipchitz constant')
        elif recalculate_lipchitz_cst == False and lipschitz_cst is not None:
            if hasattr(self, 'spec_rad'):
                warnings.warn('`recalculate_lipchitz_cst` was set to False, '
                              'and a lipschitz_cst was also specified at input. '
                              'Picking the old value of lipchitz constant!')
                lipschitz_cst = self.spec_rad
        if gradient_method == "synthesis":
            self.gradient_op = GradSynthesis(
                data=kspace_data,
                linear_op=self.linear_op,
                fourier_op=self.fourier_op,
                max_iter_spec_rad=lips_calc_max_iter,
                lipschitz_cst=lipschitz_cst,
                num_check_lips=num_check_lips,
                verbose=self.verbose)
            self.prox_op = SparseThreshold(Identity(), mu, thresh_type="soft")
        elif gradient_method == "analysis":
            self.gradient_op = GradAnalysis(
                data=kspace_data,
                fourier_op=self.fourier_op,
                max_iter_spec_rad=lips_calc_max_iter,
                lipschitz_cst=lipschitz_cst,
                num_check_lips=num_check_lips,
                verbose=self.verbose)
            self.prox_op = SparseThreshold(self.linear_op, mu,
                                           thresh_type="soft")
        else:
            raise ValueError("gradient_method must be either "
                             "'synthesis' or 'analysis'")
        self.cost_op = GenericCost(gradient_op=self.gradient_op,
                                   prox_op=self.prox_op,
                                   verbose=self.verbose >= 20)


        if optimization_alg == "fista":
            self.x_final, self.costs, self.metrics = fista(
                gradient_op=self.gradient_op,
                linear_op=self.linear_op,
                prox_op=self.prox_op,
                cost_op=self.cost_op,
                max_nb_of_iter=num_iterations,
                x_init=x_init,
                verbose=self.verbose,
                **kwargs)
        elif optimization_alg == "condatvu":
            self.x_final, self.costs, self.metrics, self.y_final = condatvu(
                gradient_op=self.gradient_op,
                linear_op=self.linear_op,
                prox_dual_op=self.prox_op,
                cost_op=self.cost_op,
                max_nb_of_iter=num_iterations,
                verbose=self.verbose,
                **kwargs)
        elif optimization_alg == "pogm":
            self.x_final, self.costs, self.metrics = pogm(
                gradient_op=self.gradient_op,
                linear_op=self.linear_op,
                prox_op=self.prox_op,
                cost_op=self.cost_op,
                max_nb_of_iter=num_iterations,
                x_init=x_init,
                verbose=self.verbose,
                **kwargs)
        else:
            raise ValueError("The optimization_alg must be either 'fista' or "
                             "'condatvu or 'pogm'")
        return self.x_final, self.costs, self.metrics
