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
from mri.optimizers import pogm, condatvu, fista

from modopt.opt.proximity import SparseThreshold
from modopt.opt.linear import Identity


class SingleChannelReconstructor(ReconstructorWaveletBase):
    """ This class implements the common parameters across different
    reconstruction methods.
    Parameters
    ----------
    kspace_data: ndarray
        the data to reconstruct: observation are expected in Fourier space.
    kspace_loc: np.ndarray
        the mask samples in the Fourier domain.
    fourier_type: str (optional, default 'cartesian')
        type of fourier operator : 'cartesian' | 'non-cartesian' | 'stack'
    uniform_data_shape: uplet (optional, default None)
        the shape of the matrix containing the uniform data.
    wavelet_name: str | int
        if implementation is with waveletN the wavelet name to be used during
        the decomposition, else implementation with waveletUD2 where the
        wavelet name is wavelet_id Refer to help of mr_transform under option
        '-t' to choose the right wavelet_id.
    padding_mode: str, default "zero"
        ways to extend the signal when computing the decomposition.
    nb_scales: int, default 4
        the number of scales in the wavelet decomposition.
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
    num_check_lips: int, default 10
        Number of iterations to check if the lipchitz constant is correct
    optimization_alg: str, default 'pogm'
        Type of optimization algorithm to use, 'pogm' | 'fista' | 'condatvu'
    lipschitz_cst: int, default None
        The user specified lipschitz constant. If this is not specified,
        it is calculated using PowerMethod
    verbose: int, default 0
        Verbosity level.
            1 => Print basic debug information
            5 => Print all initialization information
            20 => Calculate cost at the end of each iteration.
                NOTE : This is computationally intensive.
    """

    def __init__(self, kspace_data, kspace_loc, uniform_data_shape,
                 wavelet_name, mu, padding_mode="zero", nb_scale=4,
                 fourier_type='non-cartesian', gradient_method="synthesis",
                 nfft_implementation='cpu', lips_calc_max_iter=10,
                 num_check_lips=10, optimization_alg='pogm',
                 lipschitz_cst=None, verbose=0):
        self.optimization_alg = optimization_alg
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
        # Initialize gradient operator and proximity operators
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

    def reconstruct(self, x_init=None, num_iterations=100, **kwargs):
        """ This method calculates operator transform.
        Parameters
        ----------
        x_init: np.ndarray (optional, default None)
            input initial guess image for reconstruction
        num_iterations: int (optional, default 100)
            number of iterations of algorithm
        """
        if self.optimization_alg == "fista":
            self.x_final, self.costs, self.metrics = fista(
                gradient_op=self.gradient_op,
                linear_op=self.linear_op,
                prox_op=self.prox_op,
                cost_op=self.cost_op,
                max_nb_of_iter=num_iterations,
                x_init=x_init,
                verbose=0,
                **kwargs)
        elif self.optimization_alg == "condatvu":
            self.x_final, self.costs, self.metrics, self.y_final = condatvu(
                gradient_op=self.gradient_op,
                linear_op=self.linear_op,
                prox_dual_op=self.prox_op,
                cost_op=self.cost_op,
                verbose=self.verbose,
                **kwargs)
        elif self.optimization_alg == "pogm":
            self.x_final, self.costs, self.metrics = pogm(
                gradient_op=self.gradient_op,
                linear_op=self.linear_op,
                prox_op=self.prox_op,
                cost_op=self.cost_op,
                max_nb_of_iter=num_iterations,
                x_init=x_init,
                verbose=0,
                **kwargs)
        else:
            raise ValueError("The optimization_alg must be either 'fista' or "
                             "'condatvu or 'pogm'")
        return self.x_final, self.costs, self.metrics
