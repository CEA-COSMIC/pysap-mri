# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import
import warnings

# Package import
from ..operators.linear.wavelet import WaveletUD2, WaveletN
from ..optimizers import pogm, condatvu, fista
from ..optimizers.utils.cost import GenericCost

# Third party import
from modopt.opt.proximity import SparseThreshold
from modopt.opt.linear import Identity


class ReconstructorBase(object):
    """ This is the base reconstructor class for reconstruction.
    This class holds some parameters that is common for all MR Image
    reconstructors

    Parameters
    ----------
    """

    def __init__(self, fourier_op, linear_op, prox_op, mu, gradient_method,
                 grad_class, lips_calc_max_iter, num_check_lips, lipschitz_cst,
                 optimization_alg, verbose, init_gradient_op=True,
                 **extra_grad_args):
        self.fourier_op = fourier_op
        self.linear_op = linear_op
        self.prox_op = prox_op
        self.optimization_alg = optimization_alg
        self.gradient_method = gradient_method
        self.grad_class = grad_class
        self.lipschitz_cst = lipschitz_cst
        self.lips_calc_max_iter = lips_calc_max_iter
        self.num_check_lips = num_check_lips
        self.verbose = verbose
        self.extra_grad_args = extra_grad_args
        if prox_op is None and mu == 0:
            warnings.warn("The prox_op is not set and mu = 0. The result will "
                          "not be a reconstruction but an inverse")

        # If the reconstruction formulation is synthesis,
        # we send the linear operator as well.
        if gradient_method == 'synthesis':
            self.extra_grad_args['linear_op'] = self.linear_op
        if self.prox_op is None:
            if gradient_method == 'synthesis':
                self.prox_op = SparseThreshold(
                    Identity(),
                    mu,
                    thresh_type="soft",
                )
            elif gradient_method == "analysis":
                if self.prox_op is None:

                    self.prox_op = SparseThreshold(
                        self.linear_op,
                        mu,
                        thresh_type="soft",
                    )
        if init_gradient_op is True:
            self.initialize_gradient_op(**self.extra_grad_args)

    def initialize_gradient_op(self, **extra_args):
        # Initialize gradient operator and cost operators
        self.gradient_op = self.grad_class(
            fourier_op=self.fourier_op,
            lips_calc_max_iter=self.lips_calc_max_iter,
            lipschitz_cst=self.lipschitz_cst,
            num_check_lips=self.num_check_lips,
            verbose=self.verbose,
            **extra_args,
        )
        self.cost_op = GenericCost(
            gradient_op=self.gradient_op,
            prox_op=self.prox_op,
            verbose=self.verbose >= 20,
        )

    def reconstruct(self, kspace_data, x_init=None, num_iterations=100,
                    reinit_grad_op=False, **kwargs):
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
        self.gradient_op.obs_data = kspace_data
        if self.optimization_alg == "fista":
            self.x_final, self.costs, self.metrics = fista(
                gradient_op=self.gradient_op,
                linear_op=self.linear_op,
                prox_op=self.prox_op,
                cost_op=self.cost_op,
                max_nb_of_iter=num_iterations,
                x_init=x_init,
                verbose=self.verbose,
                **kwargs)
        elif self.optimization_alg == "condatvu":
            self.x_final, self.costs, self.metrics, self.y_final = condatvu(
                gradient_op=self.gradient_op,
                linear_op=self.linear_op,
                prox_dual_op=self.prox_op,
                cost_op=self.cost_op,
                max_nb_of_iter=num_iterations,
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
                verbose=self.verbose,
                **kwargs)
        else:
            raise ValueError("The optimization_alg must be either 'fista' or "
                             "'condatvu or 'pogm'")
        return self.x_final, self.costs, self.metrics


class ReconstructorWaveletBase(ReconstructorBase):
    """ This is the derived reconstructor class from `ReconstructorBase`.
    This class specifically defines some wavelet related parameters that is
    common for reconstructions involving the wavelet operator in the
    cost function.
    Parameters
    ----------
    kspace_loc: np.ndarray
        the mask samples in the Fourier domain.
    uniform_data_shape: tuple
        the shape of the matrix containing the uniform data.
    n_coils: int, default 1
        Number of coils used to acquire the signal in case of multiarray
        receiver coils acquisition. If n_coils > 1, data shape must be
        [n_coils, *data_shape]
    fourier_type: str (optional, default 'cartesian')
        type of fourier operator : 'cartesian' | 'non-cartesian' | 'stack'
    nfft_implementation: str, default 'cpu'
        way to implement NFFT : 'cpu' | 'cuda' | 'opencl'
    wavelet_name: str | int
        if implementation is with waveletN the wavelet name to be used during
        the decomposition, else implementation with waveletUD2 where the
        wavelet name is wavelet_id Refer to help of mr_transform under option
        '-t' to choose the right wavelet_id.
    padding_mode: str (optional, default zero)
        The padding mode used in the Wavelet transform,
        'zero' | 'periodization'
    nb_scale: int (optional default is 4)
        the number of scale in the used by the multi-scale wavelet
        decomposition
    wavelet_op_per_channel: bool
        whether wavelet transform should be applied on every channel. This is
        True only for Calibrationless reconstruction
    n_jobs: int, default 1
        Number of parallel jobs for linear operation
    verbose: int
        verbosity level for debug, please check dervied class for details
        on verbosity levels
    """

    def __init__(self, kspace_loc, uniform_data_shape, n_coils,
                 fourier_type, nfft_implementation, wavelet_name, padding_mode,
                 nb_scale, wavelet_op_per_channel, n_jobs=1, verbose=0,
                 **kwargs):
        super(ReconstructorWaveletBase, self).__init__(
            kspace_loc=kspace_loc,
            uniform_data_shape=uniform_data_shape,
            n_coils=n_coils,
            fourier_type=fourier_type,
            nfft_implementation=nfft_implementation,
            verbose=verbose)
        verbosity_wavelet_op = int(verbose >= 30)
        if wavelet_op_per_channel is False:
            # For Self Calibrating Reconstruction, we do not do linear
            # operator per channel
            n_coils = 1
        try:
            self.linear_op = WaveletN(
                nb_scale=nb_scale,
                wavelet_name=wavelet_name,
                padding_mode=padding_mode,
                dim=len(self.fourier_op.shape),
                n_coils=n_coils,
                n_jobs=n_jobs,
                verbose=verbosity_wavelet_op,
            )
        except ValueError:
            # TODO this is a hack and we need to have a separate WaveletUD2.
            # For Undecimated wavelets, the wavelet_name is wavelet_id
            self.linear_op = WaveletUD2(
                wavelet_id=wavelet_name,
                nb_scale=nb_scale,
                n_coils=n_coils,
                n_jobs=n_jobs,
                verbose=verbosity_wavelet_op,
            )
        if verbose >= 5:
            print("Initialized linear wavelet operator : " +
                  str(self.linear_op))

        if self.gradient_method == "synthesis":
            self.gradient_op = self.GradSynthesis(
                linear_op=self.linear_op,
                fourier_op=self.fourier_op,
                verbose=self.verbose,
                **kwargs,
            )
        elif self.gradient_method == "analysis":
            self.gradient_op = self.GradAnalysis(
                fourier_op=self.fourier_op,
                verbose=self.verbose,
                **kwargs,
            )
        else:
            raise ValueError("gradient_method must be either "
                             "'synthesis' or 'analysis'")
