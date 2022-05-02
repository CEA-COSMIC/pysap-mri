# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

from mri.operators import WaveletN, NonCartesianFFT

import itertools

from joblib import Parallel, delayed
from modopt.opt.linear import Identity
from modopt.opt.proximity import SparseThreshold
import numpy as np


class _TestCase(object):
    """Internal Class to save a test case in a format
    and call reconstruct

    Parameters
    ----------
    kspace_data: np.ndarray
            the kspace data for reconstruction
    linear_op_class: class
        linear operator initialization class
    linear_op_kwargs: dict
        kwargs for initializing linear operator
    regularizer_op_class: class
        regularizer operator initialization class
    regularizer_op_kwargs: dict
        kwargs for initializing regularizer operator
    optimizer_kwargs: dict
        kwargs for optimizer
    """
    def __init__(self, kspace_data, linear_op_class, regularizer_op_class,
                 linear_op_kwargs, regularizer_op_kwargs,
                 optimizer_kwargs):
        self.kspace_data = kspace_data
        self.linear_op = linear_op_class(**linear_op_kwargs)
        self.regularizer_op = regularizer_op_class(**regularizer_op_kwargs)
        self.optimizer_kwargs = optimizer_kwargs

    def reconstruct_case(self, fourier_op, reconstructor_class,
                         reconstructor_kwargs, fourier_params=None):
        """Internal Function to carry out reconstruction for a
        special case. This function pulls in appropriate keyword arguments
        from input and declares appropriate Linear, Fourier and Regularizer
        Operators. These operators later are used to create the image model
        by defining the reconstructor. Then the reconstruction is carried and
        results are returned

        Parameters
        ----------
        fourier_op: object of class FFT
            this defines the fourier operator. for NonCartesianFFT, please make
            fourier_op as `None` and pass fourier_params to allow
            parallel execution
        reconstructor_class: class
            reconstructor class
        reconstructor_kwargs: dict
            extra kwargs for reconstructor
        fourier_params: dict, default None
            holds dictionary with init_class pointing to fourier
            class to be used and args having keyword arguments for
            initialization
            This is passed only if fourier_op is None so that fourier_op can be
            made on spot during reconstruction.
            NOTE: We declare fourier operator inside this function to allow
            parallel execution as NonCartesianFFT cannot be pickled.
        """
        if fourier_op is None:
            fourier_op = fourier_params['init_class'](
                **fourier_params['kwargs']
            )
        reconstructor = reconstructor_class(
            fourier_op=fourier_op,
            linear_op=self.linear_op,
            regularizer_op=self.regularizer_op,
            **reconstructor_kwargs,
        )
        raw_results = reconstructor.reconstruct(
            kspace_data=self.kspace_data,
            **self.optimizer_kwargs
        )
        return raw_results


def gather_result(metric, results, metric_direction=None):
    """ Gather the best reconstruction result.

    Parameters:
    -----------
    metric: str,
        the name of the metric, it will become a dict key in the output dict.
    results: list of list,
        list of the raw results of the gridsearch
    metric_direction: bool, default None
        if True the higher the better the metric value is (like for `ssim`),
        else the lower the better.
        If None, we choose defaults as follows:
            if metric is 'ssim', 'psnr' or 'accuracy', metric_direction is True
            if metric is 'nrmse' or 'mse', metric_direction is False
    Return:
    -------
    results and location of best results in given set of raw results
    """
    # Make metric string lower case
    metric = metric.lower()
    list_metric = np.array([
        res[2][metric]['values'][-1] for res in results
    ])
    if metric_direction is None:
        # If metric_direction is None, we choose from a set of possible
        # default values
        if metric == 'ssim' or metric == 'psnr' or metric == 'accuracy':
            metric_direction = True
        elif metric == 'nrmse' or metric == 'mse':
            metric_direction = False
        else:
            raise ValueError('Cannot automatically find out metric direction, '
                             'please specify metric direction')
    # get best runs
    if metric_direction:
        best_metric = list_metric.max()
        best_idx = list_metric.argmax()
    else:
        best_metric = list_metric.min()
        best_idx = list_metric.argmin()

    return best_metric, best_idx


def launch_grid(kspace_data, reconstructor_class, reconstructor_kwargs,
                fourier_op=None, linear_params=None, regularizer_params=None,
                optimizer_params=None, compare_metric_details=None, n_jobs=1,
                verbose=0):
    """This function launches off reconstruction for a grid specified
    through use of kwarg dictionaries.

    Dictionary Convention
    ---------------------
    These dictionaries each defined to follow the convention:
    Each dictionary has a key `init_class` that specifies the
    initialization class for the operator (exception to
    this is 'optimizer_params'). Later we have key `kwargs` that holds
    all the input arguments that can be passed as a keyword dictionary.
    Each value in this keyword dictionary ,ust be a list of all
    values you want to search in gridsearch.

    This function finds the search space of parameters and
    sets up right parameters for '_reconstruct_case' function.
    Please check the example code for more details.

    Parameters
    ----------
    kspace_data: np.ndarray
        the kspace data for reconstruction
    reconstructor_class: class
        reconstructor class
    reconstructor_kwargs: dict
        extra kwargs for reconstructor
    fourier_op: object of class FFT
        this defines the fourier operator. for NonCartesianFFT, please make
        fourier_op as `None` and pass fourier_params to allow
        parallel execution
    linear_params: dict, default None
        dictionary for linear operator parameters
        if None, a sym8 wavelet is chosen
    regularizer_params: dict, default None
        dictionary for regularizer operator parameters
        if None, mu=0, ie no regularization is done
    optimizer_params: dict, default None
        dictionary for optimizer key word arguments
        if None, a FISTA optimization is done for 100 iterations
    compare_metric_details: dict default None
        dictionary that holds the metric to be compared and metric
        direction please refer to `gather_result` documentation.
        if None, all raw_results are returned and best_idx is None
    n_jobs: int, default 1
        number of parallel jobs for execution
    verbose: int default 0
        Verbosity level
        0 => No debug prints
        1 => View best results if present
    """
    # Convert non-list elements to list so that we can create
    # search space
    init_classes = []
    key_names = []
    if linear_params is None:
        linear_params = {
            'init_class': WaveletN,
            'kwargs':
                {
                    'wavelet_name': 'sym8',
                    'nb_scale': 4,
                }
        }
    if regularizer_params is None:
        regularizer_params = {
            'init_class': SparseThreshold,
            'kwargs':
                {
                    'linear': Identity(),
                    'weights': [0],
                }
        }
    if optimizer_params is None:
        optimizer_params = {
            # Just following convention
            'kwargs':
                {
                    'optimization_alg': 'fista',
                    'num_iterations': 100,
                }
        }
    for specific_params in [linear_params, regularizer_params,
                            optimizer_params]:
        for key, value in specific_params['kwargs'].items():
            if not isinstance(value, (list, tuple, np.ndarray)):
                specific_params['kwargs'][key] = [value]
        # Obtain Initialization classes
        if specific_params != optimizer_params:
            init_classes.append(specific_params['init_class'])
        # Obtain Key Names
        key_names.append(list(specific_params['kwargs'].keys()))
    # Create Search space
    cross_product_list = list(itertools.product(
        *linear_params['kwargs'].values(),
        *regularizer_params['kwargs'].values(),
        *optimizer_params['kwargs'].values(),
    ))
    test_cases = []
    number_of_test_cases = len(cross_product_list)
    if verbose > 0:
        print('Total number of gridsearch cases : ' +
              str(number_of_test_cases))
    # Reshape data such that they match values for key_names
    for test_case in cross_product_list:
        iterator = iter(test_case)
        # Add the test case after reshaping the list
        all_kwargs_values = []
        for indivitual_param_names in key_names:
            param_kwargs = {}
            for key in indivitual_param_names:
                param_kwargs[key] = next(iter(iterator))
            all_kwargs_values.append(param_kwargs)
        test_cases.append(_TestCase(
            kspace_data,
            *init_classes,
            *all_kwargs_values)
        )
    if isinstance(fourier_op, NonCartesianFFT):
        fourier_params = {
            'init_class': NonCartesianFFT,
            'kwargs':
                {
                    'samples': fourier_op.samples,
                    'shape': fourier_op.shape,
                    'n_coils': fourier_op.n_coils,
                    'implementation': fourier_op.implementation,
                    'density_comp': fourier_op.density_comp,
                    **fourier_op.kwargs,
                }
        }
        fourier_op = None
    else:
        fourier_params = None
    # Call for reconstruction
    results = Parallel(n_jobs=n_jobs)(
        delayed(test_case.reconstruct_case)(
            fourier_op=fourier_op,
            reconstructor_class=reconstructor_class,
            reconstructor_kwargs=reconstructor_kwargs,
            fourier_params=fourier_params,
        )
        for test_case in test_cases
    )
    best_idx = None
    if compare_metric_details is not None:
        best_value, best_idx = \
            gather_result(
                **compare_metric_details,
                results=results,
            )
        if verbose > 0:
            print('The best result of grid search is: '
                  + str(cross_product_list[best_idx]))
            print('The best value of metric is : '
                  + str(best_value))
    return results, cross_product_list, key_names, best_idx
