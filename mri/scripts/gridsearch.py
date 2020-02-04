# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

import itertools

from joblib import Parallel, delayed
import numpy as np


def _reconstruct_case(grid_search_test_case, key_names, init_classes,
                      kspace_data, fourier_params):
    """Internal Function to carry out reconstruction for a
    special case. This function pulls in appropriate keyword arguments
    from input and declares appropriate Linear, Fourier and Regularizer
    Operators. These operators later are used to create the image model
    by defining the reconstructor. Then the reconstruction is carried and
    results are returned

    Parameters
    ----------
    grid_search_test_case: list of parameters
        keyword arguments for different operators for reconstruction
    key_names: list of strings
        names of keyword arguments specified for reconstruction in 'case'
    init_classes: list of classes
        classes for varoius operators. must be in format
        [linear, regularizer, reconstructor]
    kspace_data: np.ndarray
        the kspace data for reconstruction
    fourier_params: dict
        holds dictionary with init_class pointing to fourier class to be used
        and args having keyword arguments for initialization
        NOTE: We declare fourier operator inside this function to allow
        parallel execution as NonCartesianFFT cannot be pickled.
    """
    fourier_op = fourier_params['init_class'](**fourier_params['kwargs'])
    linear_op = init_classes[0](
        **dict(zip(key_names[0], grid_search_test_case[0])))
    regularizer_op = init_classes[1](
        **dict(zip(key_names[1], grid_search_test_case[1]))
    )
    reconstructor = init_classes[2](
        fourier_op=fourier_op,
        linear_op=linear_op,
        regularizer_op=regularizer_op,
        **dict(zip(key_names[2], grid_search_test_case[2]))
    )
    raw_results = reconstructor.reconstruct(
        kspace_data=kspace_data,
        **dict(zip(key_names[3], grid_search_test_case[3])),
    )
    return raw_results


def gather_result(metric, metric_direction, results):
    """ Gather the best reconstruction result.

    Parameters:
    -----------
    metric: str,
        the name of the metric, it will become a dict key in the output dict.
    metric_direction: bool,
        if True the higher the better the metric value is (like for `ssim`),
        else the lower the better.
    results: list of list,
        list of the raw results of the gridsearch

    Return:
    -------
    results and location of best results in given set of raw results
    """
    list_metric = np.array([res[2][metric]['values'][-1]
                            for res in results])
    # get best runs
    if metric_direction:
        best_metric = list_metric.max()
        best_idx = list_metric.argmax()
    else:
        best_metric = list_metric.min()
        best_idx = list_metric.argmin()

    return best_metric, best_idx


def launch_grid(linear_params, regularizer_params, reconstructor_params,
                optimizer_params, compare_metric_details=None, n_jobs=1,
                verbose=0, **kwargs):
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
    linear_params: dict
        dictionary for linear operator parameters
    regularizer_params: dict
        dictionary for regularizer operator parameters
    reconstructor_params: dict
        dictionary for reconstructor operator parameters
    optimizer_params: dict
        dictionary for optimizer key word arguments
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

    **kwargs: keyword arguments that are passes to reconstruction
        these holds extra arguments that are passed to
        '_reconstruct_case' function.
        important arguments are 'ksapce_data' and 'fourier_params'
    """
    # Convert non-list elements to list so that we can create
    # search space
    all_reformatted_params = []
    init_classes = []
    key_names = []
    for specific_params in [linear_params, regularizer_params,
                            reconstructor_params, optimizer_params]:
        for key, value in specific_params['kwargs'].items():
            if not isinstance(value, list) and \
                    not isinstance(value, np.ndarray):
                specific_params['kwargs'][key] = [value]
        all_reformatted_params.append(
            list(specific_params['kwargs'].values()))
        # Obtain Initialization classes
        if specific_params != optimizer_params:
            init_classes.append(specific_params['init_class'])
        # Obtain Key Names
        key_names.append(list(specific_params['kwargs'].keys()))
    # Create Search space
    cross_product_list = list(itertools.product(*tuple(sum(
        all_reformatted_params,
        []
    ))))
    reshaped_cross_product = []
    for i in range(len(cross_product_list)):
        iterator = iter(cross_product_list[i])
        reshaped_cross_product.append(
            [[next(iter(iterator))
              for _ in sublist] for sublist in key_names]
        )
    results = Parallel(n_jobs=n_jobs)(
        delayed(_reconstruct_case)
        (reshaped_cross_product[i], key_names, init_classes, **kwargs)
        for i in range(len(cross_product_list))
    )
    best_idx = None
    if compare_metric_details is not None:
        best_value, best_idx = \
            gather_result(**compare_metric_details,
                          results=results)
        if verbose > 0:
            print('The best result of grid search is: '
                  + str(reshaped_cross_product[best_idx]))
            print('The best value of metric is : '
                  + str(best_value))
    return results, reshaped_cross_product, key_names, best_idx
