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


def _reconstruct_case(case, key_names, init_classes,
                      kspace_data, fourier_kwargs):
    """Internal Function to carry out reconstruction for a
    special case. This function pulls in appropriate keyword arguments
    from input and declares appropriate Linear, Fourier and Regularizer
    Operators. These operators later are used to create the image model
    by defining the reconstructor. Then the reconstruction is carried and
    results are returned

    Parameters
    ----------
    case: list of parameters
        keyword arguments for different operators for reconstruction
    key_names: list of strings
        names of keyword arguments specified for reconstruction in 'case'
    init_classes: list of classes
        classes for varoius operators. must be in format
        [linear, regularizer, reconstructor]
    kspace_data: np.ndarray
        the kspace data for reconstruction
    fourier_kwargs: dict
        holds dictionary with init_class pointing to fourier class to be used
        and args having keyword arguments for initialization
        NOTE: We declare fourier operator inside this function to allow
        parallel execution as NonCartesianFFT cannot be pickled.
    """
    fourier_op = fourier_kwargs['init_class'](**fourier_kwargs['args'])
    linear_op = init_classes[0](
        **dict(zip(key_names[0], case[0])))
    regularizer_op = init_classes[1](
        **dict(zip(key_names[1], case[1]))
    )
    reconstructor = init_classes[2](
        fourier_op=fourier_op,
        linear_op=linear_op,
        regularizer_op=regularizer_op,
        **dict(zip(key_names[2], case[2]))
    )
    raw_results = reconstructor.reconstruct(
        kspace_data=kspace_data,
        **dict(zip(key_names[3], case[3])),
    )
    return raw_results, case


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
    list_metric = []
    for res in results:
        list_metric.append(res[2][metric]['values'][-1])

    list_metric = np.array(list_metric)

    # get best runs
    if metric_direction:
        best_metric = list_metric.max()
        best_idx = list_metric.argmax()
    else:
        best_metric = list_metric.min()
        best_idx = list_metric.argmin()

    return best_metric, best_idx


def launch_grid(linear_kwargs, regularizer_kwargs, reconstructor_kwargs,
                optimizer_kwargs, compare_metric_details=None, n_jobs=1,
                verbose=0, **kwargs):
    """This function launches off reconstruction for a grid specified
    through use of kwarg dictionaries.

    Dictionary Convention
    ---------------------
    These dictionaries each defined to follow the convention:
    Each dictionary has a key `init_class` that specifies the
    initialization class for the operator (exception to
    this is 'optimizer_kwargs'). Later we have key `args` that holds
    all the input arguments that can be passed as a keyword dictionary.
    Each value in this keyword dictionary ,ust be a list of all
    values you want to search in gridsearch.

    This function finds the search space of parameters and
    sets up right parameters for '_reconstruct_case' function.
    Please check the example code for more details.

    Parameters
    ----------
    linear_kwargs: dict
        dictionary for linear operator parameters
    regularizer_kwargs: dict
        dictionary for regularizer operator parameters
    reconstructor_kwargs: dict
        dictionary for reconstructor operator parameters
    optimizer_kwargs: dict
        dictionary for optimizer key word arguments
    compare_metric_details: dict default None
        dictionary that holds the metric to be compared and metric
        direction please refer to `gather_result` documentation.
    n_jobs: int, default 1
        number of parallel jobs for execution
    verbose: int default 0
        Verbosity level
        0 => No debug prints
        1 => View best results if present

    **kwargs: keyword arguments that are passes to reconstruction
        these holds extra arguments that are passed to
        '_reconstruct_case' function.
        important arguments are 'ksapce_data' and 'fourier_kwargs'
    """
    # Convert non-list elements to list so that we can create
    # search space
    all_reformatted_kwargs = []
    for specific_kwargs in [linear_kwargs, regularizer_kwargs,
                            reconstructor_kwargs, optimizer_kwargs]:
        for key, value in specific_kwargs['args'].items():
            if not isinstance(value, list) and \
                    not isinstance(value, np.ndarray):
                specific_kwargs['args'][key] = [value]
        all_reformatted_kwargs.append(specific_kwargs)
    # Create Search space
    cross_product_list = list(itertools.product(*tuple(sum(
        [list(all_reformatted_kwargs[i]['args'].values())
         for i in range(len(all_reformatted_kwargs))],
        []
    ))))
    # Obtain Initialization classes
    init_classes = [
        all_reformatted_kwargs[i]['init_class']
        for i in range(len(all_reformatted_kwargs)-1)
    ]
    # Obtain key names for all cases
    key_names = list([
        list(all_reformatted_kwargs[i]['args'].keys())
        for i in range(len(all_reformatted_kwargs))
    ])
    reshaped_cross_product = []
    for i in range(len(cross_product_list)):
        iterator = iter(cross_product_list[i])
        reshaped_cross_product.append(
            [[next(iter(iterator))
              for _ in sublist] for sublist in key_names]
        )
    results, test_cases = zip(*Parallel(n_jobs=n_jobs)(
        delayed(_reconstruct_case)
        (reshaped_cross_product[i], key_names, init_classes, **kwargs)
        for i in range(len(cross_product_list))
    ))
    best_idx = None
    if compare_metric_details is not None:
        best_value, best_idx = \
            gather_result(**compare_metric_details,
                          results=results)
        if verbose > 0:
            print('The best result of grid search is: '
                  + str(test_cases[best_idx]))
            print('The best valuye of metric is : '
                  + str(best_value))
    return results, test_cases, key_names, best_idx
