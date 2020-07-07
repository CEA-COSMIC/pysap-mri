# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

""" This module defines primal dual optimizers
"""

# System import
import time

# Package import
from .utils.reweight import mReweight

# Third party import
import numpy as np
from modopt.math.stats import sigma_mad
from modopt.opt.linear import Identity
from modopt.opt.algorithms import Condat
from modopt.opt.reweight import cwbReweight


def condatvu(gradient_op, linear_op, dual_regularizer, cost_op,
             max_nb_of_iter=150, tau=None, sigma=None, relaxation_factor=1.0,
             x_init=None, std_est=None, std_est_method=None, std_thr=2.,
             nb_of_reweights=1, metric_call_period=5, metrics={}, verbose=0):
    """ The Condat-Vu sparse reconstruction with reweightings.

    Parameters
    ----------
    gradient_op: instance of class GradBase
        the gradient operator.
    linear_op: instance of LinearBase
        the linear operator: seek the sparsity, ie. a wavelet transform.
    dual_regularizer: instance of ProximityParent
        the  dual regularization operator
    cost_op: instance of costObj
        the cost function used to check for convergence during the
        optimization.
    max_nb_of_iter: int, default 150
        the maximum number of iterations in the Condat-Vu proximal-dual
        splitting algorithm.
    tau, sigma: float, default None
        parameters of the Condat-Vu proximal-dual splitting algorithm.
        If None estimates these parameters.
    relaxation_factor: float, default 0.5
        parameter of the Condat-Vu proximal-dual splitting algorithm.
        If 1, no relaxation.
    x_init: np.ndarray (optional, default None)
        the initial guess of image
    std_est: float, default None
        the noise std estimate.
        If None use the MAD as a consistent estimator for the std.
    std_est_method: str, default None
        if the standard deviation is not set, estimate this parameter using
        the mad routine in the image ('primal') or in the sparse wavelet
        decomposition ('dual') domain.
    std_thr: float, default 2.
        use this treshold expressed as a number of sigma in the residual
        proximity operator during the thresholding.
    relaxation_factor: float, default 0.5
        parameter of the Condat-Vu proximal-dual splitting algorithm.
        If 1, no relaxation.
    nb_of_reweights: int, default 1
        the number of reweightings.
    metric_call_period: int (default 5)
        the period on which the metrics are compute.
    metrics: dict (optional, default None)
        the list of desired convergence metrics: {'metric_name':
        [@metric, metric_parameter]}. See modopt for the metrics API.
    verbose: int, default 0
        the verbosity level.

    Returns
    -------
    x_final: ndarray
        the estimated CONDAT-VU solution.
    costs: list of float
        the cost function values.
    metrics: dict
        the requested metrics values during the optimization.
    y_final: ndarrat
        the estimated dual CONDAT-VU solution
    """
    # Check inputs
    start = time.perf_counter()
    if std_est_method not in (None, "primal", "dual"):
        raise ValueError(
            "Unrecognize std estimation method '{0}'.".format(std_est_method))

    # Define the initial primal and dual solutions
    if x_init is None:
        x_init = np.squeeze(np.zeros((linear_op.n_coils,
                                      *gradient_op.fourier_op.shape),
                                     dtype=np.complex))
    primal = x_init
    dual = linear_op.op(primal)
    weights = dual

    # Define the weights used during the thresholding in the dual domain,
    # the reweighting strategy, and the prox dual operator

    # Case1: estimate the noise std in the image domain
    if std_est_method == "primal":
        if std_est is None:
            std_est = sigma_mad(gradient_op.MtX(gradient_op.obs_data))
        weights[...] = std_thr * std_est
        reweight_op = cwbReweight(weights)
        dual_regularizer.weights = reweight_op.weights

    # Case2: estimate the noise std in the sparse wavelet domain
    elif std_est_method == "dual":
        if std_est is None:
            std_est = 0.0
        weights[...] = std_thr * std_est
        reweight_op = mReweight(weights, linear_op, thresh_factor=std_thr)
        dual_regularizer.weights = reweight_op.weights

    # Case3: manual regularization mode, no reweighting
    else:
        reweight_op = None
        nb_of_reweights = 0

    # Define the Condat Vu optimizer: define the tau and sigma in the
    # Condat-Vu proximal-dual splitting algorithm if not already provided.
    # Check also that the combination of values will lead to convergence.
    norm = linear_op.l2norm(x_init.shape)
    lipschitz_cst = gradient_op.spec_rad
    if sigma is None:
        sigma = 0.5
    if tau is None:
        # to avoid numerics troubles with the convergence bound
        eps = 1.0e-8
        # due to the convergence bound
        tau = 1.0 / (lipschitz_cst/2 + sigma * norm**2 + eps)
    convergence_test = (
        1.0 / tau - sigma * norm ** 2 >= lipschitz_cst / 2.0)

    # Welcome message
    if verbose > 0:
        print(" - mu: ", dual_regularizer.weights)
        print(" - lipschitz constant: ", gradient_op.spec_rad)
        print(" - tau: ", tau)
        print(" - sigma: ", sigma)
        print(" - rho: ", relaxation_factor)
        print(" - std: ", std_est)
        print(" - 1/tau - sigma||L||^2 >= beta/2: ", convergence_test)
        print(" - data: ", gradient_op.fourier_op.shape)
        if hasattr(linear_op, "nb_scale"):
            print(" - wavelet: ", linear_op, "-", linear_op.nb_scale)
        print(" - max iterations: ", max_nb_of_iter)
        print(" - number of reweights: ", nb_of_reweights)
        print(" - primal variable shape: ", primal.shape)
        print(" - dual variable shape: ", dual.shape)
        print("-" * 40)

    prox_op = Identity()

    # Define the optimizer
    opt = Condat(
        x=primal,
        y=dual,
        grad=gradient_op,
        prox=prox_op,
        prox_dual=dual_regularizer,
        linear=linear_op,
        cost=cost_op,
        rho=relaxation_factor,
        sigma=sigma,
        tau=tau,
        rho_update=None,
        sigma_update=None,
        tau_update=None,
        auto_iterate=False,
        metric_call_period=metric_call_period,
        metrics=metrics)
    cost_op = opt._cost_func

    # Perform the first reconstruction
    if verbose > 0:
        print("Starting optimization...")
    opt.iterate(max_iter=max_nb_of_iter)

    # Loop through the number of reweightings
    for reweight_index in range(nb_of_reweights):

        # Generate the new weights following reweighting prescription
        if std_est_method == "primal":
            reweight_op.reweight(linear_op.op(opt._x_new))
        else:
            std_est = reweight_op.reweight(opt._x_new)

        # Welcome message
        if verbose > 0:
            print(" - reweight: ", reweight_index + 1)
            print(" - std: ", std_est)

        # Update the weights in the dual proximity operator
        dual_regularizer.weights = reweight_op.weights

        # Perform optimisation with new weights
        opt.iterate(max_iter=max_nb_of_iter)

    # Goodbye message
    end = time.perf_counter()
    if verbose > 0:
        if hasattr(cost_op, "cost"):
            print(" - final iteration number: ", cost_op._iteration)
            print(" - final cost value: ", cost_op.cost)
        print(" - converged: ", opt.converge)
        print("Done.")
        print("Execution time: ", end - start, " seconds")
        print("-" * 40)

    # Get the final solution
    x_final = opt.x_final
    y_final = opt.y_final
    if hasattr(cost_op, "cost"):
        costs = cost_op._cost_list
    else:
        costs = None

    return x_final, costs, opt.metrics, y_final
