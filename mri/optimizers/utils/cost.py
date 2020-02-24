# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################


"""
Different cost functions for the optimization.
"""


# Third party import
from modopt.opt.cost import costObj
import numpy


class DualGapCost(costObj):
    """ Define the dual-gap cost function.
    """
    def __init__(self, linear_op, initial_cost=1e6, tolerance=1e-4,
                 cost_interval=None, test_range=4, verbose=False,
                 plot_output=None):
        """ Initialize the 'DualGapCost' class.
        Parameters
        ----------
        x: np.ndarray
            input original data array.
        costFunc: class
            Class for calculating the cost
        initial_cost: float, optional
            Initial value of the cost (default is "1e6")
        tolerance: float, optional
            Tolerance threshold for convergence (default is "1e-4")
        cost_interval: int, optional
            Iteration interval to calculate cost (default is "None")
            if None, cost is never calculated.
        test_range: int, optional
            Number of cost values to be used in test (default is "4")
        verbose: bool, optional
            Option for verbose output (default is "False")
        plot_output: str, optional
            Output file name for cost function plot
        """
        self.linear_op = linear_op
        super(DualGapCost, self).__init__(
            operators=None, initial_cost=initial_cost,
            tolerance=tolerance,
            cost_interval=cost_interval, test_range=test_range,
            verbose=verbose, plot_output=plot_output)
        self._iteration = 0

    def _calc_cost(self, x_new, y_new, *args, **kwargs):
        """ Return the dual-gap cost.
        Parameters
        ----------
        x_new: np.ndarray
            new primal solution.
        y_new: np.ndarray
            new dual solution.
        Returns
        -------
        norm: float
            the dual-gap.
        """
        x_dual_new = self.linear_op.adj_op(y_new)
        return numpy.linalg.norm(x_new - x_dual_new)


class GenericCost(costObj):
    """ Define the Generic cost function, based on the cost function of the
    gradient operator and the cost function of the proximity operator.
    """
    def __init__(self, gradient_op, prox_op, initial_cost=1e6,
                 tolerance=1e-4, cost_interval=None, test_range=4,
                 optimizer_type='forward_backward',
                 verbose=False, plot_output=None):
        """ Initialize the 'Cost' class.
        Parameters
        ----------
        gradient_op: instance of the gradient operator
            gradient operator used in the reconstruction process. It must
            implements the get_cost_function.
        prox_op: instance of the proximity operator
            proximity operator used in the reconstruction process. It must
            implements the get_cost function.
        linear_op: instance of the linear operator
            linear operator used to express the sparsity.
            If the synthesis formulation is used to solve the problem than the
            parameter has to be set to 0.
            If the analysis formultaion is used to solve the problem than the
            parameters needs to be filled
        initial_cost: float, optional
            Initial value of the cost (default is "1e6")
        tolerance: float, optional
            Tolerance threshold for convergence (default is "1e-4")
        cost_interval: int, optional
            Iteration interval to calculate cost (default is "None")
            if None, cost is never calculated.
        test_range: int, optional
            Number of cost values to be used in test (default is "4")
        optimizer_type: str, default 'forward_backward'
            Specifies the type of optimizer being used. This could be
            'primal_dual' or 'forward_backward'. The cost function being
            calculated would be different in case of 'primal_dual' as
            we receive both primal and dual intermediate solutions.
        verbose: bool, optional
            Option for verbose output (default is "False")
        plot_output: str, optional
            Output file name for cost function plot
        """
        gradient_cost = getattr(gradient_op, 'cost', None)
        prox_cost = getattr(prox_op, 'cost', None)
        if not callable(gradient_cost):
            raise RuntimeError("The gradient must implements a `cost`",
                               "function")
        if not callable(prox_cost):
            raise RuntimeError("The proximity operator must implements a",
                               " `cost` function")
        self.gradient_op = gradient_op
        self.prox_op = prox_op
        self.optimizer_type = optimizer_type

        super(GenericCost, self).__init__(
            operators=None, initial_cost=initial_cost,
            tolerance=tolerance,
            cost_interval=cost_interval, test_range=test_range,
            verbose=verbose, plot_output=plot_output)
        self._iteration = 0

    def _calc_cost(self, x_new, *args, **kwargs):
        """ Return the cost.
        Parameters
        ----------
        x_new: np.ndarray
            intermediate solution in the optimization problem.
        Returns
        -------
        cost: float
            the cost function defined by the operators (gradient + prox_op).
        """
        if self.optimizer_type == 'forward_backward':
            cost = self.gradient_op.cost(x_new) + self.prox_op.cost(x_new)
        else:
            # In primal dual algorithm, the value of args[0] is the data in
            # Wavelet Space, while x_new is data in Image space.
            # TODO, we need to generalize this
            cost = self.gradient_op.cost(x_new) + self.prox_op.cost(args[0])
        return cost
