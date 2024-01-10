# #############################################################################
#  pySAP - Copyright (C) CEA, 2017 - 2018                                     #
#  Distributed under the terms of the CeCILL-B license,                       #
#  as published by the CEA-CNRS-INRIA. Refer to the LICENSE file or to        #
#  http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html for details.   #
# #############################################################################

"""Base class for defining gradient operators."""

# Internal import
from .utils import check_lipschitz_cst

# Third party import
import numpy as np
from modopt.math.matrix import PowerMethod
from modopt.opt.gradient import GradBasic


class GradBaseMRI(GradBasic):
    """Base Gradient class for all gradient operators.

    Implement the gradient of following function with respect to x:
    .. math:: ||M x - y|| ^ 2

    Parameters
    ----------
    data: numpy.ndarray
        The input data array corresponding to observed data `y`.
    operator : function
        A function that implements `M`.
    trans_operator : function
        A function that implements `M ^ T`.
    shape : tuple
        The shape of observed data `y`.
    lips_calc_max_iter : int, default=10
        Number of iterations to calculate the lipschitz constant when
        `lipschitz_cst` is provided as None. The default value is 10.
    lipschitz_cst : int, default=None
        The lipschitz constant for the given `operator`. If not specified,
        this is calculated using the power method. The default value is
        None.
    num_check_lips : int, default=10
        Number of iterations to check if lipschitz constant is correct.
        The default value is 10.
    verbose: int, default=0
        Verbosity level for debug prints. When `verbose` and
        `num_check_lips` are both not equal to 0, it prints if the lipschitz
        constraints are satisfied. The default value is 0.
    """

    def __init__(self, operator, trans_operator, shape,
                 lips_calc_max_iter=10, lipschitz_cst=None, dtype=np.complex64, num_check_lips=10,
                 verbose=0, **kwargs):
        # Initialize the GradBase with dummy data
        super(GradBaseMRI, self).__init__(
            np.array(0),
            operator,
            trans_operator,
            **kwargs,
        )
        if lipschitz_cst is not None:
            self.spec_rad = lipschitz_cst
            self.inv_spec_rad = 1.0 / self.spec_rad
        else:
            calc_lips = PowerMethod(self.trans_op_op, shape,
                                    data_type=np.complex64, auto_run=False)
            calc_lips.get_spec_rad(extra_factor=1.1,
                                   max_iter=lips_calc_max_iter)
            self.spec_rad = calc_lips.spec_rad
            self.inv_spec_rad = calc_lips.inv_spec_rad
        if verbose > 0:
            print("Lipschitz constant is " + str(self.spec_rad))
        if num_check_lips > 0:
            is_lips = check_lipschitz_cst(f=self.trans_op_op,
                                          x_shape=shape,
                                          x_dtype=dtype,
                                          lipschitz_cst=self.spec_rad,
                                          max_nb_of_iter=num_check_lips)
            if not is_lips:
                raise ValueError('The lipschitz constraint is not satisfied')
            else:
                if verbose > 0:
                    print('The lipschitz constraint is satisfied')
