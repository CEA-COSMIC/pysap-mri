# #############################################################################
#  pySAP - Copyright (C) CEA, 2017 - 2018                                     #
#  Distributed under the terms of the CeCILL-B license,                       #
#  as published by the CEA-CNRS-INRIA. Refer to the LICENSE file or to        #
#  http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html for details.   #
# #############################################################################

# Internal Library import
from .utils import check_lipschitz_cst

# Third party import
import numpy as np
from modopt.math.matrix import PowerMethod
from modopt.opt.gradient import GradBasic


class GradBaseMRI(GradBasic):
    """ Gradient Analysis class for single channel
        Parameters
        ----------
        data: np.ndarray
            input data array.
        operator :
            a function handle that implements M X
        trans_operator :
            a function handle that implements Mt X
        shape : tuple
            shape of transformed data
        lipschitz_cst : int default None
            The lipschitz constant for for given operator.
            If not specified this is calculated using PowerMethod
        max_iter_spec_rad : int default 10
            Number of iterations to calculate the lipschitz constant
        num_check_lips : int default 10
            Number of iterations to check if lipschitz constant is correct
    """

    def __init__(self, data, operator, trans_operator, shape,
                 max_iter_spec_rad=10, lipschitz_cst=None, num_check_lips=10):
        super(GradBaseMRI, self).__init__(data, operator, trans_operator)
        if lipschitz_cst is not None:
            self.spec_rad = lipschitz_cst
            self.inv_spec_rad = 1.0 / self.spec_rad
        else:
            calc_lips = PowerMethod(self.trans_op_op, shape,
                                    data_type=np.complex, auto_run=False)
            calc_lips.get_spec_rad(extra_factor=1.1,
                                   max_iter=max_iter_spec_rad)
            self.spec_rad = calc_lips.spec_rad
            self.inv_spec_rad = calc_lips.inv_spec_rad
        if num_check_lips > 0:
            is_lips = check_lipschitz_cst(f=self.trans_op_op,
                                          x_shape=shape,
                                          lipschitz_cst=self.spec_rad,
                                          max_nb_of_iter=num_check_lips)
            if not is_lips:
                raise ValueError('The lipschitz constraint is not satisfied')
            else:
                print('The lipschitz constraint is satisfied')
