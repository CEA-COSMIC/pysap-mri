# #############################################################################
#  pySAP - Copyright (C) CEA, 2017 - 2018                                     #
#  Distributed under the terms of the CeCILL-B license,                       #
#  as published by the CEA-CNRS-INRIA. Refer to the LICENSE file or to        #
#  http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html for details.   #
# #############################################################################

"""Different tools required for the reconstruction."""


# Third party import
import numpy as np


def check_lipschitz_cst(f, x_shape, lipschitz_cst, max_nb_of_iter=10):
    """Check the validity of a Lipschitz constant for a specific function.

    This method checks that the Lipschitz constraints are statisfied
    for `max_nb_of_iter` random inputs:
    .. math:: ||f(x) - f(y)|| < lipschitz_cst ||x - y||

    Parameters
    ----------
    f: function
        A function to check for `lipschitz_cst` according to the
        above equation.
    x_shape: tuple
        Input data shape for function `f`.
    lipschitz_cst: float
        The Lischitz constant associated to the function `f`.
    max_nb_of_iter: int, default=10
        The number of random inputs used to validate the constant
        `lipschitz_cst` according to the above formula.

    Returns
    -------
    bool
        If False then `lipschitz_cst` is not respecting the above formula.
        Otherwise, `lipschitz_cst` might be an upper bound of the real
        Lipschitz constant for the function `f`.
    """
    is_lips_cst = True
    n = 0

    while is_lips_cst and n < max_nb_of_iter:
        n += 1
        x = np.random.randn(*x_shape)
        y = np.random.randn(*x_shape)
        is_lips_cst = (np.linalg.norm(f(x)-f(y)) <= (lipschitz_cst *
                                                     np.linalg.norm(x-y)))

    return is_lips_cst
