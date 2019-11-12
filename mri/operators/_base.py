# #############################################################################
#  pySAP - Copyright (C) CEA, 2017 - 2018                                     #
#  Distributed under the terms of the CeCILL-B license,                       #
#  as published by the CEA-CNRS-INRIA. Refer to the LICENSE file or to        #
#  http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html for details.   #
# #############################################################################


class OperatorBase(object):
    """ Base Operator class. Every operator inherits from this class,
    to ensure that we have all the functions rightly implemented
    as required by Modopt
    """

    def op(self, img):
        """ This method calculates operator transform.
        Parameters
        ----------
        img: np.ndarray
            input as array.
        Returns
        -------
        result: np.ndarray
            operator transform of the input.
        """
        raise NotImplementedError("'op' is an abstract method.")

    def adj_op(self, x):
        """ This method calculates adjoint operator transform of real or
        complex sequence.
        Parameters
        ----------
        x: np.ndarray
            input data array.
        Returns
        -------
        results: np.ndarray
            adjoint operator transform.
        """
        raise NotImplementedError("'adj_op' is an abstract method.")