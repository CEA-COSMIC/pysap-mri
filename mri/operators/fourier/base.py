# #############################################################################
#  pySAP - Copyright (C) CEA, 2017 - 2018                                     #
#  Distributed under the terms of the CeCILL-B license,                       #
#  as published by the CEA-CNRS-INRIA. Refer to the LICENSE file or to        #
#  http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html for details.   #
# #############################################################################

"""
Base Fourier Operator.

Every Fourier operator should have an `op` and `adj_op` methods.
Also, it should exposes a `uses_sense` property to determine
if it implement sensitivity maps support.
"""


class FourierOperatorBase:
    """Base Fourier Operator class.

    Every (Linear) Fourier operator inherits from this class,
    to ensure that we have all the functions rightly implemented
    as required by Modopt.
    """

    def op(self, data):
        """Compute operator transform.

        Parameters
        ----------
        data: np.ndarray
            input as array.

        Returns
        -------
        result: np.ndarray
            operator transform of the input.
        """
        raise NotImplementedError("'op' is an abstract method.")

    def adj_op(self, coeffs):
        """Compute adjoint operator transform.

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

    @property
    def uses_sense(self):
        """Return True if the operator uses sensitivity maps."""
        return False

    @property
    def shape(self):
        """Shape of the image space of the operator."""
        return self._shape

    @shape.setter
    def shape(self, shape):
        self._shape = tuple(shape)

    @property
    def n_coils(self):
        """Return number of coil of the image space of the operator."""
        return self._n_coils

    @n_coils.setter
    def n_coils(self, n_coils):
        if n_coils < 1 or type(n_coils) is not int:
            raise ValueError("n_coils should be a positive integer")
        self._n_coils = n_coils
