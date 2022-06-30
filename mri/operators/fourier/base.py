# #############################################################################
#  pySAP - Copyright (C) CEA, 2017 - 2018                                     #
#  Distributed under the terms of the CeCILL-B license,                       #
#  as published by the CEA-CNRS-INRIA. Refer to the LICENSE file or to        #
#  http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html for details.   #
# #############################################################################

"""
Base Fourier Operator.

"""
from ..base import OperatorBase

class FourierOperatorBase(OperatorBase):
    """Base Fourier Operator class.

    Every (Linear) Fourier operator inherits from this class,
    to ensure that we have all the functions rightly implemented
    as required by ModOpt.

    Attributes
    ----------
    shape
    n_coils
    uses_sense

    Methods
    -------
    op(data)
        The forward operation (image -> kspace)
    adj_op(coeffs)
        The adjoint operation (kspace -> image)
    """
    @property
    def uses_sense(self):
        """Check if the operator uses sensitivity maps ..cite:`Pruessmann1999`.

        Returns
        -------
        bool
            True if operator uses sensitivity maps.
        """
        return False

    @property
    def shape(self):
        """Shape of the image space of the operator.

        Returns
        -------
        tuple
            The shape of the image space
        """
        return self._shape

    @shape.setter
    def shape(self, shape):
        self._shape = tuple(shape)

    @property
    def n_coils(self):
        """Get the number of coil of the image space of the operator.

        Returns
        -------
        int
            The number of coils.

        """
        return self._n_coils

    @n_coils.setter
    def n_coils(self, n_coils):
        if n_coils < 1 or not int(n_coils) == n_coils:
            raise ValueError("n_coils should be a positive integer")
        self._n_coils = int(n_coils)
