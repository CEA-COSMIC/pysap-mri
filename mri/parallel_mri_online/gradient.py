# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
This module contains classes for defining algorithm operators and gradients.
"""

# Package import
from .utils import check_lipschitz_cst

# Third party import
import numpy as np
from modopt.math.matrix import PowerMethod
from modopt.opt.gradient import GradBasic


class Grad2D_pMRI_analysis(GradBasic, PowerMethod):
    """ Gradient 2D synthesis class.

    This class defines the grad operators for:
            (1/2) * sum(||Ft Sl x - yl||^2_2,l)

    Parameters
    ----------
    data: np.ndarray
        input 2D data array.
    fourier_op: instance
        a Fourier operator instance.
    S: np.ndarray
        sensitivity matrix
    """
    def __init__(self, data, fourier_op, max_iter, gradient_spec_rad=None):
        """ Initilize the 'GradSynthesis2' class.
        """

        self.fourier_op = fourier_op
        GradBasic.__init__(self, data, self._analy_op_method,
                           self._analy_rsns_op_method)
        PowerMethod.__init__(self,
                             self.trans_op_op,
                             (data.shape[0], *fourier_op.shape),
                             data_type="complex128",
                             auto_run=False)
        if gradient_spec_rad is None:
            self.get_spec_rad(extra_factor=1.1, max_iter=max_iter)
        else:
            self.spec_rad = gradient_spec_rad
            self.inv_spec_rad = 1.0 / self.spec_rad

    def _analy_op_method(self, x):
        """ MX operation.

        This method calculates the action of the matrix M on the data X, in
        this case fourier transform of the the input data

        Parameters
        ----------
        x: a WaveletTransformBase derived object
            the analysis coefficients.

        Returns
        -------
        result: np.ndarray
            the operation result (the recovered kspace).
        """
        data = []
        [data.append(self.fourier_op.op(x[channel]))
            for channel in range(x.shape[0])]
        return np.asarray(data)

    def _analy_rsns_op_method(self, x):
        """ MtX operation.

        This method calculates the action of the transpose of the matrix M on
        the data X, where M is the inverse fourier transform.

        Parameters
        ----------
        x: np.ndarray
            input kspace 2D array.

        Returns
        -------
        result: np.ndarray
            the operation result.
        """
        img = []
        [img.append(self.fourier_op.adj_op(x[channel]))
            for channel in range(x.shape[0])]
        return np.asarray(img)


class Grad2D_pMRI_synthesis(GradBasic, PowerMethod):
    """ Gradient 2D synthesis class.

    This class defines the grad operators for |M*F*invL*alpha - data|**2.

    Parameters
    ----------
    data: np.ndarray
        input 2D data array.
    fourier_op: instance
        a Fourier operator instance.
    linear_op: object
        a linear operator instance.
    S: np.ndarray  (image_shape, L)
        The sensitivity maps of size.
    """
    def __init__(self, data, fourier_op, linear_op, max_iter,
                 gradient_spec_rad=None):
        """ Initilize the 'GradSynthesis2' class.
        """

        self.fourier_op = fourier_op
        self.linear_op = linear_op
        GradBasic.__init__(self, data, self._synth_op_method,
                           self._synth_trans_op_method)
        coef = linear_op.op(np.zeros((data.shape[0],
                                      *fourier_op.shape)).astype(np.complex))
        self.linear_op_coeffs_shape = coef.shape
        PowerMethod.__init__(self, self.trans_op_op,
                             self.linear_op_coeffs_shape,
                             data_type="complex128",
                             auto_run=False)
        if gradient_spec_rad is None:
            self.get_spec_rad(extra_factor=1.1, max_iter=max_iter)
        else:
            self.spec_rad = gradient_spec_rad
            self.inv_spec_rad = 1.0 / self.spec_rad

    def _synth_op_method(self, x):
        """ MX operation.

        This method calculates the action of the matrix M on the data X, in
        this case fourier transform of the the input data

        Parameters
        ----------
        x: a WaveletTransformBase derived object
            the analysis coefficients.

        Returns
        -------
        result: np.ndarray
            the operation result (the recovered kspace).
        """
        rsl = []
        images = self.linear_op.adj_op(x)
        [rsl.append(self.fourier_op.op(images[channel]))
            for channel in range(x.shape[0])]
        return np.asarray(rsl)

    def _synth_trans_op_method(self, x):
        """ MtX operation.

        This method calculates the action of the transpose of the matrix M on
        the data X, where M is the inverse fourier transform.

        Parameters
        ----------
        x: np.ndarray
            input kspace 2D array.

        Returns
        -------
        result: np.ndarray
            the operation result.
        """
        images = np.asarray([self.fourier_op.adj_op(x[channel])
                             for channel in range(x.shape[0])])
        rslt = self.linear_op.op(images)
        return np.asarray(rslt)


class Gradient_pMRI_calibrationless(Grad2D_pMRI_analysis,
                                    Grad2D_pMRI_synthesis):
    """ Gradient for 2D parallel imaging reconstruction.

    This class defines the datafidelity terms methods that will be defined by
    derived gradient classes:
    It computes the gradient of the following equation for the analysis and
    synthesis cases respectively:

    * (1/2) * (||Ft x - yl||^2_2,l)
    * (1/2) * (||Ft L* alpha - yl||^2_2,l)
    """
    def __init__(self, data, fourier_op, linear_op=None, check_lips=False,
                 gradient_spec_rad=None, max_iter=5):
        """ Initilize the 'Grad2D_pMRI' class.

        Parameters
        ----------
        data: np.ndarray
            input 2D data array.
        fourier_op: instance
            a Fourier operator instance derived from the FourierBase' class.
        linear_op: instance
            a Linear operator instance.
        """
        if linear_op is None:
            Grad2D_pMRI_analysis.__init__(self, data, fourier_op,
                                          gradient_spec_rad=gradient_spec_rad,
                                          max_iter=max_iter)
            if check_lips or gradient_spec_rad is not None:
                xinit_shape = (data.shape[0], *fourier_op.shape)
            self.analysis = True
        else:
            Grad2D_pMRI_synthesis.__init__(self, data, fourier_op, linear_op,
                                           gradient_spec_rad=gradient_spec_rad,
                                           max_iter=max_iter)
            if check_lips or gradient_spec_rad is not None:
                xinit_shape = self.linear_op_coeffs_shape

            self.synthesis = False

        if check_lips or (gradient_spec_rad is not None):
            print("Checking Lipschitz constant")
            is_lips = check_lipschitz_cst(f=self.trans_op_op,
                                          x_shape=xinit_shape,
                                          lipschitz_cst=self.spec_rad,
                                          max_nb_of_iter=2)
            if not is_lips:
                raise ValueError('The lipschitz constraint is not satisfied')
            else:
                print('The lipschitz constraint is satisfied')

    def get_cost(self, x):
        """ Gettng the cost.

        This method calculates the cost function of the differentiable part of
        the objective function.

        Parameters
        ----------
        x: np.ndarray
        input 2D data array.

        Returns
        -------
        result: float
        The result of the differentiablepart.
        """
        return 0.5 * (np.abs(self.op(x).flatten() -
                      self.obs_data.flatten())**2).sum()
