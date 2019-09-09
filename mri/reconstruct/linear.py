# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
This module contains linears operators classes.
"""


# Package import
import pysap
from pysap.base.utils import flatten
from pysap.base.utils import unflatten

# Third party import
import numpy
from joblib import Parallel, delayed


class WaveletN(object):
    """ The 2D and 3D wavelet transform class.
    """
    def __init__(self, wavelet_name, nb_scale=4, verbose=0, dim=2,
                 num_channels=1, n_cpu=1, **kwargs):
        """ Initialize the 'Wavelet2' class.

        Parameters
        ----------
        wavelet_name: str
            the wavelet name to be used during the decomposition.
        nb_scales: int, default 4
            the number of scales in the decomposition.
        num_channels: int, default 1
            the number of channels in input data.
        verbose: int, default 0
            the verbosity level.
        """
        self.nb_scale = nb_scale
        self.flatten = flatten
        self.unflatten = unflatten
        self.num_channels = num_channels
        self.n_cpu = n_cpu
        if wavelet_name not in pysap.AVAILABLE_TRANSFORMS:
            raise ValueError(
                "Unknown transformation '{0}'.".format(wavelet_name))
        transform_klass = pysap.load_transform(wavelet_name)
        self.transform = transform_klass(
            nb_scale=self.nb_scale, verbose=verbose, dim=dim, **kwargs)
        self.coeffs_shape = None

    def get_coeff(self):
        return self.transform.analysis_data

    def set_coeff(self, coeffs):
        self.transform.analysis_data = coeffs

    def _op(self, data):
        self.transform.data = data
        self.transform.analysis()
        coeffs, coeffs_shape = flatten(self.transform.analysis_data)
        return coeffs, coeffs_shape

    def op(self, data):
        """ Define the wavelet operator.

        This method returns the input data convolved with the wavelet filter.

        Parameters
        ----------
        data: ndarray or Image
            input 2D data array.

        Returns
        -------
        coeffs: ndarray
            the wavelet coefficients.
        """
        coeffs = []
        if isinstance(data, numpy.ndarray):
            data = pysap.Image(data=data)
        if self.num_channels == 1:
            coeffs, self.coeffs_shape = self._op(data)
        else:
            coeffs, coeffs_shape = \
                zip(*Parallel(n_jobs=self.n_cpu)
                    (delayed(self._op)
                    (data[i])
                    for i in numpy.arange(self.num_channels)))
            self.coeffs_shape = numpy.asarray(coeffs_shape)
        return numpy.asarray(coeffs)

    def _adj_op(self, coeffs, coeffs_shape):
        self.transform.analysis_data = unflatten(coeffs, coeffs_shape)
        image = self.transform.synthesis()
        return image

    def adj_op(self, coeffs, dtype="array"):
        """ Define the wavelet adjoint operator.

        This method returns the reconsructed image.

        Parameters
        ----------
        coeffs: ndarray
            the wavelet coefficients.
        dtype: str, default 'array'
            if 'array' return the data as a ndarray, otherwise return a
            pysap.Image.

        Returns
        -------
        data: ndarray
            the reconstructed data.
        """
        if self.num_channels == 1:
            image = self._adj_op(coeffs, self.coeffs_shape)
        else:
            i = 1
            image = self._adj_op(coeffs[i], self.coeffs_shape[i])
            image = \
                zip(*Parallel(n_jobs=self.n_cpu)
                    (delayed(self._adj_op)
                    (coeffs[i], self.coeffs_shape[i])
                    for i in numpy.arange(self.num_channels)))
            image = numpy.asarray(image)
        if dtype == "array":
            return image.data
        return image

    def l2norm(self, shape):
        """ Compute the L2 norm.

        Parameters
        ----------
        shape: uplet
            the data shape.

        Returns
        -------
        norm: float
            the L2 norm.
        """
        # Create fake data
        shape = numpy.asarray(shape)
        shape += shape % 2
        fake_data = numpy.zeros(shape)
        fake_data[tuple(zip(shape // 2))] = 1

        # Call mr_transform
        data = self.op(fake_data)

        # Compute the L2 norm
        return numpy.linalg.norm(data)

class Identity(object):
    """ The 2D wavelet transform class.
    """
    def __init__(self, multichannel=False):
        self.multichannel = multichannel

    def op(self, data):
        self.coeffs_shape = data.shape
        return data

    def adj_op(self, coeffs):
        """ Define the wavelet adjoint operator.

        This method returns the reconsructed image.

        Parameters
        ----------
        coeffs: ndarray
            the wavelet coefficients.

        Returns
        -------
        data: ndarray
            the reconstructed data.
        """
        return coeffs

    def l2norm(self, shape):
        """ Compute the L2 norm.

        Parameters
        ----------
        shape: uplet
            the data shape.

        Returns
        -------
        norm: float
            the L2 norm.
        """
        # Create fake data
        shape = numpy.asarray(shape)
        shape += shape % 2
        fake_data = numpy.zeros(shape)
        fake_data[list(zip(shape // 2))] = 1

        # Call mr_transform
        data = self.op(fake_data)

        # Compute the L2 norm
        return numpy.linalg.norm(data)
