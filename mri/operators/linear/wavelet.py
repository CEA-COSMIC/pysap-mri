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
from ..base import OperatorBase
from modopt.signal.wavelet import get_mr_filters, filter_convolve
import pysap
from pysap.base.utils import flatten
from pysap.base.utils import unflatten
from pysap.utils import wavelist

# Third party import
import joblib
from joblib import Parallel, delayed
import numpy as np
import warnings


class WaveletN(OperatorBase):
    """ The 2D and 3D wavelet transform class.
    """

    def __init__(self, wavelet_name, nb_scale=4, verbose=0, dim=2,
                 n_coils=1, n_jobs=1, backend="threading", **kwargs):
        """ Initialize the 'WaveletN' class.

        Parameters
        ----------
        wavelet_name: str
            the wavelet name to be used during the decomposition.
        nb_scales: int, default 4
            the number of scales in the decomposition.
        n_coils: int, default 1
            the number of coils for multichannel reconstruction
        n_jobs: int, default 1
            the number of cores to use for multichannel.
        backend: str, default "threading"
            the backend to use for parallel multichannel linear operation.
        verbose: int, default 0
            the verbosity level.
        """
        self.nb_scale = nb_scale
        self.flatten = flatten
        self.unflatten = unflatten
        self.n_jobs = n_jobs
        self.n_coils = n_coils
        if self.n_coils == 1 and self.n_jobs != 1:
            print("Making n_jobs = 1 for WaveletN as n_coils = 1")
            self.n_jobs = 1
        self.backend = backend
        self.verbose = verbose
        if wavelet_name not in pysap.AVAILABLE_TRANSFORMS:
            raise ValueError(
                "Unknown transformation '{0}'.".format(wavelet_name))
        transform_klass = pysap.load_transform(wavelet_name)
        self.transform_queue = []
        n_proc = self.n_jobs
        if n_proc < 0:
            n_proc = joblib.cpu_count() + self.n_jobs + 1
        if n_proc > 0:
            if wavelet_name in wavelist()['isap-2d'] or \
                    wavelet_name in wavelist()['isap-3d']:
                warnings.warn("n_jobs is currently unsupported "
                              "for ISAP wavelets, setting n_jobs=1")
                self.n_jobs = 1
                n_proc = 1
        # Create transform queue for parallel execution
        for i in range(min(n_proc, self.n_coils)):
            self.transform_queue.append(transform_klass(
                nb_scale=self.nb_scale,
                verbose=verbose,
                dim=dim,
                **kwargs)
            )
        self.coeffs_shape = None

    def _op(self, data):
        if isinstance(data, np.ndarray):
            data = pysap.Image(data=data)
        # Get the transform from queue
        transform = self.transform_queue.pop()
        transform.data = data
        transform.analysis()
        coeffs, coeffs_shape = flatten(transform.analysis_data)
        # Add back the transform to the queue
        self.transform_queue.append(transform)
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
        if self.n_coils > 1:
            coeffs, self.coeffs_shape = zip(
                *Parallel(
                    n_jobs=self.n_jobs,
                    backend=self.backend,
                    verbose=self.verbose
                )(
                    delayed(self._op)
                    (data[i])
                    for i in np.arange(self.n_coils)
                )
            )
            coeffs = np.asarray(coeffs)
        else:
            coeffs, self.coeffs_shape = self._op(data)
        return coeffs

    def _adj_op(self, coeffs, coeffs_shape, dtype="array"):
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
        # Get the transform from queue
        transform = self.transform_queue.pop()
        transform.analysis_data = unflatten(coeffs, coeffs_shape)
        image = transform.synthesis()
        # Add back the transform to the queue
        self.transform_queue.append(transform)
        if dtype == "array":
            return image.data
        return image

    def adj_op(self, coefs):
        """ Define the wavelet adjoint operator.
        This method returns the reconstructed image.

        Parameters
        ----------
        coeffs: ndarray
            the wavelet coefficients.

        Returns
        -------
        data: ndarray
            the reconstructed data.
        """
        if self.n_coils > 1:
            images = Parallel(
                n_jobs=self.n_jobs,
                backend=self.backend,
                verbose=self.verbose
            )(
                delayed(self._adj_op)
                (coefs[i], self.coeffs_shape[i])
                for i in np.arange(self.n_coils)
            )
            images = np.asarray(images)
        else:
            images = self._adj_op(coefs, self.coeffs_shape)
        return images

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
        shape = np.asarray(shape)
        shape += shape % 2
        fake_data = np.zeros(shape)
        fake_data[tuple(zip(shape // 2))] = 1

        # Call mr_transform
        data = self.op(fake_data)

        # Compute the L2 norm
        return np.linalg.norm(data)


class WaveletUD2(OperatorBase):
    """The wavelet undecimated operator using pysap wrapper.
    """

    def __init__(self, wavelet_id=24, nb_scale=4, n_jobs=1,
                 backend='threading', n_coils=1, verbose=0):
        """Init function for Undecimated wavelet transform

        Parameters
        -----------
        wavelet_id: int, default 24 = undecimated (bi-) orthogonal transform
            ID of wavelet being used
        nb_scale: int, default 4
            the number of scales in the decomposition.
        multichannel: bool, default False
            Boolean value to indicate if the incoming data is from
            multiple-channels
        n_jobs: int, default 0
            Number of CPUs to run on. Only applicable if multichannel=True.
        backend: 'threading' | 'multiprocessing', default 'threading'
            Denotes the backend to use for parallel execution across
            multiple channels.
        verbose: int, default 0
            The verbosity level for Parallel operation from joblib
        Private Variables:
            _has_run: Checks if the get_mr_filters was called already
        """
        self.wavelet_id = wavelet_id
        self.n_coils = n_coils
        self.nb_scale = nb_scale
        self.n_jobs = n_jobs
        self.backend = backend
        self.verbose = verbose
        self._opt = [
            '-t{}'.format(self.wavelet_id),
            '-n{}'.format(self.nb_scale),
        ]
        self._has_run = False
        self.coeffs_shape = None

    def _get_filters(self, shape):
        """Function to get the Wavelet coefficients of Delta[0][0].
        This function is called only once and later the
        wavelet coefficients are obtained by convolving these coefficients
        with input Data
        """
        self.transform = get_mr_filters(
            tuple(shape),
            opt=self._opt,
            coarse=True,
        )
        self._has_run = True

    def _op(self, data):
        """ Define the wavelet operator for single channel.
        This is internal function that returns wavelet coefficients for a
        single channel
        Parameters
        ----------
        data: ndarray or Image
            input 2D data array.
        Returns
        -------
        coeffs: ndarray
            the wavelet coefficients.
        """
        coefs_real = filter_convolve(data.real, self.transform)
        coefs_imag = filter_convolve(data.imag, self.transform)
        coeffs, coeffs_shape = flatten(
            coefs_real + 1j * coefs_imag)
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
        if not self._has_run:
            if self.n_coils > 1:
                self._get_filters(list(data.shape)[1:])
            else:
                self._get_filters(data.shape)
        if self.n_coils > 1:
            coeffs, self.coeffs_shape = zip(*Parallel(n_jobs=self.n_jobs,
                                                      backend=self.backend,
                                                      verbose=self.verbose)(
                delayed(self._op)
                (data[i])
                for i in np.arange(self.n_coils)))
            coeffs = np.asarray(coeffs)
        else:
            coeffs, self.coeffs_shape = self._op(data)
        return coeffs

    def _adj_op(self, coefs, coeffs_shape):
        """" Define the wavelet adjoint operator.
        This method returns the reconstructed image for single channel.

        Parameters
        ----------
        coeffs: ndarray
            the wavelet coefficients.
        coeffs_shape: ndarray
            The shape of coefficients to unflatten before adjoint operation

        Returns
        -------
        data: ndarray
            the reconstructed data.
        """
        data_real = filter_convolve(
            np.squeeze(unflatten(coefs.real, coeffs_shape)),
            self.transform, filter_rot=True)
        data_imag = filter_convolve(
            np.squeeze(unflatten(coefs.imag, coeffs_shape)),
            self.transform, filter_rot=True)
        return data_real + 1j * data_imag

    def adj_op(self, coefs):
        """ Define the wavelet adjoint operator.
        This method returns the reconstructed image.

        Parameters
        ----------
        coeffs: ndarray
            the wavelet coefficients.

        Returns
        -------
        data: ndarray
            the reconstructed data.
        """
        if not self._has_run:
            raise RuntimeError(
                "`op` must be run before `adj_op` to get the data shape",
            )
        if self.n_coils > 1:
            images = Parallel(n_jobs=self.n_jobs,
                              backend=self.backend,
                              verbose=self.verbose)(
                delayed(self._adj_op)
                (coefs[i], self.coeffs_shape[i])
                for i in np.arange(self.n_coils))
            images = np.asarray(images)
        else:
            images = self._adj_op(coefs, self.coeffs_shape)
        return images

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
        shape = np.asarray(shape)
        shape += shape % 2
        fake_data = np.zeros(shape)
        fake_data[tuple(zip(shape // 2))] = 1

        # Call mr_transform
        data = self.op(fake_data)

        # Compute the L2 norm
        return np.linalg.norm(data)
