# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

from modopt.opt.proximity import ProximityParent, OrderedWeightedL1Norm

import numpy as np
from joblib import Parallel, delayed


class OWL(ProximityParent):
    """This class handles reshaping coefficients based on mode
    and feeding in right format the OWL operation to OrderedWeightedL1Norm

    Parameters
    ----------
    alpha: float
        value of alpha for parameterizing weights
    beta: float
        value of beta for parameterizing weights
    band_shape: list of tuples
        the shape of all bands, this corresponds to linear_op.coeffs_shape
    n_coils: int
        number of channels
    mode: string 'all' | 'band_based' | 'coeff_based', default 'band_based'
        Mode of operation of proximity:
        all         -> on all coefficients in all channels
        band_based  -> on all coefficients in each band
        coeff_based -> on all coefficients but across each channel
    n_jobs: int, default 1
        number of cores to be used for operation
    """
    def __init__(self, alpha, beta, bands_shape, n_coils,
                 mode='band_based', n_jobs=1):
        self.mode = mode
        self.n_jobs = n_jobs
        self.n_coils = n_coils
        if n_coils < 1:
            raise ValueError('Number of channels must be strictly positive')
        elif n_coils > 1:
            self.band_shape = bands_shape[0]
        else:
            self.band_shape = bands_shape
        self.band_sizes = np.prod(self.band_shape, axis=1)
        if self.mode == 'all':
            data_shape = np.sum(self.band_sizes)
            weights = self._oscar_weights(
                alpha,
                beta,
                data_shape * self.n_coils
            )
            self.owl_operator = OrderedWeightedL1Norm(weights)
        elif self.mode == 'band_based':
            self.owl_operator = []
            for band_size in self.band_sizes:
                weights = self._oscar_weights(
                    alpha,
                    beta,
                    self.n_coils * band_size,
                )
                self.owl_operator.append(OrderedWeightedL1Norm(weights))
        elif self.mode == 'scale_based':
            self.owl_operator = []
            for scale_band_size in np.unique(self.band_sizes):
                weights = self._oscar_weights(
                    alpha,
                    beta,
                    self.n_coils * scale_band_size *
                    np.sum(scale_band_size == self.band_sizes)
                )
                self.owl_operator.append(OrderedWeightedL1Norm(weights))
        elif self.mode == 'coeff_based':
            weights = self._oscar_weights(alpha, beta, self.n_coils)
            self.owl_operator = OrderedWeightedL1Norm(weights)
        else:
            raise ValueError('Unknow mode, please choose between `all` or '
                             '`band_based` or `coeff_based` or `scale_based`')
        self.weights = self.owl_operator
        self.op = self._op_method
        self.cost = self._cost_method

    @staticmethod
    def _oscar_weights(alpha, beta, size):
        """Here we parametrize weights based on alpha and beta"""
        w = np.arange(size-1, -1, -1, dtype=np.float64)
        w *= beta
        w += alpha
        return w

    def _reshape_band_based(self, data):
        """Function to reshape incoming data based on bands"""
        output = []
        start = 0
        n_channel = data.shape[0]
        for band_size in self.band_sizes:
            stop = start + band_size
            output.append(np.reshape(
                data[:, start: stop],
                (n_channel * band_size)))
            start = stop
        return output

    def _reshape_scale_based(self, data):
        output = []
        start = 0
        n_channel = data.shape[0]
        for scale_size in np.unique(self.band_sizes):
            num_bands = np.sum(scale_size == self.band_sizes)
            stop = start + scale_size * num_bands
            scale_size * np.sum(scale_size == self.band_sizes)
            output.append(np.reshape(
                data[:, start:stop],
                n_channel * scale_size * num_bands,
            ))
            start = stop
        return output

    def _op_method(self, data, extra_factor=1.0):
        """
        Based on mode, reshape the coefficients and call OrderedWeightedL1Norm

        Parameters
        ----------
        data: np.ndarray
            Input array of data
        """
        if self.mode == 'all':
            output = np.reshape(
                self.owl_operator.op(data.flatten(), extra_factor),
                data.shape
            )
            return output
        elif self.mode == 'band_based' or self.mode == 'scale_based':
            if self.mode == 'band_based':
                data_r = self._reshape_band_based(data)
                sizes = self.band_sizes
            else:
                data_r = self._reshape_scale_based(data)
                sizes = np.unique(self.band_sizes)
            output = Parallel(n_jobs=self.n_jobs)(
                delayed(self.owl_operator[i].op)(
                    data_band,
                    extra_factor
                )
                for i, data_band in enumerate(data_r)
            )
            reshaped_data = np.zeros(data.shape, dtype=data.dtype)
            start = 0
            n_channel = data.shape[0]
            for band_size, band_data in zip(sizes, output):
                if self.mode == 'scale_based':
                    step_size = band_size * np.sum(
                        band_size == self.band_sizes
                    )
                else:
                    step_size = band_size
                stop = start + step_size
                reshaped_data[:, start:stop] = np.reshape(
                    band_data,
                    (n_channel, step_size)
                )
                start = stop
            output = np.asarray(reshaped_data).T
            return np.asarray(output).T
        elif self.mode == 'coeff_based':
            output = Parallel(n_jobs=self.n_jobs)(
                delayed(self.owl_operator.op)(
                    data[:, i],
                    extra_factor)
                for i in range(data.shape[1]))
            return np.asarray(output).T

    def _cost_method(self, data):
        """Cost function
        Based on mode, reshape the incoming data and call cost in
        OrderedWeightedL1Norm
        This method calculate the cost function of the proximable part.

        Parameters
        ----------
        data: np.ndarray
            Input array of the sparse code.

        Returns
        -------
        The cost of this sparse code
        """
        if self.mode == 'all':
            cost = self.owl_operator.cost(data.flatten())
        elif self.mode == 'band_based' or self.mode == 'scale_based':
            if self.mode == 'band_based':
                data_r = self._reshape_band_based(data)
            else:
                data_r = self._reshape_scale_based(data)
            output = Parallel(n_jobs=self.n_jobs)(
                delayed(self.owl_operator[i].cost)(
                    data_band)
                for i, data_band in enumerate(data_r))
            cost = np.sum(output)
        elif self.mode == 'coeff_based':
            output = Parallel(n_jobs=self.n_jobs)(
                delayed(self.owl_operator.cost)(
                    data[:, i])
                for i in range(data.shape[1]))
            cost = np.sum(output)
        return cost
