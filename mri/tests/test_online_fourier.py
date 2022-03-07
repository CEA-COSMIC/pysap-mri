# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import

import unittest
import numpy as np

# Package import
from mri.operators.fourier.online import ColumnFFT
from mri.operators.fourier.cartesian import FFT


class TestOnlineFourierOperator(unittest.TestCase):
    """Test the Online Fourier Operator in 2D"""

    def setUp(self):

        self.n_coils = (0, 2)
        self.shape = (64, 64)

    def test_columnFFT_forward(self):
        """Test the forward operator of ColumnFFT."""
        column_indexes = np.arange(64)
        mask = np.ones(self.shape)
        for nc in self.n_coils:
            fourier_op = FFT(mask=mask, shape=self.shape, n_coils=nc)
            column_fft = ColumnFFT(shape=self.shape, line_index=0, n_coils=nc)
            data = np.squeeze(np.random.rand(fourier_op.n_coils, *self.shape))
            k_data = fourier_op.op(data)
            for col in column_indexes:
                column_fft.mask = col
                np.testing.assert_allclose(
                    column_fft.op(data), k_data[..., column_fft.mask])
        print("Test forward operator of columnFFT")

    def test_columnFFT_adjoint(self):
        """Test the adjoint operator of column FFT. """
        column_indexes = np.arange(0, 64)
        mask = np.ones(self.shape)
        for nc in self.n_coils:
            fft2d = FFT(mask=mask, shape=self.shape, n_coils=nc)
            column_fft = ColumnFFT(shape=self.shape, line_index=0, n_coils=nc)
            k_data = np.squeeze(np.random.rand(fft2d.n_coils, *self.shape)
                                + 1j * np.random.rand(fft2d.n_coils, *self.shape))
            for col in column_indexes:
                column_fft.mask = col
                mask = np.zeros(self.shape)
                mask[:, column_fft.mask] = 1
                fft2d.mask = mask
                print(np.nonzero(fft2d.mask[0, :]))
                print(column_fft.mask)
                print(column_fft.adj_op(k_data[..., column_fft._mask])[0, :5])
                print(fft2d.adj_op(k_data)[0, :5])
                np.testing.assert_allclose(column_fft.adj_op(k_data[..., column_fft._mask]),
                                           fft2d.adj_op(k_data))
        print("Test adjoint operator of columnFFT")

    def test_column_outside_range(self):
        column_fft = ColumnFFT(shape=self.shape, line_index=0, n_coils=1)
        self.assertRaises(IndexError, lambda: setattr(column_fft, 'mask', self.shape[1]+1))

if __name__ == "__main__":
    unittest.main()
