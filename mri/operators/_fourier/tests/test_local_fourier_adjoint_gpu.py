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
from mri.operators import NonCartesianFFT
from mri.operators.utils import convert_mask_to_locations


class TestAdjointOperatorFourierTransformGPU(unittest.TestCase):
    """ Test the adjoint operator of the NFFT both for 2D and 3D.
    """

    def setUp(self):
        """ Set the number of iterations.
        """
        self.N = 64
        self.num_channels = [1, 16]
        self.platforms = ['cuda', 'opencl']

    def test_NUFFT_3D(self):
        """Test the adjoint operator for the 3D non-Cartesian Fourier transform
        on GPU
        """
        for num_channels in self.num_channels:
            for platform in self.platforms:
                _mask = np.random.randint(2, size=(self.N, self.N, self.N))
                _samples = convert_mask_to_locations(_mask)
                fourier_op_dir = NonCartesianFFT(samples=_samples,
                                                 shape=(self.N, self.N,
                                                        self.N),
                                                 implementation=platform,
                                                 n_coils=num_channels)
                Img = (np.random.randn(num_channels, self.N, self.N, self.N) +
                       1j * np.random.randn(num_channels, self.N, self.N,
                                            self.N))
                f = (np.random.randn(num_channels, _samples.shape[0]) +
                     1j * np.random.randn(num_channels, _samples.shape[0]))
                f_p = fourier_op_dir.op(Img)
                I_p = fourier_op_dir.adj_op(f)
                x_d = np.vdot(Img, I_p)
                x_ad = np.vdot(f_p, f)
                np.testing.assert_allclose(x_d, x_ad, rtol=1e-5)
                print("NFFT in 3D adjoint test passes on GPU with"
                      "num_channels = " + str(num_channels) + " on "
                      "platform " + platform)

    def test_NUFFT_2D(self):
        """Test the adjoint operator for the 2D non-Cartesian Fourier transform
        on GPU
        """
        for num_channels in self.num_channels:
            for platform in self.platforms:
                _mask = np.random.randint(2, size=(self.N, self.N))
                _samples = convert_mask_to_locations(_mask)
                fourier_op_adj = NonCartesianFFT(samples=_samples,
                                                 shape=(self.N, self.N),
                                                 implementation=platform,
                                                 n_coils=num_channels)
                Img = (np.random.randn(num_channels, self.N, self.N) +
                       1j * np.random.randn(num_channels, self.N, self.N))
                f = (np.random.randn(num_channels, _samples.shape[0], 1) +
                     1j * np.random.randn(num_channels, _samples.shape[0], 1))
                f_p = fourier_op_adj.op(Img)
                I_p = fourier_op_adj.adj_op(f)
                x_d = np.vdot(Img, I_p)
                x_ad = np.vdot(f_p, f)
                np.testing.assert_allclose(x_d, x_ad, rtol=1e-5)
                print("NFFT in 2D adjoint test passes on GPU with"
                      "num_channels = " + str(num_channels) + " on "
                      "platform" + platform)


if __name__ == "__main__":
    unittest.main()
