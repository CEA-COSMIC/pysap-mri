# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import
from __future__ import print_function
import unittest
import numpy

# Package import
from mri.reconstruct.fourier import NUFFT
from mri.reconstruct.utils import convert_mask_to_locations


class TestAdjointOperatorFourierTransformGPU(unittest.TestCase):
    """ Test the adjoint operator of the NFFT both for 2D and 3D.
    """
    def setUp(self):
        """ Set the number of iterations.
        """
        self.N = 64
        self.max_iter = 10

    def test_NUFFT_3D(self):
        """Test the adjoint operator for the 3D non-Cartesian Fourier transform
        on GPU
        """
        for i in range(self.max_iter):
            _mask = numpy.random.randint(2, size=(self.N, self.N, self.N))
            _samples = convert_mask_to_locations(_mask)
            print("Process NFFT test in 3D '{0}'...", i)
            fourier_op_dir = NUFFT(samples=_samples,
                                   shape=(self.N, self.N, self.N),
                                   platform='gpu')
            Img = numpy.random.randn(self.N, self.N, self.N) + \
                1j * numpy.random.randn(self.N, self.N, self.N)
            f = numpy.random.randn(_samples.shape[0], 1) + \
                1j * numpy.random.randn(_samples.shape[0], 1)
            f_p = fourier_op_dir.op(Img)
            I_p = fourier_op_dir.adj_op(f)
            x_d = numpy.vdot(Img, I_p)
            x_ad = numpy.vdot(f_p, f)
            numpy.testing.assert_allclose(x_d, x_ad, rtol=1e-3)
        print(" NFFT in 3D adjoint test passes")

    def test_NUFFT_2D(self):
        """Test the adjoint operator for the 2D non-Cartesian Fourier transform
        on GPU
        """
        for i in range(self.max_iter):
            _mask = numpy.random.randint(2, size=(self.N, self.N))
            _samples = convert_mask_to_locations(_mask)
            print("Process NFFT in 2D test '{0}'...", i)
            fourier_op_adj = NUFFT(samples=_samples,
                                   shape=(self.N, self.N),
                                   platform='gpu')
            Img = numpy.random.randn(self.N, self.N) + \
                1j * numpy.random.randn(self.N, self.N)
            f = numpy.random.randn(_samples.shape[0], 1) + \
                1j * numpy.random.randn(_samples.shape[0], 1)
            f_p = fourier_op_adj.op(Img)
            I_p = fourier_op_adj.adj_op(f)
            x_d = numpy.vdot(Img, I_p)
            x_ad = numpy.vdot(f_p, f)
            numpy.testing.assert_allclose(x_d, x_ad, rtol=1e-10)
        print(" NFFT in 2D adjoint test passes")


if __name__ == "__main__":
    unittest.main()
