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

    def test_NUFFT_CUDA_3D(self):
        """Test the adjoint operator for the 3D non-Cartesian Fourier transform
        on GPU
        """
        _mask = np.random.randint(2, size=(self.N, self.N, self.N))
        _samples = convert_mask_to_locations(_mask)
        fourier_op_dir = NUFFT(samples=_samples,
                               shape=(self.N, self.N, self.N),
                               platform='cuda')
        Img = (np.random.randn(self.N, self.N, self.N) +
               1j * np.random.randn(self.N, self.N, self.N))
        f = (np.random.randn(_samples.shape[0], 1) +
             1j * np.random.randn(_samples.shape[0], 1))
        f_p = fourier_op_dir.op(Img)
        I_p = fourier_op_dir.adj_op(f)
        x_d = np.vdot(Img, I_p)
        x_ad = np.vdot(f_p, f)
        np.testing.assert_allclose(x_d, x_ad, rtol=1e-5)
        print(" NFFT in 3D adjoint test passes on GPU with CUDA")

    def test_NUFFT_CUDA_2D(self):
        """Test the adjoint operator for the 2D non-Cartesian Fourier transform
        on GPU
        """
        _mask = np.random.randint(2, size=(self.N, self.N))
        _samples = convert_mask_to_locations(_mask)
        fourier_op_adj = NUFFT(samples=_samples,
                               shape=(self.N, self.N),
                               platform='cuda')
        Img = (np.random.randn(self.N, self.N) +
               1j * np.random.randn(self.N, self.N))
        f = (np.random.randn(_samples.shape[0], 1) +
             1j * np.random.randn(_samples.shape[0], 1))
        f_p = fourier_op_adj.op(Img)
        I_p = fourier_op_adj.adj_op(f)
        x_d = np.vdot(Img, I_p)
        x_ad = np.vdot(f_p, f)
        np.testing.assert_allclose(x_d, x_ad, rtol=1e-5)
        print(" NFFT in 2D adjoint test passes on GPU with CUDA")

    def test_NUFFT_openCL_3D(self):
        """Test the adjoint operator for the 3D non-Cartesian Fourier transform
        on opencl
        """
        _mask = np.random.randint(2, size=(self.N, self.N, self.N))
        _samples = convert_mask_to_locations(_mask)
        fourier_op_dir = NUFFT(platform='cuda', samples=_samples,
                               shape=(self.N, self.N, self.N))
        Img = (np.random.randn(self.N, self.N, self.N) +
               1j * np.random.randn(self.N, self.N, self.N))
        f = (np.random.randn(_samples.shape[0], 1) +
             1j * np.random.randn(_samples.shape[0], 1))
        f_p = fourier_op_dir.op(Img)
        I_p = fourier_op_dir.adj_op(f)
        x_d = np.vdot(Img, I_p)
        x_ad = np.vdot(f_p, f)
        np.testing.assert_allclose(x_ad, x_d, rtol=1e-5)
        print(" NUFFT in 3D adjoint test passes with openCL")

    def test_NUFFT_openCL_2D(self):
        """Test the adjoint operator for the 2D non-Cartesian Fourier transform
        on opencl
        """
        _mask = np.random.randint(2, size=(self.N, self.N))
        _samples = convert_mask_to_locations(_mask)
        fourier_op_adj = NUFFT(platform='opencl', samples=_samples,
                               shape=(self.N, self.N))
        Img = (np.random.randn(self.N, self.N) +
               1j * np.random.randn(self.N, self.N))
        f = (np.random.randn(_samples.shape[0], 1) +
             1j * np.random.randn(_samples.shape[0], 1))
        f_p = fourier_op_adj.op(Img)
        I_p = fourier_op_adj.adj_op(f)
        x_d = np.vdot(Img, I_p)
        x_ad = np.vdot(f_p, f)
        del fourier_op_adj
        np.testing.assert_allclose(x_ad, x_d, rtol=1e-5)
        print(" NUFFT in 2D adjoint test passes with openCL")


if __name__ == "__main__":
    unittest.main()
