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
from mri.operators import WaveletN, WaveletUD2


class TestAdjointOperatorWaveletTransform(unittest.TestCase):
    """ Test the adjoint operator of the Wavelets both for 2D and 3D.
    """

    def setUp(self):
        """ Setup variables:
        N = Image size
        max_iter = Number of iterations to test
        num_channels = Number of channels to be tested with for
                        multichannel tests
        """
        self.N = 64
        self.max_iter = 10
        self.num_channels = 10

    def test_Wavelet2D_ISAP(self):
        """Test the adjoint operator for the 2D Wavelet transform
        """
        for i in range(self.max_iter):
            print("Process Wavelet2D_ISAP test '{0}'...", i)
            wavelet_op_adj = WaveletN(wavelet_name="HaarWaveletTransform",
                                      nb_scale=4)
            Img = (np.random.randn(self.N, self.N) +
                   1j * np.random.randn(self.N, self.N))
            f_p = wavelet_op_adj.op(Img)
            f = (np.random.randn(*f_p.shape) +
                 1j * np.random.randn(*f_p.shape))
            I_p = wavelet_op_adj.adj_op(f)
            x_d = np.vdot(Img, I_p)
            x_ad = np.vdot(f_p, f)
            np.testing.assert_allclose(x_d, x_ad, rtol=1e-6)
        print(" Wavelet2 adjoint test passes")

    def test_Wavelet2D_PyWt(self):
        """Test the adjoint operator for the 2D Wavelet transform
        """
        for i in range(self.max_iter):
            print("Process Wavelet2D PyWt test '{0}'...", i)
            wavelet_op_adj = WaveletN(wavelet_name="sym8",
                                      nb_scale=4)
            Img = (np.random.randn(self.N, self.N) +
                   1j * np.random.randn(self.N, self.N))
            f_p = wavelet_op_adj.op(Img)
            f = (np.random.randn(*f_p.shape) +
                 1j * np.random.randn(*f_p.shape))
            I_p = wavelet_op_adj.adj_op(f)
            x_d = np.vdot(Img, I_p)
            x_ad = np.vdot(f_p, f)
            np.testing.assert_allclose(x_d, x_ad, rtol=1e-6)
        print(" Wavelet2 adjoint test passes")

    def test_Wavelet3D_PyWt(self):
        """Test the adjoint operator for the 3D Wavelet transform
        """
        for i in range(self.max_iter):
            print("Process Wavelet3D PyWt test '{0}'...", i)
            wavelet_op_adj = WaveletN(wavelet_name="sym8",
                                      nb_scale=4, dim=3,
                                      padding_mode='periodization')
            Img = (np.random.randn(self.N, self.N, self.N) +
                   1j * np.random.randn(self.N, self.N, self.N))
            f_p = wavelet_op_adj.op(Img)
            f = (np.random.randn(*f_p.shape) +
                 1j * np.random.randn(*f_p.shape))
            I_p = wavelet_op_adj.adj_op(f)
            x_d = np.vdot(Img, I_p)
            x_ad = np.vdot(f_p, f)
            np.testing.assert_allclose(x_d, x_ad, rtol=1e-6)
        print(" Wavelet3 adjoint test passes")

    def test_Wavelet_UD_2D(self):
        """Test the adjoint operation for Undecimated wavelet
        """
        for i in range(self.max_iter):
            print("Process Wavelet Undecimated test '{0}'...", i)
            wavelet_op = WaveletUD2(nb_scale=4)
            img = (np.random.randn(self.N, self.N) +
                   1j * np.random.randn(self.N, self.N))
            f_p = wavelet_op.op(img)
            f = (np.random.randn(*f_p.shape) +
                 1j * np.random.randn(*f_p.shape))
            i_p = wavelet_op.adj_op(f)
            x_d = np.vdot(img, i_p)
            x_ad = np.vdot(f_p, f)
            np.testing.assert_allclose(x_d, x_ad, rtol=1e-6)
        print("Undecimated Wavelet 2D adjoint test passes")

    def test_Wavelet_UD_2D_Multichannel(self):
        """Test the adjoint operation for Undecmated wavelet Transform in
        multichannel case"""
        for i in range(self.max_iter):
            print("Process Wavelet Undecimated test '{0}'...", i)
            wavelet_op = WaveletUD2(
                nb_scale=4,
                n_coils=self.num_channels,
                n_jobs=2
            )
            img = (np.random.randn(self.num_channels, self.N, self.N) +
                   1j * np.random.randn(self.num_channels, self.N, self.N))
            f_p = wavelet_op.op(img)
            f = (np.random.randn(*f_p.shape) +
                 1j * np.random.randn(*f_p.shape))
            i_p = wavelet_op.adj_op(f)
            x_d = np.vdot(img, i_p)
            x_ad = np.vdot(f_p, f)
            np.testing.assert_allclose(x_d, x_ad, rtol=1e-6)
        print("Undecimated Wavelet 2D adjoint test passes for multichannel")


if __name__ == "__main__":
    unittest.main()
