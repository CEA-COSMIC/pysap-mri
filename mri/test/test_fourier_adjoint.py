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
from mri.reconstruct.fourier import FFT, NonCartesianFFT
from mri.reconstruct.utils import convert_mask_to_locations
from mri.reconstruct.utils import convert_locations_to_mask
from mri.reconstruct.utils import normalize_frequency_locations
import time


class TestAdjointOperatorFourierTransform(unittest.TestCase):
    """ Test the adjoint operator of the Fourier in both for 2D and 3D.
    """
    def setUp(self):
        """ Set the number of iterations.
        """
        self.N = 64
        self.max_iter = 10
        self.num_channels = [1, 2]

    def test_normalize_frequency_locations_2D(self):
        """Test the output of the normalize frequency methods and check that it
        is indeed between [-0.5; 0.5[
        """
        for _ in range(10):
            samples = np.random.randn(128*128, 2)
            normalized_samples = normalize_frequency_locations(samples)
            self.assertFalse((normalized_samples.all() < 0.5 and
                             normalized_samples.all() >= -0.5))
        print(" Test normalization function for 2D input passes")

    def test_normalize_frequency_locations_3D(self):
        """Test the output of the normalize frequency methods and check that it
        is indeed between [-0.5; 0.5[
        """
        for _ in range(10):
            samples = np.random.randn(128*128, 3)
            normalized_samples = normalize_frequency_locations(samples)
            self.assertFalse((normalized_samples.all() < 0.5 and
                             normalized_samples.all() >= -0.5))
        print(" Test normalization function for 3D input passes")

    def test_sampling_converters(self):
        """Test the adjoint operator for the 2D non-Cartesian Fourier transform
        """
        for i in range(self.max_iter):
            print("Process test convert mask to samples test '{0}'...", i)
            Nx = np.random.randint(8, 512)
            Ny = np.random.randint(8, 512)
            mask = np.random.randint(2, size=(Nx, Ny))
            samples = convert_mask_to_locations(mask)
            recovered_mask = convert_locations_to_mask(samples,
                                                       (Nx, Ny))
            self.assertEqual(mask.all(), recovered_mask.all())
            mismatch = 0. + (np.mean(
                np.allclose(mask, recovered_mask)))
            print("      mismatch = ", mismatch)
        print(" Test convert mask to samples and it's adjoint passes for",
              " the 2D cases")

    def test_sampling_converters_3D(self):
        """Test the adjoint operator for the 3D non-Cartesian Fourier
        transform
        """
        for i in range(self.max_iter):
            print("Process test convert mask to samples test '{0}'...", i)
            Nx = np.random.randint(8, 512)
            Ny = np.random.randint(8, 512)
            Nz = np.random.randint(8, 512)
            mask = np.random.randint(2, size=(Nx, Ny, Nz))
            samples = convert_mask_to_locations(mask)
            recovered_mask = convert_locations_to_mask(samples,
                                                       (Nx, Ny, Nz))
            self.assertEqual(mask.all(), recovered_mask.all())
            mismatch = 0. + (np.mean(
                np.allclose(mask, recovered_mask)))
            print("      mismatch = ", mismatch)
        print(" Test convert mask to samples and it's adjoint passes for",
              " the 3D cases")

    def test_FFT(self):
        """Test the adjoint operator for the 2D non-Cartesian Fourier transform
        """
        for i in range(self.max_iter):
            _mask = np.random.randint(2, size=(self.N, self.N))
            _samples = convert_mask_to_locations(_mask)
            print("Process FFT test '{0}'...", i)
            fourier_op_dir = FFT(samples=_samples, shape=(self.N, self.N))
            fourier_op_adj = FFT(samples=_samples, shape=(self.N, self.N))
            Img = (np.random.randn(self.N, self.N) +
                   1j * np.random.randn(self.N, self.N))
            f = (np.random.randn(self.N, self.N) +
                 1j * np.random.randn(self.N, self.N))
            f_p = fourier_op_dir.op(Img)
            I_p = fourier_op_adj.adj_op(f)
            x_d = np.vdot(Img, I_p)
            x_ad = np.vdot(f_p, f)
            np.testing.assert_allclose(x_d, x_ad, rtol=1e-10)
        print(" FFT adjoint test passes")

    def test_NFFT_2D(self):
        """Test the adjoint operator for the 2D non-Cartesian Fourier transform
        """
        for num_channels in self.num_channels:
            print("Testing with num_channels=" + str(num_channels))
            for i in range(self.max_iter):
                _mask = np.random.randint(2, size=(self.N, self.N))
                _samples = convert_mask_to_locations(_mask)
                print("Process NFFT in 2D test '{0}'...", i)
                fourier_op = NonCartesianFFT(samples=_samples,
                                             shape=(self.N, self.N),
                                             n_coils=num_channels)
                Img = np.random.randn(num_channels, self.N, self.N) + \
                    1j * np.random.randn(num_channels, self.N, self.N)
                f = np.random.randn(num_channels, _samples.shape[0]) + \
                    1j * np.random.randn(num_channels, _samples.shape[0])
                f_p = fourier_op.op(Img)
                I_p = fourier_op.adj_op(f)
                x_d = np.vdot(Img, I_p)
                x_ad = np.vdot(f_p, f)
                np.testing.assert_allclose(x_d, x_ad, rtol=1e-10)
        print(" NFFT in 2D adjoint test passes")

    def test_NFFT_3D(self):
        """Test the adjoint operator for the 3D non-Cartesian Fourier transform
        """
        for num_channels in self.num_channels:
            print("Testing with num_channels=" + str(num_channels))
            for i in range(self.max_iter):
                _mask = np.random.randint(2, size=(self.N, self.N, self.N))
                _samples = convert_mask_to_locations(_mask)
                print("Process NFFT test in 3D '{0}'...", i)
                fourier_op = NonCartesianFFT(samples=_samples,
                                             shape=(self.N, self.N, self.N),
                                             n_coils=num_channels)
                Img = np.random.randn(num_channels, self.N, self.N, self.N) + \
                    1j * np.random.randn(num_channels, self.N, self.N, self.N)
                f = np.random.randn(num_channels, _samples.shape[0]) + \
                    1j * np.random.randn(num_channels, _samples.shape[0])
                f_p = fourier_op.op(Img)
                I_p = fourier_op.adj_op(f)
                x_d = np.vdot(Img, I_p)
                x_ad = np.vdot(f_p, f)
                np.testing.assert_allclose(x_d, x_ad, rtol=1e-10)
        print(" NFFT in 3D adjoint test passes")


if __name__ == "__main__":
    unittest.main()
