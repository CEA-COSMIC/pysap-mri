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
from itertools import product

# Package import
from mri.operators import (FFT,
                           NonCartesianFFT,
                           Stacked3DNFFT,
                           ORCFFTWrapper)
from mri.operators.utils import convert_mask_to_locations, \
    convert_locations_to_mask, normalize_frequency_locations, \
    get_stacks_fourier


class TestORCFourierWrapper(unittest.TestCase):
    """ Test the ORC wrapper in Cartesian and non-Cartesian, 2D and 3D cases.
    """
    def setUp(self):
        """ Set the field map, shapes and correction variables.
        """
        self.L = [1, 2]
        self.n_bins = 1
        self.rtol = 1e-7

        self.N = 64
        self.max_iter = 5
        self.n_coils = [1, 2]

    # Cartesian tests
    def generate_test_FFT(self, shape, field_scale):
        """ Factorized code to test 2D and 3D wrapped FFT with
        different homogeneous B0 field shifts at constant time.
        """
        for L, i, n_coils in product(self.L, range(self.max_iter),
                                     self.n_coils):
            mask = np.random.randint(2, size=shape)
            field_shift = field_scale * np.random.randint(-150, 150)
            field_map = field_shift * np.ones(shape)

            # Prepare reference and wrapper operators
            fourier_op = FFT(mask=mask, shape=shape, n_coils=n_coils)
            wrapper_op = ORCFFTWrapper(fourier_op, field_map=field_map,
                                       time_vec=np.ones(shape[0]),
                                       mask=np.ones(shape),
                                       num_interpolators=L,
                                       n_bins=self.n_bins)

            # Forward operator
            img = np.squeeze(np.random.randn(n_coils, *shape) \
                      + 1j * np.random.randn(n_coils, *shape))
            ksp_fft = fourier_op.op(img)
            ksp_wra = wrapper_op.op(img * np.exp(-2j * np.pi * field_shift))
            np.testing.assert_allclose(ksp_fft, ksp_wra, rtol=self.rtol)

            # Adjoint operator
            ksp = np.squeeze(np.random.randn(n_coils, *shape) \
                      + 1j * np.random.randn(n_coils, *shape))
            img_fft = fourier_op.adj_op(ksp)
            img_wra = wrapper_op.adj_op(ksp * np.exp(2j * np.pi * field_shift))
            np.testing.assert_allclose(img_fft, img_wra, rtol=self.rtol)

    def test_FFT_2D(self):
        """ Test forward and adjoint operators for wrapped 2D Cartesian FFT
        with homogeneous B0 field at constant time.
        """
        print("Process test for 2D Cartesian FFT.")
        shape = (self.N, self.N // 2)
        self.generate_test_FFT(shape, 0)

    def test_FFT_2D_shifted(self):
        """ Test forward and adjoint operators for wrapped 2D Cartesian FFT
        with shifted homogeneous B0 field at constant time.
        """
        print("Process test for 2D Cartesian FFT with shifted field.")
        shape = (self.N, self.N // 2)
        self.generate_test_FFT(shape, 20e-3)

    def test_FFT_3D(self):
        """ Test forward and adjoint operators for wrapped 3D Cartesian FFT
        with homogeneous B0 field at constant time.
        """
        print("Process test for 3D Cartesian FFT.")
        shape = (self.N, self.N // 2, self.N // 4)
        self.generate_test_FFT(shape, 0)

    def test_FFT_3D_shifted(self):
        """ Test forward and adjoint operators for wrapped 3D Cartesian FFT
        with shifted homogeneous B0 field at constant time.
        """
        print("Process test for 3D Cartesian FFT with shifted field.")
        shape = (self.N, self.N // 2, self.N // 4)
        self.generate_test_FFT(shape, 20e-3)

    # Non-Cartesian tests
    def generate_test_NFFT(self, shape, field_scale):
        """ Factorized code to test 2D and 3D wrapped NFFT with
        different homogeneous B0 field shifts at constant time.
        """
        for L, i, n_coils in product(self.L, range(self.max_iter),
                                     self.n_coils):
            mask = np.random.randint(2, size=shape)
            samples = convert_mask_to_locations(mask)
            samples = samples[:samples.shape[0]
                            - (samples.shape[0] % shape[0])]

            field_shift = field_scale * np.random.randint(-150, 150)
            field_map = field_shift * np.ones(shape)

            # Prepare reference and wrapper operators
            fourier_op = NonCartesianFFT(
                samples=samples,
                shape=shape,
                n_coils=n_coils,
                implementation='cpu',
                density_comp=np.ones((n_coils, samples.shape[0]))
            )
            wrapper_op = ORCFFTWrapper(fourier_op, field_map=field_map,
                                       time_vec=np.ones(shape[0]),
                                       mask=np.ones(shape),
                                       num_interpolators=L,
                                       n_bins=self.n_bins)

            # Forward operator
            img = np.squeeze(np.random.randn(n_coils, *shape) \
                      + 1j * np.random.randn(n_coils, *shape))
            ksp_fft = fourier_op.op(img)
            ksp_wra = wrapper_op.op(img * np.exp(-2j * np.pi * field_shift))
            np.testing.assert_allclose(ksp_fft, ksp_wra, rtol=self.rtol)

            # Adjoint operator
            ksp = np.squeeze(np.random.randn(n_coils, samples.shape[0]) \
                      + 1j * np.random.randn(n_coils, samples.shape[0]))
            img_fft = fourier_op.adj_op(ksp)
            img_wra = wrapper_op.adj_op(ksp * np.exp(2j * np.pi * field_shift))
            np.testing.assert_allclose(img_fft, img_wra, rtol=self.rtol)

    def test_NFFT_2D(self):
        """ Test forward and adjoint operators for wrapped 2D non-Cartesian
        FFT with homogeneous B0 field at constant time.
        """
        print("Process test for 2D non-Cartesian FFT.")
        shape = (self.N, self.N // 2)
        self.generate_test_NFFT(shape, 0)

    def test_NFFT_2D_shifted(self):
        """ Test forward and adjoint operators for wrapped 2D non-Cartesian
        FFT with shifted homogeneous B0 field at constant time.
        """
        print("Process test for 2D non-Cartesian FFT with shifted field.")
        shape = (self.N, self.N // 2)
        self.generate_test_NFFT(shape, 20e-3)

    def test_NFFT_3D(self):
        """ Test forward and adjoint operators for wrapped 3D non-Cartesian
        FFT with homogeneous B0 field at constant time.
        """
        print("Process test for 3D non-Cartesian FFT.")
        shape = (self.N, self.N // 2, self.N // 4)
        self.generate_test_NFFT(shape, 0)

    def test_NFFT_3D_shifted(self):
        """ Test forward and adjoint operators for wrapped 3D non-Cartesian
        FFT with shifted homogeneous B0 field at constant time.
        """
        print("Process test for 3D non-Cartesian FFT with shifted field.")
        shape = (self.N, self.N // 2, self.N // 4)
        self.generate_test_NFFT(shape, 20e-3)

    def generate_test_stacked_NFFT(self, shape, field_scale):
        """ Factorized code to test 3D-stacked wrapped NFFT with
        different homogeneous B0 field shifts at constant time.
        """
        for L, i, n_coils in product(self.L, range(self.max_iter),
                                     self.n_coils):
            mask = np.random.randint(2, size=shape[:-1])
            mask = np.tile(mask[..., None], (1, 1, shape[-1]))
            samples = convert_mask_to_locations(mask)
            samples = samples[:samples.shape[0] \
                            - (samples.shape[0] % shape[0])]

            field_shift = field_scale * np.random.randint(-150, 150)
            field_map = field_shift * np.ones(shape)

            # Prepare reference and wrapper operators
            fourier_op = Stacked3DNFFT(
                kspace_loc=samples,
                shape=shape,
                implementation='cpu',
                n_coils=n_coils,
            )
            wrapper_op = ORCFFTWrapper(fourier_op, field_map=field_map,
                                       time_vec=np.ones(shape[0]),
                                       mask=np.ones(shape),
                                       num_interpolators=L,
                                       n_bins=self.n_bins)

            # Forward operator
            img = np.squeeze(np.random.randn(n_coils, *shape) \
                      + 1j * np.random.randn(n_coils, *shape))
            ksp_fft = fourier_op.op(img)
            ksp_wra = wrapper_op.op(img * np.exp(-2j * np.pi * field_shift))
            np.testing.assert_allclose(ksp_fft, ksp_wra, rtol=self.rtol)

            # Adjoint operator
            ksp = np.squeeze(np.random.randn(n_coils, samples.shape[0]) \
                      + 1j * np.random.randn(n_coils, samples.shape[0]))
            img_fft = fourier_op.adj_op(ksp)
            img_wra = wrapper_op.adj_op(ksp * np.exp(2j * np.pi * field_shift))
            np.testing.assert_allclose(img_fft, img_wra, rtol=self.rtol)

    def test_stacked_NFFT_3D(self):
        """ Test forward and adjoint operators for wrapped 3D-stacked
        non-Cartesian FFT with homogeneous B0 field at constant time.
        """
        print("Process test for 3D-stacked non-Cartesian FFT.")
        shape = (self.N, self.N // 2, self.N // 4)
        self.generate_test_stacked_NFFT(shape, 0)

    def test_stacked_NFFT_3D_shifted(self):
        """ Test forward and adjoint operators for wrapped 3D-stacked
        non-Cartesian FFT with shifted homogeneous B0 field at constant time.
        """
        print("Process test for 3D-stacked non-Cartesian FFT",
              "with shifted field.")
        shape = (self.N, self.N // 2, self.N // 4)
        self.generate_test_stacked_NFFT(shape, 20e-3)


if __name__ == "__main__":
    unittest.main()
