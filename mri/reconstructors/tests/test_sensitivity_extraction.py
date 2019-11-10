# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

import unittest
import numpy as np

# Package import
from mri.operators import NonCartesianFFT
from mri.operators.utils import convert_mask_to_locations
from mri.reconstructors.utils.extract_sensitivity_maps \
    import get_Smaps, extract_k_space_center_and_locations


class TestSensitivityExtraction(unittest.TestCase):
    """ Test the code for sensitivity extraction
    """

    def setUp(self):
        """ Initialization of variables for tests:
            N = Image size in 2D
            Nz = Number of slices
            num_channel = Number of channels
            percent = percent of kspace to extract while testing
        """
        self.N = 64
        self.Nz = 60
        self.num_channel = 3
        # Percent of k-space center
        self.percent = 0.5

    def test_extract_k_space_center_3D(self):
        """ This test ensures that the output of the non cartesian kspace
        extraction is same a that of mimicked cartesian extraction in 3D
        """
        _mask = np.ones((self.N, self.N, self.Nz))
        _samples = convert_mask_to_locations(_mask)
        Img = (np.random.randn(self.num_channel, self.N, self.N, self.Nz) +
               1j * np.random.randn(self.num_channel, self.N, self.N,
                                    self.Nz))
        Nby2_percent = self.N * self.percent / 2
        Nzby2_percent = self.Nz * self.percent / 2
        low = int(self.N / 2 - Nby2_percent)
        high = int(self.N / 2 + Nby2_percent + 1)
        lowz = int(self.Nz / 2 - Nzby2_percent)
        highz = int(self.Nz / 2 + Nzby2_percent + 1)
        center_Img = Img[:, low:high, low:high, lowz:highz]
        thresh = self.percent * 0.5
        data_thresholded, samples_thresholded = \
            extract_k_space_center_and_locations(
                data_values=np.reshape(Img, (self.num_channel,
                                             self.N * self.N * self.Nz)),
                samples_locations=_samples,
                thr=(thresh, thresh, thresh),
                img_shape=(self.N, self.N, self.Nz))
        np.testing.assert_allclose(
            center_Img.reshape(data_thresholded.shape),
            data_thresholded)

    def test_extract_k_space_center_2D(self):
        """ Ensure that the extracted k-space center is right"""
        _mask = np.ones((self.N, self.N))
        _samples = convert_mask_to_locations(_mask)
        Img = (np.random.randn(self.num_channel, self.N, self.N) +
               1j * np.random.randn(self.num_channel, self.N, self.N))
        Nby2_percent = self.N * self.percent / 2
        low = int(self.N / 2 - Nby2_percent)
        high = int(self.N / 2 + Nby2_percent + 1)
        center_Img = Img[:, low:high, low:high]
        thresh = self.percent * 0.5
        data_thresholded, samples_thresholded = \
            extract_k_space_center_and_locations(
                data_values=np.reshape(Img, (self.num_channel,
                                             self.N * self.N)),
                samples_locations=_samples,
                thr=(thresh, thresh),
                img_shape=(self.N, self.N))
        np.testing.assert_allclose(
            center_Img.reshape(data_thresholded.shape),
            data_thresholded)

    def test_sensitivity_extraction_2D(self):
        """ This test ensures that the output of the non cartesian kspace
        extraction is same a that of mimicked cartesian extraction in 2D
        """
        _mask = np.ones((self.N, self.N))
        _samples = convert_mask_to_locations(_mask)
        fourier_op = NonCartesianFFT(samples=_samples, shape=(self.N, self.N))
        Img = (np.random.randn(self.num_channel, self.N, self.N) +
               1j * np.random.randn(self.num_channel, self.N, self.N))
        F_img = np.asarray([fourier_op.op(Img[i])
                            for i in np.arange(self.num_channel)])
        Smaps_gridding, SOS_Smaps = get_Smaps(
            k_space=F_img,
            img_shape=(self.N, self.N),
            samples=_samples,
            thresh=(0.4, 0.4),
            mode='gridding',
            min_samples=(-0.5, -0.5),
            max_samples=(0.5, 0.5),
            n_cpu=1)
        Smaps_NFFT, SOS_Smaps = get_Smaps(
            k_space=F_img,
            img_shape=(self.N, self.N),
            thresh=(0.4, 0.4),
            samples=_samples,
            min_samples=(-0.5, -0.5),
            max_samples=(0.5, 0.5),
            mode='NFFT')
        np.testing.assert_allclose(Smaps_gridding, Smaps_NFFT)


if __name__ == "__main__":
    unittest.main()
