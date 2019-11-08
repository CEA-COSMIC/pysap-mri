# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import
import numpy as np
import unittest


# Package import
from mri.operators import FFT, NonCartesianFFT
from mri.reconstructors import SingleChannelReconstructor, \
    SelfCalibrationReconstructor
from mri.operators.utils import convert_mask_to_locations
from pysap.data import get_sample_data

from itertools import product


class TestReconstructor(unittest.TestCase):
    """ Test the FISTA's gradient descent.
    """

    def setUp(self):
        """ Get the data from the server.
        """
        self.num_iter = 50
        self.images = [get_sample_data(dataset_name="mri-slice-nifti")]
        print("[info] Image loaded for test: {0}.".format(
            [im.data.shape for im in self.images]))
        self.mask = get_sample_data("mri-mask").data
        # From WaveletN
        self.decimated_wavelets = ['sym8']
        # From WaveletUD2, tested only for analysis formulation
        self.undecimated_wavelets = [24]
        self.recon_type = ['cartesian', 'non-cartesian']
        self.optimizers = ['fista', 'condatvu', 'pogm']
        self.nb_scales = [4]
        self.test_cases = list(product(self.images,
                                       self.nb_scales,
                                       self.optimizers,
                                       self.recon_type,
                                       self.decimated_wavelets))
        self.test_cases += list(product(self.images,
                                        self.nb_scales,
                                        ['condatvu'],
                                        self.recon_type,
                                        self.undecimated_wavelets))

    def test_single_channel_reconstruction(self):
        """ Test all the registered transformations.
        """
        print("Process test for SingleChannelReconstructor ::")
        for i in range(len(self.test_cases)):
            print("Test Case " + str(i) + " " + str(self.test_cases[i]))
            image, nb_scale, optimizer, recon_type, name = self.test_cases[i]
            if optimizer == 'condatvu':
                formulation = "analysis"
            else:
                formulation = "synthesis"
            if recon_type == 'cartesian':
                fourier = FFT(
                    samples=convert_mask_to_locations(self.mask),
                    shape=image.shape)
            else:
                fourier = NonCartesianFFT(
                    samples=convert_mask_to_locations(self.mask),
                    shape=image.shape)
            kspace_data = fourier.op(image.data)
            reconstructor = SingleChannelReconstructor(
                kspace_data=kspace_data,
                kspace_loc=convert_mask_to_locations(self.mask),
                uniform_data_shape=fourier.shape,
                wavelet_name=name,
                mu=0,
                nb_scale=2,
                fourier_type=recon_type,
                gradient_method=formulation,
                optimization_alg=optimizer,
                verbose=0
            )
            x_final, costs, _ = reconstructor.reconstruct(
                num_iterations=self.num_iter)
            fourier_0 = FFT(
                samples=convert_mask_to_locations(self.mask),
                shape=image.shape)
            data_0 = fourier_0.op(image.data)
            np.testing.assert_allclose(
                x_final, fourier_0.adj_op(data_0))

    def test_self_calibrating_reconstruction(self):
        """ Test all the registered transformations.
        """
        self.num_channels = 2
        print("Process test for SelfCalibratingReconstructor ::")
        for i in range(len(self.test_cases)):
            print("Test Case " + str(i) + " " + str(self.test_cases[i]))
            image, nb_scale, optimizer, recon_type, name = self.test_cases[i]
            if recon_type == 'cartesian':
                # TODO fix SelfCalibrating recon for cartesian case
                print("Skipping Test case as SelfCalibrationReconstructor is "
                      "not compatible with cartesian")
                continue
            image_multichannel = np.repeat(image.data[np.newaxis],
                                           self.num_channels, axis=0)
            if optimizer == 'condatvu':
                formulation = "analysis"
            else:
                formulation = "synthesis"
            if recon_type == 'cartesian':
                fourier = FFT(
                    samples=convert_mask_to_locations(self.mask),
                    shape=image.shape,
                    n_coils=self.num_channels)
            else:
                fourier = NonCartesianFFT(
                    samples=convert_mask_to_locations(self.mask),
                    shape=image.shape,
                    n_coils=self.num_channels)
            kspace_data = fourier.op(image_multichannel)
            reconstructor = SelfCalibrationReconstructor(
                kspace_data=kspace_data,
                kspace_loc=convert_mask_to_locations(self.mask),
                uniform_data_shape=fourier.shape,
                wavelet_name=name,
                mu=0,
                nb_scale=2,
                fourier_type=recon_type,
                gradient_method=formulation,
                optimization_alg=optimizer,
                verbose=0,
                n_coils=2
            )
            x_final, costs, _ = reconstructor.reconstruct(
                num_iterations=self.num_iter)
            # TODO add checks on result
            # This is just an integration test, we dont have any checks for
            # now. Please refer to tests for extracting sensitivity maps
            # for more tests


if __name__ == "__main__":
    unittest.main()
