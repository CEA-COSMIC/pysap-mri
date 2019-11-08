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
from scipy.fftpack import fftshift
import unittest


# Package import
from mri.operators import FFT, NonCartesianFFT
from mri.reconstructors import SingleChannelReconstructor
from mri.operators.utils import convert_mask_to_locations
from pysap.data import get_sample_data


class TestSingleChannelReconstructor(unittest.TestCase):
    """ Test the FISTA's gradient descent.
    """

    def setUp(self):
        """ Get the data from the server.
        """
        self.images = [get_sample_data(dataset_name="mri-slice-nifti")]
        print("[info] Image loaded for test: {0}.".format(
            [im.data.shape for im in self.images]))
        self.mask = get_sample_data("mri-mask").data
        # Test a wide variety of linear operators :
        # From WaveletN
        self.decimated_wavelets = ['sym8']
        # From WaveletUD2, tested only for analysis formulation
        self.undecimated_wavelets = [24]
        self.recon_type = ['cartesian', 'non-cartesian']
        self.optimizers = ['fista', 'condatvu', 'pogm']
        print("[info] Found {0} transformations.".
              format(len(self.decimated_wavelets)))
        self.nb_scales = [4]
        self.nb_iter = 100

    def test_reconstruction(self):
        """ Test all the registered transformations.
        """
        print("Process test for SingleChannelReconstructor ::")
        for recon_type in self.recon_type:
            print("Test for " + str(recon_type) + "Reconstruction")
            for optimizer in self.optimizers:
                print("Testing optimizer : " + str(optimizer))
                for image in self.images:
                    if recon_type == 'cartesian':
                        fourier = FFT(
                            samples=convert_mask_to_locations(self.mask),
                            shape=image.shape)
                    else:
                        fourier = NonCartesianFFT(
                            samples=convert_mask_to_locations(self.mask),
                            shape=image.shape)
                    kspace_data = fourier.op(image.data)
                    print("Process test with image '{0}'...".format(
                        image.metadata["path"]))
                    for nb_scale in self.nb_scales:
                        print("- Number of scales: {0}".format(nb_scale))
                        for name in self.decimated_wavelets:
                            print("  Transform: {0}".format(name))
                            reconstructor = SingleChannelReconstructor(
                                kspace_data=kspace_data,
                                kspace_loc=convert_mask_to_locations(
                                    fftshift(self.mask)),
                                uniform_data_shape=fourier.shape,
                                wavelet_name=name,
                                mu=0, nb_scale=2, fourier_type='cartesian',
                                gradient_method="synthesis",
                                optimization_alg=optimizer,
                                verbose=0
                            )
                            x_final, costs, _ = reconstructor.reconstruct(
                                num_iterations=self.nb_iter)
                            fourier_0 = FFT(
                                samples=convert_mask_to_locations(
                                    fftshift(self.mask)),
                                shape=image.shape)
                            data_0 = fourier_0.op(np.fft.fftshift(image.data))
                            np.testing.assert_allclose(
                                x_final, np.fft.ifftshift(fourier_0.adj_op(
                                    data_0)))


if __name__ == "__main__":
    unittest.main()
