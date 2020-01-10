# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# Third-party import
import numpy as np
import unittest

# Package import
from mri.operators.fourier.cartesian import FFT
from mri.operators.fourier.non_cartesian import NonCartesianFFT, Stacked3DNFFT
from mri.operators.linear.wavelet import WaveletUD2, WaveletN
from mri.reconstructors import SingleChannelReconstructor, \
    SelfCalibrationReconstructor, CalibrationlessReconstructor
from mri.operators.utils import convert_mask_to_locations
from pysap.data import get_sample_data

from itertools import product
from modopt.opt.proximity import SparseThreshold
from modopt.opt.linear import Identity


class TestReconstructor(unittest.TestCase):
    """ Tests every reconstructor with mu=0, a value to which we know the
    solution must converge to analytical solution,
    ie the inverse fourier transform
    """
    def setUp(self):
        """ Setup common variables to be used in tests:
        num_iter : Number of iterations
        images : Ground truth images to test with, obtained from server
        mask : MRI fourier space mask
        decimated_wavelets : Decimated wavelets to test with
        undecimated_wavelets : Undecimated wavelets to test with
        optimizers : Different optimizers to test with
        nb_scales : Number of scales
        test_cases : holds the final test cases
        """
        self.num_iter = 40
        # TODO getting images from net slows down these tests,
        #  we would prefer to rather use random complex data.
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
        self.test_cases = list(product(
                self.images,
                self.nb_scales,
                self.optimizers,
                self.recon_type,
                self.decimated_wavelets,
            ))
        self.test_cases += list(product(
                self.images,
                self.nb_scales,
                ['condatvu'],
                self.recon_type,
                self.undecimated_wavelets,
            ))

    def get_linear_n_regularization_operator(
            self, gradient_formulation, wavelet_name, dimension=2, nb_scale=3,
            n_coils=1, n_jobs=1, verbose=0):
        # A helper function to obtain linear and regularization operator
        try:
            linear_op = WaveletN(
                nb_scale=nb_scale,
                wavelet_name=wavelet_name,
                dim=dimension,
                n_coils=n_coils,
                n_jobs=n_jobs,
                verbose=verbose,
            )
        except ValueError:
            # TODO this is a hack and we need to have a separate WaveletUD2.
            # For Undecimated wavelets, the wavelet_name is wavelet_id
            linear_op = WaveletUD2(
                wavelet_id=wavelet_name,
                nb_scale=nb_scale,
                n_coils=n_coils,
                n_jobs=n_jobs,
                verbose=verbose,
            )
        if gradient_formulation == 'synthesis':
            regularizer_op = SparseThreshold(Identity(), 0, thresh_type="soft")
        elif gradient_formulation == "analysis":
            regularizer_op = SparseThreshold(linear_op, 0, thresh_type="soft")
        return linear_op, regularizer_op

    def test_single_channel_reconstruction(self):
        """ Test all the registered transformations for
        single channel reconstructor.
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
            linear_op, regularizer_op = \
                self.get_linear_n_regularization_operator(
                    wavelet_name=name,
                    dimension=len(fourier.shape),
                    nb_scale=3,
                    gradient_formulation=formulation,
                )
            reconstructor = SingleChannelReconstructor(
                fourier_op=fourier,
                linear_op=linear_op,
                regularizer_op=regularizer_op,
                gradient_formulation=formulation,
                verbose=0,
            )
            x_final, costs, _ = reconstructor.reconstruct(
                kspace_data=kspace_data,
                optimization_alg=optimizer,
                num_iterations=self.num_iter,
            )
            fourier_0 = FFT(
                samples=convert_mask_to_locations(self.mask),
                shape=image.shape,
            )
            data_0 = fourier_0.op(image.data)
            # mu is 0 for above single channel reconstruction and
            # hence we expect the result to be the inverse fourier transform
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
            linear_op, regularizer_op = \
                self.get_linear_n_regularization_operator(
                    wavelet_name=name,
                    dimension=len(fourier.shape),
                    nb_scale=2,
                    n_coils=self.num_channels,
                    gradient_formulation=formulation,
                )
            # For self calibrating reconstruction the n_coils
            # for wavelet operation is 1
            linear_op.n_coils = 1
            reconstructor = SelfCalibrationReconstructor(
                fourier_op=fourier,
                linear_op=linear_op,
                regularizer_op=regularizer_op,
                gradient_formulation=formulation,
                verbose=0,
            )
            x_final, costs, _ = reconstructor.reconstruct(
                kspace_data=kspace_data,
                optimization_alg=optimizer,
                num_iterations=self.num_iter,
            )
            fourier_0 = FFT(
                samples=convert_mask_to_locations(self.mask),
                shape=image.shape,
                n_coils=self.num_channels,
            )
            recon = fourier_0.adj_op(fourier_0.op(image_multichannel))
            np.testing.assert_allclose(
                np.abs(x_final), np.sqrt(np.sum(np.abs(recon)**2, axis=0)))

    def test_sparse_calibrationless_reconstruction(self):
        """ Test all the registered transformations.
        """
        self.num_channels = 2
        print("Process test for SparseCalibrationlessReconstructor ::")
        for i in range(len(self.test_cases)):
            print("Test Case " + str(i) + " " + str(self.test_cases[i]))
            image, nb_scale, optimizer, recon_type, name = self.test_cases[i]
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
            linear_op, regularizer_op = \
                self.get_linear_n_regularization_operator(
                    wavelet_name=name,
                    dimension=len(fourier.shape),
                    nb_scale=2,
                    n_coils=2,
                    n_jobs=2,
                    gradient_formulation=formulation,
                )
            reconstructor = CalibrationlessReconstructor(
                fourier_op=fourier,
                linear_op=linear_op,
                regularizer_op=regularizer_op,
                gradient_formulation=formulation,
                verbose=1,
            )
            x_final, costs, _ = reconstructor.reconstruct(
                kspace_data=kspace_data,
                optimization_alg=optimizer,
                num_iterations=self.num_iter,
            )
            fourier_0 = FFT(
                samples=convert_mask_to_locations(self.mask),
                shape=image.shape,
                n_coils=self.num_channels,
            )
            data_0 = fourier_0.op(image_multichannel)
            # mu is 0 for above single channel reconstruction and
            # hence we expect the result to be the inverse fourier transform
            np.testing.assert_allclose(
                x_final, fourier_0.adj_op(data_0))

    def test_check_asserts(self):
        # Tests to check for asserts
        image, nb_scale, optimizer, recon_type, name = self.test_cases[0]
        fourier = NonCartesianFFT(
            samples=convert_mask_to_locations(self.mask),
            shape=image.shape,
        )
        kspace_data = fourier.op(image.data)
        linear_op, regularizer_op = \
            self.get_linear_n_regularization_operator(
                wavelet_name=name,
                dimension=len(fourier.shape),
                nb_scale=2,
                gradient_formulation="synthesis",
            )
        reconstructor = CalibrationlessReconstructor(
            fourier_op=fourier,
            linear_op=linear_op,
            regularizer_op=regularizer_op,
            gradient_formulation="synthesis",
            verbose=1,
        )
        np.testing.assert_raises(
            ValueError,
            reconstructor.reconstruct,
            kspace_data=kspace_data,
            optimization_alg="test_fail",
            num_iterations=self.num_iter,
        )
        fourier.n_coils = 10
        reconstructor = SelfCalibrationReconstructor(
            fourier_op=fourier,
            linear_op=linear_op,
            regularizer_op=regularizer_op,
            gradient_formulation="synthesis",
            verbose=1,
        )
        np.testing.assert_raises(
            ValueError,
            reconstructor.reconstruct,
            kspace_data=kspace_data,
            optimization_alg=optimizer,
            num_iterations=self.num_iter,
        )


if __name__ == "__main__":
    unittest.main()
