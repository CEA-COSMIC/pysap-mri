# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

from mri.operators import NonCartesianFFT, Stacked3DNFFT, FFT
from mri.operators import WaveletN, WaveletUD2


class ReconstructorBase(object):
    def __init__(self, kspace_loc, uniform_data_shape, n_coils,
                 fourier_type, nfft_implementation, verbose):
        # Define the linear/fourier operators
        if fourier_type == 'non-cartesian':
            self.fourier_op = NonCartesianFFT(
                samples=kspace_loc,
                shape=uniform_data_shape,
                implementation=nfft_implementation,
                n_coils=n_coils)
        elif fourier_type == 'cartesian':
            self.fourier_op = FFT(
                samples=kspace_loc,
                shape=uniform_data_shape,
                n_coils=n_coils)
        elif fourier_type == 'stack':
            self.fourier_op = Stacked3DNFFT(kspace_loc=kspace_loc,
                                            shape=uniform_data_shape,
                                            implementation=nfft_implementation,
                                            n_coils=n_coils)
        else:
            raise ValueError('The value of fourier_type must be "cartesian" | '
                             '"non-cartesian" | "stack"')
        if verbose >= 5:
            print("Initialized fourier operator : " + str(self.fourier_op))

    def reconstruct(self, x_init=None):
        """ This method calculates operator transform.
        Parameters
        ----------
        x_init: np.ndarray (optional, default None)
            input initial guess image for reconstruction
        """
        raise NotImplementedError("'reconstruct' is an abstract method.")


class ReconstructorWaveletBase(ReconstructorBase):
    def __init__(self, kspace_loc, uniform_data_shape,
                 wavelet_name, padding_mode, nb_scale,
                 wavelet_op_per_channel, n_coils,
                 fourier_type, nfft_implementation, verbose):
        super(ReconstructorWaveletBase, self).__init__(
            kspace_loc=kspace_loc,
            uniform_data_shape=uniform_data_shape,
            n_coils=n_coils,
            fourier_type=fourier_type,
            nfft_implementation=nfft_implementation,
            verbose=verbose)
        try:
            self.linear_op = WaveletN(
                nb_scale=nb_scale,
                wavelet_name=wavelet_name,
                padding_mode=padding_mode,
                dim=len(self.fourier_op.shape),
                multichannel=wavelet_op_per_channel)
        except ValueError:
            # For Undecimated wavelets, the wavelet_name is wavelet_id
            self.linear_op = WaveletUD2(wavelet_id=wavelet_name,
                                        nb_scale=nb_scale,
                                        multichannel=wavelet_op_per_channel)
        if verbose >= 5:
            print("Initialized lineat wavelet operator : " +
                  str(self.fourier_op))
