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
    """ This is the base reconstructor class for reconstruction.
    This class holds some common parameters that is common for all
    MR Image reconstructors
    Parameters
    ----------
    kspace_loc: np.ndarray
        the mask samples in the Fourier domain.
    uniform_data_shape: tuplet
        the shape of the matrix containing the uniform data.
    n_coils: int, default 1
        Number of coils used to acquire the signal in case of multiarray
        receiver coils acquisition. If n_coils > 1, data shape must be
        [n_coils, *data_shape]
    fourier_type: str (optional, default 'cartesian')
        type of fourier operator : 'cartesian' | 'non-cartesian' | 'stack'
    nfft_implementation: str, default 'cpu'
        way to implement NFFT : 'cpu' | 'cuda' | 'opencl'
    verbose: int
        verbosity level for debug, please check dervied class for details
        on verbosity levels
    """

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
    """ This is the base reconstructor class for reconstruction.
    This class holds some common parameters that is common for all
    MR Image reconstructors
    Parameters
    ----------
    kspace_loc: np.ndarray
        the mask samples in the Fourier domain.
    uniform_data_shape: tuplet
        the shape of the matrix containing the uniform data.
    n_coils: int, default 1
        Number of coils used to acquire the signal in case of multiarray
        receiver coils acquisition. If n_coils > 1, data shape must be
        [n_coils, *data_shape]
    fourier_type: str (optional, default 'cartesian')
        type of fourier operator : 'cartesian' | 'non-cartesian' | 'stack'
    nfft_implementation: str, default 'cpu'
        way to implement NFFT : 'cpu' | 'cuda' | 'opencl'
    wavelet_name: str | int
        if implementation is with waveletN the wavelet name to be used during
        the decomposition, else implementation with waveletUD2 where the
        wavelet name is wavelet_id Refer to help of mr_transform under option
        '-t' to choose the right wavelet_id.
    padding_mode: str (optional, default zero)
        The padding mode used in the Wavelet transform,
        'zero' | 'periodization'
    nb_scale: int (optional default is 4)
        the number of scale in the used by the multi-scale wavelet
        decomposition
    wavelet_op_per_channel: bool
        whether wavelet transform should be applied on every channel. This is
        True only for Calibrationless reconstruction
    n_jobs: int, default 1
        Number of parallel jobs for linear operation
    verbose: int
        verbosity level for debug, please check dervied class for details
        on verbosity levels
    """
    def __init__(self, kspace_loc, uniform_data_shape, n_coils,
                 fourier_type, nfft_implementation, wavelet_name, padding_mode,
                 nb_scale, wavelet_op_per_channel, n_jobs=1, verbose=0):
        super(ReconstructorWaveletBase, self).__init__(
            kspace_loc=kspace_loc,
            uniform_data_shape=uniform_data_shape,
            n_coils=n_coils,
            fourier_type=fourier_type,
            nfft_implementation=nfft_implementation,
            verbose=verbose)
        verbose = int(verbose > 20)
        if wavelet_op_per_channel is False:
            # For Self Calibrating Reconstruction, we do not do linear
            # operator per channel
            n_coils = 1
        try:
            self.linear_op = WaveletN(
                nb_scale=nb_scale,
                wavelet_name=wavelet_name,
                padding_mode=padding_mode,
                dim=len(self.fourier_op.shape),
                n_coils=n_coils,
                n_jobs=n_jobs,
                verbose=verbose,
            )
        except ValueError:
            # For Undecimated wavelets, the wavelet_name is wavelet_id
            self.linear_op = WaveletUD2(
                wavelet_id=wavelet_name,
                nb_scale=nb_scale,
                n_coils=n_coils,
                n_jobs=n_jobs,
                verbose=verbose,
            )
        if verbose >= 5:
            print("Initialized linear wavelet operator : " +
                  str(self.linear_op))
