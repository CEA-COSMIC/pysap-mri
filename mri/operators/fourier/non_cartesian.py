# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Fourier operators for cartesian and non-cartesian space.
"""

# System import
import warnings
import numpy as np

# Package import
from ..base import OperatorBase
from .utils import normalize_frequency_locations, get_stacks_fourier
from modopt.interface.errors import warn

# Third party import
try:
    import pynfft
except Exception:
    warnings.warn("pynfft python package has not been found. If needed use "
                  "the master release.")
    pass
try:
    from pynufft import NUFFT_hsa, NUFFT_cpu
except Exception:
    warnings.warn("pynufft python package has not been found. If needed use "
                  "the master release. Till then you cannot use NUFFT on GPU")
    pass


class NFFT:
    """ ND non catesian Fast Fourrier Transform class
    The NFFT will normalize like the FFT i.e. in a symetric way.
    This means that both direct and adjoint operator will be divided by the
    square root of the number of samples in the fourier domain.

    Attributes
    ----------
    samples: np.ndarray
        the samples locations in the Fourier domain between [-0.5; 0.5[.
    shape: tuple of int
        shape of the image (not necessarly a square matrix).
    n_coils: int default 1
        Number of coils used to acquire the signal in case of multiarray
        receiver coils acquisition
    """

    def __init__(self, samples, shape, n_coils=1):
        """ Initilize the 'NFFT' class.

        Parameters
        ----------
        samples: np.ndarray (Mxd)
            the samples locations in the Fourier domain where M is the number
            of samples and d is the dimensionnality of the output data
            (2D for an image, 3D for a volume).
        shape: tuple of int
            shape of the image (not necessarly a square matrix).
        n_coils: int, default 1
            Number of coils used to acquire the signal in case of multiarray
            receiver coils acquisition. If n_coils > 1, please organize data as
            n_coils X data_per_coil

        Exemple
        -------
        >>> import numpy as np
        >>> from pysap.data import get_sample_data
        >>> from mri.numerics.fourier import NFFT, FFT
        >>> from mri.reconstruct.utils import \
        convert_mask_to_locations

        >>> I = get_sample_data("2d-pmri").data.astype("complex128")
        >>> I = I[0]
        >>> samples = convert_mask_to_locations(np.ones(I.shape))
        >>> fourier_op = NFFT(samples=samples, shape=I.shape)
        >>> cartesian_fourier_op = FFT(samples=samples, shape=I.shape)
        >>> x_nfft = fourier_op.op(I)
        >>> x_fft = np.fft.ifftshift(cartesian_fourier_op.op(
            np.fft.fftshift(I))).flatten()
        >>> np.mean(np.abs(x_fft / x_nfft))
        1.000000000000005
        """
        if samples.shape[-1] != len(shape):
            raise ValueError("Samples and Shape dimension doesn't correspond")
        self.samples = samples
        if samples.min() < -0.5 or samples.max() >= 0.5:
            warnings.warn("Samples will be normalized between [-0.5; 0.5[")
            self.samples = normalize_frequency_locations(self.samples)
        # TODO Parallelize this if possible
        self.nb_coils = n_coils
        self.plan = pynfft.NFFT(N=shape, M=len(samples))
        self.plan.x = self.samples
        self.plan.precompute()
        self.shape = shape

    def _op(self, img):
        self.plan.f_hat = img
        return np.copy(self.plan.trafo()) / np.sqrt(self.plan.M)

    def op(self, img):
        """ This method calculates the masked non-cartesian Fourier transform
        of a N-D data.

        Parameters
        ----------
        img: np.ndarray
            input ND array with the same shape as the mask.

        Returns
        -------
        x: np.ndarray
            masked Fourier transform of the input image.
        """
        if self.nb_coils == 1:
            coeff = self._op(img)
        else:
            coeff = [self._op(img[i])
                     for i in range(self.nb_coils)]
            coeff = np.asarray(coeff)
        return coeff

    def _adj_op(self, x):
        self.plan.f = x
        return np.copy(self.plan.adjoint()) / np.sqrt(self.plan.M)

    def adj_op(self, x):
        """ This method calculates inverse masked non-cartesian Fourier
        transform of a 1-D coefficients array.

        Parameters
        ----------
        x: np.ndarray
            masked non-cartesian Fourier transform 1D data.

        Returns
        -------
        img: np.ndarray
            inverse 2D discrete Fourier transform of the input coefficients.
        """
        if self.nb_coils == 1:
            img = self._adj_op(x)
        else:
            img = [self._adj_op(x[i])
                   for i in range(self.nb_coils)]
            img = np.asarray(img)
        return img


class Singleton:
    """ This is an internal class used by GPU based NUFFT,
    to hold a count of instances of GPU NUFFT Class.
    We raise an error if we have more than one"""
    numOfInstances = 0

    def countInstances(cls):
        """ This function increments each time an object is created"""
        cls.numOfInstances += 1

    countInstances = classmethod(countInstances)

    def getNumInstances(cls):
        return cls.numOfInstances

    getNumInstances = classmethod(getNumInstances)

    def __init__(self):
        self.countInstances()


class NUFFT(Singleton):
    """  GPU implementation of N-D non uniform Fast Fourrier Transform class.

    Attributes
    ----------
    samples: np.ndarray
        the mask samples in the Fourier domain.
    shape: tuple of int
        shape of the image (necessarly a square/cubic matrix).
    nufftObj: The pynufft object
        depending on the required computational platform
    platform: string, 'opencl' or 'cuda'
        string indicating which hardware platform will be used to compute the
        NUFFT
    Kd: int or tuple
        int or tuple indicating the size of the frequency grid, for regridding.
        if int, will be evaluated to (Kd,)*nb_dim of the image
    Jd: int or tuple
        Size of the interpolator kernel. If int, will be evaluated
        to (Jd,)*dims image
    n_coils: int default 1
            Number of coils used to acquire the signal in case of multiarray
            receiver coils acquisition. If n_coils > 1, please organize data as
            n_coils X data_per_coil
    """
    numOfInstances = 0

    def __init__(self, samples, shape, platform='cuda', Kd=None, Jd=None,
                 n_coils=1, verbosity=0):
        """ Initilize the 'NUFFT' class.

        Parameters
        ----------
        samples: np.ndarray
            the mask samples in the Fourier domain.
        shape: tuple of int
            shape of the image (necessarly a square/cubic matrix).
        platform: string, 'cpu', 'opencl' or 'cuda'
            string indicating which hardware platform will be used to
            compute the NUFFT
        Kd: int or tuple
            int or tuple indicating the size of the frequency grid,
            for regridding. If int, will be evaluated
            to (Kd,)*nb_dim of the image
        Jd: int or tuple
            Size of the interpolator kernel. If int, will be evaluated
            to (Jd,)*dims image
        n_coils: int
            Number of coils used to acquire the signal in case of multiarray
            receiver coils acquisition
        """
        if (n_coils < 1) or (type(n_coils) is not int):
            raise ValueError('The number of coils should be an integer >= 1')
        self.nb_coils = n_coils
        self.shape = shape
        self.platform = platform
        self.samples = samples * (2 * np.pi)  # Pynufft use samples in
        # [-pi, pi[ instead of [-0.5, 0.5[
        self.dim = samples.shape[1]  # number of dimensions of the image

        if type(Kd) == int:
            self.Kd = (Kd,) * self.dim
        elif type(Kd) == tuple:
            self.Kd = Kd
        elif Kd is None:
            # Preferential option
            self.Kd = tuple([2 * ix for ix in shape])

        if type(Jd) == int:
            self.Jd = (Jd,) * self.dim
        elif type(Jd) == tuple:
            self.Jd = Jd
        elif Jd is None:
            # Preferential option
            self.Jd = (5,) * self.dim

        for (i, s) in enumerate(shape):
            assert (self.shape[i] <= self.Kd[i]), 'size of frequency grid' + \
                                                  'must be greater or equal ' \
                                                  'than the image size'
        if verbosity > 0:
            print('Creating the NUFFT object...')
        if self.platform == 'opencl':
            warn('Attemping to use OpenCL plateform. Make sure to '
                 'have  all the dependecies installed')
            Singleton.__init__(self)
            if self.getNumInstances() > 1:
                warn('You have created more than one NUFFT object. '
                     'This could cause memory leaks')
            self.nufftObj = NUFFT_hsa(API='ocl',
                                      platform_number=None,
                                      device_number=None,
                                      verbosity=verbosity)

            self.nufftObj.plan(om=self.samples,
                               Nd=self.shape,
                               Kd=self.Kd,
                               Jd=self.Jd,
                               batch=1,  # TODO self.nb_coils,
                               ft_axes=tuple(range(samples.shape[1])),
                               radix=None)

        elif self.platform == 'cuda':
            warn('Attemping to use Cuda plateform. Make sure to '
                 'have  all the dependecies installed and '
                 'to create only one instance of NUFFT GPU')
            Singleton.__init__(self)
            if self.getNumInstances() > 1:
                warn('You have created more than one NUFFT object. '
                     'This could cause memory leaks')
            self.nufftObj = NUFFT_hsa(API='cuda',
                                      platform_number=None,
                                      device_number=None,
                                      verbosity=verbosity)

            self.nufftObj.plan(om=self.samples,
                               Nd=self.shape,
                               Kd=self.Kd,
                               Jd=self.Jd,
                               batch=1,  # TODO self.nb_coils,
                               ft_axes=tuple(range(samples.shape[1])),
                               radix=None)

        else:
            raise ValueError('Wrong type of platform. Platform must be'
                             '\'opencl\' or \'cuda\'')

    def __del__(self):
        # This is an important desctructor to ensure that the device memory
        # is freed
        # TODO this is still not freeing the memory right on device.
        # Mostly issue with reikna library.
        # Refer : https://github.com/fjarri/reikna/issues/53
        if self.platform == 'opencl' or self.platform == 'cuda':
            self.nufftObj.release()

    def op(self, img):
        """ This method calculates the masked non-cartesian Fourier transform
        of a 3-D image.

        Parameters
        ----------
        img: np.ndarray
            input 3D array with the same shape as shape.

        Returns
        -------
        x: np.ndarray
            masked Fourier transform of the input image.
        """
        if self.nb_coils == 1:
            dtype = np.complex64
            # Send data to the mCPU/GPU platform
            self.nufftObj.x_Nd = self.nufftObj.thr.to_device(
                img.astype(dtype))
            gx = self.nufftObj.thr.copy_array(self.nufftObj.x_Nd)
            # Forward operator of the NUFFT
            gy = self.nufftObj.forward(gx)
            y = np.squeeze(gy.get())
        else:
            dtype = np.complex64
            # Send data to the mCPU/GPU platform
            y = []
            for ch in range(self.nb_coils):
                self.nufftObj.x_Nd = self.nufftObj.thr.to_device(
                    np.copy(img[ch]).astype(dtype))
                gx = self.nufftObj.thr.copy_array(self.nufftObj.x_Nd)
                # Forward operator of the NUFFT
                gy = self.nufftObj.forward(gx)
                y.append(np.squeeze(gy.get()))
            y = np.asarray(y)
        return y * 1.0 / np.sqrt(np.prod(self.Kd))

    def adj_op(self, x):
        """ This method calculates inverse masked non-uniform Fourier
        transform of a 1-D coefficients array.

        Parameters
        ----------
        x: np.ndarray
            masked non-uniform Fourier transform 1D data.

        Returns
        -------
        img: np.ndarray
            inverse 3D discrete Fourier transform of the input coefficients.
        """
        if self.nb_coils == 1:
            dtype = np.complex64
            cuda_array = self.nufftObj.thr.to_device(x.astype(dtype))
            gx = self.nufftObj.adjoint(cuda_array)
            img = np.squeeze(gx.get())
        else:
            dtype = np.complex64
            img = []
            for ch in range(self.nb_coils):
                cuda_array = self.nufftObj.thr.to_device(np.copy(
                    x[ch]).astype(dtype))
                gx = self.nufftObj.adjoint(cuda_array)
                img.append(gx.get())
            img = np.asarray(np.squeeze(img))
        return img * np.sqrt(np.prod(self.Kd))


class NonCartesianFFT(OperatorBase):
    """This class wraps around different implementation algorithms for NFFT"""
    def __init__(self, samples, shape, implementation='cpu', n_coils=1):
        """ Initialize the class.

        Parameters
        ----------
        samples: np.ndarray (Mxd)
            the samples locations in the Fourier domain where M is the number
            of samples and d is the dimensionnality of the output data
            (2D for an image, 3D for a volume).
        shape: tuple of int
            shape of the image (not necessarly a square matrix).
        implementation: str 'cpu' | 'cuda' | 'opencl', default 'cpu'
            which implementation of NFFT to use.
        n_coils: int default 1
            Number of coils used to acquire the signal in case of multiarray
            receiver coils acquisition
        """
        self.shape = shape
        self.samples = samples
        self.n_coils = n_coils
        if implementation == 'cpu':
            self.implementation = NFFT(samples=samples, shape=shape,
                                       n_coils=self.n_coils)
        elif implementation == 'cuda' or implementation == 'opencl':
            self.implementation = NUFFT(samples=samples, shape=shape,
                                        platform=implementation,
                                        n_coils=self.n_coils)
        else:
            raise ValueError('Bad implementation ' + implementation +
                             ' chosen. Please choose between "cpu" | "cuda" |'
                             '"opencl"')

    def op(self, data):
        """ This method calculates the masked non-cartesian Fourier transform
        of an image.

        Parameters
        ----------
        img: np.ndarray
            input N-D array with the same shape as shape.

        Returns
        -------
            masked Fourier transform of the input image.
        """
        return self.implementation.op(data)

    def adj_op(self, coeffs):
        """ This method calculates inverse masked non-uniform Fourier
        transform of a 1-D coefficients array.

        Parameters
        ----------
        x: np.ndarray
            masked non-uniform Fourier transform 1D data.

        Returns
        -------
            inverse discrete Fourier transform of the input coefficients.
        """
        return self.implementation.adj_op(coeffs)


class Stacked3DNFFT(OperatorBase):
    """"  3-D non uniform Fast Fourier Transform class,
    fast implementation for Stacked samples. Note that the kspace locations
    must be in the form of a stack along z, with same locations in
    each plane.

    Attributes
    ----------
    samples: np.ndarray
        the mask samples in the Fourier domain.
    shape: tuple of int
        shape of the image (necessarly a square/cubic matrix).
    implementation: string, 'cpu', 'cuda' or 'opencl' default 'cpu'
        string indicating which implemenmtation of Noncartesian FFT
        must be carried out
    n_coils: int default 1
        Number of coils used to acquire the signal in case of multiarray
        receiver coils acquisition
    """

    def __init__(self, kspace_loc, shape, implementation='cpu', n_coils=1):
        """ Init function for Stacked3D class.

        Parameters
        ----------
        kspace_loc: np.ndarray
            the position of the samples in the k-space
        shape: tuple of int
            shape of the image stack in 3D. (N x N x Nz)
        implementation: string, 'cpu', 'cuda' or 'opencl' default 'cpu'
            string indicating which implemenmtation of Noncartesian FFT
            must be carried out. Please refer to Documentation of
            NoncartesianFFT
        n_coils: int default 1
            Number of coils used to acquire the signal in case of multiarray
            receiver coils acquisition
        """
        self.num_slices = shape[2]
        self.shape = shape
        (kspace_plane_loc, self.z_sample_loc, self.sort_pos, self.idx_mask_z) \
            = \
            get_stacks_fourier(
            kspace_loc,
            self.shape,
            )
        self.acq_num_slices = len(self.z_sample_loc)
        self.stack_len = len(kspace_plane_loc)
        self.plane_fourier_operator = \
            NonCartesianFFT(samples=kspace_plane_loc, shape=shape[0:2],
                            implementation=implementation)
        self.n_coils = n_coils

    def _op(self, data):
        fft_along_z_axis = np.fft.fftshift(np.fft.fft(
            np.fft.ifftshift(data, axes=2),
            norm="ortho"),
            axes=2)
        stacked_kspace_sampled = np.asarray(
            [self.plane_fourier_operator.op(fft_along_z_axis[:, :, stack])
             for stack in self.idx_mask_z])
        stacked_kspace_sampled = np.reshape(
            stacked_kspace_sampled,
            self.acq_num_slices * self.stack_len)
        # Unsort the Coefficients
        inv_idx = np.zeros_like(self.sort_pos)
        inv_idx[self.sort_pos] = np.arange(len(self.sort_pos))
        # Return kspace unsorted and normalised by the ratio of slices acquired
        return stacked_kspace_sampled[inv_idx] * \
            np.sqrt(self.num_slices / self.acq_num_slices)

    def op(self, data):
        """ This method calculates Fourier transform.

        Parameters
        ----------
        data: np.ndarray
            input image as array.

        Returns
        -------
        result: np.ndarray
            Forward 3D Fourier transform of the image.
        """
        if self.n_coils == 1:
            coeff = self._op(np.squeeze(data))
        else:
            coeff = [self._op(data[i])
                     for i in range(self.n_coils)]
        coeff = np.asarray(coeff)
        return coeff

    def _adj_op(self, coeff):
        coeff = coeff[self.sort_pos]
        stacks = np.reshape(coeff, (self.acq_num_slices, self.stack_len))
        # Receive First Fourier transformed data (per plane) in N x N x Nz
        adj_fft_along_z_axis = np.zeros((*self.plane_fourier_operator.shape,
                                         self.num_slices),
                                        dtype=coeff.dtype)
        for idxs, idxm in enumerate(self.idx_mask_z):
            adj_fft_along_z_axis[:, :, idxm] = \
                self.plane_fourier_operator.adj_op(stacks[idxs])

        stacked_images = np.fft.ifftshift(np.fft.ifft(
                np.asarray(np.fft.fftshift(adj_fft_along_z_axis, axes=-1)),
                axis=-1, norm="ortho"),
            axes=-1)

        return stacked_images * np.sqrt(self.num_slices / self.acq_num_slices)

    def adj_op(self, coeff):
        """ This method calculates inverse masked non-uniform Fourier
        transform of a 1-D coefficients array.

        Parameters
        ----------
        coeff: np.ndarray
            masked non-uniform Fourier transform 1D data.

        Returns
        -------
        img: np.ndarray
            inverse 3D discrete Fourier transform of the input coefficients.
        """
        if self.n_coils == 1:
            img = self._adj_op(np.squeeze(coeff))
        else:
            img = [self._adj_op(coeff[i])
                   for i in range(self.n_coils)]
        img = np.asarray(img)
        return img
