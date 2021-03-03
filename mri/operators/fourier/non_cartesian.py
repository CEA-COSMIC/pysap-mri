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
pynufft_available = False
try:
    from pynufft import NUFFT_hsa, NUFFT_cpu
except Exception:
    pass
else:
    pynufft_available = True

gpunufft_available = False
try:
    from gpuNUFFT import NUFFTOp
except ImportError:
    warnings.warn("gpuNUFFT python package has not been found. If needed "
                  "please check on how to install in README")
else:
    gpunufft_available = True


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
        if not pynufft_available:
            raise ValueError('PyNUFFT Package is not installed, please '
                             'consider using `gpuNUFFT` or install the '
                             'PyNUFFT package')
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


class gpuNUFFT:
    """  GPU implementation of N-D non uniform Fast Fourrier Transform class.

    Attributes
    ----------
    samples: np.ndarray
        the normalized kspace location values in the Fourier domain.
    shape: tuple of int
        shape of the image
    operator: The NUFFTOp object
        to carry out operation
    n_coils: int default 1
            Number of coils used to acquire the signal in case of multiarray
            receiver coils acquisition. If n_coils > 1, please organize data as
            n_coils X data_per_coil
    """
    def __init__(self, samples, shape, n_coils=1, density_comp=None,
                 kernel_width=3, sector_width=8, osf=2, balance_workload=True,
                 smaps=None):
        """ Initilize the 'NUFFT' class.

        Parameters
        ----------
        samples: np.ndarray
            the kspace sample locations in the Fourier domain,
            normalized between -0.5 and 0.5
        shape: tuple of int
            shape of the image
        n_coils: int
            Number of coils used to acquire the signal in case of multiarray
            receiver coils acquisition
        density_comp: np.ndarray default None.
            k-space weighting, density compensation, if not specified
            equal weightage is given.
        kernel_width: int default 3
            interpolation kernel width (usually 3 to 7)
        sector_width: int default 8
            sector width to use
        osf: int default 2
            oversampling factor (usually between 1 and 2)
        balance_workload: bool default True
            whether the workloads need to be balanced
        smaps: np.ndarray default None
            Holds the sensitivity maps for SENSE reconstruction
        """
        if gpunufft_available is False:
            raise ValueError('gpuNUFFT library is not installed, '
                             'please refer to README')
        if (n_coils < 1) or (type(n_coils) is not int):
            raise ValueError('The number of coils should be an integer >= 1')
        self.n_coils = n_coils
        self.shape = shape
        if samples.min() < -0.5 or samples.max() >= 0.5:
            warnings.warn("Samples will be normalized between [-0.5; 0.5[")
            samples = normalize_frequency_locations(samples)
            self.samples = samples
        if density_comp is None:
            density_comp = np.ones(samples.shape[0])
        if smaps is None:
            self.uses_sense = False
        else:
            smaps = np.asarray(
                [np.reshape(smap_ch.T, smap_ch.size) for smap_ch in smaps]
            ).T
            self.uses_sense = True
        self.operator = NUFFTOp(
            np.reshape(samples, samples.shape[::-1], order='F'),
            shape,
            n_coils,
            smaps,
            density_comp,
            kernel_width,
            sector_width,
            osf,
            balance_workload
        )

    def op(self, image, interpolate_data=False):
        """ This method calculates the masked non-cartesian Fourier transform
        of a 2D / 3D image.

        Parameters
        ----------
        image: np.ndarray
            input array with the same shape as shape.
        interpolate_data: bool, default False
            if set to True, the image is just apodized and interpolated to
            kspace locations. This is used for density estimation.

        Returns
        -------
        np.ndarray
            Non Uniform Fourier transform of the input image.
        """
        # Base gpuNUFFT Operator is written in CUDA and C++, we need to
        # reorganize data to follow a different memory hierarchy
        # TODO we need to update codes to use np.reshape for all this directly
        if self.n_coils > 1 and not self.uses_sense:
            coeff = self.operator.op(np.asarray(
                [np.reshape(image_ch.T, image_ch.size) for image_ch in image]
            ).T, interpolate_data)
        else:
            coeff = self.operator.op(
                np.reshape(image.T, image.size),
                interpolate_data
            )
            # Data is always returned as num_channels X coeff_array,
            # so for single channel, we extract single array
            if not self.uses_sense:
                coeff = coeff[0]
        return coeff

    def adj_op(self, coeff, grid_data=False):
        """ This method calculates adjoint of non-uniform Fourier
        transform of a 1-D coefficients array.

        Parameters
        ----------
        coeff: np.ndarray
            masked non-uniform Fourier transform 1D data.
        grid_data: bool, default False
            if True, the kspace data is gridded and returned,
            this is used for density compensation
        Returns
        -------
        np.ndarray
            adjoint operator of Non Uniform Fourier transform of the
            input coefficients.
        """
        image = self.operator.adj_op(coeff, grid_data)
        if self.n_coils > 1 and not self.uses_sense:
            image = np.asarray(
                [image_ch.T for image_ch in image]
            )
        else:
            image = np.squeeze(image).T
        # The recieved data from gpuNUFFT is num_channels x Nx x Ny x Nz,
        # hence we use squeeze
        return np.squeeze(image)


class NonCartesianFFT(OperatorBase):
    """This class wraps around different implementation algorithms for NFFT"""
    def __init__(self, samples, shape, implementation='cpu', n_coils=1,
                 density_comp=None, **kwargs):
        """ Initialize the class.

        Parameters
        ----------
        samples: np.ndarray (Mxd)
            the samples locations in the Fourier domain where M is the number
            of samples and d is the dimensionnality of the output data
            (2D for an image, 3D for a volume).
        shape: tuple of int
            shape of the image (not necessarly a square matrix).
        implementation: str 'cpu' | 'cuda' | 'opencl' | 'gpuNUFFT',
        default 'cpu'
            which implementation of NFFT to use.
        n_coils: int default 1
            Number of coils used to acquire the signal in case of multiarray
            receiver coils acquisition
        kwargs: extra keyword args
            these arguments are passed to gpuNUFFT operator. This is used
            only in gpuNUFFT
        """
        self.shape = shape
        self.samples = samples
        self.n_coils = n_coils
        self.implementation = implementation
        self.density_comp = density_comp
        self.kwargs = kwargs
        if self.implementation == 'cpu':
            self.density_comp = density_comp
            self.impl = NFFT(samples=samples, shape=shape,
                             n_coils=self.n_coils)
        elif self.implementation == 'cuda' or self.implementation == 'opencl':
            self.density_comp = density_comp
            self.impl = NUFFT(samples=samples, shape=shape,
                              platform=self.implementation,
                              n_coils=self.n_coils)
        elif self.implementation == 'gpuNUFFT':
            if gpunufft_available is False:
                raise ValueError('gpuNUFFT library is not installed, '
                                 'please refer to README'
                                 'or use cpu for implementation')
            self.impl = gpuNUFFT(
                samples=self.samples,
                shape=self.shape,
                n_coils=self.n_coils,
                density_comp=self.density_comp,
                **self.kwargs
            )
        else:
            raise ValueError('Bad implementation ' + implementation +
                             ' chosen. Please choose between "cpu" | "cuda" |'
                             '"opencl" | "gpuNUFFT"')

    def op(self, data, *args):
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
        return self.impl.op(data, *args)

    def adj_op(self, coeffs, *args):
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
        if not isinstance(self.impl, gpuNUFFT) and \
                self.density_comp is not None:
            return self.impl.adj_op(
                coeffs * self.density_comp,
                *args
            )
        else:
            return self.impl.adj_op(coeffs, *args)


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
        implementation: string, 'cpu', 'cuda' or 'opencl'  or 'gpuNUFFT'
        default 'cpu'
            string indicating which implemenmtation of Noncartesian FFT
            must be carried out. Please refer to Documentation of
            NoncartesianFFT
        n_coils: int default 1
            Number of coils used to acquire the signal in case of multiarray
            receiver coils acquisition
        """
        self.num_slices = shape[2]
        self.shape = shape
        self.samples = kspace_loc
        self.implementation = implementation
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
                            implementation=self.implementation)
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
