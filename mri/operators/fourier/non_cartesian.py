# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""Fourier operators for cartesian and non-cartesian space."""

# System import
import warnings

import numpy as np

# Package import
from ..base import OperatorBase
from .utils import get_stacks_fourier, normalize_frequency_locations

GPUNUFFT_AVAILABLE = False
CUFINUFFT_AVAILABLE = False
CPU_NUFFT_AVAILABLE = False
# Nufft Librairies
try:
    import pynfft
except ImportError:
    warnings.warn(
        "pynfft python package has not been found. If needed use " "the master release."
    )
else:
    CPU_NUFFT_AVAILABLE = True

try:
    from gpuNUFFT import NUFFTOp
except ImportError:
    warnings.warn(
        "gpuNUFFT python package has not been found. If needed "
        "please check on how to install in README"
    )
else:
    GPUNUFFT_AVAILABLE = True

try:
    from cufinufft import cufinufft
except ImportError:
    warnings.warn(
        "cufinufft python package has not been found. If needed "
        "please check on how to install in README"
    )
else:
    CUFINUFFT_AVAILABLE = True
    from pycuda.gpuarray import GPUArray, to_gpu


class NFFT:
    """ND non catesian Fast Fourrier Transform class.

    The NFFT will normalize like the FFT i.e. in a symetric way.
    This means that both direct and adjoint operator will be divided by the
    square root of the number of samples in the fourier domain.


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

    Attributes
    ----------
    samples: np.ndarray
        the samples locations in the Fourier domain between [-0.5; 0.5[.
    shape: tuple of int
        shape of the image (not necessarly a square matrix).
    n_coils: int default 1
        Number of coils used to acquire the signal in case of multiarray
        receiver coils acquisition

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

    def __init__(self, samples, shape, n_coils=1):
        if CPU_NUFFT_AVAILABLE is False:
            raise ValueError(
                "pyNFFT library is not installed, "
                "please refer to README"
            )
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
        """Compute the masked non-cartesian Fourier transform.

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
            coeff = [self._op(img[i]) for i in range(self.nb_coils)]
            coeff = np.asarray(coeff)
        return coeff

    def _adj_op(self, x):
        self.plan.f = x
        return np.copy(self.plan.adjoint()) / np.sqrt(self.plan.M)

    def adj_op(self, x):
        """Compute inverse masked non-cartesian Fourier.

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
            img = [self._adj_op(x[i]) for i in range(self.nb_coils)]
            img = np.asarray(img)
        return img

class gpuNUFFT:
    """GPU implementation of N-D non uniform Fast Fourrier Transform class.

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

    def __init__(
        self,
        samples,
        shape,
        n_coils=1,
        density_comp=None,
        kernel_width=3,
        sector_width=8,
        osf=2,
        balance_workload=True,
        smaps=None,
    ):
        if GPUNUFFT_AVAILABLE is False:
            raise ValueError(
                "gpuNUFFT library is not installed, " "please refer to README"
            )
        if (n_coils < 1) or not isinstance(n_coils, int):
            raise ValueError("The number of coils should be an integer >= 1")
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
            smaps = np.moveaxis(smaps, 0, -1)
            self.uses_sense = True
        self.operator = NUFFTOp(
            np.reshape(samples, samples.shape[::-1], order="F"),
            shape,
            n_coils,
            smaps,
            density_comp,
            kernel_width,
            sector_width,
            osf,
            balance_workload,
        )

    def op(self, image, interpolate_data=False):
        """This method calculates the masked non-cartesian Fourier transform.

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
        # the following reshape is equivalent to:
        # np.asarray([np.reshape(image_ch.T, image_ch.size) for image_ch in image]).T
        if self.n_coils > 1 and not self.uses_sense:
            coeff = self.operator.op(
                np.reshape(
                    image.T, (np.prod(image.shape[1:]), image.shape[0])),
                interpolate_data,
            )
        else:
            coeff = self.operator.op(
                np.reshape(image.T, image.size),
                interpolate_data,
            )
            # Data is always returned as num_channels X coeff_array,
            # so for single channel, we extract single array
            if not self.uses_sense:
                coeff = coeff[0]
        return coeff

    def adj_op(self, coeff, grid_data=False):
        """Compute adjoint of non-uniform Fourier.

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
        image: np.ndarray
            adjoint operator of Non Uniform Fourier transform.
        """
        image = self.operator.adj_op(coeff, grid_data)
        if self.n_coils > 1 and not self.uses_sense:
            image = image.transpose(0, *range(image.ndim - 1, 0, -1))
        else:
            image = np.squeeze(image).T
        # The recieved data from gpuNUFFT is num_channels x Nx x Ny x Nz,
        # hence we use squeeze
        return np.squeeze(image)

    def data_consistency(self, input_image, coeffs):
        """Compute the data consistency term using gpu only functions.

        It performs: adj_op(op(input_image) - coeffs) on the gpu.

        Parameters
        ----------
        input_image: np.ndarray
            Input image
        coeffs: np.ndarray
            data coefficient in kspace.

        Returns
        -------
        np.ndarray
            Gradient estimation of the Non Uniform Fourier transform.
        """
        if self.n_coils > 1 and not self.uses_sense:
            image = self.operator.data_consistency(
                np.reshape(
                    input_image.T,
                    (np.prod(input_image.shape[1:]), input_image.shape[0]),
                ),
                coeffs,
            )
            image = image.transpose(0, *range(image.ndim - 1, 0, -1))
        else:
            image = self.operator.data_consistency(
                np.reshape(input_image.T, input_image.size), coeffs
            )
            image = np.squeeze(image).T
        # The recieved data from gpuNUFFT is num_channels x Nx x Ny x Nz,
        # hence we use squeeze
        return np.squeeze(image)

    def estimate_density_compensation(self, n_iter=10):
        """Estimate density_compensation using gpu."""
        return self.operator.estimate_density_comp(n_iter)


class cufiNUFFT:
    """GPU implementation of N-D non uniform Fast Fourrier Transform class.

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

    def __init__(self, samples, shape, n_coils=1, smaps=None):
        if CUFINUFFT_AVAILABLE is False:
            raise ValueError(
                "cufinufft library is not installed, " "please refer to README"
            )

        import pycuda.autoinit
        if (n_coils < 1) or not isinstance(n_coils, int):
            raise ValueError("The number of coils should be an integer >= 1")
        self.n_coils = n_coils
        self.shape = shape
        if samples.min() < -0.5 or samples.max() >= 0.5:
            warnings.warn("Samples will be normalized between [-0.5; 0.5[")
            samples = normalize_frequency_locations(samples)
            samples = samples * 2.0 * np.pi
            samples = np.float32(samples)
        if smaps is None:
            self.uses_sense = False
        else:
            self.uses_sense = True
            self.smaps = smaps

        samples_x = to_gpu(samples[:, 0])
        samples_y = to_gpu(samples[:, 1])
        if len(shape) == 2:
            self.plan_op = cufinufft(
                2, shape, n_coils, eps=1e-3, dtype=np.float32)
            self.plan_op.set_pts(samples_x, samples_y)

            self.plan_adj_op = cufinufft(
                1, shape, n_coils, eps=1e-3, dtype=np.float32)
            self.plan_adj_op.set_pts(samples_x, samples_y)
        elif len(shape) == 3:
            self.plan_op = cufinufft(
                2, shape, n_coils, eps=1e-3, dtype=np.float32)
            self.plan_op.set_pts(samples_x, samples_y, to_gpu(samples[:, 2]))

            self.plan_adj_op = cufinufft(
                1, shape, n_coils, eps=1e-3, dtype=np.float32)
            self.plan_adj_op.set_pts(
                samples_x, samples_y, to_gpu(samples[:, 2]))

        else:
            raise ValueError("Unsupported number of dimension. ")

        self.kspace_data_gpu = GPUArray(
            (self.n_coils, len(samples)), dtype=np.complex64
        )
        self.image_gpu_multicoil = GPUArray(
            (self.n_coils, *self.shape), dtype=np.complex64
        )

    def op(self, image):
        """This method calculates the masked non-cartesian Fourier transform.

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
        if self.uses_sense:
            image_mc = np.complex64(image * self.smaps)
            self.image_gpu_multicoil.set(image_mc)
        else:
            self.image_gpu_multicoil.set(image)

        self.plan_op.execute(self.kspace_data_gpu, self.image_gpu_multicoil)

        return self.kspace_data_gpu.get()

    def adj_op(self, coeff):
        """Compute adjoint of non-uniform Fourier.

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
        image: np.ndarray
            adjoint operator of Non Uniform Fourier transform.
        """
        self.plan_adj_op.execute(to_gpu(coeff), self.image_gpu_multicoil)
        image_mc = self.image_gpu_multicoil.get()
        if self.uses_sense:
            # TODO: Do this on device.
            return np.sum(np.conjugate(self.smaps) * image_mc, axis=0)
        return image_mc

    def data_consistency(self, input_image, coeffs):
        """Compute the data consistency term using gpu only functions.

        It performs: adj_op(op(input_image) - coeffs) on the gpu.

        Parameters
        ----------
        input_image: np.ndarray
            Input image
        coeffs: np.ndarray
            data coefficient in kspace.

        Returns
        -------
        np.ndarray
            Gradient estimation of the Non Uniform Fourier transform.
        """
        if self.uses_sense:
            image_mc = input_image * self.smaps

        self.image_gpu_multicoil.set(image_mc)
        self.plan_op.execute(self.kspace_data_gpu, self.image_gpu_multicoil)

        self.plan_adj_op.execute(
            self.kspace_data_gpu - to_gpu(coeffs), self.image_gpu_multicoil
        )

        # The recieved data from gpuNUFFT is num_channels x Nx x Ny x Nz,
        # hence we use squeeze
        image_mc = self.image_gpu_multicoil.get()
        # TODO: Do this on device.
        return np.sum(np.conjugate(self.smaps) * image_mc, axis=0)


class NonCartesianFFT(OperatorBase):
    """Wrap around different implementation algorithms for NFFT.

    Parameters
    ----------
    samples: np.ndarray (Mxd)
        the samples locations in the Fourier domain where M is the number
        of samples and d is the dimensionnality of the output data
        (2D for an image, 3D for a volume).
    shape: tuple of int
        shape of the image (not necessarly a square matrix).
    implementation: str 'cpu' or  'gpuNUFFT', default 'cpu'
        which implementation of NFFT to use.
    n_coils: int default 1
        Number of coils used to acquire the signal in case of multiarray
        receiver coils acquisition
    kwargs: extra keyword args
        these arguments are passed to gpuNUFFT operator. This is used
        only in gpuNUFFT
    """

    def __init__(
        self,
        samples,
        shape,
        implementation="cpu",
        n_coils=1,
        density_comp=None,
        **kwargs,
    ):
        self.shape = shape
        self.samples = samples
        self.n_coils = n_coils
        self.implementation = implementation
        self.density_comp = density_comp
        self.kwargs = kwargs
        if self.implementation == "cpu":
            self.density_comp = density_comp
            self.impl = NFFT(samples=samples, shape=shape,
                             n_coils=self.n_coils)
        elif self.implementation == "gpuNUFFT":
            if GPUNUFFT_AVAILABLE is False:
                raise ValueError(
                    "gpuNUFFT library is not installed, "
                    "please refer to README"
                    "or use cpu for implementation"
                )
            self.impl = gpuNUFFT(
                samples=self.samples,
                shape=self.shape,
                n_coils=self.n_coils,
                density_comp=self.density_comp,
                **self.kwargs,
            )
        elif self.implementation == "cufiNUFFT":
            if not CUFINUFFT_AVAILABLE:
                raise ValueError(
                    "cufiNUFFT library is not installed"
                    "please refer to README"
                    "or use other implementation"
                )
            self.impl = cufiNUFFT(
                samples=self.samples,
                shape=self.shape,
                n_coils=self.n_coils,
                **self.kwargs,
            )
        else:
            raise ValueError(
                f"Bad implementation {implementation}"
                " chosen. Please choose between "
                '"cpu" | "gpuNUFFT" | "cufiNUFFT"'
            )

    def op(self, data, *args):
        """Compute the masked non-cartesian Fourier transform.

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
        """Compute inverse masked non-uniform Fourier transform.

        Parameters
        ----------
        x: np.ndarray
            masked non-uniform Fourier transform 1D data.

        Returns
        -------
            inverse discrete Fourier transform of the input coefficients.
        """
        if not isinstance(self.impl, gpuNUFFT) and self.density_comp is not None:
            return self.impl.adj_op(coeffs * self.density_comp, *args)
        else:
            return self.impl.adj_op(coeffs, *args)


class Stacked3DNFFT(OperatorBase):
    """3-D non uniform Fast Fourier Transform class.

    Fast implementation for Stacked samples. Note that the kspace locations
    must be in the form of a stack along z, with same locations in
    each plane.

    Parameters
    ----------
    kspace_loc: np.ndarray
        the position of the samples in the k-space
    shape: tuple of int
        shape of the image stack in 3D. (N x N x Nz)
    implementation: string, 'cpu' or 'gpuNUFFT' default 'cpu'
        string indicating which implemenmtation of Noncartesian FFT
        must be carried out. Please refer to Documentation of
        NoncartesianFFT
    n_coils: int default 1
        Number of coils used to acquire the signal in case of multiarray
        receiver coils acquisition

    Attributes
    ----------
    samples: np.ndarray
        the mask samples in the Fourier domain.
    shape: tuple of int
        shape of the image (necessarly a square/cubic matrix).
    implementation: string, 'cpu' or 'gpuNUFFT'
        string indicating which implemenmtation of Noncartesian FFT
        must be carried out
    n_coils: int default 1
        Number of coils used to acquire the signal in case of multiarray
        receiver coils acquisition
    """

    def __init__(self, kspace_loc, shape, implementation="cpu", n_coils=1):
        self.num_slices = shape[2]
        self.shape = shape
        self.samples = kspace_loc
        self.implementation = implementation
        (
            kspace_plane_loc,
            self.z_sample_loc,
            self.sort_pos,
            self.idx_mask_z,
        ) = get_stacks_fourier(
            kspace_loc,
            self.shape,
        )
        self.acq_num_slices = len(self.z_sample_loc)
        self.stack_len = len(kspace_plane_loc)
        self.plane_fourier_operator = NonCartesianFFT(
            samples=kspace_plane_loc,
            shape=shape[0:2],
            implementation=self.implementation,
        )
        self.n_coils = n_coils

    def _op(self, data):
        fft_along_z_axis = np.fft.fftshift(
            np.fft.fft(np.fft.ifftshift(data, axes=2), norm="ortho"), axes=2
        )
        stacked_kspace_sampled = np.asarray(
            [
                self.plane_fourier_operator.op(fft_along_z_axis[:, :, stack])
                for stack in self.idx_mask_z
            ]
        )
        stacked_kspace_sampled = np.reshape(
            stacked_kspace_sampled, self.acq_num_slices * self.stack_len
        )
        # Unsort the Coefficients
        inv_idx = np.zeros_like(self.sort_pos)
        inv_idx[self.sort_pos] = np.arange(len(self.sort_pos))
        # Return kspace unsorted and normalised by the ratio of slices acquired
        return stacked_kspace_sampled[inv_idx] * np.sqrt(
            self.num_slices / self.acq_num_slices
        )

    def op(self, data):
        """Compute the Fourier transform.

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
            coeff = [self._op(data[i]) for i in range(self.n_coils)]
        coeff = np.asarray(coeff)
        return coeff

    def _adj_op(self, coeff):
        coeff = coeff[self.sort_pos]
        stacks = np.reshape(coeff, (self.acq_num_slices, self.stack_len))
        # Receive First Fourier transformed data (per plane) in N x N x Nz
        adj_fft_along_z_axis = np.zeros(
            (*self.plane_fourier_operator.shape,
             self.num_slices), dtype=coeff.dtype
        )
        for idxs, idxm in enumerate(self.idx_mask_z):
            adj_fft_along_z_axis[:, :, idxm] = self.plane_fourier_operator.adj_op(
                stacks[idxs]
            )

        stacked_images = np.fft.ifftshift(
            np.fft.ifft(
                np.asarray(np.fft.fftshift(adj_fft_along_z_axis, axes=-1)),
                axis=-1,
                norm="ortho",
            ),
            axes=-1,
        )

        return stacked_images * np.sqrt(self.num_slices / self.acq_num_slices)

    def adj_op(self, coeff):
        """Compute the inverse masked non-uniform Fourier.

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
            img = [self._adj_op(coeff[i]) for i in range(self.n_coils)]
        img = np.asarray(img)
        return img
