"""Fourier operators for online reconstructions."""

import numpy as np
import scipy as sp

from mri.operators.base import OperatorBase


class ColumnFFT(OperatorBase):
    """
    Fourier operator optimized to compute the 2D FFT + selection of various line of the kspace.

    The FFT will be normalized in a symmetric way.
    Currently work only in 2D or stack of 2D.

    Attributes:
    -----------
    shape: tuple of int
        shape of the image (not necessarly a square matrix).
    n_coils: int, default 1
        Number of coils used to acquire the signal in case of multiarray
        receiver coils acquisition. If n_coils > 1, data shape must be
        [n_coils, Nx, Ny, NZ]
    n_jobs: int, default 1
        Number of parallel workers to use for fourier computation
    mask: int
        The column of the kspace which is kept.

    Notes:
    ------
    This Operator performs a 1D FFT on the column axis of the provided data
    and then perform a classical DFT operation (there is only one frequency to compute).

    This method is faster and cheaper than the regular 2D FFT+mask operation.
    """

    def __init__(self, shape, line_index=0, n_coils=1):
        """Initilize the 'FFT' class.

        Parameters:
        ----------
        shape: tuple of int
            shape of the image (not necessarly a square matrix).
        n_coils: int, default 1
            Number of coils used to acquire the signal in case of
            multiarray receiver coils acquisition. If n_coils > 1,
            data shape must be equal to [n_coils, Nx, Ny, NZ]
        line_index: int
            The index of the column onto the line_axis of the kspace
        n_jobs: int, default 1
            Number of parallel workers to use for fourier computation
            All cores are used if -1
        package: str
            The plateform on which to run the computation. can be either 'numpy', 'numba', 'cupy'
        """
        self.shape = shape
        if n_coils <= 0:
            n_coils = 1
        self.n_coils = n_coils
        self._exp_f = np.zeros(shape[1], dtype=complex)
        self._exp_b = np.zeros(shape[1], dtype=complex)
        self._mask = line_index

    @property
    def mask(self):
        """Return the column index of the mask."""
        return self._mask

    @mask.setter
    def mask(self, val: int, shift=True):
        if shift:
            val = (self.shape[1] // 2 + val) % self.shape[1]
        if val >= self.shape[1]:
            raise IndexError("Index out of range")
        self._mask = val
        cos = np.cos(2 * np.pi * val / self.shape[1])
        sin = np.sin(2 * np.pi * val / self.shape[1])
        exp_f = cos - 1j * sin
        exp_b = cos + 1j * sin
        self._exp_f = (1 / np.sqrt(self.shape[1])) * exp_f ** np.arange(self.shape[1])
        self._exp_b = (1 / np.sqrt(self.shape[1])) * exp_b ** np.arange(self.shape[1])

    def op(self, img):
        """Compute the masked 2D Fourier transform of a 2d or 3D image.

        Parameters
        ----------
        img: np.ndarray
            input ND array with the same shape as the mask. For multichannel
            images the coils dimension is put first

        Returns
        -------
        x: np.ndarray
            masked Fourier transform of the input image. For multichannel
            images the coils dimension is put first
        """
        # if self.n_coils == 1:
        #     return self._op(img)
        # return np.array(self._op(img_slice) for img_slice in img)
        return sp.fft.ifftshift(
            sp.fft.fft(
                np.dot(sp.fft.fftshift(img, axes=[-1, -2]), self._exp_f),
                axis=-1,
                norm="ortho",
            ),
            axes=[-1],
        )

    def adj_op(self, x):
        """Compute inverse masked Fourier transform of a ND image.

        Parameters
        ----------
        x: np.ndarray
            masked Fourier transform data. For multichannel
            images the coils dimension is put first

        Returns
        -------
        img: np.ndarray
            inverse ND discrete Fourier transform of the input coefficients.
            For multichannel images the coils dimension is put first
        """
        # if self.n_coils == 1:
        #     return self._adj_op(x)
        # return np.array(self._op(x_slice) for x_slice in x)

        return sp.fft.fftshift(
            np.multiply.outer(
                sp.fft.ifft(sp.fft.ifftshift(x, axes=[-1]), axis=-1, norm="ortho"),
                self._exp_b,
            ),
            axes=[-1, -2],
        )
