"""Density Compensation estimation."""
import numpy as np

from ..non_cartesian import NonCartesianFFT, gpunufft_available


def estimate_density_compensation(kspace_loc, volume_shape, num_iterations=10):
    """Estimate the density compensator for a given set of kspace locations.

    Parameters
    ----------
    kspace_loc: np.ndarray
        the kspace locations
    volume_shape: np.ndarray
        the volume shape
    num_iterations: int default 10
        the number of iterations for density estimation

    Returns
    -------
    np.ndarray: the density compensation vector
    """
    if gpunufft_available is False:
        raise ValueError("gpuNUFFT is not available, cannot "
                         "estimate the density compensation")
    grid_op = NonCartesianFFT(
        samples=kspace_loc,
        shape=volume_shape,
        implementation='gpuNUFFT',
        osf=1,
    )
    density_comp = np.ones(kspace_loc.shape[0])
    for _ in range(num_iterations):
        density_comp = (
            density_comp / np.abs(grid_op.op(grid_op.adj_op(density_comp,
                                                            True),
                                             True)))
    return density_comp


def estimate_density_compensation_gpu(kspace_loc,
                                      volume_shape,
                                      num_iterations=10):
    """Estimate the density compensation using gpu-only functions.

    This is faster than the non gpu implementation


    Parameters
    ----------
    kspace_loc: np.ndarray
        the kspace locations
    volume_shape: np.ndarray
        the volume shape
    num_iterations: int default 10
        the number of iterations for density estimation

    Returns
    -------
    np.ndarray: the density compensation vector

    See Also
    --------
    estimate_density_compensation
    """
    if gpunufft_available is False:
        raise ValueError("gpuNUFFT is not available, cannot "
                         "estimate the density compensation")
    grid_op = NonCartesianFFT(
        samples=kspace_loc,
        shape=volume_shape,
        implementation='gpuNUFFT',
        osf=1,
    )
    return grid_op.impl.estimate_density_compensation(num_iterations)
