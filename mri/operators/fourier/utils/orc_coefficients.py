# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Functions to compute different kinds of coefficients used in
Off-Resonance Correction.
"""


import numpy as np
import scipy.sparse.linalg as ssl
import sklearn.cluster as sc


def compute_orc_coefficients(field_map, time_vec, mask, coefficients="svd",
                             num_interpolators=15, weights="full", n_bins=1000):
    """
    This function computes different off-resonance correction weights used in
    the ORCFFTWrapper, as described in:
    - :cite: `man1997` for Multi-Frequency Interpolation (MFI)
    - :cite: `sutton2003` for Multi-Temporal Interpolation (MTI)
    - :cite: `fessler2005` for (Fast) Singular Value Decomposition (SVD, FSVD)

    Parameters
    ----------
    field_map: numpy.ndarray
        B0 field inhomogeneity map (in Hz)
    time_vec: numpy.ndarray
        1D vector indicating time after pulse in each shot (in s)
    mask: numpy.ndarray
        Mask describing the regions to consider during correction
    coefficients: {'mfi', 'mti', 'svd', 'fsvd'}
        Computation method used to obtain the interpolation weights.
    L: int
        Number of interpolators used for multi-linear correction (default 15)
    weights: {'full', 'sqrt', 'log', 'ones'}
        Weightning policy for the field map histogram (default 'full')
    n_bins: int
        Number of bins for the field map histogram (default 1000)

    Returns
    -------
    B: numpy.ndarray
        (len(time_vec), L) array correspondig to k-space coefficients
    C: numpy.ndarray
        (L, n_bins) array corresponding to volume coefficients
    E: numpy.ndarray
        (len(time_vec), n_bins) array corresponding to the target matrix
    """

    if (coefficients == "svd"):
        compute_coefficients = _compute_svd_coefficients
    elif (coefficients == "fsvd"):
        compute_coefficients = _compute_fsvd_coefficients
    elif (coefficients == "mfi"):
        compute_coefficients = _compute_mfi_coefficients
    elif (coefficients == "mti"):
        compute_coefficients = _compute_mti_coefficients
    else:
        raise NotImplementedError(
            "Unknown B0 correction coefficients: {}".format(coefficients))
    return compute_coefficients(field_map, time_vec, mask, num_interpolators,
                                weights, n_bins)


def _create_histogram(field_map, mask, weights="full", n_bins=1000):
    """ Create the weighted histogram of the field map covered by the mask

    Parameters
    ----------
    field_map: numpy.ndarray
        B0 field inhomogeneity map (in Hz)
    mask: numpy.ndarray
        Mask describing the regions to consider during correction
    weights: {'full', 'sqrt', 'log', 'ones'}
        Weightning policy for the field map histogram (default 'full')
    n_bins: int
        Number of bins for the field map histogram (default 1000)

    Returns
    -------
    histogram_centers: numpy.ndarray
        Central values of each bin
    histogram_counts: numpy.ndarray
        Weights of elements in each bin according to args
    """
    # Compute the histogram centers and counts
    field_map = field_map[np.where(mask)]
    histogram_counts, histogram_edges = np.histogram(field_map, n_bins)
    histogram_centers = (histogram_edges + (histogram_edges[1]
                                            - histogram_edges[0]) / 2)[:-1]

    # Change the weightning according to args
    if (weights == "ones"):
        histogram_counts = np.array(histogram_counts != 0).astype(int)
    elif (weights == "sqrt"):
        histogram_counts = np.sqrt(histogram_counts)
    elif (weights == "log"):
        histogram_counts = np.log(1 + histogram_counts)
    elif (weights != "full"):
        raise NotImplementedError("Unknown weightning: {}".format(weights))
    return histogram_centers, histogram_counts


def _create_variable_density(centers, counts, L):
    """ Find a reduced histogram with variable density from previous histogram

    Parameters
    ----------
    centers: numpy.ndarray
        Central value of each bin from previous histogram
    counts: numpy.ndarray
        Number of elements in each bin from previous histogram
    L: int
        Number of interpolators defining the size of the reduced histogram

    Returns
    -------
    centers: numpy.ndarray
        New bin centers of the reduced histogram
    """

    # Compute kmeans to get custom centers
    km = sc.KMeans(n_clusters=L, random_state=0)
    km = km.fit(centers.reshape((-1, 1)), sample_weight=counts)
    centers = np.array(sorted(km.cluster_centers_)).flatten()
    return centers


def _compute_mfi_coefficients(field_map, time_vec, mask, L=15,
                              weights="full", n_bins=1000):
    # Format the input and apply the weight option
    field_map = 2 * np.pi * field_map
    w_k, h_k = _create_histogram(field_map, mask, weights, n_bins)
    if (weights == "ones"):
        w_l = np.linspace(np.min(field_map), np.max(field_map), L)
    else:
        w_l = _create_variable_density(w_k, h_k, L)
    h_k = h_k.reshape((1, -1))

    # Compute B as a phase shift
    B = np.exp(1j * np.outer(time_vec, w_l))
    E = np.exp(1j * np.outer(time_vec, w_k))

    # Compute C with a Least Squares interpolation
    C, _, _, _ = np.linalg.lstsq(B, E, rcond=None)
    return B, C, E


def _compute_mti_coefficients(field_map, time_vec, mask, L=15,
                              weights="full", n_bins=1000):
    # Format the input and apply the weight option
    field_map = 2 * np.pi * field_map
    w_k, h_k = _create_histogram(field_map, mask, weights, n_bins)
    h_k = h_k.reshape((-1, 1))
    t_l = np.linspace(np.min(time_vec), np.max(time_vec), L)

    # Compute C as a phase shift
    C = np.exp(1j * np.outer(w_k, t_l))
    E = np.exp(1j * np.outer(w_k, time_vec))

    # Compute B with a Least Squares interpolation
    B, _, _, _ = np.linalg.lstsq(np.sqrt(h_k) * C,
                                 np.sqrt(h_k) * E, rcond=None)
    return B.T, C.T, E.T


def _compute_svd_coefficients(field_map, time_vec, mask, L=15,
                              weights="full", n_bins=1000):
    # Format the input and apply the weight option
    field_map = 2 * np.pi * field_map
    w_k, h_k = _create_histogram(field_map, mask, weights, n_bins)
    h_k = h_k.reshape((1, -1))

    # Compute B with a Singular Value Decomposition
    E = np.exp(1j * np.outer(time_vec, w_k))
    u, _, _ = np.linalg.svd(np.sqrt(h_k) * E)
    B = u[:, :L]

    # Compute C with a Least Squares interpolation
    # (Redundant with C=DV from E=UDV using L singular values
    # when weights are set to "ones")
    C, _, _, _ = np.linalg.lstsq(B, E, rcond=None)
    return B, C, E


def _compute_fsvd_coefficients(field_map, time_vec, mask, L=15,
                               weights="full", n_bins=1000):
    # Format the input and apply the weight option
    field_map = 2 * np.pi * field_map
    h_k, w_k = _create_histogram(field_map, mask, weights, n_bins)
    h_k = h_k.reshape((1, -1))

    # Compute B with an approximative Singular Value Decomposition
    E = np.exp(1j * np.outer(time_vec, w_k))
    B, _, _ = ssl.svds(np.sqrt(h_k) * E, L)

    # Compute C with a Least Square interpolation
    # (Redundant with C=DV from E=UDV using L singular values
    # but it avoids 0 division issues when weighted)
    C, _, _, _ = np.linalg.lstsq(B, E, rcond=None)
    return B, C, E
