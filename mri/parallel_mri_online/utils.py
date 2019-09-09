# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
This module contains all the utils tools needed in the p_MRI reconstruction.
"""


# System import

# Package import

# Third party import
import numpy as np
from itertools import product
from sklearn.feature_extraction.image import extract_patches
from sklearn.feature_extraction.image import _compute_n_patches
# from sklearn.feature_extraction.image import
from skimage.measure import compare_ssim
import warnings


def mat2gray(image):
    """Rescale the image between 0 and 1

    Parameters:
    ----------
    image: np.ndarray
        The image complex or not that has to be rescaled. If the image is
        complex the returned image will be taken by the abs of the image
    Returns:
    -------
    out: np.ndarray
        The returned image
    """
    abs_image = np.abs(image)
    return (abs_image - abs_image.min())/(abs_image.max() - abs_image.min())


def compute_ssim(ref, image, mask=None):
    """Compute the SSIM from the refernce on a rescaled image

    Parameter:
    ----------
    ref: np.ndarray
        The reference image
    image: np.ndarray

    mask: np.ndarray
        A binary mask where the ssim should be calvulated
    Output:
    ------
    ssim; np.float
        SSIM value between 0 and 1
    """
    if mask is None:
        return compare_ssim(mat2gray(ref), mat2gray(image))
    else:
        _, maps_ssim = compare_ssim(mat2gray(ref), mat2gray(image),
                                    full=True)
        maps_ssim = mask*maps_ssim
        return maps_ssim.sum()/mask.sum()


def prod_over_maps(S, X):
    """
    Computes the element-wise product of the two inputs over the first two
    direction

    Parameters
    ----------
    S: np.ndarray
        The sensitivity maps of size [N,M,L]
    X: np.ndarray
        An image of size [N,M]

    Returns
    -------
    Sl: np.ndarray
        The product of every L element of S times X
    """
    Sl = np.copy(S)
    if Sl.shape == X.shape:
        for i in range(S.shape[2]):
            Sl[:, :, i] *= X[:, :, i]
    else:
        for i in range(S.shape[2]):
            Sl[:, :, i] *= X
    return Sl


def check_lipschitz_cst(f, x_shape, lipschitz_cst, max_nb_of_iter=10):
    """
    This methods check that for random entrees the lipschitz constraint are
    statisfied:

    * ||f(x)-f(y)|| < lipschitz_cst ||x-y||

    Parameters
    ----------
    f: callable
        This lipschitzien function
    x_shape: tuple
        Input data shape
    lipschitz_cst: float
        The Lischitz constant for the function f
    max_nb_of_iter: int
        The number of time the constraint must be satisfied

    Returns
    -------
    out: bool
        If is True than the lipschitz_cst given in argument seems to be an
        upper bound of the real lipschitz constant for the function f
    """
    is_lips_cst = True
    n = 0

    while is_lips_cst and n < max_nb_of_iter:
        n += 1
        x = np.random.randn(*x_shape)
        y = np.random.randn(*x_shape)
        is_lips_cst = (np.linalg.norm(f(x)-f(y)) <= (lipschitz_cst *
                                                     np.linalg.norm(x-y)))

    return is_lips_cst


def extract_patches_2d(image, patch_shape, overlapping_factor=1):

    i_h, i_w = image.shape[:2]
    p_h, p_w = patch_shape[:2]
    patch_step_size = (int(p_h/overlapping_factor),
                       int(p_w/overlapping_factor),
                       image.shape[-1])
    if p_h > i_h:
        raise ValueError("Height of the patch should be less than the height"
                         " of the image.")

    if p_w > i_w:
        raise ValueError("Width of the patch should be less than the width"
                         " of the image.")
    image = image.reshape((i_h, i_w, -1))
    n_colors = image.shape[-1]
    extracted_patches = extract_patches(image,
                                        patch_shape=(p_h, p_w, n_colors),
                                        extraction_step=patch_step_size)
    patches = extracted_patches

    patches = patches.reshape(-1, p_h, p_w, n_colors)
    # remove the color dimension if useless
    if patches.shape[-1] == 1:
        return np.squeeze(patches)
    else:
        return patches


def reconstruct_non_overlapped_patches_2d(patches, img_size):
    patches_size = patches.shape[1:-1]
    number_of_patch_x = int(img_size[0]/patches_size[0])
    number_of_patch_y = int(img_size[1]/patches_size[1])
    IMG = np.zeros((patches.shape[-1], *img_size), dtype=patches.dtype)
    for idx_y in range(number_of_patch_y):
        for idx_x in range(number_of_patch_x):
            patch_idx = idx_x*number_of_patch_y+idx_y
            patch_n = np.moveaxis(patches[patch_idx], -1, 0)
            IMG[:, idx_x*patches_size[0]: (idx_x+1)*patches_size[0],
                idx_y*patches_size[1]: (idx_y+1)*patches_size[1]] = patch_n
    return IMG


def reconstruct_overlapped_patches_2d(patches, img_size, extraction_step_size):
    i_h, i_w = img_size[:2]
    p_h, p_w = patches.shape[1:3]
    img = np.zeros(img_size).astype(patches.dtype)
    # compute the dimensions of the patches array
    n_h = i_h - p_h + 1
    n_w = i_w - p_w + 1
    ratio_h = p_h * 1.0 / extraction_step_size[0]
    ratio_w = p_w * 1.0 / extraction_step_size[1]
    stop_h = int((ratio_h - 1) * extraction_step_size[0])
    stop_w = int((ratio_w - 1) * extraction_step_size[1])
    vect_n_h = np.arange(0, i_h-stop_h, extraction_step_size[0])
    vect_n_w = np.arange(0, i_w-stop_w, extraction_step_size[1])
    weights = np.zeros(img_size)
    for p, (i, j) in zip(patches, product(vect_n_h, vect_n_w)):
        img[i:i + p_h, j:j + p_w] += p
        weights[i:i + p_h, j:j + p_w] += 1
    return img/weights


def _oscar_weights(alpha, beta, size):
    w = np.arange(size-1, -1, -1, dtype=np.double)
    w *= beta
    w += alpha
    return w


def compress_data(kspace, nb_of_channel_to_keep):
    if nb_of_channel_to_keep < kspace.shape[0]:
        U, s, V = np.linalg.svd(kspace, full_matrices=False)
        matrix_compression = np.dot(S[:, :nb_of_channel_to_keep],
                                    np.diag[:nb_of_channel_to_keep])
        return np.dot(kspace.T, matrix_compression).T
    else:
        warnings.warn("The data doesn't need to be compressed")
        return kspace


def flatten_swt2(x):
    """ Flatten list an array.

    Parameters
    ----------
    x: list of dict or ndarray
        the input data

    Returns
    -------
    y: ndarray 1D
        the flatten input list of array.
    shape: list of dict
        the input list of array structure.
    """
    # Check input
    if not isinstance(x, list):
        x = [x]
    elif len(x) == 0:
        return None, None

    # Flatten the dataset
    y = []
    coeffs_shape = []
    for cA, cD in x:
        level_shape = []
        y = np.concatenate((y, cA.flatten()))
        level_shape.append(cA.shape)
        shape_details = []
        for cD_ in cD:
            y = np.concatenate((y, cD_.flatten()))
            shape_details.append(cD_.shape)
        level_shape.append(shape_details)
        coeffs_shape.append(level_shape)

    return np.asarray(y), coeffs_shape


def unflatten_swt2(y, shape):
    """ Unflatten a flattened array.

    Parameters
    ----------
    y: ndarray 1D
        a flattened input array.
    shape: list of dict
        the output structure information.

    Returns
    -------
    x: list of ndarray
        the unflattened dataset.
    """
    # Unflatten the dataset
    x = []
    offset = 0
    for cA_shape, cD_shape in shape:
        level = []
        cA = np.reshape(y[offset: offset + np.prod(cA_shape)], cA_shape)
        offset += np.prod(cA_shape)
        level.append(cA)
        cD = []
        for cD_ in cD_shape:
            cD.append(np.reshape(y[offset: offset + np.prod(cD_)], cD_))
            offset += np.prod(cA_shape)
        level.append(cD)
        x.append(tuple(level))
    return x


def flatten_wave2(x):
    """ Flatten list an array.

    Parameters
    ----------
    x: list of dict or ndarray
        the input data

    Returns
    -------
    y: ndarray 1D
        the flatten input list of array.
    shape: list of dict
        the input list of array structure.
    """

    # Flatten the dataset
    if not isinstance(x, list):
        x = [x]
    elif len(x) == 0:
        return None, None

    # Flatten the dataset
    y = x[0].flatten()
    shape_dict = [x[0].shape]
    for x_i in x[1:]:
        shape_coeffs = []
        for cn in x_i:
            y = np.concatenate((y, cn.flatten()))
            shape_coeffs.append(cn.shape)
        shape_dict.append(shape_coeffs)

    return y, shape_dict


def unflatten_wave2(y, shape):
    """ Unflatten a flattened array.

    Parameters
    ----------
    y: ndarray 1D
        a flattened input array.
    shape: list of dict
        the output structure information.

    Returns
    -------
    x: list of ndarray
        the unflattened dataset.
    """
    # Unflatten the dataset
    start = 0
    stop = np.prod(shape[0])
    x = [y[start:stop].reshape(shape[0])]
    offset = stop
    for shape_i in shape[1:]:
        sublevel = []
        for value in shape_i:
            start = offset
            stop = offset + np.prod(value)
            offset = stop
            sublevel.append(y[start: stop].reshape(value))
        x.append(sublevel)
    return x


def flatten_dwt2(x):
    """ Flatten list an array.

    Parameters
    ----------
    x: list of dict or ndarray
        the input data

    Returns
    -------
    y: ndarray 1D
        the flatten input list of array.
    shape: list of dict
        the input list of array structure.
    """

    # Flatten the dataset
    if not isinstance(x, tuple):
        x = [x]
    elif len(x) == 0:
        return None, None

    # Flatten the dataset
    y = x[0].flatten()
    shape_dict = [x[0].shape]
    (cHn, cVn, cDn) = x[1]
    y = np.concatenate((y, cHn.flatten()))
    y = np.concatenate((y, cVn.flatten()))
    y = np.concatenate((y, cDn.flatten()))
    shape_dict.append((cHn.shape, cVn.shape, cDn.shape))

    return y, shape_dict


def unflatten_dwt2(y, shape):
    """ Unflatten a flattened array.

    Parameters
    ----------
    y: ndarray 1D
        a flattened input array.
    shape: list of dict
        the output structure information.

    Returns
    -------
    x: list of ndarray
        the unflattened dataset.
    """
    # Unflatten the dataset
    start = 0
    stop = np.prod(shape[0])
    x = [y[start:stop].reshape(shape[0])]
    offset = stop
    sublevel = []
    for shape_i in shape[1]:
        start = offset
        stop = offset + np.prod(shape_i)
        offset = stop
        sublevel.append(y[start: stop].reshape(shape_i))
    x.append(tuple(sublevel))
    return tuple(x)


def reshape_dwt2_coeff_channel(wavelet_coeff, linear_op):
    """ Reshape the list of wavelet coeffiscients of length (channel number)
    on a list of wavelet coeffiscient of length (number of band) where each
    band is the concatenation of all the channel
    Parameters:
    ----------
    wavelet_coeff: np.ndarray
        The flattened wavelet coeffiscient
    linear_op: instance of PyWavelet2 or Wavelet2
        The linear op used to generate the wavelet coeffiscients
    Return:
        list of all the band
    ------
    """
    wavelet_coeff_unflattened = [linear_op.unflatten(
        wavelet_coeff[ch],
        linear_op.coeffs_shape[ch]) for ch in range(wavelet_coeff.shape[0])]
    coeffs_cA = []
    coeffs_cDh = []
    coeffs_cDv = []
    coeffs_cDd = []

    for cA_per_channel, cD_per_channel in wavelet_coeff_unflattened:
        coeffs_cA.append(cA_per_channel)
        cDh, cDv, cDd = cD_per_channel
        coeffs_cDh.append(cDh)
        coeffs_cDv.append(cDv)
        coeffs_cDd.append(cDd)
    return [np.asarray(coeffs_cA), np.asarray(coeffs_cDh),
            np.asarray(coeffs_cDv), np.asarray(coeffs_cDd)]


def reshape_dwt2_channel_coeff(wavelet_coeff, linear_op):
    """ Reshape a list of band wavevelet coeffiscient into same shape as the
    wavelet coefficsient given by the pywt.dwt2
     Parameters:
     ----------
     wavelet_coeff: np.ndarray
         The flattened wavelet coeffiscient
     linear_op: instance of PyWavelet2 or Wavelet2
         The linear op used to generate the wavelet coeffiscients
     Return:
         list of all the wavelet coeffiscient per channel
    """
    channel_nb = wavelet_coeff[0].shape[0]
    coeffs = []
    cA_all = wavelet_coeff[0]
    cDh_all = wavelet_coeff[1]
    cDv_all = wavelet_coeff[2]
    cDd_all = wavelet_coeff[3]
    for ch in range(channel_nb):
        cA = cA_all[ch]
        cD = tuple([cDh_all[ch], cDv_all[ch], cDd_all[ch]])
        coeffs.append((cA, cD))
    return coeffs
