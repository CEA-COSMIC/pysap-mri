# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Overload the proximity class from modopt.
"""

import numpy as np
import warnings
from modopt.opt.proximity import SparseThreshold
from mri.parallel_mri_online.utils import extract_patches_2d
from mri.parallel_mri_online.utils import \
    reconstruct_non_overlapped_patches_2d
from mri.parallel_mri_online.utils import \
    reconstruct_overlapped_patches_2d
from mri.reconstruct.linear import Identity
from joblib import Parallel, delayed
from mri.parallel_mri_online.utils import _oscar_weights
from sklearn.isotonic import isotonic_regression


class ElasticNet(object):
    """The proximity of the lasso regularisation
    This class defines the group-lasso penalization
    Parameters
    ----------
    weights : np.ndarray
        Input array of weights
    """

    def __init__(self, weights_lasso, weights_ridge, linear_op=Identity):
        """
        Parameters:
        -----------
        """
        self.prox_op = SparseThreshold(linear=linear_op,
                                       weights=weights_lasso,
                                       thresh_type='soft')
        self.weights_lasso = weights_lasso
        self.weights_ridge = weights_ridge

    def op(self, data, extra_factor=1.0):
        """ Operator
        This method returns the input data thresholded by the weights
        Parameters
        ----------
        data : DictionaryBase
            Input data array
        extra_factor : float
            Additional multiplication factor
        Returns
        -------
        DictionaryBase thresholded data
        """
        return np.reshape((1.0 / (1 + self.weights_lasso
                                  * 2 * self.weights_ridge)) *
                          self.prox_op._op_method(
                              data.flatten(), extra_factor),
                          data.shape)

    def cost(self, data):
        """Cost function
        This method calculate the cost function of the proximable part.
        Parameters
        ----------
        x: np.ndarray
            Input array of the sparse code.
        Returns
        -------
        The cost of this sparse code
        """
        return (self.weights_lasso * np.sum(np.abs(data.flatten())) +
                self.weights_ridge *
                np.sqrt(np.sum(np.abs(data.flatten()) ** 2)))


class NuclearNorm(object):
    """The proximity of the nuclear norm operator
    This class defines the nuclear norm proximity operator on a patch based
    method
    Parameters
    ----------
    weights : np.ndarray
        Input array of weights
    thresh_type : str {'hard', 'soft'}, optional
        Threshold type (default is 'soft')
    patch_size: int
        Size of the patches to impose the low rank constraints
    overlapping_factor: int
        if 1 no overlapping will be made,
        if = 2,means 2 patches overlaps
    """

    def __init__(self, weights, patch_shape, overlapping_factor=1,
                 num_cores=1, mode="image", linear_op=None):
        """
        Parameters:
        -----------
        """
        if mode not in ["image", "sparse"]:
            raise ValueError('The specified mode should be either image or',
                             'sparse coefficients')
        self.mode = mode
        if mode == "sparse" and linear_op is None:
            raise ValueError("The linear operator should be specified for the",
                             "sparse mode penalization")
        self.linear_op = linear_op
        self.weights = weights
        self.patch_shape = patch_shape
        self.overlapping_factor = overlapping_factor
        self.num_cores = num_cores
        if self.overlapping_factor == 1:
            print("Patches doesn't overlap")

    def _prox_nuclear_norm(self, patch, threshold):
        u, s, vh = np.linalg.svd(np.reshape(
            patch,
            (np.prod(patch.shape[:-1]), patch.shape[-1])),
            full_matrices=False)
        s = s * np.maximum(1 - threshold / np.maximum(
            np.finfo(np.float32).eps,
            np.abs(s)), 0)
        patch = np.reshape(
            np.dot(u * s, vh),
            patch.shape)
        return patch

    def _nuclear_norm_cost(self, patch):
        _, s, _ = np.linalg.svd(np.reshape(
            patch,
            (np.prod(patch.shape[:-1]), patch.shape[-1])),
            full_matrices=False)
        return np.sum(np.abs(s.flatten()))

    def op(self, data, extra_factor=1.0):
        """ Operator
        This method returns the input data thresholded by the weights
        Parameters
        ----------
        data : DictionaryBase
            Input data array
        extra_factor : float
            Additional multiplication factor
        Returns
        -------
        DictionaryBase thresholded data
        """
        threshold = self.weights * extra_factor
        if self.mode == "image":
            if data.shape[1:] == self.patch_shape:
                images = np.moveaxis(data, 0, -1)
                images = self._prox_nuclear_norm(patch=np.reshape(
                    np.moveaxis(data, 0, -1),
                    (np.prod(self.patch_shape), data.shape[0])),
                    threshold=threshold)
                return np.moveaxis(images, -1, 0)
            elif self.overlapping_factor == 1:
                P = extract_patches_2d(
                    np.moveaxis(data, 0, -1),
                    self.patch_shape,
                    overlapping_factor=self.overlapping_factor)
                number_of_patches = P.shape[0]
                if self.num_cores == 1:
                    for idx in range(number_of_patches):
                        P[idx, :, :, :] = self._prox_nuclear_norm(
                            patch=P[idx, :, :, :, ],
                            threshold=threshold)
                else:
                    P = Parallel(n_jobs=self.num_cores)(
                        delayed(self._prox_nuclear_norm)(
                            patch=P[idx, :, :, :],
                            threshold=threshold)
                        for idx in range(number_of_patches))
                output = reconstruct_non_overlapped_patches_2d(
                    patches=np.asarray(P),
                    img_size=data.shape[1:])
                return output
            else:
                P = extract_patches_2d(
                    np.moveaxis(data, 0, -1), self.patch_shape,
                    overlapping_factor=self.overlapping_factor)
                number_of_patches = P.shape[0]
                threshold = self.weights * extra_factor
                extraction_step_size = [
                    int(P_shape / self.overlapping_factor)
                    for P_shape in self.patch_shape]
                if self.num_cores == 1:
                    for idx in range(number_of_patches):
                        P[idx, :, :, :] = self._prox_nuclear_norm(
                            patch=P[idx, :, :, :],
                            threshold=threshold)
                else:
                    P = Parallel(n_jobs=self.num_cores)(
                        delayed(self._prox_nuclear_norm)(
                            patch=P[idx, :, :, :],
                            threshold=threshold)
                        for idx in range(number_of_patches))
                image = reconstruct_overlapped_patches_2d(
                    img_size=np.moveaxis(data, 0, -1).shape,
                    patches=np.asarray(P),
                    extraction_step_size=extraction_step_size)
                return np.moveaxis(image, -1, 0)
        elif self.mode == 'sparse':
            # have to do something
            coeffs = [self.linear_op.unflatten(
                data[ch],
                self.linear_op.coeffs_shape[ch])
                for ch in range(data.shape[0])]
            reordered_coeffs = []
            for coeff_idx in range(len(coeffs[0])):
                tmp = []
                for ch in range(data.shape[0]):
                    tmp.append(coeffs[ch][coeff_idx])
                reordered_coeffs.append(np.moveaxis(np.asarray(tmp), 0, -1))
            number_of_patches = len(reordered_coeffs)
            P = Parallel(n_jobs=self.num_cores)(delayed(
                self._prox_nuclear_norm)(
                patch=reordered_coeffs[idx],
                threshold=threshold)
                for idx in range(number_of_patches))
            # Equivalent to reconstruct patches
            output = []
            for ch in range(data.shape[0]):
                tmp = []
                for coeff_idx in range(len(coeffs[0])):
                    tmp.append(P[coeff_idx][..., ch])
                r_coeffs, _ = self.linear_op.flatten(tmp)
                output.append(r_coeffs)
            return np.asarray(output)

    def cost(self, data, extra_factor=1.0):
        """Cost function
        This method calculate the cost function of the proximable part.
        Parameters
        ----------
        x: np.ndarray
            Input array of the sparse code.
        Returns
        -------
        The cost of this sparse code
        """
        cost = 0
        threshold = self.weights * extra_factor
        if self.mode == "image":
            if data.shape[1:] == self.patch_shape:
                cost += self._nuclear_norm_cost(patch=np.reshape(
                    np.moveaxis(data, 0, -1),
                    (np.prod(self.patch_shape), data.shape[0])))
            elif self.overlapping_factor == 1:
                P = extract_patches_2d(
                    np.moveaxis(data, 0, -1),
                    self.patch_shape,
                    overlapping_factor=self.overlapping_factor)
                number_of_patches = P.shape[0]
                cost_list = Parallel(n_jobs=self.num_cores)(delayed(
                    self._cost_nuclear_norm)(
                    patch=P[idx, :, :, :]
                ) for idx in range(number_of_patches))
                cost = np.asarray(cost_list).sum()
            else:
                P = extract_patches_2d(
                    np.moveaxis(data, 0, -1), self.patch_shape,
                    overlapping_factor=self.overlapping_factor)
                number_of_patches = P.shape[0]
                threshold = self.weights * extra_factor
                cost_list = Parallel(n_jobs=self.num_cores)(delayed(
                    self._nuclear_norm_cost)(
                    patch=P[idx, :, :, :])
                    for idx in range(number_of_patches))
                cost = np.asarray(cost_list).sum()
        elif self.mode == "sparse":
            coeffs = [self.linear_op.unflatten(
                data[ch],
                self.linear_op.coeffs_shape[ch])
                for ch in range(data.shape[0])]
            reordered_coeffs = []
            for coeff_idx in range(len(coeffs[0])):
                tmp = []
                for ch in range(data.shape[0]):
                    tmp.append(coeffs[ch][coeff_idx])
                reordered_coeffs.append(np.moveaxis(np.asarray(tmp), 0, -1))
            number_of_patches = len(reordered_coeffs)
            cost_list = Parallel(n_jobs=self.num_cores)(
                delayed(self._nuclear_norm_cost)(
                    patch=reordered_coeffs[idx])
                for idx in range(number_of_patches))
            cost = np.asarray(cost_list).sum()
        return cost * threshold


class GroupLasso(object):
    """The proximity of the group-lasso regularisation
    This class defines the group-lasso penalization
    Parameters
    ----------
    weights : np.ndarray
        Input array of weights
    """

    def __init__(self, weights):
        """
        Parameters:
        -----------
        """
        self.weights = weights

    def op(self, data, extra_factor=1.0):
        """ Operator
        This method returns the input data thresholded by the weights
        Parameters
        ----------
        data : DictionaryBase
            Input data array
        extra_factor : float
            Additional multiplication factor
        Returns
        -------
        DictionaryBase thresholded data
        """
        norm_2 = np.linalg.norm(data, axis=0)
        return data * np.maximum(0, 1.0 - self.weights * extra_factor /
                                 np.maximum(norm_2, np.finfo(np.float32).eps))

    def cost(self, data):
        """Cost function
        This method calculate the cost function of the proximable part.
        Parameters
        ----------
        x: np.ndarray
            Input array of the sparse code.
        Returns
        -------
        The cost of this sparse code
        """
        return np.sum(np.linalg.norm(data, axis=0))


class SparseGroupLasso(SparseThreshold, GroupLasso):
    """The proximity of the sparse group-lasso regularisation
    This class defines the sparse group-lasso penalization
    Parameters
    ----------
    weights : np.ndarray
        Input array of weights
    """

    def __init__(self, weights_l1, weights_l2, linear_op=Identity):
        """
        Parameters:
        -----------
        """
        self.prox_op_l1 = SparseThreshold(linear=linear_op,
                                          weights=weights_l1,
                                          thresh_type='soft')
        self.prox_op_l2 = GroupLasso(weights=weights_l2)
        self.weights_l1 = weights_l1
        self.weights_l2 = weights_l2

    def op(self, data, extra_factor=1.0):
        """ Operator
        This method returns the input data thresholded by the weights
        Parameters
        ----------
        data : DictionaryBase
            Input data array
        extra_factor : float
            Additional multiplication factor
        Returns
        -------
        DictionaryBase thresholded data
        """

        return self.prox_op_l2.op(self.prox_op_l1.op(
            data,
            extra_factor=extra_factor),
            extra_factor=extra_factor)

    def cost(self, data):
        """Cost function
        This method calculate the cost function of the proximable part.
        Parameters
        ----------
        x: np.ndarray
            Input array of the sparse code.
        Returns
        -------
        The cost of this sparse code
        """
        return self.prox_op_l1.cost(data) + self.prox_op_l2.cost(data)


class OWL(object):
    """The proximity of the OWL regularisation
    This class defines the OWL penalization
    Parameters
    ----------
    weights : np.ndarray
        Input array of weights
    """

    def __init__(self, alpha, beta=None, bands_shape=None, mode='all',
                 n_channel=1, num_cores=1):
        """
        Parameters:
        -----------
        """
        self.weights = alpha
        self.mode = mode
        self.num_cores = num_cores
        if beta is not None:
            print("Uses OSCAR: Octogonal Shrinkage and Clustering Algorithm"
                  " for Regression")
            if bands_shape is None:
                raise ('Data size must be specified if OSCAR is used')
            else:
                if self.mode is 'all':
                    data_shape = 0
                    if n_channel > 1:
                        for band_shape in bands_shape[0]:
                            data_shape += np.prod(band_shape)
                    elif n_channel == 1:
                        for band_shape in bands_shape:
                            data_shape += np.prod(band_shape)
                    self.weights = np.reshape(
                        _oscar_weights(alpha, beta,
                                       data_shape * n_channel),
                        (n_channel, data_shape))
                elif self.mode is 'band_based':
                    if n_channel > 1:
                        self.band_shape = bands_shape[0]
                    elif n_channel == 1:
                        self.band_shape = bands_shape
                    else:
                        raise ValueError('Number of channels '
                                         'must be strictly positive')
                    self.weights = []
                    for band_shape in self.band_shape:
                        self.weights.append(_oscar_weights(
                            alpha, beta, n_channel * np.prod(band_shape)))
                elif self.mode is 'coeff_based':
                    self.weights = _oscar_weights(alpha, beta, n_channel)
                else:
                    raise ('Unknow mode')

    def _prox_owl(self, data, threshold):
        data_abs = np.abs(data)
        ix = np.argsort(np.squeeze(data_abs))[::-1]
        data_abs = data_abs[ix]  # Sorted absolute value of the data

        # Project on the monotone non-negative deacresing cone
        data_abs = isotonic_regression(data_abs - threshold, y_min=0,
                                       increasing=False)
        # Undo the sorting
        inv_x = np.zeros_like(ix)
        inv_x[ix] = np.arange(len(data))
        data_abs = data_abs[inv_x]

        sign_data = data / np.abs(data)
        sign_data[np.isnan(sign_data)] = 0

        return sign_data * data_abs

    def _reshape_mode_based(self, data):
        output = []
        start = 0
        n_channel = data.shape[0]
        for band_shape_idx in self.band_shape:
            n_coeffs = np.prod(band_shape_idx)
            stop = start + n_coeffs
            output.append(np.reshape(data[:, start: stop],
                                     (n_channel * n_coeffs)))
            start = stop
        return output

    def _prox_OWL_cost(self, data, weights):
        """Cost function of OWL
        :param Data : ndArray, holds array of observed data"""
        # Sort the data
        data_abs = np.sort(np.abs(data))
        return np.sum(data_abs * weights)

    def op(self, data, extra_factor=1.0):
        """
        Define the proximity operator of the OWL norm
        """
        if self.mode is 'all':
            threshold = self.weights.flatten() * extra_factor
            output = np.reshape(self._prox_owl(data.flatten(), threshold),
                                data.shape)
            return output
        elif self.mode is 'band_based':
            data_r = self._reshape_mode_based(data)
            output = []
            output = Parallel(n_jobs=self.num_cores)(delayed(self._prox_owl)(
                data=data_band,
                threshold=weights * extra_factor)
                for data_band, weights in zip(data_r, self.weights))
            reshaped_data = np.zeros(data.shape, dtype=data.dtype)
            start = 0
            n_channel = data.shape[0]

            for band_shape_idx, band_data in zip(self.band_shape, output):
                stop = start + np.prod(band_shape_idx)
                reshaped_data[:, start: stop] = \
                    np.reshape(band_data, (n_channel, np.prod(band_shape_idx)))
                start = stop
            output = np.asarray(reshaped_data).T
        elif self.mode is 'coeff_based':
            threshold = self.weights * extra_factor
            output = Parallel(n_jobs=self.num_cores)(delayed(self._prox_owl)(
                data=np.squeeze(data[:, idx]),
                threshold=threshold) for idx in range(data.shape[1]))
        return np.asarray(output).T

    def cost(self, data):
        """Cost function
        This method calculate the cost function of the proximable part.
        Parameters
        ----------
        x: np.ndarray
            Input array of the sparse code.
        Returns
        -------
        The cost of this sparse code
        """
        if self.mode is 'all':
            cost = self._prox_OWL_cost(data, self.weights)
        elif self.mode is 'band_based':
            data_r = self._reshape_mode_based(data)
            cost = 0
            for data_band, weights in zip(data_r, self.weights):
                cost = cost + self._prox_OWL_cost(data_band, weights)
            cost = 0
        elif self.mode is 'coeff_based':
            cost = 0
            for idx in range(data.shape[1]):
                cost = cost + \
                       self._prox_OWL_cost(
                           np.squeeze(data[:, idx]), self.weights)
        return cost


class k_support_norm(object):
    """The proximity of the squarre k-support norm regularisation
    This class defines the squarre of k-support norm regularization
    Parameters
    ----------
        weights: np.ndarray or float
            Hyperparameters for the regularisation
        k : float
            Parameter of the k-support norm.
            if k = 1 this is equivalent to an l_1 norm
            if k = d with d the dimension of input vector,
                it's equivalement to the l_2 norm
        Notes
        -----
        **References:**
        A.M. McDonald, M. Pontil, D.Stamos 2016: New perspective on k-support
        and cluster norm (http://jmlr.org/papers/volume17/15-151/15-151.pdf)
    """

    def __init__(self, k, lmbda):
        """
        Parameters:
        -----------
        weights: np.ndarray or float
            Hyperparameters for the regularisation
        k : float
            Parameter of the k-support norm.
            if k = 1 this is equivalent to an the squarre of the l_1 norm
            if k = d with d the dimension of input vector,
                it's equivalement to the squarre of the l_2 norm
        """
        self.weights = lmbda * 1.
        self.k = k

    def _compute_theta(self, w, alpha, extra_factor=1.0):
        """ Compute theta
        This method compute theta from Corollary 16
                    |1                        if Alpha |w_i| - 2 * lambda > 1
        Theta_i =   |Alpha |w_i| - 2 * lambda if 1 >= Alpha |w_i| -2*lambda>= 0
                    |0                        if 0 > Alpha |w_i| - 2 * lambda
        Parameters:
        ----------
        w: np.ndarray
            Input data
        alpha: float
            Parameter choosen such that sum(theta_i) = k
        extra_factor: float
            Potential extra factor comming from the optimization process
        Return:
        -------
        theta: np.ndarray
            Same size as w and each component is equal to theta_i
        """
        theta = np.zeros(w.shape)
        theta += 1.0 * ((np.abs(w) * alpha
                         - 2 * self.weights * extra_factor) > 1)
        theta += (alpha * np.abs(w) - 2 * self.weights * extra_factor) * (
                ((np.abs(w) * alpha - 2 * self.weights *
                  extra_factor) <= 1) &
                ((np.abs(w) * alpha - 2 * self.weights *
                  extra_factor) >= 0))
        return theta

    def _interpolate(self, alpha_0, alpha_1, sum_0, sum_1):
        """ Linear interpolation of alpha
        This method estimate alpha* such that sum(theta(alpha*))=k via a linear
        interpolation.
        Parameters:
        -----------
        alpha_0: float
            A value for wich sum(theta(alpha_0)) <= k
        alpha_1: float
            A value for which sum(theta(alpha_1)) <= k
        sum_0: float
            Value of sum(theta(alpha_0))
        sum_1:
            Value of sum(theta(alpha_0))
        Return:
        -------
        alpha_star: float
            An interpolation for wich sum(theta(alpha_star)) = k
        """
        if sum_0 == self.k:
            return alpha_0
        elif sum_1 == self.k:
            return alpha_1
        else:
            slope = (sum_1 - sum_0) / (alpha_1 - alpha_0)
            b = sum_0 - slope * sum_1
            alpha_star = (self.k - b) / slope
            return alpha_star

    def _binary_search(self, data, alpha, extra_factor=1.0):
        """ Binary search method
        This method finds i the coordinate of alpha such that
        sum(theta(alpha[i])) =<k and sum(theta(alpha[i+1]))>=k via binary
        search method
        Parameters:
        -----------
        data_abs: np.ndarray
            absolute avlue of the input data
        alpha: np.ndarray
            Array same size as the input data
        extra_factor: float
            Potential extra factor comming from the optimization process
        Returns:
        --------
        idx: int
            the index where: sum(theta(alpha[index])) <= k and
                             sum(theta(alpha[index+1]))>=k
        """
        first_idx = 0
        data_abs = np.abs(data)
        last_idx = data_abs.shape[0] - 1
        found = False
        prev_midpoint = 0
        cnt = 0  # Avoid infinite looops

        # Checking particular to be sure that the solution is in the array
        sum_0 = self._compute_theta(data_abs, alpha[0], extra_factor).sum()
        sum_1 = self._compute_theta(data_abs, alpha[data_abs.shape[0] - 1],
                                    extra_factor).sum()
        if sum_1 < self.k:
            midpoint = data_abs.shape[0] - 2
            found = True
        if sum_0 >= self.k:
            found = True
            midpoint = 0

        while (first_idx <= last_idx) and not found and (cnt < alpha.shape[0]):

            midpoint = (first_idx + last_idx) // 2
            cnt += 1

            if prev_midpoint == midpoint:

                # Particular case
                sum_0 = self._compute_theta(data_abs, alpha[first_idx],
                                            extra_factor).sum()
                sum_1 = self._compute_theta(data_abs, alpha[last_idx],
                                            extra_factor).sum()

                if (np.abs(sum_0 - 1e-4) == self.k):
                    found = True
                    midpoint = first_idx

                if (np.abs(sum_1 - 1e-4) == self.k):
                    found = True
                    midpoint = last_idx - 1
                    # -1 because output is index such that
                    # sum(theta(alpha[index])) <= k

                if first_idx - last_idx == 2 or first_idx - last_idx == 1:
                    sum_0 = self._compute_theta(data_abs, alpha[first_idx],
                                                extra_factor).sum()
                    sum_1 = self._compute_theta(data_abs, alpha[last_idx],
                                                extra_factor).sum()
                    if (sum_0 <= self.k) or (sum_1 >= self.k):
                        found = True

            sum_0 = self._compute_theta(data_abs, alpha[midpoint],
                                        extra_factor).sum()
            sum_1 = self._compute_theta(data_abs, alpha[midpoint + 1],
                                        extra_factor).sum()

            if (sum_0 <= self.k) & (sum_1 >= self.k):
                found = True

            elif sum_1 < self.k:
                first_idx = midpoint

            elif sum_0 > self.k:
                last_idx = midpoint

            prev_midpoint = midpoint

        if found:
            return alpha[midpoint], alpha[midpoint + 1], sum_0, sum_1
        else:
            return -1

    def _find_alpha(self, w, extra_factor=1.0):
        """ Find alpha value to compute theta.
        This method aim at finding alpha such that sum(theta(alpha)) = k
        Parameters:
        -----------
        w: np.ndarray
            Input data
        extra_factor: float
            Potential extra factor for the weights
        Return:
        -------
            alpha: float
                An interpolation of alpha such that sum(theta(alpha)) = k
        """
        alpha = np.zeros((w.shape[0] * 2))
        data_abs = np.abs(w)
        alpha[:w.shape[0]] = (self.weights * 2 * extra_factor) / data_abs
        alpha[w.shape[0]:] = (self.weights * 2 * extra_factor + 1) / data_abs
        alpha = np.sort(np.unique(alpha))
        alpha_0, alpha_1, sum_0, sum_1 = self._binary_search(w, alpha,
                                                             extra_factor)
        alpha_star = self._interpolate(alpha_0, alpha_1, sum_0, sum_1)
        return alpha_star

    def op(self, data, extra_factor=1.0):
        """
        Define the proximity operator of the k-support norm
        from Algorithm1 http://jmlr.org/papers/v17/15-151.html
        """
        data_shape = data.shape
        alpha = self._find_alpha(np.abs(data.flatten()), extra_factor)
        theta = self._compute_theta(np.abs(data.flatten()), alpha)
        rslt = ((data.flatten() * theta) /
                (theta + self.weights * 2 * extra_factor))
        return rslt.reshape(data_shape)

    def _find_q(self, sorted_data):
        """ Find q index value
        This method finds the value of q such that:
            sorted_data[q] >=
                    sum(sorted_data[q+1:]) / (k - q)>= sorted_data[q+1]
        Parameters:
        -----------
        sorted_data = np.ndarray
            Absolute value of the input data sorted in a non-decreasing order
        Return:
        -------
        q: int
            index such that
            sorted_data[q] >=
                sum(sorted_data[q+1:]) / (k - q)>= sorted_data[q+1]
        """
        first_idx = 0
        last_idx = self.k - 1
        found = False
        q = (first_idx + last_idx) // 2
        cnt = 0

        # Particular case
        if (sorted_data[0:].sum() / (self.k)) >= sorted_data[0]:
            found = True
            q = 0
        elif (sorted_data[self.k - 1:].sum()) <= sorted_data[self.k - 1]:
            found = True
            q = self.k - 1
        while (not found and cnt == self.k and (first_idx <= last_idx) and
               last_idx < self.k):
            q = (first_idx + last_idx) // 2
            cnt += 1
            l1_part = sorted_data[q:].sum() / (self.k - q)
            if sorted_data[q] >= l1_part and l1_part >= sorted_data[q + 1]:
                found = True
            else:
                if sorted_data[q] <= l1_part:
                    last_idx = q
                if l1_part <= sorted_data[q + 1]:
                    first_idx = q
        return q

    def cost(self, data):
        """Cost function
        This method calculate the cost function of the proximable part.
        Parameters
        ----------
        x: np.ndarray
            Input array of the sparse code.
        Returns
        -------
        The cost of this sparse code
        """
        data_abs = np.abs(data.flatten())
        ix = np.argsort(data_abs)[::-1]
        data_abs = data_abs[ix]  # Sorted absolute value of the data
        q = self._find_q(data_abs)
        rslt = (np.sum(data_abs[:q] ** 2) +
                data_abs[q + 1:].sum() / (self.k - q))
        return rslt
