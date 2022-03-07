# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""Provide linear operators classes dedicated to dictionary learning."""


# Third party import
import numpy
from sklearn.feature_extraction.image import reconstruct_from_patches_2d

# Package import
from ..base import OperatorBase
from .utils import extract_patches_from_2d_images


class DictionaryLearning(OperatorBase):
    """Sparse encoder using a learnt dictionary.

    This implementation relies on `MiniBatchDictionaryLearning`
    and its back projection from sklearn

    Parameters
    ----------
    img_shape: tuple of int,
        shape of the image (not necessarly a square image).
    dictonary_r: sklearn MiniBatchDictionaryLearning object
        containing the 'transform' method.
    dictonary_i: (default=None)
        sklearn MiniBatchDictionaryLearning object if images are
        complex-valued, None if images are real-valued.

    Attributes
    ----------
    two_dico: bool

    """

    def __init__(self, img_shape, dictionary_r, dictionary_i=None):
        self.dictionary_r = dictionary_r
        self.is_complex = False
        self.two_dico = False

        if dictionary_i is not None:
            if (dictionary_r.components_.shape !=
                    dictionary_i.components_.shape):
                raise ValueError(
                    "Real and imaginary atoms should have the same dimension, "
                    f"found {dictionary_r.components_.shape} for real"
                    f"and {dictionary_i.components_.shape} for imaginary."
                )
            self.two_dico = True
            self.dictionary_i = dictionary_i
        elif dictionary_r.components_.dtype is "complex":
            self.two_dico = False

        if numpy.sqrt(dictionary_r.components_.shape[1]) % 1 != 0:
            raise ValueError("Patches should have iso-dimension.")

        patches_size = int(numpy.sqrt(dictionary_r.components_.shape[1]))
        self.patches_shape = (patches_size, patches_size)
        self.img_shape = img_shape
        self.coeff = None

    def _op(self, dictionary, image):  # XXX works for square patches only!
        """Private operator for real-valued images and dictionaries.

        This method returns the representation of the input data in the
        learnt dictionary, e.g the sparse coefficients.

        Parameters
        ----------
        dictionary: sklearn MiniBatchDictionaryLearning object
            containing the 'transform' method
        image: ndarray
            Input data array, a 2D image.

        Returns
        -------
        coeffs: ndarray of floats, 2d matrix dim nb_patches*nb_components,
                the sparse coefficients.
        """
        patches = extract_patches_from_2d_images(image, self.patches_shape)
        return dictionary.transform(numpy.nan_to_num(patches))

    def op(self, data):
        """Operator.

        This method returns the representation of the input data in the
        learnt dictionary, that is to say the sparse coefficients.

        Remark: This method only works for squared patches

        Parameters
        ----------
        data: ndarray
            Input data array, a 2D image.

        Returns
        -------
        coeffs: ndarray of complex if is_complex, default(float)
                2d matrix dim nb_patches*nb_components, the sparse
                coefficients.
        """
        if self.is_complex:
            return self._op(self.dictionary_r, data)
        if self.two_dico:
            coeff_r = self._op(self.dictionary_r, numpy.real(data))
            return coeff_r + 1j * self._op(self.dictionary_i,
                                           numpy.imag(data))
        return self._op(self.dictionary_r, numpy.real(data))

    def _adj_op(self, coeffs, atoms):
        """Private Adjoint operator.

        This method returns the reconsructed image from the sparse
        coefficients.

        Parameters
        ----------
        coeffs: ndarray of floats,
                2d matrix dim nb_patches*nb_components,
                the sparse coefficients.
        atoms: ndarray of floats,
                2d matrix dim nb_components*nb_pixels_per_patch,
                the dictionary components.

        Returns
        -------
        ndarray, the reconstructed data.

        Notes
        -----
        This method only works for squared patches
        """
        image = numpy.dot(coeffs, atoms)
        image = image.reshape((image.shape[0], *self.patches_shape))
        return reconstruct_from_patches_2d(image, self.img_shape)

    def adj_op(self, coeffs):
        """Adjoint operator.

        This method returns the reconsructed image from the sparse
        coefficients.

        Parameters
        ----------
        coeffs: ndarray of floats,
                2d matrix dim nb_patches*nb_components,
                the sparse coefficients.

        Returns
        -------
        ndarray, the reconstructed data.

        Notes
        -----
        This method only works for squared patches
        """
        image_r = self._adj_op(
            numpy.real(coeffs),
            self.dictionary_r.components_,
        )
        if self.is_complex:
            return image_r + 1j * self._adj_op(
                numpy.imag(coeffs),
                self.dictionary_i.components_,
            )
        return image_r

    def l2norm(self, data_shape):
        """Compute the L2 norm.

        Parameters
        ----------
        data_shape: uplet
            the data shape.

        Returns
        -------
        norm: float
            the L2 norm.
        """
        # Create fake data.
        data_shape = numpy.asarray(data_shape)
        data_shape += data_shape % 2
        fake_data = numpy.zeros(data_shape)
        fake_data[zip(data_shape / 2)] = 1

        # Call the direct operator.
        data = self.op(fake_data)

        # Compute the L2 norm
        return numpy.linalg.norm(data)
