# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

""" This module defines the common operators.
"""

from ..fourier.utils import convert_mask_to_locations, \
    convert_locations_to_mask, normalize_frequency_locations, \
    get_stacks_fourier, gridded_inverse_fourier_transform_nd, \
    gridded_inverse_fourier_transform_stack, check_if_fourier_op_uses_sense
from ..linear.utils import extract_patches_from_2d_images, min_max_normalize, \
    learn_dictionary
from ..gradient.utils import check_lipschitz_cst
