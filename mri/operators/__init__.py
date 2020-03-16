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

from .fourier.cartesian import FFT
from .fourier.non_cartesian import NonCartesianFFT, Stacked3DNFFT
from .gradient.gradient import GradAnalysis, GradSynthesis, \
    GradSelfCalibrationAnalysis, GradSelfCalibrationSynthesis
from .linear.wavelet import WaveletN, WaveletUD2
from .linear.dictionary import DictionaryLearning
from .proximity.ordered_weighted_l1_norm import OWL
