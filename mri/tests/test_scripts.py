# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import
import unittest
import numpy as np

# Package import
from mri.operators import FFT, WaveletN
from mri.operators.utils import convert_mask_to_locations
from mri.reconstructors import SingleChannelReconstructor
from mri.scripts.gridsearch import launch_grid

# Third party import
from modopt.math.metrics import ssim
from modopt.opt.linear import Identity
from modopt.opt.proximity import SparseThreshold
from pysap.data import get_sample_data


class TestScripts(unittest.TestCase):
    """ Test the scripts in pysap mri like gridsearch
    """
    def setUp(self):
        """ Setup the test variables.
        """
        self.n_jobs = 2

    def test_gridsearch_single_channel(self):
        """Test Gridsearch script in mri.scripts for
        single channel reconstruction this is a test of sanity
        and not if the reconstruction is right.
        """
        image = get_sample_data('2d-mri')
        mask = get_sample_data('cartesian-mri-mask')
        kspace_loc = convert_mask_to_locations(mask.data)
        fourier_op = FFT(samples=kspace_loc, shape=image.shape)
        kspace_data = fourier_op.op(image.data)
        # Define the keyword dictionaries based on convention
        metrics = {'ssim': {'metric': ssim,
                            'mapping': {'x_new': 'test', 'y_new': None},
                            'cst_kwargs': {'ref': image, 'mask': None},
                            'early_stopping': True,
                            },
                   }
        fourier_kwargs = {
            'init_class': FFT,
            'args':
                {
                    'samples': kspace_loc,
                    'shape': image.shape,
                }
        }
        linear_kwargs = {
            'init_class': WaveletN,
            'args':
                {
                    'wavelet_name': ['sym8'],
                    'nb_scale': [4]
                }
        }
        regularizer_kwargs = {
            'init_class': SparseThreshold,
            'args':
                {
                    'linear': Identity(),
                    'weights': np.geomspace(1e-8, 1e-6, 10),
                }
        }
        optimizer_kwargs = {
            # Just following convention
            'args':
                {
                    'optimization_alg': 'fista',
                    'num_iterations': 10,
                    'metrics': metrics,
                }
        }
        reconstructor_kwargs = {
            'init_class': SingleChannelReconstructor,
            'args':
                {
                    'gradient_formulation': 'synthesis',
                }
        }
        # Call the launch grid function and obtain results
        raw_results, test_cases, key_names, best_idx = launch_grid(
            kspace_data=kspace_data,
            fourier_kwargs=fourier_kwargs,
            linear_kwargs=linear_kwargs,
            regularizer_kwargs=regularizer_kwargs,
            optimizer_kwargs=optimizer_kwargs,
            reconstructor_kwargs=reconstructor_kwargs,
            compare_metric_details={'metric': 'ssim',
                                    'metric_direction': True},
            n_jobs=self.n_jobs,
            verbose=1,
        )


if __name__ == "__main__":
    unittest.main()
