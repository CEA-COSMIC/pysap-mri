"""
Neuroimaging cartesian reconstruction
=====================================

Author: Chaithya G R

In this tutorial we will use the pysap-mri's launch grid helper function
to carry out grid search. We will search for best regularisation weight
and the best wavelet for reconstruction.
For this the search space works on :
mu          ==> 5 Values on log scale between 1e-8 and 1e-9
Wavelets    ==> sym8 and sym12
nb_scale    ==> 3 and 4
"""

# Imports
from mri.operators import FFT, WaveletN
from mri.operators.utils import convert_mask_to_locations
from mri.reconstructors import SingleChannelReconstructor
from mri.scripts.gridsearch import launch_grid

from pysap.data import get_sample_data

from modopt.opt.proximity import SparseThreshold
from modopt.opt.linear import Identity
from modopt.math.metrics import ssim
import numpy as np


# Load MR data and obtain kspace
image = get_sample_data('2d-mri')
mask = get_sample_data("cartesian-mri-mask")
kspace_loc = convert_mask_to_locations(mask.data)
fourier_op = FFT(samples=kspace_loc, shape=image.shape)
kspace_data = fourier_op.op(image.data)
# Define the keyword dictionaries based on convention
ref = image
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
            'wavelet_name': ['sym8', 'sym12'],
            'nb_scale': [3, 4]
        }
}
regularizer_kwargs = {
    'init_class': SparseThreshold,
    'args':
        {
            'linear': Identity(),
            'weights': np.geomspace(1e-8, 1e-6, 5),
        }
}
optimizer_kwargs = {
    # Just following convention
    'args':
        {
            'optimization_alg': 'fista',
            'num_iterations': 20,
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
    n_jobs=32,
    verbose=1,
)
image_rec = raw_results[best_idx][0]
