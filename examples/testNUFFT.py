"""
Neuroimaging cartesian reconstruction
=====================================

Credit: L. El Gueddari

In this tutorial we will reconstruct an MRI image from the sparse kspace
measurments.

"""

# Package import
from pysap.data import get_sample_data
from mri.reconstruct.fourier import NUFFT, NFFT
from mri.reconstruct.utils import normalize_frequency_locations

# Third party import
import numpy as np

samples = get_sample_data("mri-radial-3d-samples").data
samples = normalize_frequency_locations(samples)
#############################################################################
# Generate the kspace
# -------------------
#
# From the 2D brain slice and the acquistion mask, we generate the acquisition
# measurments, the observed kspace.
# We then reconstruct the zero order solution.

# Generate the subsampled kspace

gen_fourier_op = NUFFT(samples=samples,
                      shape=(128, 128, 160),
                       platform='gpu')

data = np.random.random((128, 128, 160))

op = gen_fourier_op.op(data)
