"""
Neuroimaging cartesian reconstruction
=====================================

Credit: Chaithya G R
Load results
"""

# Package import
from mri.reconstruct.utils import imshow3D

# Third party import
import numpy as np

A = np.load("/neurospin/optimed/Chaithya/Results/D6M9Y2019_runtest.npy")
imshow3D(np.abs(A))
