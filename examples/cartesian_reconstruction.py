"""
Neuroimaging cartesian reconstruction
=====================================

Credit: A Grigis, L Elgueddari, H Carrie

In this tutorial we will reconstruct an MRI image from the sparse kspace
measurments.

Import neuroimaging data
------------------------

We use the toy datasets available in pysap, more specifically a 2D brain slice
and the acquistion cartesian scheme.
We also add some gaussian noise in the image space.
"""

# Package import
import pysap
from pysap.data import get_sample_data
from mri.numerics.reconstruct import sparse_rec_fista
from mri.numerics.fourier import FFT2
from mri.numerics.utils import generate_operators
from mri.numerics.utils import convert_mask_to_locations, \
    convert_locations_to_mask
from modopt.math.metrics import ssim

# Third party import
import numpy as np


def generate_cartesian_mask(img_size, acceleration_factor=3, decay=1, save_filename=None):
    kspace_lines = np.linspace(-1/2., 1/2., img_size) * img_size
    # Define the sampling density
    p_decay = np.power(np.abs(kspace_lines),-decay)
    p_decay = p_decay/np.sum(p_decay)
    cdf_pdecay = np.cumsum(p_decay)
    # generate its CDF
    nb_samples = (int)(img_size/acceleration_factor)
    samples = np.random.uniform(0, 1, nb_samples)
    gen_klines = [int(kspace_lines[np.argwhere(cdf_pdecay == min(cdf_pdecay[(cdf_pdecay - r) > 0]))])
                  for r in samples]
    gen_klines_int = ((np.array(gen_klines) - 1) / 1).astype(int) + (int)(img_size/2)
    sampled_klines = np.array(np.unique(gen_klines_int))
    kspace_mask = np.zeros((img_size,img_size), dtype="float64")
    nblines = np.size(sampled_klines)
    kspace_mask[sampled_klines, :] = np.ones((nblines,img_size) , dtype="float64")
    mask = pysap.Image(data=kspace_mask)
    if save_filename is not None:
        pysap.utils.save_image(mask, save_filename)
    return mask

# Loading input data
image = pysap.utils.load_image('../../../Data/Pysap_examples/base_image.npy')

# Obtain K-Space Cartesian Mask
try:
    mask = pysap.utils.load_image('../../../Data/Pysap_examples/mask_cartesian.npy')
except:
    mask = generate_cartesian_mask(image.shape[0], save_filename='../../../Data/Pysap_examples/mask_cartesian.npy')

# View Input
image.show()
mask.show()

#############################################################################
# Generate the kspace
# -------------------
#
# From the 2D brain slice and the acquistion mask, we generate the acquisition
# measurments, the observed kspace.
# We then reconstruct the zero order solution.


# Get the locations of the kspace samples
kspace_loc = convert_mask_to_locations(np.fft.fftshift(mask.data))
# Generate the subsampled kspace
fourier_op = FFT2(samples=kspace_loc, shape=image.shape)
kspace_data = fourier_op.op(image)

# Zero order solution
image_rec0 = pysap.Image(data=fourier_op.adj_op(kspace_data),
                         metadata=image.metadata)
image_rec0.show()
pysap.utils.save_image(image_rec0, '../../../Data/Pysap_examples/zero_order_reconstruct.npy')

base_ssim = ssim(image_rec0, image)
print(base_ssim)

#############################################################################
# FISTA optimization
# ------------------
#
# We now want to refine the zero order solution using a FISTA optimization.
# Here no cost function is set, and the optimization will reach the
# maximum number of iterations.

# Generate operators
my_ssims =[]
mus = []
gradient_op, linear_op, prox_op, cost_op = generate_operators(
    data=kspace_data,
    wavelet_name="sym8",
    samples=kspace_loc,
    nb_scales=4,
    mu=1e-06,
    non_cartesian=False,
    uniform_data_shape=None,
    gradient_space="synthesis",
    padding_mode="periodization")

# Start the FISTA reconstruction
max_iter = 200
x_final, transform, costs, metrics = sparse_rec_fista(
    gradient_op,
    linear_op,
    prox_op,
    cost_op,
    lambda_init=1.0,
    max_nb_of_iter=max_iter,
    atol=1e-4,
    verbose=1)
image_rec = pysap.Image(data=np.abs(x_final))
image_rec.show()
recon_ssim = ssim(image_rec, image)
print('The Reconstruction SSIM is : ' + str(recon_ssim))

