from scipy.io import savemat
import pickle as pkl
import nibabel as nib


def save_data(filename, recon, header=None):
    """Save reconstructed data to a file.

    Parameters
    ----------
    filename : str
        Path to the output file.
    recon : array_like
        Reconstructed data to be saved.
    header : object, optional
        Header information for the file format (default: None).

    """
    extension = filename.split('.')[-1].lower()
    save_dict = {'recon': recon, 'header': header}
    if extension == 'pkl':
        with open(filename, 'wb') as f:
            pkl.dump(save_dict, f)
    elif extension == 'mat':
        savemat(filename, save_dict)
    elif extension == 'nii':
        img = nib.Nifti1Image(recon)
        nib.save(img, filename)
    else:
        raise ValueError("Unsupported file extension.")
