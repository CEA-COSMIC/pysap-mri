"""
Base Class for k-space generators.

Kspace Generators emulates the acquisition of MRI data, by providing sequential access to data.
"""

import numpy as np
import progressbar


class KspaceGeneratorBase:
    """
    Basic K-space Generator emulate the acquisition of an MRI.

    K-space generator are regular Python generators, with extra function to access data property.

    At each iteration the relevant kspace data and mask is returned.

    Parameters
    ----------
    full_kspace: numpy.ndarray
        The fully sampled kspace, which will be returned incrementally,
        use for the Fourier transform.
    mask: numpy.ndarray
        A binary mask, giving the sampled location for the kspace
    """

    def __init__(self, full_kspace: np.ndarray, mask: np.ndarray, max_iter: int = 1):
        """Create a Basic k-space generator.

        Parameters
        -----------
        full_kspace: ndarray
            The full kspace data
        mask: ndarray
            The mask for undersampling the k-space data.
        max_iter: int
            Maximum number of iterations to be yields.
        """
        self._full_kspace = full_kspace
        self.kspace = full_kspace.copy()
        self.mask = mask
        self._len = max_iter
        self.iter = 0

    @property
    def shape(self):
        """Return shape of full_kspace."""
        return self._full_kspace.shape

    @property
    def dtype(self):
        """Return dtype of full_kspace."""
        return self._full_kspace.dtype

    def __len__(self):
        """The Number of kspace acquisitions available."""
        return self._len

    def __iter__(self):
        return self

    def __getitem__(self, idx):
        if idx >= self._len:
            raise IndexError
        return self._full_kspace, self.mask

    def __next__(self):
        if self.iter < self._len:
            self.iter += 1
            return self._full_kspace, self.mask
        raise StopIteration

    def reset(self):
        """Reset the Generator to its initial state."""
        self.iter = 0

    def opt_iterate(self, opt, estimate_call_period=None):
        """Run a optimizer with updates provided by the generator.

        Parameters
        ----------
        opt: Instance of SetUp
            The optimisation algorithm to run.
        estimate_call_period: int, optional
            The period over which to retrieve an estimate of the online algorithm.
            If None, only the last estimate is retrieved.

        Returns
        -------
        x_new_list: array_like
            array of the different reconstructions estimations.

        See Also
        --------
        module modopt.opt.algorithms

        """
        x_new_list = []
        #FIXME Modopt Setup Operator does not systematically have idx defined
        opt.idx = 0
        for (kspace, mask) in progressbar.progressbar(self):
            opt.idx += 1
            opt._grad.obs_data = kspace
            opt._grad.fourier_op.mask = mask
            opt._update()
            if opt.metrics and opt.metric_call_period is not None:
                if opt.idx % opt.metric_call_period == 0 or opt.idx == (self._len - 1):
                    opt._compute_metrics()
            if estimate_call_period is not None:
                if opt.idx % estimate_call_period == 0 or opt.idx == (self._len - 1):
                    x_new_list.append(opt.get_notify_observers_kwargs()["x_new"])
        opt.retrieve_outputs()
        return np.asarray(x_new_list)
