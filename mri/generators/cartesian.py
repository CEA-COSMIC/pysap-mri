"""
Cartesian Kspace-Generators.

At each iteration, a column of kspace is yielded.
Transpose your data if you want to work row-wise.
"""

import numpy as np

from .base import KspaceGeneratorBase


class Column2DKspaceGenerator(KspaceGeneratorBase):
    """K-Space generator, at each step a new column fills the existing k-space.

    Parameters
    ----------
    full_kspace: ndarray
        Complete kspace_data.
    mask_cols: array_like
        List of the column indices to use for the mask.
    max_iter: int, optional
        The maximum number of iteration to do, default is
        the number of column provided.
    mode: {"line", "current", "memory"}
        If "line" : at step k, the generator yields only the
        data of k-th column, and the column number associated.
        If "current": at step k, the generator yields an array
        with the shape of full_kspace, but containing only the data
        of step k, and the mask associated.
        If "memory", same as "current", but all the data that have
        been previously acquired fills the kspace-array.
        Default is "current".
    start_center: bool, optional
        Should the acquisition defined by mask_cols be reordered
        to start from the center of kspace and move outward by alternating
        left and right. (default: True)
    """

    def __init__(self, full_kspace, mask_cols, max_iter=0, mode="current", start_center=True):
        mask = np.zeros(full_kspace.shape[-2:])

        if mode == "line":
            self.acquire = self._getitem_line
        elif mode == "current":
            self.acquire = self._getitem_current
        elif mode == "memory":
            self.acquire = self._getitem_memory
        else:
            raise ValueError("Unknown mode of acquisition.")

        self.cols = np.asarray(mask_cols)
        if start_center:
            center_pos = np.argmin(np.abs(mask_cols - full_kspace.shape[-1] // 2))
            mask_cols = list(mask_cols)
            left = mask_cols[center_pos::-1]
            right = mask_cols[center_pos + 1:]
            new_cols = []
            while left or right:
                if left:
                    new_cols.append(left.pop(0))
                if right:
                    new_cols.append(right.pop(0))
            self.cols = np.array(new_cols)

        if max_iter == 0:
            max_iter = len(self.cols)
        super().__init__(full_kspace, mask, max_iter=max_iter)

    def _getitem_memory(self, idx):
        mask = np.zeros(self.shape[-2:])
        mask[:, self.cols[:idx+1]] = 1
        kspace = np.squeeze(self._full_kspace * mask[np.newaxis, ...])
        return kspace, mask

    def _getitem_current(self, idx: int):
        mask = np.zeros(self.shape[-2:])
        mask[:, self.cols[idx]] = 1
        kspace = np.squeeze(self._full_kspace * mask[np.newaxis, ...])
        return kspace, mask

    def _getitem_line(self, idx: int):
        col = self.cols[idx]
        kspace = self.kspace[..., col]
        return kspace, col

    def __getitem__(self, it):
        if it > self._len:
            raise IndexError
        idx = min(it, len(self.cols)-1)
        return self.acquire(idx)

    def __next__(self):
        if self.iter > self._len:
            raise StopIteration
        self.iter += 1
        return self[self.iter-1]
