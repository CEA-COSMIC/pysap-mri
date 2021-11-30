"""
Cartesian Kspace-Generators.

At each iteration, a column of kspace is yielded.
Transpose your data if you want to work row-wise.
"""

import numpy as np

from .base import KspaceGeneratorBase


class Column2DKspaceGenerator(KspaceGeneratorBase):
    """k-space Generator, at each step a new column fills the existing k-space."""

    def __init__(self, full_kspace, mask_cols, max_iter=0):
        mask = np.zeros(full_kspace.shape[-2:])

        def flip2center(mask_cols, center_pos):
            """reorder a list by starting by a center_position and alternating left/right"""
            mask_cols = list(mask_cols)
            left = mask_cols[center_pos::-1]
            right = mask_cols[center_pos + 1:]
            new_cols = []
            while left or right:
                if left:
                    new_cols.append(left.pop(0))
                if right:
                    new_cols.append(right.pop(0))
            return np.array(new_cols)

        self.cols = flip2center(
            mask_cols, np.argmin(np.abs(mask_cols - full_kspace.shape[-1] // 2))
        )
        if max_iter == 0:
            max_iter = len(self.cols)
        super().__init__(full_kspace, mask, max_iter=max_iter)

    def _get_kspace_and_mask(self, idx):
        mask = np.zeros(self.shape[-2:])
        mask[:, self.cols[:idx]] = 1
        kspace = np.squeeze(self._full_kspace * mask[np.newaxis, ...])
        return kspace, mask

    def __getitem__(self, it):
        if it > self._len:
            raise IndexError
        idx = min(it, len(self.cols))
        return self._get_kspace_and_mask(idx)

    def __next__(self):
        if self.iter > self._len:
            raise StopIteration
        idx = min(self.iter + 1, len(self.cols))
        self.iter += 1
        return self._get_kspace_and_mask(idx)


class OneColumn2DKspaceGenerator(Column2DKspaceGenerator):
    """
    K-space Generator yielding only the newly acquired line.

    To be used with  classical FFT operator.
    """

    def __getitem__(self, it: int):
        if it >= self._len:
            raise IndexError
        idx = min(it, len(self.cols) - 1)
        return self.get_kspace_and_mask(idx)

    def __next__(self):
        if self.iter >= self._len:
            raise StopIteration
        idx = min(self.iter, len(self.cols) - 1)
        self.iter += 1
        return self._get_kspace_and_mask(idx)

    def _get_kspace_and_mask(self, idx: int):
        mask = np.zeros(self.shape[-2:])
        mask[:, self.cols[idx]] = 1
        kspace = np.squeeze(self._full_kspace * mask[np.newaxis, ...])
        return kspace, mask


class DataOnlyKspaceGenerator(OneColumn2DKspaceGenerator):
    """
    Kspace Generator to be used with a ColumnFFT Operator.
    """

    def _get_kspace_and_mask(self, idx: int):
        col = self.cols[idx]
        kspace = self.kspace[..., col]
        return kspace, col
