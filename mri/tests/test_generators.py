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
from mri.generators import DataOnlyKspaceGenerator, OneColumn2DKspaceGenerator, Column2DKspaceGenerator


class TestKspaceGenerator(unittest.TestCase):
    """Test the kspace Generator"""

    def setUp(self):
        self.n_coils = (1, 2)
        self.shape = (64, 64)

    def test_kspace_generator(self):
        mask_cols = np.random.randint(0, 64, size=16)
        for nc in self.n_coils:
            full_kspace = np.squeeze(np.random.rand(nc, *self.shape))
            data_gen = DataOnlyKspaceGenerator(full_kspace, mask_cols)
            data_gen2 = OneColumn2DKspaceGenerator(full_kspace, mask_cols)
            data_gen3 = Column2DKspaceGenerator(full_kspace, mask_cols)

            for (line, col), (kspace, mask), (kspace2, mask2) in zip(data_gen, data_gen2, data_gen3):
                np.testing.assert_equal(line, kspace[..., col])
                np.testing.assert_equal(line, kspace2[..., col])

        print("Test Column Generators iteration")

    def test_kspace_generator_getitem(self):
        mask_cols = np.random.randint(0, 64, size=16)
        for nc in self.n_coils:
            full_kspace = np.squeeze(np.random.rand(nc, *self.shape))
            data_gen = DataOnlyKspaceGenerator(full_kspace, mask_cols)
            data_gen2 = OneColumn2DKspaceGenerator(full_kspace, mask_cols)
            data_gen3 = Column2DKspaceGenerator(full_kspace, mask_cols)

            for idx, (line, col) in enumerate(data_gen):
                print(idx, col)
                np.testing.assert_equal(data_gen[idx][0], data_gen3[idx+1][0][..., col])
                np.testing.assert_equal(data_gen[idx][0], data_gen2[idx][0][..., col])

        print("Test Column Generators getter")
