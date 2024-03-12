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
from mri.generators import Column2DKspaceGenerator, KspaceGeneratorBase

class TestKspaceGenerator(unittest.TestCase):
    """Test the kspace Generator"""

    def setUp(self):
        self.n_coils = (1, 2)
        self.shape = (64, 64)

    def test_kspace_generator(self):
        """Test the k-space generators."""
        mask_cols = np.arange(64)
        mask = np.ones(self.shape)
        for n_c in self.n_coils:
            full_kspace = np.squeeze(np.random.rand(n_c, *self.shape))
            gen_base = KspaceGeneratorBase(full_kspace, mask)
            gen_line = Column2DKspaceGenerator(
                full_kspace,
                mask_cols,
                mode="line",
            )
            gen_current = Column2DKspaceGenerator(
                full_kspace,
                mask_cols,
                mode="current",
            )
            gen_memory = Column2DKspaceGenerator(
                full_kspace,
                mask_cols,
                mode="memory",
            )
            for line, current, memory, base in zip(gen_line, gen_current, gen_memory, gen_base):
                col = line[1]
                np.testing.assert_equal(
                    line[0],
                    full_kspace[..., col],
                    err_msg="Line not matching kspace column",
                )
                np.testing.assert_equal(
                    line[0],
                    current[0][..., col],
                    err_msg="Line not matching current column",
                )
                np.testing.assert_equal(
                    line[0],
                    memory[0][..., col],
                    err_msg="Line not matching memory column",
                )
                np.testing.assert_equal(
                    np.nonzero(memory[1][0, :]),
                    np.asarray(line[1]),
                    err_msg="Mask not matching column",
                )
                np.testing.assert_equal(
                    current[1][:, col],
                    memory[1][:, col],
                    err_msg="current mask not matching memory",
                )
        print("Test Column Generators iteration")

    def test_getitem_iterator(self):
        """Test getitem function is synced with iterator."""
        mask_cols = np.arange(64)
        for n_c in self.n_coils:
            full_kspace = np.squeeze(np.random.rand(n_c, *self.shape))
            for mode in ("line", "current", "memory"):
                data_gen = Column2DKspaceGenerator(
                    full_kspace,
                    mask_cols,
                    mode=mode,
                )
                self.assertEqual(data_gen.dtype, full_kspace.dtype)

                for idx, (kspace, mask) in enumerate(data_gen):
                    np.testing.assert_equal(kspace, data_gen[idx][0])
                    np.testing.assert_equal(mask, data_gen[idx][1])

    def test_raises(self):
        """Test Exceptions."""
        self.assertRaises(
            ValueError,
            Column2DKspaceGenerator,
            np.arange(10),
            np.arange(10),
            mode="test",
        )
        gen = Column2DKspaceGenerator(
            np.arange(10),
            np.arange(10),
            mode="line",
        )
        self.assertRaises(
            IndexError,
            gen.__getitem__,
            len(gen)+1,
        )
