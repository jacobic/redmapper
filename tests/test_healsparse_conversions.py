from __future__ import division, absolute_import, print_function
from past.builtins import xrange

import unittest
import numpy.testing as testing
import numpy as np
import fitsio
from numpy import random
import healpy as hp
import healsparse
import os
import tempfile
import shutil

import redmapper

class HealsparseConversionsTestCase(unittest.TestCase):
    """
    Tests for converting old maps to healsparse maps
    """

    def test_mask_conversion(self):
        """
        Test mask conversion
        """

        file_path = 'data_for_tests'

        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestRedmapper-')

        redmapper.mask.convert_maskfile_to_healsparse(os.path.join(file_path, 'test_dr8_mask.fit'), os.path.join(self.test_dir, 'test_dr8_mask_hs.fit'), 64)

        sparsemask = healsparse.HealSparseMap.read(os.path.join(self.test_dir, 'test_dr8_mask_hs.fit'))

        RAs = np.array([142.10934, 142.04090, 142.09242, 142.11448, 50.0])
        Decs = np.array([65.022666, 65.133844, 65.084844, 65.109541, 50.0])

        comp = np.array([1.0, 1.0, 1.0, 1.0, hp.UNSEEN], dtype=np.float32)

        testing.assert_almost_equal(sparsemask.get_values_pos(RAs, Decs, lonlat=True), comp)

    def test_depth_conversion(self):
        """
        Test depth conversion
        """

        file_path = 'data_for_tests'

        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestRedmapper-')

        redmapper.depthmap.convert_depthfile_to_healsparse(os.path.join(file_path, 'test_dr8_depth.fit'), os.path.join(self.test_dir, 'test_dr8_depth_hs.fit'), 64)

        sparsemask = healsparse.HealSparseMap.read(os.path.join(self.test_dir, 'test_dr8_depth_hs.fit'))

        RAs = np.array([142.10934, 142.04090, 142.09242, 142.11448, 50.0])
        Decs = np.array([65.022666, 65.133844, 65.084844, 65.109541, 50.0])

        comp_limmag = np.array([20.6847, 20.5915, 20.5966, 20.5966, -1.63750e+30], dtype='f4')
        comp_exptime = np.array([70.3742, 63.5621, 63.5621, 63.5621, -1.63750e+30], dtype='f4')
        comp_m50 = np.array([20.8964, 20.8517, 20.8568, 20.8568, -1.63750e+30], dtype='f4')

        values = sparsemask.get_values_pos(RAs, Decs, lonlat=True)

        testing.assert_almost_equal(values['limmag'], comp_limmag, 4)
        testing.assert_almost_equal(values['exptime'], comp_exptime, 4)
        testing.assert_almost_equal(values['m50'], comp_m50, 4)

        redmapper.depthmap.convert_depthfile_to_healsparse(os.path.join(file_path, 'test_dr8_depth_r.fit'), os.path.join(self.test_dir, 'test_dr8_depth_r_hs.fit'), 64)

    def setUp(self):
        self.test_dir = None

    def tearDown(self):
        if self.test_dir is not None:
            if os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir, True)

if __name__=='__main__':
    unittest.main()


