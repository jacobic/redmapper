from __future__ import division, absolute_import, print_function
from past.builtins import xrange

import unittest
import numpy.testing as testing
import numpy as np
import healpy as hp
import fitsio

from redmapper import DepthMap
from redmapper import Configuration

class DepthMapTestCase(unittest.TestCase):
    """
    Test reading and using redmapper.DepthMap class.
    """
    def runTest(self):
        """
        Run redmapper.DepthMap tests.
        """
        file_path = "data_for_tests"
        conf_filename = "testconfig.yaml"
        config = Configuration(file_path + "/" + conf_filename)

        # Check the regular depth
        depthstr = DepthMap(config)

        RAs = np.array([140.00434405, 142.04090, 142.09242, 142.11448, 50.0])
        Decs = np.array([63.47175301, 65.133844, 65.084844, 65.109541, 50.0])

        comp_limmag = np.array([20.810108, 20.59153, 20.59663, 20.59663, -1.63750e+30], dtype='f4')
        comp_exptime = np.array([78.849754, 63.56208, 63.56208, 63.56209, -1.63750e+30], dtype='f4')
        comp_m50 = np.array([20.967576, 20.85170, 20.85677, 20.85677, -1.63750e+30], dtype='f4')

        limmag, exptime, m50 = depthstr.get_depth_values(RAs, Decs)

        testing.assert_almost_equal(limmag, comp_limmag, 4)
        testing.assert_almost_equal(exptime, comp_exptime, 4)
        testing.assert_almost_equal(m50, comp_m50, 4)

        # And check the areas...
        mags = np.array([20.0, 20.2, 20.4, 20.6, 20.8, 21.0])
        areas_idl = np.array([3.29709, 3.29709, 3.29709, 3.29709, 2.86089, 0.0603447])
        areas = depthstr.calc_areas(mags)
        testing.assert_almost_equal(areas, areas_idl, 4)

        config2 = Configuration(file_path + "/" + conf_filename)
        config2.d.hpix = [582972]
        config2.d.nside = 1024
        config2.border = 0.02
        depthstr2 = DepthMap(config2)

        limmag2, exptime2, m502 = depthstr2.get_depth_values(RAs, Decs)
        comp_limmag[0] = hp.UNSEEN
        comp_exptime[0] = hp.UNSEEN
        comp_m50[0] = hp.UNSEEN

        testing.assert_almost_equal(limmag2, comp_limmag, 4)
        testing.assert_almost_equal(exptime2, comp_exptime, 4)
        testing.assert_almost_equal(m502, comp_m50, 4)

        config3 = Configuration(file_path + "/" + conf_filename)
        config3.d.hpix = [8421]
        config3.d.nside = 128
        config3.border = 0.0
        depthstr3 = DepthMap(config3)
        areas3 = depthstr3.calc_areas(np.array([20.0, 20.5, 21.0]))
        testing.assert_almost_equal(areas3[0], 0.20457271, 6)

if __name__=='__main__':
    unittest.main()
