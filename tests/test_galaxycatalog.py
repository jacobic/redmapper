from __future__ import division, absolute_import, print_function
from past.builtins import xrange

import unittest
import numpy.testing as testing
import numpy as np
import fitsio
from numpy import random
import healpy as hp
import tempfile
import shutil
import os
import esutil

from redmapper import Configuration
from redmapper import GalaxyCatalog
from redmapper import GalaxyCatalogMaker
from redmapper import Catalog, Entry


class GalaxyCatalogTestCase(unittest.TestCase):
    """
    Tests for redmapper.GalaxyCatalog class, including reading full catalogs,
    reading sub-regions, and matching neighbors.
    """
    def test_galaxycatalog_read(self):
        """
        Run redmapper.GalaxyCatalog tests,
        """

        file_path = 'data_for_tests'

        galfile = 'pixelized_dr8_test/dr8_test_galaxies_master_table.fit'

        gals_all = GalaxyCatalog.from_galfile(file_path + '/' + galfile)

        # check that we got the expected number...
        testing.assert_equal(gals_all.size, 14449)

        # read in a subregion, no border
        hpix = 2163
        gals_sub = GalaxyCatalog.from_galfile(file_path + '/' + galfile,
                                              hpix=hpix, nside=64)

        theta = (90.0 - gals_all.dec) * np.pi/180.
        phi = gals_all.ra * np.pi/180.
        ipring_all = hp.ang2pix(64, theta, phi)
        use, = np.where(ipring_all == hpix)

        testing.assert_equal(gals_sub.size, use.size)

        # Read in a subregion that's made of two pixels, no border
        hpix = [2163, 2296]
        gals_sub = GalaxyCatalog.from_galfile(file_path + '/' + galfile,
                                              hpix=hpix, nside=64)
        a, b = esutil.numpy_util.match(hpix, ipring_all)

        testing.assert_equal(gals_sub.size, a.size)

        # read in a subregion, with border
        gals_sub = GalaxyCatalog.from_galfile(file_path + '/' + galfile,
                                              hpix=9218, nside=128, border=0.1)

        # this isn't really a big enough sample catalog to fully test...
        testing.assert_equal(gals_sub.size, 2511)

        # and test the matching...

        indices, dists = gals_all.match_one(140.5, 65.0, 0.2)
        testing.assert_equal(indices.size, 521)
        testing.assert_array_less(dists, 0.2)

        i0, i1, dists = gals_all.match_many([140.5,141.2],
                                            [65.0, 65.2], [0.2,0.1])
        testing.assert_equal(i0.size, 666)
        test, = np.where(i0 == 0)
        testing.assert_equal(test.size, 521)
        testing.assert_array_less(dists[test], 0.2)
        test, = np.where(i0 == 1)
        testing.assert_equal(test.size, 666 - 521)
        testing.assert_array_less(dists[test], 0.1)

    def test_galaxycatalog_create(self):
        """
        Run `redmapper.GalaxyCatalogMaker` tests.
        """

        # Make a test directory
        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestRedmapper-')

        info_dict = {'LIM_REF': 21.0,
                     'REF_IND': 3,
                     'AREA': 25.0,
                     'NMAG': 5,
                     'MODE': 'SDSS',
                     'ZP': 22.5,
                     'U_IND': 0,
                     'G_IND': 1,
                     'R_IND': 2,
                     'I_IND': 3,
                     'Z_IND': 3}

        # Test 1: read in catalog, write it to a single file, and then
        # try to split it up
        # make sure that it makes it properly, and that the output number
        # of files is the same as input number of files.

        configfile = os.path.join('data_for_tests', 'testconfig.yaml')
        config = Configuration(configfile)
        gals = GalaxyCatalog.from_galfile(config.galfile)
        tab = Entry.from_fits_file(config.galfile)

        maker = GalaxyCatalogMaker(os.path.join(self.test_dir, 'test_working'), info_dict, nside=tab.nside)
        maker.append_galaxies(gals._ndarray)
        maker.finalize_catalog()

        tab2 = Entry.from_fits_file(os.path.join(self.test_dir, 'test_working_master_table.fit'))
        self.assertEqual(tab.nside, tab2.nside)
        self.assertEqual(tab.filenames.size, tab2.filenames.size)

        for filename in tab2.filenames:
            try:
                fname = os.path.join(self.test_dir, filename.decode())
            except AttributeError:
                fname = os.path.join(self.test_dir, filename)
            self.assertTrue(os.path.isfile(fname))

        # Test 2: Make a catalog that has an incomplete dtype
        dtype = [('id', 'i8'),
                 ('ra', 'f8')]
        maker = GalaxyCatalogMaker(os.path.join(self.test_dir, 'test'), info_dict)
        testgals = np.zeros(10, dtype=dtype)
        self.assertRaises(RuntimeError, maker.append_galaxies, testgals)

        # Test 3: make a catalog that has the wrong number of magnitudes
        dtype = GalaxyCatalogMaker.get_galaxy_dtype(3)
        testgals = np.zeros(10, dtype=dtype)
        self.assertRaises(RuntimeError, maker.append_galaxies, testgals)

        # Test 4: make a catalog that has some NaNs
        dtype = GalaxyCatalogMaker.get_galaxy_dtype(info_dict['NMAG'])
        testgals = np.ones(10, dtype=dtype)
        testgals['mag'][0, 1] = np.nan
        self.assertRaises(RuntimeError, maker.append_galaxies, testgals)

        # Test 5: make a catalog that has ra/dec out of range
        testgals = np.ones(10, dtype=dtype)
        testgals['ra'][1] = -1.0
        self.assertRaises(RuntimeError, maker.append_galaxies, testgals)

        testgals = np.ones(10, dtype=dtype)
        testgals['dec'][1] = -100.0
        self.assertRaises(RuntimeError, maker.append_galaxies, testgals)

        # Test 6: make a catalog that has mag > 90.0
        testgals = np.ones(10, dtype=dtype)
        testgals['mag'][0, 1] = 100.0
        self.assertRaises(RuntimeError, maker.append_galaxies, testgals)

        # Test 7: make a catalog that has mag_err == 0.0
        testgals = np.ones(10, dtype=dtype)
        testgals['mag_err'][0, 1] = 0.0
        self.assertRaises(RuntimeError, maker.append_galaxies, testgals)

    def setUp(self):
        self.test_dir = None

    def tearDown(self):
        if self.test_dir is not None:
            if os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir, True)


if __name__=='__main__':
    unittest.main()
