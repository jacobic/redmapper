from __future__ import division, absolute_import, print_function
from past.builtins import xrange

import unittest
import numpy.testing as testing
import numpy as np
import fitsio
import tempfile
import os

from redmapper import ColorBackground
from redmapper import ColorBackgroundGenerator
from redmapper import Configuration

class ColorBackgroundTestCase(unittest.TestCase):
    """
    Tests for the redmapper.ColorBackground and
    redmapper.ColorBackgroundGenerator classes.
    """
    def runTest(self):
        """
        Run the ColorBackground and ColorBackgroundGenerator tests.
        """

        file_name = 'test_dr8_col_bkg.fit'
        file_path = 'data_for_tests'

        cbkg = ColorBackground('%s/%s' % (file_path, file_name))

        col1 = np.array([0.572300, 1.39560])
        col2 = np.array([0.7894, 0.9564])
        refmags = np.array([17.587, 18.956])

        refmagindex = np.array([258, 395])
        col1index = np.array([1, 17])
        col2index = np.array([15, 19])

        idl_bkg1 = np.array([0.148366, 0.165678])
        idl_bkg2 = np.array([0.00899471, 0.0201531])
        idl_bkg12 = np.array([0.0111827, 0.0719981])

        # Test color1
        py_outputs = cbkg.lookup_diagonal(1, col1, refmags)
        testing.assert_almost_equal(py_outputs, idl_bkg1, decimal=5)

        # Test color2
        py_outputs = cbkg.lookup_diagonal(2, col2, refmags)
        testing.assert_almost_equal(py_outputs, idl_bkg2, decimal=5)

        # Test off-diagonal
        py_outputs = cbkg.lookup_offdiag(1, 2, col1, col2, refmags)
        testing.assert_almost_equal(py_outputs, idl_bkg12, decimal=5)

        # And a test sigma_g with the usehdrarea=True
        cbkg2 = ColorBackground('%s/%s' % (file_path, file_name), usehdrarea=True)

        col1 = np.array([0.572300, 1.39560, 1.0])
        col2 = np.array([0.7894, 0.9564, 1.0])
        refmags = np.array([17.587, 18.956, 25.0])

        idl_sigma_g1 = np.array([123.382, 611.711, np.inf])
        idl_sigma_g2 = np.array([8.48481, 82.8938, np.inf])

        # Test color1
        py_outputs = cbkg2.sigma_g_diagonal(1, col1, refmags)
        testing.assert_almost_equal(py_outputs, idl_sigma_g1, decimal=3)

        # Test color2
        py_outputs = cbkg2.sigma_g_diagonal(2, col2, refmags)
        testing.assert_almost_equal(py_outputs, idl_sigma_g2, decimal=3)


        #####################################################
        # Now a test of the generation of a color background
        conf_filename = 'testconfig.yaml'
        config = Configuration(file_path + "/" + conf_filename)

        tfile = tempfile.mkstemp()
        os.close(tfile[0])
        config.bkgfile_color = tfile[1]
        config.d.nside = 128
        config.d.hpix = [8421]
        config.border = 0.0

        cbg = ColorBackgroundGenerator(config, minrangecheck=5)
        # Need to set clobber=True because the tempfile was created
        cbg.run(clobber=True)

        fits = fitsio.FITS(config.bkgfile_color)

        # Make sure we have 11 extensions
        testing.assert_equal(len(fits), 11)

        # Check the 01_01 and 01_02
        bkg11 = fits['01_01_REF'].read()
        bkg11_compare = fitsio.read(file_path + "/test_dr8_bkg_zredc_sub.fits", ext='01_01_REF')
        testing.assert_almost_equal(bkg11['BC'], bkg11_compare['BC'], 3)
        testing.assert_almost_equal(bkg11['N'], bkg11_compare['N'], 3)

        bkg12 = fits['01_02_REF'].read()
        bkg12_compare = fitsio.read(file_path + "/test_dr8_bkg_zredc_sub.fits", ext='01_02_REF')

        testing.assert_almost_equal(bkg12['BC'], bkg12_compare['BC'], 2)
        testing.assert_almost_equal(bkg12['N'], bkg12_compare['N'], 4)

        # And delete the tempfile
        os.remove(config.bkgfile_color)

if __name__=='__main__':
    unittest.main()
