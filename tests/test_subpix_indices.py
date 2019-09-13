from __future__ import division, absolute_import, print_function
from past.builtins import xrange

import unittest
import numpy.testing as testing
import numpy as np
import healpy as hp
import os
import esutil

from redmapper.utilities import get_healsparse_subpix_indices

class SubpixIndexTestCase(unittest.TestCase):
    """
    Tests for get_healsparse_subpix_indices
    """

    def test_subpix_nside_lt_cov_nside(self):
        """
        Test when subpix_nside is less than coverage nside
        """

        # Test with no border...
        subpix_nside = 2
        subpix_hpix = [16]
        subpix_border = 0.0
        coverage_nside = 32

        covpix = get_healsparse_subpix_indices(subpix_nside, subpix_hpix, subpix_border, coverage_nside)

        # Test that we have the right number
        testing.assert_equal(covpix.size, int(hp.nside2npix(coverage_nside) / hp.nside2npix(subpix_nside)))

        # Test that they are all in the correct pixel
        theta, phi = hp.pix2ang(coverage_nside, covpix, nest=True)
        ipring = hp.ang2pix(subpix_nside, theta, phi)

        testing.assert_equal(ipring, subpix_hpix[0])

        covpix_temp = covpix

        # Test with two pixels
        subpix_hpix = [16, 17]

        covpix = get_healsparse_subpix_indices(subpix_nside, subpix_hpix, subpix_border, coverage_nside)

        # Test that they are all in the correct pixel
        theta, phi = hp.pix2ang(coverage_nside, covpix, nest=True)
        ipring = hp.ang2pix(subpix_nside, theta, phi)

        testing.assert_equal(ipring[0: 256], subpix_hpix[0])
        testing.assert_equal(ipring[256: ], subpix_hpix[1])

        # Test with border (most typical value)...
        subpix_hpix = [16]
        subpix_border = 2.5

        covpix2 = get_healsparse_subpix_indices(subpix_nside, subpix_hpix, subpix_border, coverage_nside)

        # There should be more of these pixels
        self.assertTrue(covpix2.size > covpix_temp.size)

        theta2, phi2 = hp.pix2ang(coverage_nside, covpix2, nest=True)

        matcher = esutil.htm.Matcher(10, np.degrees(phi), 90.0 - np.degrees(theta))
        m1, m2, dist = matcher.match(np.degrees(phi2), 90.0 - np.degrees(theta2), subpix_border*3.0, maxmatch=1)

        # This checks that they are all close
        self.assertTrue(dist.max() < subpix_border*2.0)

        # And that there are some greater than 0 dist (this is redundant with
        # the number test above)
        test, = np.where(dist > 0.0)
        self.assertTrue(test.size > 0)

        # Note that border + multiple pixels is not supported.

    def test_subpix_nside_eq_cov_nside(self):
        """
        Test when subpix_nside is equal to coverage nside
        """

        # Test with no border
        subpix_nside = 32
        subpix_hpix = [16]
        subpix_border = 0.0
        coverage_nside = 32

        covpix = get_healsparse_subpix_indices(subpix_nside, subpix_hpix, subpix_border, coverage_nside)

        # Test that we have the right number
        testing.assert_equal(covpix.size, 1)

        # Test that they are all in the correct pixel
        theta, phi = hp.pix2ang(coverage_nside, covpix, nest=True)
        ipring = hp.ang2pix(subpix_nside, theta, phi)

        testing.assert_equal(ipring, subpix_hpix)

        # Test with two pixels
        subpix_hpix = [16, 17]

        covpix = get_healsparse_subpix_indices(subpix_nside, subpix_hpix, subpix_border, coverage_nside)

        # Test that they are all in the correct pixel
        theta, phi = hp.pix2ang(coverage_nside, covpix, nest=True)
        ipring = hp.ang2pix(subpix_nside, theta, phi)

        testing.assert_equal(ipring[0], subpix_hpix[0])
        testing.assert_equal(ipring[1], subpix_hpix[1])

        # Test with border...
        subpix_hpix = [16]
        subpix_border = 2.5

        covpix2 = get_healsparse_subpix_indices(subpix_nside, subpix_hpix, subpix_border, coverage_nside)

        # There should be more of these pixels
        self.assertTrue(covpix2.size > covpix.size)

        theta2, phi2 = hp.pix2ang(coverage_nside, covpix2, nest=True)

        matcher = esutil.htm.Matcher(10, np.degrees(phi), 90.0 - np.degrees(theta))
        m1, m2, dist = matcher.match(np.degrees(phi2), 90.0 - np.degrees(theta2), subpix_border*3.0, maxmatch=1)

        # This checks that they are all close
        self.assertTrue(dist.max() < subpix_border*2.0)

        # And that there are some greater than 0 dist (this is redundant with
        # the number test above)
        test, = np.where(dist > 0.0)
        self.assertTrue(test.size > 0)

    def test_subpix_nside_gt_cov_nside(self):
        """
        Test when subpix_nside is greater than coverage_nside
        """

        # Test with no border...
        subpix_nside = 128
        subpix_hpix = [16]
        subpix_border = 0.0
        coverage_nside = 32

        covpix = get_healsparse_subpix_indices(subpix_nside, subpix_hpix, subpix_border, coverage_nside)

        # Test that we have the right number
        testing.assert_equal(covpix.size, 1)

        # Test that they are all in the correct pixel
        theta, phi = hp.pix2ang(subpix_nside, subpix_hpix)
        ipnest = hp.ang2pix(coverage_nside, theta, phi, nest=True)

        testing.assert_equal(covpix, ipnest)

        # Test with two pixels
        subpix_hpix = [16, 17]

        covpix = get_healsparse_subpix_indices(subpix_nside, subpix_hpix, subpix_border, coverage_nside)

        # Test that we have the right number
        testing.assert_equal(covpix.size, 2)

        # Test that they are all in the correct pixel
        theta, phi = hp.pix2ang(subpix_nside, subpix_hpix)
        ipnest = hp.ang2pix(coverage_nside, theta, phi, nest=True)

        testing.assert_equal(covpix, ipnest)

        # Test with border...
        subpix_hpix = [16]
        subpix_border = 2.5

        covpix2 = get_healsparse_subpix_indices(subpix_nside, subpix_hpix, subpix_border, coverage_nside)

        # There should be more of these pixels
        self.assertTrue(covpix2.size > covpix.size)

        theta2, phi2 = hp.pix2ang(coverage_nside, covpix2, nest=True)

        matcher = esutil.htm.Matcher(10, np.degrees(phi), 90.0 - np.degrees(theta))
        m1, m2, dist = matcher.match(np.degrees(phi2), 90.0 - np.degrees(theta2), subpix_border*3.0, maxmatch=1)

        # This checks that they are all close
        self.assertTrue(dist.max() < subpix_border*2.0)

        # And that there are some greater than 0 dist (this is redundant with
        # the number test above)
        test, = np.where(dist > 0.0)
        self.assertTrue(test.size > 0)


if __name__=='__main__':
    unittest.main()

