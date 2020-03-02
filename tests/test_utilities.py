from __future__ import division, absolute_import, print_function
from past.builtins import xrange

import unittest
import numpy.testing as testing
import numpy as np
import fitsio
import esutil

import redmapper
from redmapper.utilities import CubicSpline, sample_from_pdf

class SplineTestCase(unittest.TestCase):
    """
    Tests of cubic-spline interpolator, redmapper.utilities.CubicSpline
    """
    def runTest(self):
        """
        Run tests on redmapper.utilities.CubicSpline
        """
        # create test data
        # these numbers are from redMaPPer 6.3.1, DR8
        xx = np.array([0.05,0.15,0.25,0.35,0.45,0.60],dtype=np.float64)
        yy = np.array([15.685568,17.980721,18.934799,19.671671,19.796223,20.117981], dtype=np.float64)

        # will want to add error checking and exceptions to CubicSpline

        spl=CubicSpline(xx, yy)
        vals=spl(np.array([0.01, 0.44, 0.55, 0.665]))

        # these numbers are also from redMaPPer 6.3.1, DR8
        testing.assert_almost_equal(vals,np.array([14.648017,19.792828,19.973761,20.301322],dtype=np.float64),decimal=6)

        # Test the pdf inverter
        def power(x, exp=1.0):
            return x ** exp

        np.random.seed(seed=1000)
        vals = sample_from_pdf(power, [0.0, 10.0], 0.001, 10000, exp=1.0)
        h = esutil.stat.histogram(vals, min=0.0, max=10.0-0.003, more=True, binsize=0.1)
        fit = np.polyfit(np.log10(h['center']), np.log10(h['hist']), 1)
        # first component should be ~1 for a log-log plot
        testing.assert_almost_equal(fit[0], 0.9836134)

        vals = sample_from_pdf(power, [0.0, 10.0], 0.001, 10000, exp=2.0)
        h = esutil.stat.histogram(vals, min=0.0, max=10.0-0.003, more=True, binsize=0.1)
        ok, = np.where(h['hist'] > 10.0)
        fit = np.polyfit(np.log10(h['center'][ok]), np.log10(h['hist'][ok]), 1)
        # first component should be ~2 for a log-log plot
        testing.assert_almost_equal(fit[0], 1.98994079)

class MStarTestCase(unittest.TestCase):
    """
    Tests of mstar(z) redmapper.utilities.MStar
    """
    def runTest(self):
        """
        Run tests of redmapper.utilities.MStar
        """
        # make sure invalid raises proper exception
        self.assertRaises(IOError,redmapper.utilities.MStar,'blah','junk')

        # make an SDSS test...
        ms = redmapper.utilities.MStar('sdss','i03')

        mstar = ms([0.1,0.2,0.3,0.4,0.5])
        # test against IDL...
        testing.assert_almost_equal(mstar,np.array([16.2375,17.8500,18.8281,19.5878,20.1751]),decimal=4)
        # and against regressions...
        testing.assert_almost_equal(mstar,np.array([ 16.23748776,  17.85000035,  18.82812871,  19.58783337,  20.17514801]))

class AstroToSphereTestCase(unittest.TestCase):
    """
    Tests of redmapper.utilities.astro_to_sphere
    """
    def runTest(self):
        """
        Test conversion of ra/dec to healpix spherical coordinates theta/phi
        """
        ra,dec = 40.1234, 55.9876
        testing.assert_almost_equal(redmapper.utilities.astro_to_sphere(ra,dec),np.array([ 0.5936,  0.7003]),decimal=4)


class RedGalInitialColorsTestCase(unittest.TestCase):
    """
    Tests of color(z) redmapper.utilities.RedGalInitialColors
    """
    def runTest(self):
        """
        Run tests of redmapper.utilities.RedGalInitialColors
        """

        # Make sure invalid file raises proper exception
        self.assertRaises(IOError, redmapper.utilities.RedGalInitialColors, 'notafile')

        # Make an SDSS test...
        rg = redmapper.utilities.RedGalInitialColors('bc03_colors_sdss.fit')

        redshifts = np.array([0.1, 0.2, 0.3, 0.4])

        # Check that u-g (not there) raises an exception
        self.assertRaises(ValueError, rg, 'u', 'g', redshifts)

        gmr = rg('g', 'r', redshifts)
        testing.assert_almost_equal(gmr, np.array([1.05341685, 1.3853203 , 1.65178809, 1.71100367]))
        rmi = rg('r', 'i', redshifts)
        testing.assert_almost_equal(rmi, np.array([0.42778724, 0.49461159, 0.57792853, 0.68157081]))
        imz = rg('i', 'z', redshifts)
        testing.assert_almost_equal(imz, np.array([0.35496065, 0.35439262, 0.35918364, 0.41805246]))


class CicTestCase(unittest.TestCase):
    """
    Tests of redmapper.utilities.cic cloud-in-cell code.
    """
    def runTest(self):
        """
        Run tests on redmapper.utilities.cic
        """
        incat = fitsio.read('data_for_tests/test_cic_small.fits', ext=1)

        posx = incat[0]['POSX']
        nx = incat[0]['NX']
        posy = incat[0]['POSY']
        ny = incat[0]['NY']
        posz = incat[0]['POSZ']
        nz = incat[0]['NZ']
        value = np.ones(posx.size)

        field = redmapper.utilities.cic(value, posx, nx, posy, ny, posz, nz)
        avfield = redmapper.utilities.cic(value, posx, nx, posy, ny, posz, nz, average=True)
        testing.assert_almost_equal(field, incat[0]['FIELD'], decimal=6)
        testing.assert_almost_equal(avfield, incat[0]['AVFIELD'], decimal=6)


# copy this for a new utility test
class UtilityTemplateTestCase(unittest.TestCase):
    def runTest(self):
        pass


if __name__=='__main__':
    unittest.main()
