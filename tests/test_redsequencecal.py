from __future__ import division, absolute_import, print_function
from past.builtins import xrange

import matplotlib
matplotlib.use('Agg')

import unittest
import os
import shutil
import numpy.testing as testing
import numpy as np
import fitsio
import tempfile
from numpy import random

from redmapper.configuration import Configuration
from redmapper.calibration import RedSequenceCalibrator

class RedSequenceCalTestCase(unittest.TestCase):
    """
    Tests of redmapper.calibration.RedSequenceCalibrator which calibrates the
    red sequence.
    """
    def test_redsequencecal(self):
        """
        Run tests of redmapper.calibration.RedSequenceCalibrator
        """
        file_path = 'data_for_tests'
        conf_filename = 'testconfig.yaml'
        config = Configuration(os.path.join(file_path, conf_filename))
        config.calib_use_pcol = True

        config.zrange = [0.1, 0.2]

        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestRedmapper-')
        config.outpath = self.test_dir

        config.parfile = os.path.join(config.outpath, '%s_testpars.fit' % (config.d.outbase))

        gal_filename = 'test_dr8_trainredseq_gals.fit'
        galfile = os.path.join(file_path, gal_filename)

        redsequencecal = RedSequenceCalibrator(config, galfile)
        redsequencecal.run(doRaise=False)

        # make sure that the pars and the plots were made
        self.assertTrue(os.path.isfile(config.parfile))
        for i in xrange(config.nmag - 1):
            plotfile = os.path.join(config.outpath, config.plotpath, '%s_%s-%s.png' % (config.d.outbase, config.bands[i], config.bands[i + 1]))
            self.assertTrue(plotfile)
        plotfile = os.path.join(config.outpath, config.plotpath, '%s_zred_plots.png' % (config.d.outbase))
        self.assertTrue(plotfile)
        plotfile = os.path.join(config.outpath, config.plotpath, '%s_zred2_plots.png' % (config.d.outbase))
        self.assertTrue(plotfile)
        plotfile = os.path.join(config.outpath, config.plotpath, '%s_offdiags.png' % (config.d.outbase))
        self.assertTrue(plotfile)

        # Read in the pars and check numbers
        pars = fitsio.read(config.parfile, ext=1)
        testing.assert_almost_equal(pars[0]['pivotmag'], np.array([17.111357, 18.587465]), 4)
        testing.assert_almost_equal(pars[0]['medcol'], np.array([[1.890542, 0.9216751, 0.40069848, 0.33520406],
                                                                 [2.003969, 1.237295, 0.47768143, 0.32698348]]), 4)
        testing.assert_almost_equal(pars[0]['c01'], np.array([0.92373127, 1.079002, 1.2382987]), 4)
        testing.assert_almost_equal(pars[0]['slope01'], np.array([-0.00438811, -0.02315307]), 4)
        testing.assert_almost_equal(pars[0]['covmat_amp'][1: 3, 1: 3, :],
                                    np.array([[[0.00134602, 0.00357424],
                                               [0.00059764, 0.00118457]],
                                              [[0.00059764, 0.00118457],
                                               [0.00032759, 0.00048468]]]), 4)

    def setUp(self):
        self.test_dir = None

    def tearDown(self):
        if self.test_dir is not None:
            if os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir, True)


if __name__=='__main__':
    unittest.main()
