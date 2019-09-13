from __future__ import division, absolute_import, print_function
from past.builtins import xrange

import matplotlib
matplotlib.use('Agg')

import unittest
import numpy.testing as testing
import numpy as np
import fitsio
import copy
from numpy import random
import time
import tempfile
import shutil
import os
import esutil

from redmapper import Configuration
from redmapper import GalaxyCatalog
from redmapper import RunColormem
from redmapper.calibration import SelectSpecRedGalaxies
from redmapper import Catalog

class RunColormemTestCase(unittest.TestCase):
    """
    Tests of redmapper.RunColormem, which computes richness by fitting the
    red-sequence for each cluster (for calibration)
    """

    def test_run_colormem(self):
        """
        Run tests of redmapper.RunColormem
        """

        random.seed(seed=12345)

        file_path = 'data_for_tests'
        configfile = 'testconfig.yaml'

        config = Configuration(os.path.join(file_path, configfile))

        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestRedmapper-')
        config.outpath = self.test_dir

        # First, we need the red galaxy model

        config.specfile_train = os.path.join(file_path, 'test_dr8_spec.fit')
        config.zrange = [0.1,0.2]

        config.redgalfile = config.redmapper_filename('test_redgals')
        config.redgalmodelfile = config.redmapper_filename('test_redgalmodel')

        selred = SelectSpecRedGalaxies(config)
        selred.run()

        # Main test...
        config.zmemfile = config.redmapper_filename('test_zmem')

        rcm = RunColormem(config)
        rcm.run()
        rcm.output_training()

        # Check that the files are there...
        self.assertTrue(os.path.isfile(config.zmemfile))

        mem = fitsio.read(config.zmemfile, ext=1)
        testing.assert_equal(mem.size, 16)
        testing.assert_array_almost_equal(mem['pcol'][0:3], np.array([0.94829756, 0.83803916, 0.88315928]))
        testing.assert_array_almost_equal(mem['z'][0:3], np.array([0.191887, 0.18744484, 0.19445464]))

    def setUp(self):
        self.test_dir = None

    def tearDown(self):
        if self.test_dir is not None:
            if os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir, True)


if __name__=='__main__':
    unittest.main()
