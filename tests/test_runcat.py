from __future__ import division, absolute_import, print_function
from past.builtins import xrange

import unittest
import numpy.testing as testing
import numpy as np
import fitsio
from numpy import random
import healpy as hp

from redmapper import Cluster
from redmapper import ClusterCatalog
from redmapper import Configuration
from redmapper import GalaxyCatalog
from redmapper import DataObject
from redmapper import RedSequenceColorPar
from redmapper import Background
from redmapper import HPMask
from redmapper import DepthMap
from redmapper import RunCatalog

class RuncatTestCase(unittest.TestCase):
    """
    Tests of redmapper.RunCatalog, which computes richness for an input catalog with
    ra/dec/z.
    """
    def runTest(self):
        """
        Run the redmapper.RunCatalog tests.
        """
        random.seed(seed=12345)

        file_path = 'data_for_tests'
        conffile = 'testconfig.yaml'
        catfile = 'test_cluster_pos.fit'

        config = Configuration(file_path + '/' + conffile)
        config.catfile = file_path + '/' + catfile

        runcat = RunCatalog(config)

        runcat.run(do_percolation_masking=False)

        testing.assert_equal(runcat.cat.mem_match_id, [1, 2, 3])
        testing.assert_almost_equal(runcat.cat.Lambda, [24.16809273, 26.85296822, 13.36757088])
        testing.assert_almost_equal(runcat.cat.lambda_e, [2.50003219, 4.83695221, 2.4651196])
        testing.assert_almost_equal(runcat.cat.z_lambda, [0.22785459, 0.32256541, 0.2176394])
        testing.assert_almost_equal(runcat.cat.z_lambda_e, [0.00631017, 0.01353367, 0.00984607])

        runcat.run(do_percolation_masking=True)

        testing.assert_equal(runcat.cat.mem_match_id, [1, 2, 3])
        testing.assert_almost_equal(runcat.cat.Lambda, [24.22911263, 26.85296822, -1.])
        testing.assert_almost_equal(runcat.cat.lambda_e, [2.50442076, 4.83695221, -1.])
        testing.assert_almost_equal(runcat.cat.z_lambda, [0.22785437, 0.32256407, -1.])
        testing.assert_almost_equal(runcat.cat.z_lambda_e, [0.00630675, 0.0135363, -1.])

if __name__=='__main__':
    unittest.main()
