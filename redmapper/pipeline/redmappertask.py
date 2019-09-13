"""Class to run redmapper on a single pixel, for distributed runs.
"""

from __future__ import division, absolute_import, print_function

import os
import numpy as np
import glob

from ..configuration import Configuration
from ..utilities import make_lockfile
from ..run_firstpass import RunFirstPass
from ..run_likelihoods import RunLikelihoods
from ..run_percolation import RunPercolation

class RunRedmapperPixelTask(object):
    """
    Class to run redmapper on a single healpix pixel, for distributed runs.
    """

    def __init__(self, configfile, pixel, nside, path=None):
        """
        Instantiate a RunRedmapperPixelTask.

        Parameters
        ----------
        configfile: `str`
           Configuration yaml filename.
        pixel: `int`
           Healpix pixel to run on.
        nside: `int`
           Healpix nside associated with pixel.
        path: `str`, optional
           Output path.  Default is None, use same absolute
           path as configfile.
        """
        if path is None:
            outpath = os.path.dirname(os.path.abspath(configfile))
        else:
            outpath = path

        self.config = Configuration(configfile, outpath=path)
        self.pixel = pixel
        self.nside = nside

    def run(self):
        """
        Run redmapper on a single healpix pixel.

        This method will check if files already exist, and will
        skip any steps that already exist.  The border radius
        will automatically be calculated based on the richest
        possible cluster at the lowest possible redshift.

        All files will be placed in self.config.outpath (see
        self.__init__)
        """

        # need to think about outpath

        # Make sure all files are here and okay...

        if not self.config.galfile_pixelized:
            raise ValueError("Code only runs with pixelized galfile.")

        self.config.check_files(check_zredfile=True, check_bkgfile=True, check_bkgfile_components=True, check_parfile=True, check_zlambdafile=True)

        # Compute the border size

        self.config.border = self.config.compute_border()

        self.config.d.hpix = [self.pixel]
        self.config.d.nside = self.nside
        self.config.d.outbase = '%s_%d_%05d' % (self.config.outbase, self.nside, self.pixel)

        # Do the run

        self.config.logger.info("Running redMaPPer on pixel %d" % (self.pixel))

        firstpass = RunFirstPass(self.config)

        if not os.path.isfile(firstpass.filename):
            firstpass.run()
            firstpass.output(savemembers=False, withversion=False)
        else:
            self.config.logger.info("Firstpass file %s already present.  Skipping..." % (firstpass.filename))

        self.config.catfile = firstpass.filename

        like = RunLikelihoods(self.config)

        if not os.path.isfile(like.filename):
            like.run()
            like.output(savemembers=False, withversion=False)
        else:
            self.config.logger.info("Likelihood file %s already present.  Skipping..." % (like.filename))

        self.config.catfile = like.filename

        perc = RunPercolation(self.config)

        if not os.path.isfile(perc.filename):
            perc.run()
            perc.output(savemembers=True, withversion=False)
        else:
            self.config.logger.info("Percolation file %s already present.  Skipping..." % (perc.filename))

