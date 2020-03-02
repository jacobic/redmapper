"""Class to compute zreds on a single pixel, for distributed runs.
"""

from __future__ import division, absolute_import, print_function

import os
import numpy as np
import glob
import re

from ..configuration import Configuration
from ..zred_runner import ZredRunPixels
from ..utilities import make_lockfile

class RunZredPixelTask(object):
    """
    Class to compute zreds on a single healpix pixel, for distributed runs.
    """

    def __init__(self, configfile, pixel, nside, path=None):
        """
        Instantiate a RunZredPixelTask.

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
           path as configfile.  I think this is unused.
        """
        if path is None:
            outpath = os.path.dirname(os.path.abspath(configfile))
        else:
            outpath = path

        self.config = Configuration(configfile, outpath=outpath)
        self.pixel = pixel
        self.nside = nside

    def run(self):
        """
        Calculate zreds on a single healpix pixel.

        All files will be placed in the path in self.config.zredfile, and when
        the final pixel is run the self.config.zredfile master table will be
        created.
        """

        # Make sure galaxy file exists, and is pixelized

        if not self.config.galfile_pixelized:
            raise ValueError("Code only runs with pixelized galfile.")

        # Create output path if necessary (with checks)

        zredpath = os.path.dirname(self.config.zredfile)
        galpath = os.path.dirname(self.config.galfile)

        test = re.search('^(.*)_zreds_master_table.fit',
                         os.path.basename(self.config.zredfile))
        if test is None:
            raise ValueError("zredfile filename not in proper format (must end with _zreds_master_table.fit)")

        self.config.outbase = test.groups()[0]

        if not os.path.exists(zredpath):
            try:
                os.makedirs(zredpath)
            except OSError:
                # Make sure that the path exists (From another run), if so we're good
                if not os.path.exists(zredpath):
                    raise IOError("Could not create %s directory" % (zredpath))

        # Configure the config to run only this pixel
        self.config.d.hpix = [self.pixel]
        self.config.d.nside = self.nside
        self.config.d.outbase = '%s_%05d' % (self.config.outbase, self.pixel)
        self.config.border = 0.0

        # Create a pixel lockfile
        # Note that the pixel number will probably contain many sub-pixels, but
        # this is fine because we just don't want these jobs to have the possibility
        # of stepping on each other
        writelock = '%s/%s_zreds_%07d.lock' % (zredpath, self.config.outbase, self.pixel)
        test = make_lockfile(writelock, block=False)
        if not test:
            raise IOError("Failed to get lock on pixel %d" % (self.pixel))

        # Compute all the zreds and output pixels
        runPixels = ZredRunPixels(self.config)
        runPixels.run(single_process=True, no_zred_table=True, verbose=True)

        # We are done writing, so we can clear the lockfile
        os.unlink(writelock)

        # Make a lockfile and check what's been output already.

        lockfile = '%s.lock' % (self.config.zredfile)
        locktest = make_lockfile(lockfile, block=True, maxtry=60, waittime=2)
        if locktest:
            self.config.logger.info("Created lock file: %s" % (lockfile))
            self.config.logger.info("Checking for zred completion...")

            test_files = glob.glob('%s/%s_zreds_???????.fit' % (zredpath, self.config.outbase))
            test_locks = glob.glob('%s/%s_zreds_???????.lock' % (zredpath, self.config.outbase))
            if (len(test_files) == len(runPixels.galtable.filenames) and
                len(test_locks) == 0):
                # We have written all the files, and there are no locks left.
                self.config.logger.info("All zred files have been found!  Creating master table.")

                indices = np.arange(len(runPixels.galtable.filenames))
                filenames = []
                for i in indices:
                    filenames.append('%s/%s_zreds_%07d.fit' % (zredpath, self.config.outbase, runPixels.galtable.hpix[i]))

                indices_and_filenames = list(zip(indices, filenames))

                runPixels.make_zred_table(indices_and_filenames)
            elif len(test_locks) > 0:
                pass

            # clear the lockfile
            os.unlink(lockfile)
        else:
            self.config.logger.info("Failed to get a consolidate lock.  That's okay.")

