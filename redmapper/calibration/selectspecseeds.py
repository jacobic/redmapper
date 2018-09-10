from __future__ import division, absolute_import, print_function

import os
import numpy as np
import fitsio
import esutil

from ..configuration import Configuration
from ..galaxy import GalaxyCatalog
from ..catalog import Catalog

class SelectSpecSeeds(object):
    """
    """

    def __init__(self, conf):
        if not isinstance(conf, Configuration):
            self.config = Configuration(conf)
        else:
            self.config = conf

    def run(self):
        """
        """

        # Read in the galaxies, but check if we have zreds
        if os.path.isfile(self.config.zredfile):
            zredfile = self.config.zredfile
            has_zreds = True
        else:
            zredfile = None
            has_zreds = False

        gals = GalaxyCatalog.from_galfile(self.config.galfile,
                                          nside=self.config.d.nside,
                                          hpix=self.config.h.hpix,
                                          border=self.config.border,
                                          zredfile=zredfile)

        # Read in the spectroscopic catalog
        spec = Catalog.from_fits_file(self.config.specfile_train)

        # Limit the redshift range
        zrange = self.config.zrange_cushioned

        # select good spectra
        use, = np.where((spec.z >= zrange[0]) & (spec.z <= zrange[1]) & (spec.z_err < 0.001))
        spec = spec[use]

        # Match spectra to galaxies
        i0, i1, dists = gals.match_many(spec.ra, spec.dec, 3./3600., maxmatch=1)

        gals = gals[i1]
        spec = spec[i0]

        # Ensure it has a valid zred
        if has_zreds:
            use, = np.where(gals.zred > 0.0)
        else:
            use, = np.where(gals.refmag > 0.0)

        cat = Catalog(np.zeros(use.size, dtype=[('ra', 'f8'),
                                                ('dec', 'f8'),
                                                ('model_mag', 'f4', self.config.nmag),
                                                ('model_magerr', 'f4', self.config.nmag),
                                                ('refmag', 'f4'),
                                                ('refmag_err', 'f4'),
                                                ('zred', 'f4'),
                                                ('zred_e', 'f4'),
                                                ('zred_chisq', 'f4'),
                                                ('zspec', 'f4'),
                                                ('ebv', 'f4')]))
        cat.ra[:] = gals.ra[use]
        cat.dec[:] = gals.dec[use]
        cat.model_mag[:, :] = gals.model_mag[use, :]
        cat.model_magerr[:, :] = gals.model_magerr[use, :]
        cat.refmag[:] = gals.refmag[use]
        cat.refmag_err[:] = gals.refmag_err[use]
        if (has_zreds):
            cat.zred[:] = gals.zred[use]
            cat.zred_e[:] = gals.zred_e[use]
            cat.zred_chisq[:] = gals.zred_chisq[use]
        else:
            cat.zred[:] = -1.0
            cat.zred_e[:] = -1.0
            cat.zred_chisq[:] = -1.0
        cat.zspec[:] = spec.z[use]
        cat.ebv[:] = gals.ebv[use]

        cat.to_fits_file(self.config.specseeds)
