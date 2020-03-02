"""Class to generate redmagic randoms"""

from __future__ import division, absolute_import, print_function
from past.builtins import xrange

import os
import copy
import numpy as np
import healsparse

from ..configuration import Configuration
from ..catalog import Catalog
from ..galaxy import GalaxyCatalog
from ..volumelimit import VolumeLimitMask

class RedmagicGenerateRandoms(object):
    """
    Class to generate redmagic randoms from a redmagic volume limit mask.
    """

    def __init__(self, config, vlim_mask_or_file, redmagic_cat_or_file):
        """
        Instantiate a RedmagicGenerateRandoms object

        Parameters
        ----------
        config: `redmapper.Configuration`
           Configuration object
        vlim_mask_or_file: `str` or `redmapper.VolumeLimitMask`
           Name of a file with the volume-limited mask information or
           a volume-limit mask.
        redmagic_cat_or_file: `str` or `redmapper.Catalog`
           Name of redmagic file or redmagic catalog.
        """

        self.config = config

        if isinstance(vlim_mask_or_file, VolumeLimitMask):
            self.vlim_mask = vlim_mask_or_file
        elif isinstance(vlim_mask_or_file, str):
            # This 0.2 is a dummy value
            self.vlim_mask = VolumeLimitMask(config, 0.2, vlimfile=self.vlim_mask_or_file)
        else:
            raise RuntimeError("vlim_mask_or_file must be a redmapper.VolumeLimitMask or a filename")

        if isinstance(redmagic_cat_or_file, GalaxyCatalog):
            self.redmagic_cat = redmagic_cat_or_file
        elif isinstance(redmagic_cat_or_file, str):
            self.redmagic_cat = GalaxyCatalog.from_fits_file(redmagic_file)
        else:
            raise RuntimeError("redmagic_cat_or_file must be a redmapper.GalaxyCatalog")

    def generate_randoms(self, nrandoms, filename, clobber=False):
        """
        Generate random points, and save to filename

        Parameters
        ----------
        nrandoms: `int`
           Number of randoms to generate
        filename: `str`
           Output filename
        clobber: `bool`
           Clobber output file?  Default is False.
        """

        if not clobber and os.path.isfile(filename):
            raise RuntimeError("Random file %s already exists and clobber is False." % (filename))

        min_gen = 10000
        max_gen = 1000000

        n_left = copy.copy(nrandoms)
        ctr = 0

        dtype = [('ra', 'f8'),
                 ('dec', 'f8'),
                 ('z', 'f4'),
                 ('weight', 'f4')]

        randcat = Catalog(np.zeros(nrandoms, dtype=dtype))

        # Not used at the moment
        randcat.weight[:] = 1.0

        while (n_left > 0):
            n_gen = np.clip(n_left * 3, min_gen, max_gen)
            ra_rand, dec_rand = healsparse.make_uniform_randoms(self.vlim_mask.sparse_vlimmap,
                                                                n_gen)

            # What are the associated z_max and fracgood?
            zmax, fracgood = self.vlim_mask.calc_zmax(ra_rand, dec_rand, get_fracgood=True)

            # Down-select from fracgood
            r = np.random.uniform(size=n_gen)
            gd, = np.where(r < fracgood)

            # Go back and generate more if all bad
            if gd.size == 0:
                continue

            tempcat = Catalog(np.zeros(gd.size, dtype=dtype))
            tempcat.ra = ra_rand[gd]
            tempcat.dec = dec_rand[gd]
            tempcat.z[:] = -1.0

            zz = np.random.choice(self.redmagic_cat.zredmagic, size=gd.size)
            zmax = zmax[gd]

            # This essentially takes each redshift and then finds a random
            # point where it fits within the redshift envelope.  It's a bit
            # inefficient, but it preserves the redshift distribution.

            zctr = 0
            for i in xrange(tempcat.size):
                if zz[zctr] < zmax[i]:
                    # This redshift is okay!
                    tempcat.z[i] = zz[zctr]
                    zctr += 1

            gd, = np.where(tempcat.z > 0.0)
            n_good = gd.size

            if n_good == 0:
                continue

            if n_good > n_left:
                n_good = n_left

            randcat._ndarray[ctr: ctr + n_good] = tempcat._ndarray[:n_good]

            ctr += n_good
            n_left -= n_good

        randcat.to_fits_file(filename, clobber=True)
