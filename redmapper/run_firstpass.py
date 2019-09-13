"""Class to run the first pass through a catalog for cluster finding.
"""

from __future__ import division, absolute_import, print_function
from past.builtins import xrange

import fitsio
import numpy as np
import esutil
import os

from .cluster import ClusterCatalog
from .catalog import Catalog
from .background import Background
from .mask import HPMask
from .galaxy import GalaxyCatalog
from .cluster import Cluster
from .cluster import ClusterCatalog
from .depthmap import DepthMap
from .zlambda import Zlambda
from .zlambda import ZlambdaCorrectionPar
from .cluster_runner import ClusterRunner

###################################################
# Order of operations:
#  __init__()
#    _additional_initialization() [override this]
#  run()
#    _setup()
#    _more_setup() [override this]
#    _process_cluster() [override this]
#    _postprocess() [override this]
#  output()
###################################################

class RunFirstPass(ClusterRunner):
    """
    The RunFirstPass class is derived from ClusterRunner, and will compute
    richness, redshift (z_lambda) and other associated values for the first
    pass of the cluster finder.

    The specific configuration variables used for the firstpass run are:

    firstpass_r0: `float`
       r0 value for radius/richness relation
    firstpass_beta: `float`
       beta value for radius/richness relation
    firstpass_niter: `int`
       Number of iterations to converge on z_lambda
    firstpass_minlambda: `float`
       Minimum richness (lambda/scaleval) to record in output catalog

    The firstpass run requires a list of "seeds".  This can either be derived
    from a spectroscopic catalog (specmode) or from all zred galaxies brighter
    than the luminosity threshold.

    Parameters
    ----------
    specmode: `bool`, optional
       Run in spectroscopic-seed mode.  Default is False.
    """

    def _additional_initialization(self, specmode=False):
        """
        Additional initialization for RunCatalog.

        Parameters
        ----------
        specmode: `bool`, optional
           Run in spectroscopic seed mode.  Default is False.
        """
        # This is the runmode and where we get the mask/radius config vars from
        self.runmode = 'firstpass'

        self.read_zreds = True
        self.zreds_required = False
        self.cutgals_bkgrange = True
        self.cutgals_chisqmax = True
        self.specmode = specmode
        if specmode:
            self.filetype = 'firstpass_spec'
        else:
            self.filetype = 'firstpass'

    def run(self, *args, **kwargs):
        """
        Run a catalog through RunCatalog.

        Loop over all clusters and perform RunCatalog computations on each cluster.

        Parameters
        ----------
        keepz: `bool`, optional
           Keep input redshifts?  (Otherwise use z_lambda).
           Default is False.
        cleaninput: `bool`, optional
           Clean seed clusters that are out of the footprint?  Default is False.
        """

        return super(RunFirstPass, self).run(*args, **kwargs)

    def _more_setup(self, *args, **kwargs):
        """
        More setup for RunFirstPass.

        Parameters
        ----------
        keepz: `bool`, optional
           Keep input redshifts?  (Otherwise use z_lambda).
           Default is False.
        cleaninput: `bool`, optional
           Clean seed clusters that are out of the footprint?  Default is False.
        """

        incat = None

        self.keepz = kwargs.pop('keepz', False)
        self.cleaninput = kwargs.pop('cleaninput', False)

        # Check if there's a seedfile
        if self.config.seedfile is not None and os.path.isfile(self.config.seedfile):
            self.config.logger.info("%s: Firstpass using seedfile: %s" % (self.hpix_logstr, self.config.seedfile))
            incat = Catalog.from_fits_file(self.config.seedfile)
        else:
            if self.specmode:
                raise RuntimeError("Must have config.seedfile for run_firstpass in specmode.")
            elif not self.did_read_zreds:
                raise RuntimeError("Must have config.seedfile for run_firstpass with no zreds.")
            self.config.logger.info("%s: Firstpass using zreds as input" % (self.hpix_logstr))

        if incat is not None:
            # We have an input catalog
            # generate a cluster catalog from incat
            self.cat = ClusterCatalog.zeros(incat.size,
                                            zredstr=self.zredstr,
                                            config=self.config,
                                            bkg=self.bkg,
                                            cosmo=self.cosmo,
                                            r0=self.r0,
                                            beta=self.beta)

            self.cat.ra = incat.ra
            self.cat.dec = incat.dec
            self.cat.mag = incat.mag
            self.cat.mag_err = incat.mag_err
            self.cat.refmag = incat.refmag
            self.cat.refmag_err = incat.refmag_err
            self.cat.zred = incat.zred
            self.cat.zred_e = incat.zred_e
            self.cat.zred_chisq = incat.zred_chisq
            self.cat.chisq = incat.zred_chisq
            self.cat.ebv_mean = incat.ebv
            self.cat.z_spec_init = incat.zspec

            if self.specmode:
                self.cat.z_init = incat.zspec
                self.cat.z = incat.zspec
            else:
                self.cat.z_init = incat.zred
                self.cat.z = incat.zred

            cuse = ((self.cat.zred >= self.config.zrange[0]) &
                    (self.cat.zred <= self.config.zrange[1]))

            if self.cutgals_chisqmax:
                cuse &= (self.cat.chisq < self.config.chisq_max)

            use, = np.where(cuse)

            if use.size == 0:
                self.cat = None
                self.config.logger.info("No good zred inputs for firstpass on pixel %s" % (self.hpix_logstr))
                return False

            self.cat = self.cat[use]

            if self.cleaninput:
                catmask = self.mask.compute_radmask(self.cat.ra, self.cat.dec)
                self.cat = self.cat[catmask]

                if self.cat.size == 0:
                    self.cat = None
                    self.config.logger.info("No zred positions are in the mask on pixel %s" % (self.hpix_logstr))
                    return False
        else:
            # We do not have an input catalog
            # we must not have a seedfile, and did_read_zreds
            # so generate a cluster catalog from self.gals

            cuse = ((self.gals.zred >= self.config.zrange[0]) &
                    (self.gals.zred <= self.config.zrange[1]))

            if self.cutgals_chisqmax:
                cuse &= (self.gals.chisq < self.config.chisq_max)

            use, = np.where(cuse)

            if use.size == 0:
                self.cat = None
                self.config.logger.info("No usable zred inputs for firstpass on pixel %s" % (self.hpix_logstr))
                return False

            mstar = self.zredstr.mstar(self.gals.zred[use])
            mlim = mstar - 2.5 * np.log10(self.config.lval_reference)

            good, = np.where(self.gals.refmag[use] < mlim)

            if good.size == 0:
                self.cat = None
                self.config.logger.info("No good zred inputs for firstpass on pixel %s" % (self.hpix_logstr))
                return False

            self.cat = ClusterCatalog.zeros(good.size,
                                            zredstr=self.zredstr,
                                            config=self.config,
                                            bkg=self.bkg,
                                            cosmo=self.cosmo)

            self.cat.ra = self.gals.ra[use[good]]
            self.cat.dec = self.gals.dec[use[good]]
            self.cat.mag = self.gals.mag[use[good]]
            self.cat.mag_err = self.gals.mag_err[use[good]]
            self.cat.refmag = self.gals.refmag[use[good]]
            self.cat.refmag_err = self.gals.refmag_err[use[good]]
            self.cat.zred = self.gals.zred[use[good]]
            self.cat.zred_e = self.gals.zred_e[use[good]]
            self.cat.zred_chisq = self.gals.chisq[use[good]]
            self.cat.chisq = self.gals.chisq[use[good]]

            self.cat.z_init = self.cat.zred
            self.cat.z = self.cat.zred

            # self.match_cat_to_spectra()

        self.do_lam_plusminus = False
        self.match_centers_to_galaxies = False
        self.do_percolation_masking = False
        self.do_correct_zlambda = False
        self.do_pz = False

        #self.limlum = np.clip(self.config.lval_reference - 0.1, 0.01, None)

        self.maxiter = self.config.firstpass_niter
        if self.specmode:
            self.maxiter = 1

        self.min_lambda = self.config.firstpass_minlambda

        return True

    def _process_cluster(self, cluster):
        """
        Process a single cluster with RunFirstpass.

        Parameters
        ----------
        cluster: `redmapper.Cluster`
           Cluster to compute richness.
        """
        bad = False
        iteration = 0
        done = False

        zuse = cluster.z_init.copy()

        for i in xrange(self.maxiter):
            if bad:
                done = True
                continue

            lam = cluster.calc_richness(self.mask, calc_err=False)

            if (lam < np.abs(self.config.firstpass_minlambda / cluster.scaleval)):
                bad = True
                done = True
                self._reset_bad_values(cluster)
                continue

            if i < self.maxiter:
                # only on first iteration, compute z_lambda
                # Really, this should be on at most n-1th iteration
                zlam = Zlambda(cluster)
                z_lambda, z_lambda_e = zlam.calc_zlambda(cluster.redshift, self.mask,
                                                         calc_err=True, calcpz=False)

                if z_lambda < self.config.zrange[0] or z_lambda > self.config.zrange[1]:
                    bad = True
                    done = True
                    self._reset_bad_values(cluster)
                    continue

                if not self.keepz:
                    cluster.redshift = z_lambda

        if bad:
            cluster.z_lambda = -1.0
            cluster.z_lambda_e = -1.0
            cluster.z_lambda_niter = 0
        else:
            cluster.z_lambda = z_lambda
            cluster.z_lambda_e = z_lambda_e
            cluster.z_lambda_niter = zlam.niter

        cind = np.argmin(cluster.neighbors.r)
        cluster.chisq = cluster.neighbors.chisq[cind]

        # and record the .z for the next round
        if (self.specmode):
            cluster.z = cluster.z_spec_init
        else:
            cluster.z = cluster.z_lambda

        # All done
        return bad
