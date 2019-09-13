"""Class to run the second (likelihood) pass through a catalog for cluster finding.
"""

from __future__ import division, absolute_import, print_function
from past.builtins import xrange

import fitsio
import numpy as np
import esutil
import copy

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
from .utilities import chisq_pdf, interpol

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

class RunLikelihoods(ClusterRunner):
    """
    The RunLikelihoods class is derived from a ClusterRunner, and will compute
    richness and cluster likelihood (including centering) and other associated
    values for the second "likelihood" pass of the cluster finder.

    The specific configuration variables used in the likelihood run are:

    likelihoods_r0: `float`
       r0 value for radius/richness relation
    likelihoods_beta: `float`
       beta value for radius/richness relation
    likelihoods_use_zred: `bool`
       Centering likelihood should use zred filter rather than chisq filter.
    likelihoods_minlambda: `float`
       Minimum richness (lambda/scaleval) to record in output catalog
    """

    def _additional_initialization(self):
        """
        Additional initialization for RunLikelihoods.
        """
        self.runmode = 'likelihoods'

        if self.config.likelihoods_use_zred:
            self.read_zreds = True
            self.zreds_required = True
        else:
            self.read_zreds = False
            self.zreds_required = False

        self.cutgals_bkgrange = True
        self.cutgals_chisqmax = True

        self.filetype = 'like'

    def run(self, *args, **kwargs):
        """
        Run a catalog through RunLikelihoods.

        Loop over all clusters and perform RunLikelihoods computations on each cluster.

        Parameters
        ----------
        keepz: `bool`, optional
           Keep input redshifts?  (Otherwise use z_lambda).
           Default is False.
        cleaninput: `bool`, optional
           Clean seed clusters that are out of the footprint?  Default is False.
        """

        return super(RunLikelihoods, self).run(*args, **kwargs)

    def _more_setup(self, *args, **kwargs):
        """
        More setup for RunLikelihoods

        Parameters
        ----------
        keepz: `bool`, optional
           Keep input redshifts?  (Otherwise use z_lambda).
           Default is False.
        cleaninput: `bool`, optional
           Clean seed clusters that are out of the footprint?  Default is False.
        """

        self.cleaninput = kwargs.pop('cleaninput', False)

        self.config.logger.info("%s: Likelihoods using catfile: %s" % (self.hpix_logstr, self.config.catfile))

        self.cat = ClusterCatalog.from_catfile(self.config.catfile,
                                               zredstr=self.zredstr,
                                               config=self.config,
                                               bkg=self.bkg,
                                               cosmo=self.cosmo,
                                               r0=self.r0,
                                               beta=self.beta)

        keepz = kwargs.pop('keepz', False)

        zrange = copy.copy(self.config.zrange)
        if keepz:
            zrange[0] -= self.config.calib_zrange_cushion
            zrange[0] = zrange[0] if zrange[0] > 0.05 else 0.05
            zrange[1] += self.config.calib_zrange_cushion

        use, = np.where((self.cat.z > zrange[0]) &
                        (self.cat.z < zrange[1]) &
                        (self.cat.Lambda > 0.0))

        if use.size == 0:
            self.cat = None
            self.config.logger.info("No usable inputs for likelihood on pixel %s" % (self.hpix_logstr))
            return False

        mstar = self.zredstr.mstar(self.cat.z[use])
        mlim = mstar - 2.5 * np.log10(self.limlum)

        good, = np.where(self.cat.refmag[use] < mlim)

        if good.size == 0:
            self.cat = None
            self.config.logger.info("No good inputs for likelihood on pixel %s" % (self.hpix_logstr))
            return False

        self.cat = self.cat[use[good]]

        if self.cleaninput:
            catmask = self.mask.compute_radmask(self.cat.ra, self.cat.dec)
            self.cat = self.cat[catmask]

            if self.cat.size == 0:
                self.cat = None
                self.config.logger.info("No input cluster positions are in the mask on pixel %s" % (self.hpix_logstr))
                return False

        self.do_lam_plusminux = False
        self.match_centers_to_galaxies = False
        self.record_members = False
        self.do_correct_zlambda = False
        self.do_pz = False

        return True

    def _reset_bad_values(self, cluster):
        """
        Internal method to reset all cluster values to "bad" values.
        """
        cluster.lnlamlike = -1e11
        cluster.lnbcglike = -1e11
        cluster.lnlike = -1e11

        super(RunLikelihoods, self)._reset_bad_values(cluster)

    def _process_cluster(self, cluster):
        """
        Process a single cluster with RunLikelihoods.

        Parameters
        ----------
        cluster: `redmapper.Cluster`
           Cluster to compute richness.
        """

        bad = False

        maxmag = cluster.mstar - 2.5*np.log10(self.limlum)

        lam = cluster.calc_richness(self.mask)

        minrind = np.argmin(cluster.neighbors.r)
        incut, = np.where((cluster.neighbors.pmem > 0.0) &
                          (cluster.neighbors.r > cluster.neighbors.r[minrind]))

        if cluster.Lambda < self.config.likelihoods_minlambda or incut.size < 3:
            # this is a bad cluster
            self._reset_bad_values(cluster)
            bad = True
            return bad

        # compute the cluster member likelihood
        cluster.lnlamlike = (-cluster.Lambda / cluster.scaleval -
                              np.sum(np.log(1.0 - cluster.neighbors.pmem[incut])))

        # And the central likelihood
        if self.config.lnw_cen_sigma <= 0.0:
            # We do not have a calibration yet
            cluster.lnbcglike = 0.0
        else:
            # First phi_cen
            mbar = (cluster.mstar + self.config.wcen_Delta0 +
                    self.config.wcen_Delta1 * np.log(cluster.Lambda / self.config.wcen_pivot))
            phi_cen = ((1. / (np.sqrt(2. * np.pi) * self.config.wcen_sigma_m)) *
                       np.exp(-0.5 * (cluster.neighbors.refmag[minrind] - mbar)**2. / self.config.wcen_sigma_m**2.))

            # And the zred or chisq filter
            if self.config.likelihoods_use_zred:
                # We use "z_lambda" here because this is specifically a correction on z_lambda
                # (but should be equal to cluster.redshift)
                if self.zlambda_corr is not None:
                    zrmod = interpol(self.zlambda_corr.zred_uncorr, self.zlambda_corr.z, cluster.z_lambda)
                else:
                    zrmod = cluster.z_lambda

                g = ((1./(np.sqrt(2. * np.pi) * cluster.neighbors.zred_e[minrind])) *
                     np.exp(-0.5 * (cluster.neighbors.zred[minrind] - zrmod)**2. / cluster.neighbors.zred_e[minrind]**2.))
            else:
                # chisq filter
                g = chisq_pdf(cluster.neighbors.chisq[minrind], self.zredstr.dof)

            # and the w filter
            lum = 10.**((cluster.mstar - cluster.neighbors.refmag) / 2.5)
            u, = np.where((cluster.neighbors.r > 1e-5) & (cluster.neighbors.pmem > 0.0))
            w = np.log(np.sum(cluster.neighbors.pmem[u] * lum[u] /
                              np.sqrt(cluster.neighbors.r[u]**2. + self.config.wcen_rsoft**2.)) / ((1. / cluster.r_lambda) * np.sum(cluster.neighbors.pmem[u] * lum[u])))
            sig = self.config.lnw_cen_sigma / np.sqrt(((np.clip(cluster.Lambda, None, self.config.wcen_maxlambda)) / cluster.scaleval) / self.config.wcen_pivot)
            fw = (1. / (np.sqrt(2. * np.pi) * sig)) * np.exp(-0.5 * (np.log(w) - self.config.lnw_cen_mean)**2. / (sig**2.))

            with np.warnings.catch_warnings():
                np.warnings.simplefilter("error")

                cluster.lnbcglike = np.log(phi_cen * g * fw)

            cluster.lnlike = cluster.lnbcglike + cluster.lnlamlike

        return bad


    def _postprocess(self):
        """
        RunLikelihoods post-processing.

        This will select clusters where they have a valid likelihood computed.
        """
        # For this catalog we're cutting on failed likelihood

        use, = np.where(self.cat.lnlamlike > -1e11)
        self.cat = self.cat[use]

        # should I delete unused columns here before saving?
