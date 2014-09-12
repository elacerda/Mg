#!/usr/bin/python
import sys
import numpy as np
from scipy import stats as st
from matplotlib import pyplot as plt
from pycasso import fitsQ3DataCube

c = 299792.5  # km/s
Zsun = 0.019
imageDirectory = '/Users/lacerda/CALIFA/images/'

#EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
# Worthy et al. 1994 http://adsabs.harvard.edu/abs/1994ApJS...94..687W
# Trager el al. 1998 - http://arxiv.org/abs/astro-ph/9712258
#
# MgH         IndexBandpass:  5069.125-5134.125
#             Pseudocontinua: 4895.125-4957.625
#                             5301.125-5366.125
# MgH + Mgb   IndexBandpass:  5154.125-5196.625
#             Pseudocontinua: 4895.125-4957.625
#                             5301.125-5366.125
# Mgb         IndexBandpass:  5160.125-5192.625
#             Pseudocontinue: 5142.625-5161.375
#                             5191.375-5206.375
#EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE

def add_subplot_axes(ax, rect, axisbg = 'w'):
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    subax = fig.add_axes([x, y, width, height], axisbg = axisbg)
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2] ** 0.5
    y_labelsize *= rect[3] ** 0.5
    subax.xaxis.set_tick_params(labelsize = x_labelsize)
    subax.yaxis.set_tick_params(labelsize = y_labelsize)

    return subax

class line:
    def __init__(self, name, centralWavelenght = 0, lineExtents = [], sideBandExtents = []):
        self.name = name
        self.centralWavelenght = centralWavelenght
        self.lineExtents = lineExtents
        self.sideBandExtents = sideBandExtents

    def get_lineEdge_low(self):
        return self.lineExtents[0]

    def get_lineEdge_top(self):
        return self.lineExtents[1]

    def get_sideBandLeftEdge_low(self):
        return self.sideBandExtents[0]

    def get_sideBandLeftEdge_top(self):
        return self.sideBandExtents[1]

    def get_sideBandRightEdge_low(self):
        return self.sideBandExtents[2]

    def get_sideBandRightEdge_top(self):
        return self.sideBandExtents[3]

def v0RestFrame(K, f_interp):
    # Rest-frame spectra
    O_rf__lz = np.zeros((K.Nl_obs, K.N_zone))
    M_rf__lz = np.zeros((K.Nl_obs, K.N_zone))

    # Rest-frame wavelength
    l_rf__lz = (K.l_obs * np.ones((K.N_zone, K.Nl_obs))).T / (1 + K.v_0 / c)

    # linear-interpolating new l_obs fluxes
    # from this point afterward in the code one can use O_RF and M_RF with K.l_obs instead K.f_obs and K.f_syn.
    for z in range(K.N_zone):
        O_rf__lz[:, z] = f_interp(K.l_obs, l_rf__lz[:, z], K.f_obs[:, z])
        M_rf__lz[:, z] = f_interp(K.l_obs, l_rf__lz[:, z], K.f_syn[:, z])

    return O_rf__lz, M_rf__lz

# distributionPercentiles() divides X in bins with edges XEDGES, using
# numpy.histogram(). After that, uses the intervals in X as a MASK to calculate
# the percentiles of distribution of Y in this X interval.
def distributionPercentiles(x, y, bins, q):
    H, xedges = np.histogram(x, bins = bins)

    prc = np.zeros((bins, len(q)))

    for i, xe in enumerate(xedges[1:]):
        xe_l = xedges[i]
        xe_r = xe

        mask = (x > xe_l) & (x < xe_r)
        ym = y[mask]

        if len(ym) == 0:
            prc[i, :] = np.asarray([np.nan, np.nan, np.nan])
        else:
            prc[i, :] = np.percentile(ym, q = q)

    return H, xedges, prc

if __name__ == '__main__':
    fitsfile = sys.argv[1]
    # Load FITS
    K = fitsQ3DataCube(fitsfile)

    # Define log of base metallicities **in solar units** for convenience
    logZBase__Z = np.log10(K.metBase / Zsun)
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    # Define alogZ_****__z: flux & mass weighted average logZ for each zone
    muCor__tZz = K.popmu_cor / K.popmu_cor.sum(axis = 1).sum(axis = 0)
    alogZ_mass__z = np.tensordot(muCor__tZz, np.log10(K.metBase / 0.019) , (1, 0)).sum(axis = 0)

    x__tZz = K.popx / K.popx.sum(axis = 1).sum(axis = 0)
    alogZ_flux__z = np.tensordot(x__tZz, np.log10(K.metBase / 0.019) , (1, 0)).sum(axis = 0)
    #--------------------------------------------------------------------------

    O_rf__lz, M_rf__lz = v0RestFrame(K, np.interp)

    leftOffset = 0
    rightOffset = 0
    # leftOffset = -2
    # rightOffset = 2

    lineEdges = [5154 + leftOffset, 5196 + rightOffset]
    lineSideBandsEdges = [4894, 4960, 5300, 5366]
    centralWavelenght = 5175
    Mg = line('Mg_2', centralWavelenght, lineEdges, lineSideBandsEdges)

    # Mg Masks
    Mg_mask = ((K.l_obs < Mg.get_lineEdge_low()) | (K.l_obs > Mg.get_lineEdge_top()))
    MgSideBandLow_mask = ((K.l_obs < Mg.get_sideBandLeftEdge_low()) | (K.l_obs > Mg.get_sideBandLeftEdge_top()))
    MgSideBandUp_mask = ((K.l_obs < Mg.get_sideBandRightEdge_low()) | (K.l_obs > Mg.get_sideBandRightEdge_top()))
    MgSideBands_mask = MgSideBandLow_mask & MgSideBandUp_mask
    MgSideBandsAndLine_mask = ((K.l_obs < Mg.get_sideBandLeftEdge_low()) | (K.l_obs > Mg.get_sideBandRightEdge_top()))

    # Median flux in every zone spectra
    MgSideBands_median__z = np.median(M_rf__lz[~MgSideBands_mask, :], axis = 0)

    # Mg Delta EW
    l_step = 2
    Mg_deltaW__z = ((O_rf__lz[~Mg_mask] - M_rf__lz[~Mg_mask]) * l_step).sum(axis = 0) / MgSideBands_median__z

    # Calculate the distribution percentiles
    bins = 20
    q = [5, 50, 95]

    prop = {
    'arr'   : [ K.at_flux__z,
                np.log10(K.aZ_flux__z / 0.019),
                alogZ_mass__z,
                alogZ_flux__z,
                K.A_V,
                K.v_0,
                K.v_d,
                np.log10(K.Mcor__z),
              ],
    'label' : [ r'$\langle \log\ t \rangle_L\ [yr]$',
                r'$\log\ \langle Z \rangle_L\ [Z_\odot]$',
                r'$\langle \log\ Z_{\star} \rangle_M\ [Z_\odot]$',
                r'$\langle \log\ Z_{\star} \rangle_L\ [Z_\odot]$',
                r'$A_V\ [mag]$',
                r'$v_\star\ [km/s]$',
                r'$\sigma_\star\ [km/s]$',
                r'$\log\ M_\star\ [M_\odot]$',
              ],
    'fname' : [ 'logt',
                'logZ_L',
                'alogZ_M',
                'alogZ_L',
                'AV',
                'v0',
                'vd',
                'logM',
              ]
    }

    for i, p in enumerate(prop['arr']):
        H, xedges, prc = distributionPercentiles(p, Mg_deltaW__z, bins, q)

        f = plt.figure()
        plt.title(r'%s - %s' % (K.galaxyName, K.califaID))
        plt.xlabel(prop['label'][i])
        plt.ylabel(r'$\Delta W_{%s}$ [$\AA$]' % Mg.name)
        plt.scatter(p, Mg_deltaW__z, marker = 'o', s = 0.5)

        p5 = prc[:, 0]
        median = prc[:, 1]
        p95 = prc[:, 2]
        rhoSpearman, pvalSpearman = st.spearmanr(p, Mg_deltaW__z)
        #txt = r'$\Delta W_{%s}\ \to\ %.2f$ as $A_V^{\star} \ \to\ 0$ ($R_s$: $%.3f$)' % (Mg.name, median[0], rhoSpearman)
        txt = r'$\Delta W_{%s}\ \to\ %.2f$ as $x \ \to\ 0$ ($R_s$: $%.3f$)' % (Mg.name, median[0], rhoSpearman)
        plt.text(0.10, 0.9, txt, fontsize = 12, backgroundcolor = 'w', transform = plt.gca().transAxes, verticalalignment = 'center', horizontalalignment = 'left')

        step = (xedges[1] - xedges[0]) / 2.
        plt.plot(xedges[:-1] + step, p5, 'k--')
        plt.plot(xedges[:-1] + step, median, 'k-')
        plt.plot(xedges[:-1] + step, p95, 'k--')

        if len(sys.argv) > 2:
            # Subplot with the Galaxy Image
            subpos = [0.7, 0.55, 0.26, 0.45]
            subax = add_subplot_axes(plt.gca(), subpos)
            plt.setp(subax.get_xticklabels(), visible = False)
            plt.setp(subax.get_yticklabels(), visible = False)
            galimg = plt.imread('%s%s.jpg' % (imageDirectory, K.califaID))
            subax.imshow(galimg)

        f.savefig('%s-%s_DeltaEW_%s.png' % (K.califaID, prop['fname'][i], Mg.name))
