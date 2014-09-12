#!/usr/bin/python
import sys
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from pycasso import fitsQ3DataCube
from Mg import line, v0RestFrame

#EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
# Worthy et al. 1994 http://adsabs.harvard.edu/abs/1994ApJS...94..687W
# Trager el al. 1998 - http://arxiv.org/abs/astro-ph/9712258
#
# MgH (Mg1)           IndexBandpass:  5069.125-5134.125
#                     Pseudocontinua: 4895.125-4957.625
#                                     5301.125-5366.125
# MgH + Mgb (Mg2)     IndexBandpass:  5154.125-5196.625
#                     Pseudocontinua: 4895.125-4957.625
#                                     5301.125-5366.125
# Mgb                 IndexBandpass:  5160.125-5192.625
#                     Pseudocontinue: 5142.625-5161.375
#                                     5191.375-5206.375
#EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
fitsfile = sys.argv[1]
# Load FITS
K = fitsQ3DataCube(fitsfile)

O_rf__lz, M_rf__lz = v0RestFrame(K, np.interp)

#leftOffset = 0
#rightOffset = 0
leftOffset = 2
rightOffset = 0

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

l_obs = K.l_obs[~MgSideBandsAndLine_mask]
f_obs = O_rf__lz[~MgSideBandsAndLine_mask] / K.fobs_norm
f_syn = M_rf__lz[~MgSideBandsAndLine_mask] / K.fobs_norm
f_res = f_obs - f_syn

#EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
# max_f_obs = f_obs.max()
# max_f_syn = f_syn.max()
# max_f_res = f_res.max()
# min_f_obs = f_obs.min()
# min_f_syn = f_syn.min()
# min_f_res = f_res.min()
#
# max_f = max_f_obs
#
# if max_f_syn > max_f:
#     max_f = max_f_syn
#
# if max_f_res > max_f:
#     max_f = max_f_res
#
# min_f = min_f_res
#
# if min_f_obs < min_f:
#     min_f = min_f_obs
#
# if min_f_syn < min_f:
#     min_f = min_f_syn
#EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE

max_f = 1.5
min_f = -0.5

for i, z in enumerate(np.argsort(K.v_0)):
    f = plt.figure()
    plt.title(r'%s - %s - zone %4d - $\Delta W_{%s} = %.2f$' % (K.galaxyName, K.califaID, z, Mg.name, Mg_deltaW__z[z]))
    plt.xlim([Mg.get_sideBandLeftEdge_low() - 2, Mg.get_sideBandRightEdge_top() + 2])
    plt.ylim([min_f, max_f])
    plt.xlabel(r'$\lambda$ $[\AA]$')

    txt = r'$\sigma_\star$: %.2f km/s - $v_\star$: %.2f km/s' % (K.v_d[z], K.v_0[z])
    plt.text(0.5, 0.05, txt,
             fontsize = 12,
             backgroundcolor = 'w',
             transform = plt.gca().transAxes,
             verticalalignment = 'center',
             horizontalalignment = 'left')

    plt.plot(l_obs, f_obs[:, z], label = r'$O_\lambda$')
    plt.plot(l_obs, f_syn[:, z], label = r'$M_\lambda$')
    plt.plot(l_obs, f_res[:, z], label = r'$R_\lambda$')

    x = K.l_obs[~Mg_mask]
    y1 = O_rf__lz[~Mg_mask, z] / K.fobs_norm[z]
    y2 = M_rf__lz[~Mg_mask, z] / K.fobs_norm[z]
    plt.fill_between(x, y1, y2, where = y1 >= y2, edgecolor = 'gray', facecolor = 'lightgray', interpolate = True)
    plt.fill_between(x, y1, y2, where = y1 <= y2, edgecolor = 'gray', facecolor = 'darkgray', interpolate = True)

    plt.axvline(x = Mg.get_lineEdge_low())
    plt.axvline(x = Mg.get_lineEdge_top())

    plt.axvline(x = Mg.get_sideBandLeftEdge_low(), c = 'r')
    plt.axvline(x = Mg.get_sideBandLeftEdge_top(), c = 'r')
    plt.axvline(x = Mg.get_sideBandRightEdge_low(), c = 'r')
    plt.axvline(x = Mg.get_sideBandRightEdge_top(), c = 'r')

    plt.axhline(y = MgSideBands_median__z[z] / K.fobs_norm[z])
    plt.axvline(x = Mg.centralWavelenght, ls = ':')
    plt.legend(loc = 5)

    nMinorTicks = 4 + (Mg.get_sideBandRightEdge_top() - Mg.get_sideBandLeftEdge_low())
    plt.gca().xaxis.set_minor_locator(mpl.ticker.MaxNLocator(nbins = nMinorTicks))

    plt.grid()
    f.savefig('%s-%s-%04d.png' % (K.califaID, Mg.name, i))
    plt.close(f)
