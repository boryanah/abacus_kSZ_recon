import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table

def surface_to_comoving_density(zmin, zmax, surf_dens, H0=100, Om0=0.3):

    from astropy.cosmology import FlatLambdaCDM
    import astropy.units as u

    area_sphere = 41252.96  # Total area of the sky in sq. deg.
    cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)

    densities = np.zeros(len(zmin))

    for index in range(len(zmin)):
        volume = cosmo.comoving_volume(zmax[index]) - cosmo.comoving_volume(zmin[index])
        # convert to Python scalar
        volume = volume / (1*u.Mpc)**3
        densities[index] = surf_dens[index]*area_sphere/volume

    return densities

# read correct file
tracer = sys.argv[1]
if tracer == "ELG":
    fn = 'elg_main-800coaddefftime1200-nz-zenodo.ecsv'
    perc = 30.
elif tracer == "BGS":
    #fn = 'BGS_BRIGHT-21.5_full_nz.txt'
    fn = 'BGS_BRIGHT_full_N_nz.txt'
    perc = 100.
elif tracer == "LRG":
    fn = 'lrg_fig_1_histograms.csv'
    perc = 100.
elif tracer == "LRG_main":
    fn = '/global/cfs/cdirs/desi/users/rongpu/data/lrg_xcorr/dndz/iron_v0.2/main_lrg_pz_dndz_iron_v0.2_dz_0.02.txt'
    perc = 100.
elif tracer == "LRG_extended":
    fn = '/global/cfs/cdirs/desi/users/rongpu/data/lrg_xcorr/dndz/iron_v0.2/extended_lrg_pz_dndz_iron_v0.2_dz_0.02.txt'
    perc = 100.
perc_str = f"_perc{perc:.1f}" if not np.isclose(perc, 100.) else ""

if tracer == "LRG_main" or tracer == "LRG_extended" or tracer == "BGS":
    cat = Table.read(fn, format='ascii.commented_header')
else:
    cat = Table.read(fn)

# convert to comoving density in h^3 Mpc^-3
if tracer == "ELG":
    comov_dens = surface_to_comoving_density(cat['ZMIN'], cat['ZMAX'], cat['ELG_LOP_SOUTH_DES'])
    z_edges = np.append(cat['ZMIN'], cat['ZMAX'][-1])
elif tracer == "LRG":
    comov_dens = surface_to_comoving_density(cat['zmin'], cat['zmax'], cat['n_desi_lrg'])
    z_edges = np.append(cat['zmin'], cat['zmax'][-1])
elif tracer == "LRG_main" or tracer == "LRG_extended":
    comov_dens = surface_to_comoving_density(cat['zmin'], cat['zmax'], cat['all_combined'])
    z_edges = np.append(cat['zmin'], cat['zmax'][-1])
elif tracer == "BGS":
    comov_dens = cat['n(z)']
    z_edges = np.append(cat['zlow'], cat['zhigh'][-1])

# save file
print("number density", comov_dens, comov_dens.shape)
print("z edges", z_edges, z_edges.shape)
np.savez(f"{fn.split('/')[-1].split('.')[0]}{perc_str}.npz", comov_dens=comov_dens*perc/100., z_edges=z_edges)

plt.figure(figsize=(9, 7))
plt.plot((z_edges[1:]+z_edges[:-1])*.5, comov_dens)
plt.ylim([1.e-6, 1.e-3])
plt.xlim([0., 1.4])
if "ELG" in tracer:
    plt.yscale('log')
plt.savefig(f"{fn.split('/')[-1].split('.')[0]}.png")
plt.close()
