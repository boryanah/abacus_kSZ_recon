import sys

import numpy as np
from astropy.io import ascii
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from colossus.lss import bias
from colossus.cosmology import cosmology

#cosmo = cosmology.setCosmology('myCosmo', params)
cosmo = cosmology.setCosmology('planck18')
box_lc = "lc"
mdef = '200c'
sim_name = "AbacusSummit_base_c000_ph004"
#sim_name = "AbacusSummit_huge_c000_ph201"
redshifts = np.array([0.300, 0.350, 0.400, 0.450, 0.500, 0.575, 0.650, 0.725, 0.800, 0.875, 0.950, 1.025, 1.100])
#tracers = ["LRG_main", "ELG"]
tracers = ["LRG_main", "LRG_extended", "ELG", "BGS"]
#tracers = ["BGS"]
#tracers = ["LRG_main"]
#tracers = ["LRG_extended"]

def get_z_comov(tracer):
    if tracer == "ELG":
        fn = 'elg_main-800coaddefftime1200-nz-zenodo.ecsv'
        perc = 30.
        #extra = ""
        extra = "_uchuu"
    elif tracer == "BGS":
        #fn = 'fig19_nz_bgs_svda.dat'
        fn = 'BGS_BRIGHT_full_N_nz.txt'
        perc = 100.
        #extra = "_bgs"
        extra = "_uchuu"
        tracer = "LRG"
    elif tracer == "LRG":
        fn = 'lrg_fig_1_histograms.csv'
        perc = 100.
        extra = ""
    elif tracer == "LRG_main":
        fn = '/global/cfs/cdirs/desi/users/rongpu/data/lrg_xcorr/dndz/iron_v0.2/main_lrg_pz_dndz_iron_v0.2_dz_0.02.txt'
        tracer = "LRG"
        extra = ""
        perc = 100.
    elif tracer == "LRG_extended":
        fn = '/global/cfs/cdirs/desi/users/rongpu/data/lrg_xcorr/dndz/iron_v0.2/extended_lrg_pz_dndz_iron_v0.2_dz_0.02.txt'
        tracer = "LRG"
        extra = "_high_density"
        perc = 100.
    perc_str = f"_perc{perc:.1f}" if not np.isclose(perc, 100.) else ""
    com_fn = f"{fn.split('/')[-1].split('.')[0]}{perc_str}.npz"
    bias_fn = f"bias{extra}_{fn.split('/')[-1].split('.')[0]}{perc_str}.npz"
    data = np.load(com_fn)
    return data['z_edges'], data['comov_dens'], tracer, extra, bias_fn

def get_bias(z_binc, tracer, extra):
    b_eff = np.zeros(len(z_binc))
    b_avg = np.zeros(len(z_binc))
    b_med = np.zeros(len(z_binc))
    for i in range(len(z_binc)):
        z = z_binc[i]
        print("redshift", z)
        fn = f"/pscratch/sd/b/boryanah/AbacusHOD_scratch/mocks_{box_lc}_output_kSZ_recon{extra}/{sim_name}/z{z:.3f}/galaxies/{tracer}s.dat"
        f = ascii.read(fn)
        mass = f['mass'] # Msun/h

        b_eff[i] = np.mean(bias.haloBias(mass, model='tinker10', z=z, mdef=mdef))
        print("int b(M) dN/dM / int dN/dM", b_eff[i])

        b_avg[i] = bias.haloBias(np.mean(mass), model='tinker10', z=z, mdef=mdef)
        print("b(mean M)", b_avg[i])

        b_med[i] = bias.haloBias(np.median(mass), model='tinker10', z=z, mdef=mdef)
        print("b(median M)", b_med[i])
        print("--------------")

    return b_eff, b_avg, b_med

bs = 0.
s = 0.
zs = np.linspace(0., 1.4, 29)
choice = (zs > redshifts.min()) & (zs < redshifts.max())
for tracer in tracers:
    # density
    z_edges, comov_dens, tracer, extra, bias_fn = get_z_comov(tracer)
    z_binc = (z_edges[1:]+z_edges[:-1])*.5
    n_fun = interp1d(z_binc, comov_dens, bounds_error=False, fill_value=0.)

    # bias
    b_eff, b_avg, b_med = get_bias(redshifts, tracer, extra)
    b_fun = interp1d(redshifts, b_avg, bounds_error=False, fill_value=0.)

    # compute effective bias for this tracer
    bias_sum = np.sum(n_fun(zs[choice])*b_fun(zs[choice]))
    sum = np.sum(n_fun(zs[choice]))
    bs += bias_sum
    s += sum
    print("effective bias", tracer, bias_sum/sum)

    plt.figure(figsize=(9, 7))
    plt.plot(z_binc, b_fun(z_binc))
    plt.xlim([0., 1.4])
    plt.savefig(f"bias_{tracer}.png")
    plt.close()

    np.savez(bias_fn, z=z_binc, bias=b_fun(z_binc))

print("effective bias", tracers, bs/s)
