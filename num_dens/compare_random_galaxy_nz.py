from pathlib import Path
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

#extra = "_high_density"
extra = "_uchuu"
#extra = ""

mock_dir = Path("/global/cfs/cdirs/desi/users/boryanah/kSZ_recon/new/")
sub_dir = mock_dir / f"mocks_lc_output_kSZ_recon{extra}"
sim_name = "AbacusSummit_base_c000_ph002"
#sim_name = "AbacusSummit_base_c000_ph000"
fns = glob.glob(str(sub_dir / sim_name)+"/galaxies_*.npz") # sorry ama me murzi

z_bins = np.linspace(0.2, 1.2, 1001)
z_binc = (z_bins[1:]+z_bins[:-1])*.5

for i, fn in enumerate(fns):

    #/global/cfs/cdirs/desi/users/boryanah/kSZ_recon/new/recon/AbacusSummit_base_c000_ph002/z0.500/displacements_LRG_uchuu_zerr0.0_postrecon_R12.50_b2.2_nmesh1024_recsym_MG_BGS_BRIGHT_full_N_nz.npz"
    #if ("BGS_BRIGHT_full_N_nz.npz" not in fn) or ("zerr0" not in fn): continue
    if ("elg_main-800coaddefftime1200-nz-zenodo_perc30.0" not in fn) or ("zerr0" not in fn): continue
    if "fakelc" in fn: continue
    if "mask" in fn: continue

    fig_name = fn.split("galaxies_")[-1].split(".npz")[0]+f"{extra}.png"
    
    print(fig_name, i, len(fns))
    #rand_fn = fn.split("galaxies_")[0] + "randoms_" + fn.split("galaxies_")[-1]
    rand_fn = fn.split("galaxies_")[0] + "new_randoms_" + fn.split("galaxies_")[-1]

    data = np.load(fn)
    Z = data['Z']
    Z_RSD = data['Z_RSD']

    data = np.load(rand_fn)
    RAND_Z = data['RAND_Z']

    factor = len(RAND_Z)/len(Z)
    print("factor", factor)
    
    hist_Z, _ = np.histogram(Z, z_bins)
    hist_Z_RSD, _ = np.histogram(Z_RSD, z_bins)
    hist_RAND_Z, _ = np.histogram(RAND_Z, z_bins)
    hist_RAND_Z = hist_RAND_Z/factor
    
    plt.figure(figsize=(9, 7))
    plt.plot(z_binc, hist_Z, label="Z")
    plt.plot(z_binc, hist_Z_RSD, label="Z RSD")
    plt.plot(z_binc, hist_RAND_Z, label="RAND Z")
    plt.legend()
    plt.savefig("test_figs/"+fig_name)
    plt.close()
