import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pypower import CatalogFFTPower

"""
python visualize_velocity_power.py box 0
python visualize_velocity_power.py box 1
python visualize_velocity_power.py lc 0
python visualize_velocity_power.py lc 1
"""

save_dir = "/global/cfs/cdirs/desi/users/boryanah/kSZ_recon/"
sim_name = "AbacusSummit_base_c000_ph002"
box_lc = sys.argv[1]
want_rsd = int(sys.argv[2])
rsd_str = "_rsd" if want_rsd else ""
ells = [0, 2, 4]
cs = ['c', 'm', 'y']
want_poles = True

if box_lc == "lc":
    want_poles = True
    print("need to use poles for lc for god knows what reason")

def read_dat(dat_file, want_poles=True):
    result = CatalogFFTPower.load(dat_file)
    poles = result.poles
    if not want_poles:
        wedges = result.wedges
        print(wedges.shape)
    else:
        wedges = None
    print(poles.shape)
    return poles, wedges

poles_cross, wedges_cross = read_dat(f"{save_dir}/{sim_name}/tmp/vel{rsd_str}_cross_poles_{box_lc}.npy", want_poles=want_poles)
poles_truth, wedges_truth = read_dat(f"{save_dir}/{sim_name}/tmp/vel{rsd_str}_truth_poles_{box_lc}.npy", want_poles=want_poles)
poles_recon, wedges_recon = read_dat(f"{save_dir}/{sim_name}/tmp/vel{rsd_str}_recon_poles_{box_lc}.npy", want_poles=want_poles)

plt.figure(figsize=(9, 7))
if want_poles:
    for ill in range(len(ells)):
        k = poles_cross.k
        p_cross_ell = poles_cross(ell=ells[ill], complex=False)
        p_truth_ell = poles_truth(ell=ells[ill], complex=False)
        p_recon_ell = poles_recon(ell=ells[ill], complex=False)
        r_ell = p_cross_ell/np.sqrt(p_truth_ell*p_recon_ell)
        plt.plot(k, np.ones(len(k)), 'k--')
        plt.plot(k, r_ell, c=cs[ill], label=f"l={ells[ill]:d}")
else:
    print("nmodes", wedges_cross.nmodes)
    muavg = wedges_cross.muavg
    k, Pk_cross = wedges_cross(mu=np.max(muavg), return_k=True, complex=False)
    k, Pk_truth = wedges_truth(mu=np.max(muavg), return_k=True, complex=False)
    k, Pk_recon = wedges_recon(mu=np.max(muavg), return_k=True, complex=False)
    rk = Pk_cross/np.sqrt(Pk_truth*Pk_recon)
    print("k", k)
    print("rk", rk)
    plt.plot(k, np.ones(len(k)), 'k--')
    plt.plot(k, rk, c='k')
    
plt.xscale('log')
#plt.yscale('log')
plt.ylim([0.4, 1])
plt.legend()
if want_poles:
    plt.savefig(f"figs/r_ell{rsd_str}_{box_lc}.png")
else:
    plt.savefig(f"figs/r{rsd_str}_{box_lc}.png")
plt.close()
