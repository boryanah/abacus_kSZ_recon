import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pypower import CatalogFFTPower

save_dir = "/global/cfs/cdirs/desi/users/boryanah/kSZ_recon/"
sim_name = "AbacusSummit_base_c000_ph002"
mode = sys.argv[1] #"pre" # "pre" # "post"
box_lc = sys.argv[2] #"box"
dat_file = f"{save_dir}/{sim_name}/tmp/poles_{box_lc}_{mode}.npy"
ells = [0, 2, 4]

result = CatalogFFTPower.load(dat_file)
poles = result.poles

for ill in range(len(ells)):
    plt.plot(poles.k, poles.k*poles(ell=ells[ill], complex=False), label=f"l={ells[ill]:d}")
    #plt.plot(result.modes, result.modes*result.power_nonorm[ill], label=f"l={ells[ill]:d}")
plt.xscale('log')
#plt.yscale('log')
plt.legend()
plt.ylim([-250, 1600])
plt.savefig(f"figs/pk_{box_lc}_{mode}.png")
plt.close()
