import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pycorr import TwoPointCorrelationFunction, project_to_multipoles

save_dir = "/global/cfs/cdirs/desi/users/boryanah/kSZ_recon/"
sim_name = "AbacusSummit_base_c000_ph002"
mode = sys.argv[1] #"post" #"pre" # "post"
box_lc = sys.argv[2] #"box"
dat_file = f"{save_dir}/{sim_name}/tmp/result_cross_{box_lc}_{mode}.npy"
poles = [0, 2, 4]

result_ab = TwoPointCorrelationFunction.load(dat_file)
s_cross, xiell_cross = project_to_multipoles(result_ab, ells=poles)

for ill in range(len(poles)):
    plt.plot(s_cross, xiell_cross[ill]*s_cross**2, label=f"l={poles[ill]:d}")
plt.legend()
plt.ylim([-80, 110])
plt.savefig(f"figs/xi_{box_lc}_{mode}.png")
plt.close()
