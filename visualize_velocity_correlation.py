import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pycorr import TwoPointCorrelationFunction, project_to_multipoles

"""
python visualize_velocity_correlation.py box 0
python visualize_velocity_correlation.py box 1
python visualize_velocity_correlation.py lc 0
python visualize_velocity_correlation.py lc 1
"""

save_dir = "/global/cfs/cdirs/desi/users/boryanah/kSZ_recon/"
sim_name = "AbacusSummit_base_c000_ph002"
box_lc = sys.argv[1]
want_rsd = int(sys.argv[2])
rsd_str = "_rsd" if want_rsd else ""

def read_dat(dat_file):
    result_ab = TwoPointCorrelationFunction.load(dat_file).wrap()
    rpedges, piedges = result_ab.edges # rp, pi
    pibinc = result_ab.sepavg(axis=1)
    #print(pibinc)
    #pibinc = (piedges[1:]+piedges[:-1])*.5
    corr = result_ab.corr
    #print(rpedges)
    #return (rpedges[:-1]+rpedges[1:])*.5, corr[:, 50]
    # wp but (Pi)
    return pibinc, corr[0, :]

pi, corr_cross = read_dat(f"{save_dir}/{sim_name}/tmp/result_vel{rsd_str}_cross_{box_lc}.npy")
pi, corr_truth = read_dat(f"{save_dir}/{sim_name}/tmp/result_vel{rsd_str}_truth_{box_lc}.npy")
pi, corr_recon = read_dat(f"{save_dir}/{sim_name}/tmp/result_vel{rsd_str}_recon_{box_lc}.npy")
print(pi[:4])
print("xi los 0 cross", corr_cross[:4])
print("xi los 0 recon", corr_recon[:4])
print("xi los 0 truth", corr_truth[:4])

plt.figure(figsize=(16, 7))
plt.plot(pi, np.zeros(len(pi)), 'k--')
plt.plot(pi, corr_truth, label="truth")
plt.plot(pi, corr_recon, label="recon")
plt.plot(pi, corr_cross, label="cross")
plt.legend()
plt.ylim([-100000, 400000])
plt.savefig(f"figs/xi_vv{rsd_str}_{box_lc}.png")
plt.close()

plt.figure(figsize=(16, 7))
r_coeff = corr_cross/np.sqrt(corr_recon*corr_truth)
plt.plot(pi, np.ones(len(pi)), 'k--')
plt.plot(pi, r_coeff, c='black', ls='-')
plt.ylim([0, 2])
plt.savefig(f"figs/r_xi_vv{rsd_str}_{box_lc}.png")
#plt.show()
plt.close()
