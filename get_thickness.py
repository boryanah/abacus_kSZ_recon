import numpy as np

from prepare_lc_catalog import relate_chi_z

sim_name = "AbacusSummit_base_c000_ph000"
L = 2000.

# functions relating chi and z
chi_of_z, z_of_chi = relate_chi_z(sim_name)

z_min = 0.4
z_maxs = np.array([0.5, 0.6, 0.7, 0.8])

for z_max in z_maxs:
    print("chi_min, chi_max", chi_of_z(z_min), chi_of_z(z_max))
    print("z_min, z_max", z_min, z_max)
    V = 1./8 * 4./3 * np.pi * (chi_of_z(z_max)**3 - chi_of_z(z_min)**3)
    print(f"total volume = {V:.2f}")
    print(f"fraction of box = {(V/L**3):.2f}")
    print(f"thickness = {(-chi_of_z(z_min)+chi_of_z(z_max)):.2f}")


