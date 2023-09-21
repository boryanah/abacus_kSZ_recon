from pathlib import Path
import gc

import numpy as np
from astropy.io import ascii

tracer = "LRG"
out_dir = "/pscratch/sd/b/boryanah/AbacusHOD_scratch/mocks_box_output_kSZ_recon/"
sim_name = "AbacusSummit_base_c000_ph000"
z_mock = 0.500
loc = "galaxies"

mock_dir = Path(out_dir) / sim_name / f"z{z_mock:.3f}" / loc
f = ascii.read(mock_dir / f"{tracer}s.dat")
vel = np.vstack((f['vx'], f['vy'], f['vz'])).T
halo_vel = np.load(mock_dir / f"{tracer}s_halo_vel.npy")

vel_r = vel[:, 2]
halo_vel_r = halo_vel[:, 2]
assert len(vel_r) == len(halo_vel_r)

vel_rms = np.sqrt(np.mean(vel[:, 0]**2+vel[:, 1]**2+vel[:, 2]**2))/np.sqrt(3.)
vel_3d_rms = np.sqrt(np.mean(vel**2, axis=0))
vel_r_rms = np.sqrt(np.mean(vel_r**2))

print("galaxy")
print("1D RMS", vel_rms)
print("3D RMS", vel_3d_rms)
print("LOS RMS", vel_r_rms)

halo_vel_rms = np.sqrt(np.mean(halo_vel[:, 0]**2+halo_vel[:, 1]**2+halo_vel[:, 2]**2))/np.sqrt(3.)
halo_vel_3d_rms = np.sqrt(np.mean(halo_vel**2, axis=0))
halo_vel_r_rms = np.sqrt(np.mean(halo_vel_r**2))

print("halo")
print("1D RMS", halo_vel_rms)
print("3D RMS", halo_vel_3d_rms)
print("LOS RMS", halo_vel_r_rms)

print("cross")
print("1D R", np.mean(halo_vel*vel)/(vel_rms*halo_vel_rms))
print("3D R", np.mean(halo_vel*vel, axis=0)/(vel_3d_rms*halo_vel_3d_rms))
print("LOS R", np.mean(halo_vel_r*vel_r)/(vel_r_rms*halo_vel_r_rms))

