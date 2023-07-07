from pathlib import Path
import gc

import numpy as np
from astropy.io import ascii
import h5py

from match_searchsorted import match


redshift = 0.5
sim_name = "AbacusSummit_base_c000_ph002"
tracer = "LRG"
extra = "" # "_high_density" "_bgs"

mock_dir = Path(f"/pscratch/sd/b/boryanah/AbacusHOD_scratch/mocks_box_output_kSZ_recon{extra}/")
mock_dir = mock_dir / f"{sim_name}/z{redshift:.3f}/"
sub_dir = Path("/pscratch/sd/s/sihany/summit_subsamples_cleaned_desi/")
sub_dir = sub_dir / f"{sim_name}/z{redshift:.3f}/"

gal_fn = mock_dir / f"galaxies/{tracer}s.dat"
print("Loading file:", str(gal_fn))
f = ascii.read(gal_fn)
header = f.meta
gal_id = f['id']
gal_vx = f['vx']
gal_vy = f['vy']
gal_vz = f['vz']
gal_v = np.vstack((gal_vx, gal_vy, gal_vz)).T
print("Done loading")

Ncent = header['Ncent']
Ncent = 0 # note that this is not a bad choice since we add dispersion so this should perhaps be "true"
new_sat_v = np.zeros((len(gal_id) - Ncent, 3))
check = np.zeros(len(gal_id) - Ncent, dtype=bool)

sat_id = gal_id[Ncent:]
sat_v = gal_v[Ncent:]

n_chunks = 34
print("Starting the matching")
for i_chunk in range(n_chunks):
    print(i_chunk)
    
    halo_fn = sub_dir / f"halos_xcom_{i_chunk:d}_seed600_abacushod_oldfenv_new.h5"
    part_fn = sub_dir / f"particles_xcom_{i_chunk:d}_seed600_abacushod_oldfenv_withranks_new.h5"
    
    halos = h5py.File(halo_fn, 'r')['halos']
    #particles = h5py.File(part_fn, 'r')['particles']
    #particles['halo_vel']
    #particles['halo_id']
    #particles['vel']

    halo_id = halos['id']
    halo_v_L2com = halos['v_L2com']

    i_sort = np.argsort(halo_id)
    halo_id = halo_id[i_sort]
    halo_v_L2com = halo_v_L2com[i_sort]
    
    mt_in_lc = match(sat_id, halo_id, arr2_sorted=True)
    choice = mt_in_lc > -1
    comm2 = mt_in_lc[choice]
    #comm1 = np.arange(len(sat_pid), dtype=np.int64)[choice]
    
    #new_sat_v[comm1] = halo_v_L2com[comm2]
    new_sat_v[choice] = halo_v_L2com[comm2]
    check[choice] = True
    #print("in this file we found", np.sum(choice), len(choice), np.sum(choice)/len(choice), 1./n_chunks)
    #print(sat_v[choice][:10], new_sat_v[choice][:10])
    
    del choice, comm2, mt_in_lc
    del halos
    del halo_id, halo_v_L2com, i_sort
    gc.collect()

assert np.sum(check) == len(check)

new_gal_v = np.vstack((gal_v[:Ncent], new_sat_v))
print("Done and saving")
np.save(mock_dir / f"galaxies/{tracer}s_halo_vel.npy", new_gal_v)
