from pathlib import Path
import gc
import sys

import numpy as np
from astropy.io import ascii
import h5py

from match_searchsorted import match

"""
python match_velocity.py LRG "" lc
python match_velocity.py ELG "" lc
python match_velocity.py LRG _bgs lc
python match_velocity.py LRG _high_density lc
python match_velocity.py LRG _uchuu lc
python match_velocity.py ELG _uchuu lc
"""

tracer = sys.argv[1] #"ELG" "LRG"
extra = sys.argv[2] #"" "_high_density" "_bgs"
box_lc = sys.argv[3] #"box" #"lc"

sim_type = "base"
#sim_type = "huge"
if sim_type == "base":
    #num_sims = 10
    num_sims = 1
    phase_start = 2
    #phase_start = 15
else:
    num_sims = 1
    phase_start = 201

par_mock_dir = Path(f"/pscratch/sd/b/boryanah/AbacusHOD_scratch/mocks_{box_lc}_output_kSZ_recon{extra}/")
if box_lc == "box":
    par_sub_dir = Path("/pscratch/sd/s/sihany/summit_subsamples_cleaned_desi/")
elif box_lc == "lc":
    par_sub_dir = Path("/pscratch/sd/b/boryanah/AbacusHOD_scratch/mocks_lc_subsample/")

if box_lc == "lc":
    redshifts = np.array([0.300, 0.350, 0.400, 0.450, 0.500, 0.575, 0.650, 0.725, 0.800, 0.875, 0.950, 1.025, 1.100])
else:
    redshifts = np.array([0.8]) #0.5])

sim_names = []
for i in range(num_sims):
    sim_names.append(f"AbacusSummit_{sim_type}_c000_ph{phase_start+i:03d}")
print(sim_names)

for sim_name in sim_names:
    for redshift in redshifts:

        mock_dir = par_mock_dir / f"{sim_name}/z{redshift:.3f}/"
        sub_dir = par_sub_dir / f"{sim_name}/z{redshift:.3f}/"

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
        #if box_lc == "box":
        #    Ncent = len(gal_id) # note that this is not a bad choice since we add dispersion so this should perhaps be "true"
        Ncent = len(gal_id) # note that this is not a bad choice since we add dispersion so this should perhaps be "true"

        new_cent_v = np.zeros((Ncent, 3))
        cent_check = np.zeros(Ncent, dtype=bool)
            
        new_sat_v = np.zeros((len(gal_id) - Ncent, 3))
        sat_check = np.zeros(len(gal_id) - Ncent, dtype=bool)

        cent_id = gal_id[:Ncent]
        cent_v = gal_v[:Ncent]
        
        sat_id = gal_id[Ncent:]
        sat_v = gal_v[Ncent:]
        print("central percentage", Ncent/len(gal_id))
        
        if box_lc == "box":
            n_chunks = 34
        elif box_lc == "lc":
            n_chunks = 1


        print("Starting the matching")
        for i_chunk in range(n_chunks):
            print(i_chunk, "\r")

            if box_lc == "box":
                halo_fn = sub_dir / f"halos_xcom_{i_chunk:d}_seed600_abacushod_oldfenv_new.h5"
                part_fn = sub_dir / f"particles_xcom_{i_chunk:d}_seed600_abacushod_oldfenv_withranks_new.h5"
            elif box_lc == "lc":
                halo_fn = sub_dir / f"halos_xcom_0_seed600_abacushod_oldfenv_MT_new.h5"
                part_fn = sub_dir / f"particles_xcom_0_seed600_abacushod_oldfenv_MT_withranks_new.h5"

            halos = h5py.File(halo_fn, 'r')['halos']
            if False:#box_lc == "lc":
                particles = h5py.File(part_fn, 'r')['particles']
                part_vel = particles['halo_vel']
                #part_vel = particles['vel']
                part_id = particles['halo_id']
                

            halo_id = halos['id']
            halo_v_L2com = halos['v_L2com']

            i_sort = np.argsort(halo_id)
            halo_id = halo_id[i_sort]
            halo_v_L2com = halo_v_L2com[i_sort]
            if False:# box_lc == "lc":
                i_sort = np.argsort(part_id)
                part_id = part_id[i_sort]
                part_vel = part_vel[i_sort]

            mt_in_lc = match(cent_id, halo_id, arr2_sorted=True)
            choice = mt_in_lc > -1
            comm2 = mt_in_lc[choice]

            new_cent_v[choice] = halo_v_L2com[comm2]
            cent_check[choice] = True
            #print("in this file we found", np.sum(choice), len(choice), np.sum(choice)/len(choice), 1./n_chunks)
            #print(cent_v[choice][:10], new_cent_v[choice][:10])

            # og
            """
            if False:#box_lc == "lc":
                mt_in_lc = match(sat_id, part_id, arr2_sorted=True)
                choice = mt_in_lc > -1
                comm2 = mt_in_lc[choice]

                new_sat_v[choice] = part_vel[comm2]
                sat_check[choice] = True
            # TESTING # is the same
            """
            if False:#box_lc == "lc":
                mt_in_lc = match(sat_id, halo_id, arr2_sorted=True)
                choice = mt_in_lc > -1
                comm2 = mt_in_lc[choice]

                new_sat_v[choice] = halo_v_L2com[comm2]
                sat_check[choice] = True
                        
            del choice, comm2, mt_in_lc
            del halos
            del halo_id, halo_v_L2com, i_sort
            gc.collect()

        # TESTING!!!!!!!
        #assert np.sum(cent_check) == len(cent_check)
        if box_lc == "lc":
            assert np.sum(sat_check) == len(sat_check)

        new_gal_v = np.vstack((new_cent_v, new_sat_v))
        print("Done and saving")
        #print("new cent", new_gal_v[:5])
        #print("old cent", gal_v[:5])
        #print("new sats", new_gal_v[-5:])
        #print("old sats", gal_v[-5:]);
        np.save(mock_dir / f"galaxies/{tracer}s_halo_vel.npy", new_gal_v)
