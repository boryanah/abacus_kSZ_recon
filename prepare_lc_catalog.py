import os
import gc
import sys
import glob
from pathlib import Path
import argparse
sys.path.append("/global/homes/b/boryanah/abacus_lensing/maps")
sys.path.append("/global/homes/b/boryanah/abacus_lensing/post")

import numpy as np
import healpy as hp
import asdf
from astropy.io import fits
from astropy.io import ascii
from scipy.interpolate import interp1d

from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog
from generate_randoms import gen_rand
from tools import compress_asdf

DEFAULTS = {}
DEFAULTS['sim_name'] = "AbacusSummit_base_c000_ph002" # "AbacusSummit_huge_c000_ph201"
DEFAULTS['stem'] = 'DESI_LRG' # 'DESI_ELG' 

def get_norm(gals_pos, origin):
    """ get normal vector and chis"""
    gals_norm = gals_pos - origin
    gals_chis = np.linalg.norm(gals_norm, axis=1)
    assert len(gals_pos) == len(gals_chis)
    gals_norm /= gals_chis[:, None]
    gals_min = np.min(gals_chis)
    gals_max = np.max(gals_chis)

    return gals_norm, gals_chis, gals_min, gals_max

def get_ra_dec_chi(norm, chis):
    """ given normal vector and chis, return, ra, dec, chis"""
    theta, phi = hp.vec2ang(norm)
    ra = phi
    dec = np.pi/2. - theta
    ra *= 180./np.pi
    dec *= 180./np.pi
    
    return ra, dec, chis

def relate_chi_z(sim_name):
    # load zs from high to low
    data_path = Path("/global/homes/b/boryanah/repos/abacus_lc_cat/data_headers/")

    # all redshifts, steps and comoving distances of light cones files; high z to low z
    zs_all = np.load(data_path / sim_name / "redshifts.npy")
    chis_all = np.load(data_path / sim_name / "coord_dist.npy")
    zs_all[-1] = float('%.1f'%zs_all[-1])

    # get functions relating chi and z
    chi_of_z = interp1d(zs_all,chis_all)
    z_of_chi = interp1d(chis_all,zs_all)
    return chi_of_z, z_of_chi

def read_dat(fn):
    # load file with catalog
    f = ascii.read(fn)
    gals_pos = np.vstack((f['x'], f['y'], f['z'])).T
    gals_vel = np.vstack((f['vx'], f['vy'], f['vz'])).T
    N_gals = gals_pos.shape[0]
    print("galaxy number loaded = ", N_gals)
    return gals_pos, gals_vel

def main(sim_name, stem):
    # parameter choices
    if stem == 'DESI_LRG': 
        redshifts = [0.300, 0.350, 0.400, 0.450, 0.500, 0.575, 0.650, 0.725, 0.800, 0.875, 0.950, 1.025, 1.100]
    elif stem == 'DESI_ELG':
        redshifts = [0.800]
        
    # select tracer
    if "LRG" in stem.upper():
        tracer = "LRG"
    elif "ELG" in stem.upper():
        tracer = "ELG"
    else:
        print("tracer must be either LRG or ELG"); quit()
        #tracer = "halos"

    # number of randoms
    want_rands = True    
    if want_rands:
        rands_fac = 20
    
    # immutables
    sim_dir = "/global/cfs/cdirs/desi/cosmosim/Abacus/halo_light_cones/"
    #lens_dir = f"/global/cfs/cdirs/desi/cosmosim/AbacusLensing/{sim_name}/"
    lens_dir = f"/global/cfs/cdirs/desi/public/cosmosim/boryanah_AbacusLensing/{sim_name}/"
    save_dir = Path("/global/project/projectdirs/desi/users/boryanah/kSZ_recon/")
    cat_dir = "/global/cscratch1/sd/boryanah/abacus_lensing/"
    offset = 10. # Mpc/h

    # read from simulation header
    header = asdf.open(f"{sim_dir}/{sim_name}/z{redshifts[0]:.3f}/lc_halo_info.asdf")['header']
    Lbox = header['BoxSizeHMpc']
    mpart = header['ParticleMassHMsun'] # 5.7e10, 2.1e9
    origins = np.array(header['LightConeOrigins']).reshape(-1,3)
    origin = origins[0]
    print(f"mpart = {mpart:.2e}")

    # functions relating chi and z
    chi_of_z, z_of_chi = relate_chi_z(sim_name)

    # loop over all catalog redshifts        
    for redshift in redshifts:
        # directory for saving stuff
        save_sub_dir = Path(save_dir) / sim_name / (f"z{redshift:.3f}") 
        os.makedirs(save_sub_dir, exist_ok=True)

        # file with galaxy mock
        #gal_rsd_fn = f"{cat_dir}/mocks_box_output/{sim_name}/z{redshift:.3f}/galaxies{rsd_str}/{tracer}s.dat"
        gal_rsd_fn = f"{cat_dir}/mocks_lc_output_test_paper/z{redshift:.3f}/galaxies_rsd/{tracer}s.dat"
        gal_real_fn = f"{cat_dir}/mocks_lc_output_test_paper/z{redshift:.3f}/galaxies/{tracer}s.dat"
        print(gal_rsd_fn)
        
        # read files
        gals_real_pos, gals_real_vel = read_dat(gal_real_fn)
        gals_rsd_pos, gals_rsd_vel = read_dat(gal_rsd_fn)
        
        # get the unit vectors and comoving distances to the observer
        gals_norm, gals_chis, _, _ = get_norm(gals_real_pos, origin)
        _, CZ_rsd, gals_min, gals_max = get_norm(gals_rsd_pos, origin)
        del _; gc.collect()
        print("closest and furthest distance of gals = ", gals_min, gals_max)

        # fixing when there are no halos at low halo mass
        gals_min = np.min(gals_chis[gals_chis > 0.])
        print("GALS MIN = ", gals_min)
        
        if want_rands:
            # generate randoms in L shape
            rands_pos, rands_norm, rands_chis = gen_rand(len(gals_chis), gals_min, gals_max, rands_fac, Lbox, offset, origins)
        del gals_min, gals_max; gc.collect()

        # convert the unit vectors into RA and DEC
        RA, DEC, CZ = get_ra_dec_chi(gals_norm, gals_chis)
        if want_rands:
            rands_RA, rands_DEC, rands_CZ = get_ra_dec_chi(rands_norm, rands_chis)
            del rands_norm, rands_chis; gc.collect()
        del gals_chis; gc.collect()
        
        # fixing when there are no halos at low halo mass
        choice = ~np.isnan(RA)
        RA = RA[choice]
        DEC = DEC[choice]
        CZ = CZ[choice]
        CZ_rsd = CZ_rsd[choice]
        gals_real_vel = gals_real_vel[choice]
        gals_rsd_vel = gals_rsd_vel[choice]
        gals_real_pos = gals_real_pos[choice]
        gals_rsd_pos = gals_rsd_pos[choice]
        gals_norm = gals_norm[choice]
        print("how many RAs were none = ", np.sum(~choice))
        print("RA min/max", RA.min(), RA.max())
        print("DEC min/max", DEC.min(), DEC.max())

        # convert chi to redshift
        Z = z_of_chi(CZ)
        Z_rsd = z_of_chi(CZ_rsd)
        if want_rands:
            rands_Z = z_of_chi(rands_CZ)

        # save in the cosmosim directory as npz arrays
        np.savez(save_sub_dir / f"galaxies_{tracer}_pos.npz", RA=RA, DEC=DEC, CZ=CZ, CZ_RSD=CZ_rsd, Z=Z, Z_RSD=Z_rsd, VEL=gals_real_vel, VEL_RSD=gals_rsd_vel, POS=gals_real_pos, POS_RSD=gals_rsd_pos, UNIT_LOS=gals_norm)
        np.savez(save_sub_dir / f"randoms_{tracer}_pos.npz", RAND_RA=rands_RA, RAND_DEC=rands_DEC, RAND_CZ=rands_CZ, RAND_Z=rands_Z, RAND_POS=rands_pos)
        
class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--sim_name', help='Simulation name', default=DEFAULTS['sim_name'])
    parser.add_argument('--stem', help='Stem file name', default=DEFAULTS['stem'])

    args = vars(parser.parse_args())
    main(**args)
