import os
import gc
import glob
from pathlib import Path
import argparse

import numpy as np
import healpy as hp
import asdf
from astropy.io import ascii
from scipy.interpolate import interp1d

from abacusnbody.metadata import get_meta
from generate_randoms import gen_rand, is_in_lc
from apply_dndz_mask import parse_nztype, get_dndz_mask
from prepare_lc_catalog import relate_chi_z, read_dat, get_norm, get_ra_dec_chi

DEFAULTS = {}
DEFAULTS['sim_name'] = "AbacusSummit_base_c000_ph002"
DEFAULTS['stem'] = 'DESI_LRG' # 'DESI_ELG' 
DEFAULTS['nz_type'] = 'Gaussian(0.5, 0.4)'

"""
Usage:
python create_fake_lc.py --stem DESI_LRG --sim_name AbacusSummit_base_c000_ph002 --nz_type 'Gaussian(0.5,0.4)'
python create_fake_lc.py --stem DESI_LRG --sim_name AbacusSummit_base_c000_ph002 --nz_type 'Step(0.3,0.7)'
python create_fake_lc.py --stem DESI_LRG --sim_name AbacusSummit_huge_c000_ph201
"""

np.random.seed(300)

def apply_rsd(gals_pos, gals_vel, inv_velz2kms, norm):
    nx = norm[:, 0]
    ny = norm[:, 1]
    nz = norm[:, 2]
    proj = inv_velz2kms * (gals_vel[:, 0]*nx + gals_vel[:, 1]*ny + gals_vel[:, 2]*nz)
    gals_pos[:, 0] += proj*nx
    gals_pos[:, 1] += proj*ny
    gals_pos[:, 2] += proj*nz
    return gals_pos

def main(sim_name, stem, nz_type):

    # parse the dndz instructions
    nz_dict = parse_nztype(nz_type)
    if nz_dict['Type'] == "Gaussian":
        nz_str = f"_meanz{nz_dict['Mean_z']:.3f}_deltaz{nz_dict['Delta_z']:.3f}"
    elif nz_dict['Type'] == "StepFunction":
        nz_str = f"_minz{nz_dict['Min_z']:.3f}_maxz{nz_dict['Max_z']:.3f}"
    elif nz_dict['Type'] == "FromFile":
        nz_str = f"_{nz_dict['File']}"
    
    # additional specs of the tracer
    extra = '_'.join(stem.split('_')[2:])
    stem = '_'.join(stem.split('_')[:2])
    if extra != '': extra = '_'+extra
    
    # parameter choices
    if stem == 'DESI_LRG': 
        Mean_z = 0.5
        z_min = 0.275
        if "base" in sim_name:
            z_max = 0.8 #1.12
        else:
            z_max = 1.12
        
    elif stem == 'DESI_ELG':
        Mean_z = 0.8
        z_min = 0.45
        z_max = 1.45
        
    # select tracer
    if "LRG" in stem.upper():
        tracer = "LRG"
    elif "ELG" in stem.upper():
        tracer = "ELG"
    else:
        print("Other tracers not yet implemented"); quit()

    # number of randoms
    want_rands = True
    if want_rands:
        rands_fac = 20
    
    # halo light cone parameters
    offset = 10. # Mpc/h

    # directory where mock catalogs are saved
    mock_dir = Path(f"/pscratch/sd/b/boryanah/AbacusHOD_scratch/mocks_box_output_kSZ_recon{extra}/")
    os.makedirs(mock_dir, exist_ok=True)
    save_dir = Path(f"/global/cfs/cdirs/desi/users/boryanah/kSZ_recon/new/mocks_lc_output_kSZ_recon{extra}/") 
    save_sub_dir = save_dir / sim_name
    os.makedirs(save_sub_dir, exist_ok=True)
    
    # read from simulation header
    header = get_meta(sim_name, 0.1)
    Lbox = header['BoxSizeHMpc'] # cMpc/h
    mpart = header['ParticleMassHMsun'] # 5.7e10, 2.1e9
    inv_velz2kms = 1./(header['VelZSpace_to_kms']/Lbox)
    origins = np.array(header['LightConeOrigins']).reshape(-1,3)
    origin = origins[0]
    print(f"mpart = {mpart:.2e}")

    # functions relating chi and z
    chi_of_z, z_of_chi = relate_chi_z(sim_name)
    
    # TESTING!
    """
    Ngal = 1068565 # true lc number
    print("0.4, 0.8", chi_of_z(0.4), chi_of_z(0.8))
    V = 1./8 * 4./3 * np.pi * (chi_of_z(0.8)**3 - chi_of_z(0.4)**3)
    print("total volume", V)
    print("average num dens", Ngal/V)
    print("average num L^3", Ngal/V*2000.**3)
    """
    
    # read in file names to determine all the available z's
    mask_dir = f"/global/cfs/cdirs/desi/public/cosmosim/AbacusLensing/v1/{sim_name}/"
    mask_fns = sorted(glob.glob(mask_dir+f"mask_0*.asdf"))
    z_srcs = []
    for i in range(len(mask_fns)):
        z_srcs.append(asdf.open(mask_fns[i])['header']['SourceRedshift'])
    z_srcs = np.sort(np.array(z_srcs))
    print("redshift sources = ", z_srcs)

    # directory for saving stuff
    mock_sub_dir = Path(mock_dir) / sim_name / (f"z{Mean_z:.3f}")

    # file with galaxy mock
    gal_rsd_fn = mock_sub_dir / f"galaxies_rsd/{tracer}s.dat"
    gal_real_fn = mock_sub_dir / f"galaxies/{tracer}s.dat"
    print(gal_rsd_fn)

    # read files
    gals_real_pos, gals_real_vel = read_dat(gal_real_fn)
    gals_rsd_pos, gals_rsd_vel = read_dat(gal_real_fn) # because we apply rsd
    gals_real_vel = np.load(mock_sub_dir / f"galaxies/{tracer}s_halo_vel.npy")
    
    # get the unit vectors and comoving distances to the observer (no RSD)
    gals_norm, gals_chis, _, _ = get_norm(gals_real_pos, origin)

    # apply RSD
    gals_rsd_pos = apply_rsd(gals_rsd_pos, gals_rsd_vel, inv_velz2kms, gals_norm)

    # get the unit vectors and comoving distances to the observer (RSD)
    _, CZ_rsd, gals_min, gals_max = get_norm(gals_rsd_pos, origin)
    del _; gc.collect()
    print("closest and furthest distance of gals = ", gals_min, gals_max)

    # fixing when there are no halos at low halo mass
    #gals_min = np.min(gals_chis[gals_chis > 0.])
    #print("GALS MIN = ", gals_min)

    # apply redshift cuts
    gals_min = chi_of_z(z_min)
    gals_max = chi_of_z(z_max)
    choice = (gals_chis < gals_max) & (gals_chis >= gals_min)
    gals_norm = gals_norm[choice]
    gals_chis = gals_chis[choice]
    gals_real_pos = gals_real_pos[choice]
    gals_rsd_pos = gals_rsd_pos[choice]
    gals_real_vel = gals_real_vel[choice]
    CZ_rsd = CZ_rsd[choice]

    # apply geometry cuts
    gals_real_pos_offset = gals_real_pos - origins[0]
    choice = is_in_lc(*gals_real_pos_offset.T, gals_max, Lbox, offset, origins)
    del gals_real_pos_offset; gc.collect()
    gals_norm = gals_norm[choice]
    gals_chis = gals_chis[choice]
    gals_real_pos = gals_real_pos[choice]
    gals_rsd_pos = gals_rsd_pos[choice]
    gals_real_vel = gals_real_vel[choice]
    CZ_rsd = CZ_rsd[choice]
    
    if want_rands:
        # generate randoms in L shape
        rands_pos, rands_norm, rands_chis = gen_rand(len(gals_chis), gals_min, gals_max, rands_fac, Lbox, offset, origins)
    del gals_min, gals_max; gc.collect()

    # convert the unit vectors into RA and DEC
    RA, DEC, CZ = get_ra_dec_chi(gals_norm, gals_chis)
    if want_rands:
        rands_RA, rands_DEC, rands_CZ = get_ra_dec_chi(rands_norm, rands_chis)
        del rands_chis; gc.collect()
    del gals_chis; gc.collect()

    if want_rands:
        # apply RSD  TESTING!!!!!!! maybe don't use in reconstruction cause we don't do this for the real data and it doesn't seem to matter (0.01)
        rands_rsd_vel = np.random.choice(gals_rsd_vel.shape[0], rands_pos.shape[0]) 
        rands_rsd_vel = gals_rsd_vel[rands_rsd_vel]
        rands_rsd_pos = apply_rsd(rands_pos, rands_rsd_vel, inv_velz2kms, rands_norm)
        del rands_norm; gc.collect()
        _, rands_CZ_rsd, _, _ = get_norm(rands_rsd_pos, origin)
        
    # convert chi to redshift
    Z = z_of_chi(CZ)
    Z_rsd = z_of_chi(CZ_rsd)
    if want_rands:
        rands_Z = z_of_chi(rands_CZ)
        rands_Z_rsd = z_of_chi(rands_CZ_rsd)

    # renaming
    Z_RSD = Z_rsd
    VEL = gals_real_vel
    POS = gals_real_pos
    POS_RSD = gals_rsd_pos
    UNIT_LOS = gals_norm
    if want_rands:
        RAND_Z = rands_Z
        RAND_Z_RSD = rands_Z_rsd
        RAND_RA = rands_RA
        RAND_DEC = rands_DEC
        RAND_POS = rands_pos

    # get masks
    gal_mask, rand_mask = get_dndz_mask(RA, DEC, Z, RAND_RA, RAND_DEC, RAND_Z, nz_dict, z_srcs, mask_fns, chi_of_z) # og
    #gal_mask, rand_mask = get_dndz_mask(RA, DEC, Z_RSD, RAND_RA, RAND_DEC, RAND_Z_RSD, nz_dict, z_srcs, mask_fns, chi_of_z) # makes things worse for real and 0.001 for rsd
    
    # apply the mask cuts
    RA = RA[gal_mask]
    DEC = DEC[gal_mask]
    Z = Z[gal_mask]
    Z_RSD = Z_RSD[gal_mask]
    POS = POS[gal_mask]
    POS_RSD = POS_RSD[gal_mask]
    VEL = VEL[gal_mask]
    UNIT_LOS = UNIT_LOS[gal_mask]

    # apply the mask cuts
    RAND_RA = RAND_RA[rand_mask]
    RAND_DEC = RAND_DEC[rand_mask]
    RAND_Z = RAND_Z[rand_mask]
    RAND_Z_RSD = RAND_Z_RSD[rand_mask]
    RAND_POS = RAND_POS[rand_mask]
    
    # TESTING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    z_edges = np.linspace(0.1, 2.5, 101)
    z_cent = 0.5*(z_edges[1:] + z_edges[:-1])
    dNdz, _ = np.histogram(Z, bins=z_edges)
    dNdz_rand, _ = np.histogram(RAND_Z, bins=z_edges)
    print(dNdz[(z_cent < 0.9) & (z_cent > 0.2)])
    print(dNdz_rand[(z_cent < 0.9) & (z_cent > 0.2)])
    print("should be constant and large (20-60)", (dNdz_rand/dNdz)[(z_cent < 0.9) & (z_cent > 0.2)])
    dNdz, _ = np.histogram(Z_RSD, bins=z_edges)
    dNdz_rand, _ = np.histogram(RAND_Z_RSD, bins=z_edges)
    print("should be constant and large (20-60)", (dNdz_rand/dNdz)[(z_cent < 0.9) & (z_cent > 0.2)])
    
    """
    # TESTING please get rid
    z_edges = np.linspace(0.1, 2.5, 1001)
    z_cent = 0.5*(z_edges[1:] + z_edges[:-1])
    dNdz_gal, _ = np.histogram(Z, bins=z_edges)
    n_per = len(RAND_Z)/len(z_cent)
    dNdz, _ = np.histogram(RAND_Z, bins=z_edges)
    inds = np.digitize(RAND_Z, bins=z_edges) - 1
    w_bin = dNdz/n_per # this kinda gives you the weighting you need to get perfectly uniform stuff?
    fac = np.interp(RAND_Z, z_cent, dNdz_gal/np.max(dNdz_gal))
    fac /= w_bin[inds] # downweight many galaxies in bin
    #fac /= np.max(fac[(Z_l > Mean_z-Delta_z/2.) & (Z_l < Mean_z+Delta_z/2.)]) # og
    #fac /= np.max(fac[(RAND_Z > 0.4) & (RAND_Z < 0.6)]) # og
    fac /= np.max(fac[(RAND_Z > nz_dict['Min_z']) & (RAND_Z < nz_dict['Max_z'])])
    rand_mask = np.random.rand(len(RAND_Z)) < fac
    """
    
    # save to file
    np.savez(save_sub_dir / f"galaxies_{tracer}_fakelc_zerr0.0_prerecon{nz_str}.npz", RA=RA, DEC=DEC, Z_RSD=Z_RSD, Z=Z, VEL=VEL, POS=POS, POS_RSD=POS_RSD, UNIT_LOS=UNIT_LOS)
    #np.savez(save_sub_dir / f"randoms_{tracer}_fakelc_zerr0.0_prerecon{nz_str}.npz", RAND_RA=RAND_RA, RAND_DEC=RAND_DEC, RAND_Z=RAND_Z, RAND_POS=RAND_POS) # og
    np.savez(save_sub_dir / f"randoms_{tracer}_fakelc_zerr0.0_prerecon{nz_str}.npz", RAND_RA=RAND_RA, RAND_DEC=RAND_DEC, RAND_Z=RAND_Z, RAND_Z_RSD=RAND_Z_RSD, RAND_POS=RAND_POS)

class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--sim_name', help='Simulation name', default=DEFAULTS['sim_name'])
    parser.add_argument('--stem', help='Stem file name', default=DEFAULTS['stem'])
    parser.add_argument('--nz_type', help='Type of N(z) distribution: Gaussian(mean, 2sigma), Step(min, max), file name', default=DEFAULTS['nz_type'])
    args = vars(parser.parse_args())
    main(**args)
