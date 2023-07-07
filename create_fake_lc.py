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

DEFAULTS = {}
DEFAULTS['sim_name'] = "AbacusSummit_base_c000_ph002"
DEFAULTS['stem'] = 'DESI_LRG' # 'DESI_ELG' 

"""
Usage:
python create_fake_lc.py --stem DESI_LRG --sim_name AbacusSummit_base_c000_ph002
python create_fake_lc.py --stem DESI_LRG --sim_name AbacusSummit_huge_c000_ph201
"""

def get_down_choice(Z_l, z_l, Delta_z):
    # apply cuts
    z_edges = np.linspace(0.1, 2.5, 1001)
    z_cent = 0.5*(z_edges[1:] + z_edges[:-1])
    n_tot = len(Z_l)
    n_per = n_tot/len(z_cent)
    dNdz, _ = np.histogram(Z_l, bins=z_edges)
    inds = np.digitize(Z_l, bins=z_edges) - 1
    w_bin = dNdz/n_per
    fac = gaussian(Z_l, z_l, Delta_z/2.)
    fac /= w_bin[inds] # downweight many galaxies in bin
    #fac /= np.max(fac) # normaliza so that probabilities don't exceed 1
    print("mean/min/max z", np.mean(Z_l), np.min(Z_l), np.max(Z_l))
    fac /= np.max(fac[(Z_l > z_l-Delta_z/2.) & (Z_l < z_l+Delta_z/2.)])
    #fac = 1. # heaviside
    down_choice = np.random.rand(n_tot) < fac
    return down_choice

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def get_mask_ang(mask, RA, DEC, nest, nside, lonlat=True):
    """
    function that returns a boolean mask given RA and DEC
    """
    ipix = hp.ang2pix(nside, theta=RA, phi=DEC, nest=nest, lonlat=lonlat) # RA, DEC degrees (math convention)
    return mask[ipix] == 1.

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

def apply_rsd(gals_pos, gals_vel, inv_velz2kms, norm):
    nx = norm[:, 0]
    ny = norm[:, 1]
    nz = norm[:, 2]
    proj = inv_velz2kms * (gals_vel[:, 0]*nx + gals_vel[:, 1]*ny + gals_vel[:, 2]*nz)
    gals_pos[:, 0] += proj*nx
    gals_pos[:, 1] += proj*ny
    gals_pos[:, 2] += proj*nz
    return gals_pos

def main(sim_name, stem):

    # additional specs of the tracer
    extra = '_'.join(stem.split('_')[2:])
    stem = '_'.join(stem.split('_')[:2])
    if extra != '': extra = '_'+extra
    
    # parameter choices
    if stem == 'DESI_LRG': 
        Mean_z = 0.5
        Delta_z = 0.4
        z_min = 0.275
        if "base" in sim_name:
            z_max = 0.8 #1.12
        else:
            z_max = 1.12
        
    elif stem == 'DESI_ELG':
        Mean_z = 0.8
        Delta_z = 0.3
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
    save_dir = Path(f"/global/cfs/cdirs/desi/users/boryanah/kSZ_recon/mocks_lc_output_kSZ_recon{extra}/") 
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
    Ngal = 1068565 # true lc number
    print("0.4, 0.8", chi_of_z(0.4), chi_of_z(0.8))
    V = 1./8 * 4./3 * np.pi * (chi_of_z(0.8)**3 - chi_of_z(0.4)**3)
    print("total volume", V)
    print("average num dens", Ngal/V)
    print("average num L^3", Ngal/V*2000.**3)
    quit()
    
    
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
    # this is gonna be super wrong cause boundary conditions and origin
    #gals_rsd_pos, gals_rsd_vel = read_dat(gal_rsd_fn)
    gals_rsd_pos, gals_rsd_vel = read_dat(gal_real_fn)

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
        del rands_norm, rands_chis; gc.collect()
    del gals_chis; gc.collect()
        
    # convert chi to redshift
    Z = z_of_chi(CZ)
    Z_rsd = z_of_chi(CZ_rsd)
    if want_rands:
        rands_Z = z_of_chi(rands_CZ)

    # renaming
    Z_RSD = Z_rsd
    VEL = gals_real_vel
    POS = gals_real_pos
    POS_RSD = gals_rsd_pos
    UNIT_LOS = gals_norm
    if want_rands:
        RAND_Z = rands_Z
        RAND_RA = rands_RA
        RAND_DEC = rands_DEC
        RAND_POS = rands_pos
    
    # find highest redshift
    #z_max = np.max(redshifts)
    z_max = Mean_z+Delta_z
    print("number of galaxies", len(Z))

    # apply masking
    mask_fn = mask_fns[np.argmax(z_srcs - z_max >= 0.)]
    mask = asdf.open(mask_fn)['data']['mask']
    nside = asdf.open(mask_fn)['header']['HEALPix_nside']
    order = asdf.open(mask_fn)['header']['HEALPix_order']
    nest = True if order == 'NESTED' else False

    # mask lenses and randoms
    print("applying mask")
    choice = get_mask_ang(mask, RA, DEC, nest, nside)
    print("masked fraction gals", np.sum(choice)/len(choice)*100.)
    RA, DEC, Z_RSD, Z, VEL, POS, POS_RSD, UNIT_LOS = RA[choice], DEC[choice], Z_RSD[choice], Z[choice], VEL[choice], POS[choice], POS_RSD[choice], UNIT_LOS[choice]
    choice = get_mask_ang(mask, RAND_RA, RAND_DEC, nest, nside)
    print("masked fraction rands", np.sum(choice)/len(choice)*100.)
    RAND_RA, RAND_DEC, RAND_Z, RAND_POS = RAND_RA[choice], RAND_DEC[choice], RAND_Z[choice], RAND_POS[choice]
    del mask; gc.collect()

    # apply redshift selection
    print("apply redshift downsampling")
    choice = get_down_choice(Z, Mean_z, Delta_z)
    RA, DEC, Z_RSD, Z, VEL, POS, POS_RSD, UNIT_LOS = RA[choice], DEC[choice], Z_RSD[choice], Z[choice], VEL[choice], POS[choice], POS_RSD[choice], UNIT_LOS[choice]
    print("kept fraction (20-30%) gals", np.sum(choice)/len(choice)*100.)
    choice = get_down_choice(RAND_Z, Mean_z, Delta_z)
    RAND_RA, RAND_DEC, RAND_Z, RAND_POS = RAND_RA[choice], RAND_DEC[choice], RAND_Z[choice], RAND_POS[choice]
    print("kept fraction (20-30%) rands", np.sum(choice)/len(choice)*100.)

    # save to file
    np.savez(save_sub_dir / f"galaxies_{tracer}_fakelc_prerecon_meanz{Mean_z:.3f}_deltaz{Delta_z:.3f}.npz", RA=RA, DEC=DEC, Z_RSD=Z_RSD, Z=Z, VEL=VEL, POS=POS, POS_RSD=POS_RSD, UNIT_LOS=UNIT_LOS)
    np.savez(save_sub_dir / f"randoms_{tracer}_fakelc_prerecon_meanz{Mean_z:.3f}_deltaz{Delta_z:.3f}.npz", RAND_RA=RAND_RA, RAND_DEC=RAND_DEC, RAND_Z=RAND_Z, RAND_POS=RAND_POS)

class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--sim_name', help='Simulation name', default=DEFAULTS['sim_name'])
    parser.add_argument('--stem', help='Stem file name', default=DEFAULTS['stem'])

    args = vars(parser.parse_args())
    main(**args)
