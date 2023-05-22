import os
import gc
import glob
from pathlib import Path
import argparse

import numpy as np
import healpy as hp
import asdf

from abacusnbody.metadata import get_meta

DEFAULTS = {}
DEFAULTS['sim_name'] = "AbacusSummit_base_c000_ph002"
DEFAULTS['stem'] = 'DESI_LRG' # 'DESI_ELG' 

"""
Usage:
python join_lc_catalog.py --stem DESI_LRG --sim_name AbacusSummit_base_c000_ph002
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
    #fac = 1. # heaviside
    fac /= w_bin[inds] # downweight many galaxies in bin
    #fac /= np.max(fac) # normaliza so that probabilities don't exceed 1
    print(np.mean(Z_l), np.min(Z_l), np.max(Z_l))
    fac /= np.max(fac[(Z_l > z_l-Delta_z/2.) & (Z_l < z_l+Delta_z/2.)])
    down_choice = np.random.rand(n_tot) < fac
    #down_choice[:] = True
    print("kept fraction (20-30%) = ", np.sum(down_choice)/len(down_choice)*100.)
    return down_choice

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def get_mask_ang(mask, RA, DEC, nest, nside, lonlat=True):
    """
    function that returns a boolean mask given RA and DEC
    """
    ipix = hp.ang2pix(nside, theta=RA, phi=DEC, nest=nest, lonlat=lonlat) # RA, DEC degrees (math convention)
    return mask[ipix] == 1.

def main(sim_name, stem):

    # additional specs of the tracer
    extra = '_'.join(stem.split('_')[2:])
    stem = '_'.join(stem.split('_')[:2])
    if extra != '': extra = '_'+extra
    
    # parameter choices
    if stem == 'DESI_LRG': 
        redshifts = [0.300, 0.350, 0.400, 0.450, 0.500, 0.575, 0.650, 0.725, 0.800, 0.875, 0.950, 1.025, 1.100]
        Mean_z = 0.5
        Delta_z = 0.4
    elif stem == 'DESI_ELG':
        redshifts = [0.800]
        Mean_z = 0.8
        Delta_z = 0.1
        
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
        
    # immutables
    sim_dir = "/global/cfs/cdirs/desi/cosmosim/Abacus/halo_light_cones/"
    mask_dir = f"/global/cfs/cdirs/desi/public/cosmosim/AbacusLensing/v1/{sim_name}/"
    mock_dir = Path("/global/project/projectdirs/desi/users/boryanah/kSZ_recon/")
    offset = 10. # Mpc/h    

    # directory where mock catalogs are saved
    save_dir = Path("/global/cfs/cdirs/desi/users/boryanah/kSZ_recon/") # old
    #save_dir = Path("/global/cfs/cdirs/desi/users/boryanah/kSZ_recon/mocks_lc_output_kSZ_recon{extra}/")
    mock_dir = save_dir / sim_name / "tmp" # old # remove tmp
    os.makedirs(mock_dir, exist_ok=True)
    
    # read from simulation header
    header = get_meta(sim_name, 0.1)
    Lbox = header['BoxSizeHMpc'] # cMpc/h
    mpart = header['ParticleMassHMsun'] # 5.7e10, 2.1e9
    origins = np.array(header['LightConeOrigins']).reshape(-1,3)
    origin = origins[0]
    print(f"mpart = {mpart:.2e}")
    
    # read in file names to determine all the available z's
    mask_fns = sorted(glob.glob(mask_dir+f"mask_0*.asdf"))
    z_srcs = []
    for i in range(len(mask_fns)):
        z_srcs.append(asdf.open(mask_fns[i])['header']['SourceRedshift'])
    z_srcs = np.sort(np.array(z_srcs))
    print("redshift sources = ", z_srcs)
    
    # combine together the redshifts
    for i, redshift in enumerate(redshifts):
        save_sub_dir = Path(save_dir) / sim_name / (f"z{redshift:.3f}") 
        data = np.load(save_sub_dir / f"galaxies_{tracer}_pos.npz")
        if i != 0:
            RA = np.hstack((RA, data['RA']))
            DEC = np.hstack((DEC, data['DEC']))
            Z = np.hstack((Z, data['Z']))
            Z_RSD = np.hstack((Z_RSD, data['Z_RSD']))
            VEL = np.vstack((VEL, data['VEL']))
            POS = np.vstack((POS, data['POS']))
            POS_RSD = np.vstack((POS_RSD, data['POS_RSD']))
            UNIT_LOS = np.vstack((UNIT_LOS, data['UNIT_LOS']))
        else:
            RA = data['RA']
            DEC = data['DEC']
            Z = data['Z']
            Z_RSD = data['Z_RSD']
            VEL = data['VEL']
            POS = data['POS']
            POS_RSD = data['POS_RSD']
            UNIT_LOS = data['UNIT_LOS']
        data = np.load(save_sub_dir / f"randoms_{tracer}_pos.npz")
        if i != 0:
            RAND_RA = np.hstack((RAND_RA, data['RAND_RA']))
            RAND_DEC = np.hstack((RAND_DEC, data['RAND_DEC']))
            RAND_Z = np.hstack((RAND_Z, data['RAND_Z']))
            RAND_POS = np.vstack((RAND_POS, data['RAND_POS']))
        else:
            RAND_RA = data['RAND_RA']
            RAND_DEC = data['RAND_DEC']
            RAND_Z = data['RAND_Z']
            RAND_POS = data['RAND_POS']
        
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
    print("apply mask")
    choice = get_mask_ang(mask, RA, DEC, nest, nside)
    print("masked fraction", np.sum(choice)/len(choice)*100.)
    RA, DEC, Z_RSD, Z, VEL, POS, POS_RSD, UNIT_LOS = RA[choice], DEC[choice], Z_RSD[choice], Z[choice], VEL[choice], POS[choice], POS_RSD[choice], UNIT_LOS[choice]
    choice = get_mask_ang(mask, RAND_RA, RAND_DEC, nest, nside)
    print("masked fraction", np.sum(choice)/len(choice)*100.)
    RAND_RA, RAND_DEC, RAND_Z, RAND_POS = RAND_RA[choice], RAND_DEC[choice], RAND_Z[choice], RAND_POS[choice]
    del mask; gc.collect()
                
    # apply redshift selection
    print("apply redshift downsampling")
    choice = get_down_choice(Z, Mean_z, Delta_z)
    RA, DEC, Z_RSD, Z, VEL, POS, POS_RSD, UNIT_LOS = RA[choice], DEC[choice], Z_RSD[choice], Z[choice], VEL[choice], POS[choice], POS_RSD[choice], UNIT_LOS[choice]
    print("masked fraction", np.sum(choice)/len(choice)*100.)
    choice = get_down_choice(RAND_Z, Mean_z, Delta_z)
    RAND_RA, RAND_DEC, RAND_Z, RAND_POS = RAND_RA[choice], RAND_DEC[choice], RAND_Z[choice], RAND_POS[choice]
    print("masked fraction", np.sum(choice)/len(choice)*100.)
    
    # save to file
    np.savez(mock_dir / f"galaxies_{tracer}_prerecon_meanz{Mean_z:.3f}_deltaz{Delta_z:.3f}.npz", RA=RA, DEC=DEC, Z_RSD=Z_RSD, Z=Z, VEL=VEL, POS=POS, POS_RSD=POS_RSD, UNIT_LOS=UNIT_LOS)
    np.savez(mock_dir / f"randoms_{tracer}_prerecon_meanz{Mean_z:.3f}_deltaz{Delta_z:.3f}.npz", RAND_RA=RAND_RA, RAND_DEC=RAND_DEC, RAND_Z=RAND_Z, RAND_POS=RAND_POS)
    
class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--sim_name', help='Simulation name', default=DEFAULTS['sim_name'])
    parser.add_argument('--stem', help='Stem file name', default=DEFAULTS['stem'])

    args = vars(parser.parse_args())
    main(**args)
