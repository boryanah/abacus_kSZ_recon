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
python join_lc_catalog.py --stem DESI_LRG --sim_name AbacusSummit_huge_c000_ph201
"""

def main(sim_name, stem):

    # additional specs of the tracer
    extra = '_'.join(stem.split('_')[2:])
    stem = '_'.join(stem.split('_')[:2])
    if extra != '': extra = '_'+extra
    
    # parameter choices
    if stem == 'DESI_LRG':
        redshifts = np.array([0.300, 0.350, 0.400, 0.450, 0.500, 0.575, 0.650, 0.725, 0.800, 0.875, 0.950, 1.025, 1.100])
    elif stem == 'DESI_ELG':
        #redshifts = np.array([0.500, 0.575, 0.650, 0.725, 0.800, 0.875, 0.950, 1.025, 1.100, 1.175, 1.250, 1.325, 1.400])
        redshifts = np.array([0.300, 0.350, 0.400, 0.450, 0.500, 0.575, 0.650, 0.725, 0.800, 0.875, 0.950, 1.025, 1.100])
    min_z = np.min(redshifts)
    max_z = np.max(redshifts)
        
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
        
    # directory with healpix masks and halo light cone parameters
    mask_dir = f"/global/cfs/cdirs/desi/public/cosmosim/AbacusLensing/v1/{sim_name}/"
    offset = 10. # Mpc/h    

    # directory where mock catalogs are saved
    mock_dir = Path(f"/pscratch/sd/b/boryanah/AbacusHOD_scratch/mocks_lc_output_kSZ_recon{extra}/")
    save_dir = Path(f"/global/cfs/cdirs/desi/users/boryanah/kSZ_recon/mocks_lc_output_kSZ_recon{extra}/") 
    save_sub_dir = save_dir / sim_name
    os.makedirs(save_sub_dir, exist_ok=True)
    
    # read from simulation header
    header = get_meta(sim_name, 0.1)
    Lbox = header['BoxSizeHMpc'] # cMpc/h
    mpart = header['ParticleMassHMsun'] # 5.7e10, 2.1e9
    origins = np.array(header['LightConeOrigins']).reshape(-1,3)
    origin = origins[0]
    print(f"mpart = {mpart:.2e}")
    
    
    # combine together the redshifts
    for i, redshift in enumerate(redshifts):
        mock_sub_dir = Path(mock_dir) / sim_name / (f"z{redshift:.3f}") 
        data = np.load(mock_sub_dir / f"galaxies_{tracer}_pos.npz")
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

        if want_rands:
            data = np.load(mock_sub_dir / f"randoms_{tracer}_pos.npz")
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
        del data; gc.collect()
        
    # save to file
    np.savez(save_sub_dir / f"galaxies_{tracer}_prerecon_minz{min_z:.3f}_maxz{max_z:.3f}.npz", RA=RA, DEC=DEC, Z_RSD=Z_RSD, Z=Z, VEL=VEL, POS=POS, POS_RSD=POS_RSD, UNIT_LOS=UNIT_LOS)
    if want_rands:
        np.savez(save_sub_dir / f"randoms_{tracer}_prerecon_minz{min_z:.3f}_maxz{max_z:.3f}.npz", RAND_RA=RAND_RA, RAND_DEC=RAND_DEC, RAND_Z=RAND_Z, RAND_POS=RAND_POS)
    
class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--sim_name', help='Simulation name', default=DEFAULTS['sim_name'])
    parser.add_argument('--stem', help='Stem file name', default=DEFAULTS['stem'])

    args = vars(parser.parse_args())
    main(**args)
