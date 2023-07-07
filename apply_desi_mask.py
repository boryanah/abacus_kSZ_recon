from pathlib import Path
import os, gc

import numpy as np
import asdf
import argparse
import fitsio

from apply_dndz_mask import parse_nztype

DEFAULTS = {}
DEFAULTS['sim_name'] = "AbacusSummit_base_c000_ph002"
DEFAULTS['stem'] = 'DESI_LRG' # 'DESI_ELG'
DEFAULTS['nz_type'] = 'Gaussian(0.5, 0.4)'
DEFAULTS['photoz_error'] = 0.

"""
Usage:
python apply_desi_mask.py --sim_name AbacusSummit_base_c000_ph002 --stem DESI_LRG
python apply_desi_mask.py --sim_name AbacusSummit_huge_c000_ph201 --stem DESI_LRG
"""

def main(sim_name, stem, nz_type, photoz_error, want_fakelc=False):

    # parse the dndz instructions
    nz_dict = parse_nztype(nz_type)
    if nz_dict['Type'] == "Gaussian":
        nz_str = f"_meanz{nz_dict['Mean_z']:.3f}_deltaz{nz_dict['Delta_z']:.3f}"
    elif nz_dict['Type'] == "StepFunction":
        nz_str = f"_minz{nz_dict['Min_z']:.3f}_maxz{nz_dict['Max_z']:.3f}"
    elif nz_dict['Type'] == "FromFile":
        nz_str = f"_{nz_dict['File']}"

    # redshift error string
    photoz_str = f"_zerr{photoz_error:.1f}"
    
    # fake light cones
    fakelc_str = "_fakelc" if want_fakelc else ""
    mask_str = "_mask"
    
    # how many processes to use for reconstruction: 32, 128 physical cpu per node for cori, perlmutter (hyperthreading doubles)
    ncpu = 128

    # additional specs of the tracer
    extra = '_'.join(stem.split('_')[2:])
    stem = '_'.join(stem.split('_')[:2])
    if extra != '': extra = '_'+extra

    # parameter choices
    if stem == 'DESI_LRG': 
        Mean_z = 0.5
        tracer = "LRG"
        bias = 2.2 # +/- 10%
    elif stem == 'DESI_ELG':
        Mean_z = 0.8
        tracer = "ELG"
        bias = 1.3
    else:
        print("Other tracers not yet implemented"); quit()
    
    # directory where mock catalogs are saved
    mock_dir = Path(f"/global/cfs/cdirs/desi/users/boryanah/kSZ_recon/mocks_lc_output_kSZ_recon{extra}/")
    mock_dir = mock_dir / sim_name 

    # name of the file
    maskbit_fn = mock_dir / f"galaxies_{tracer}{fakelc_str}{photoz_str}_prerecon{nz_str}_{tracer.lower()}mask.fits.gz"
    maskgrz_fn = mock_dir / f"galaxies_{tracer}{fakelc_str}{photoz_str}_prerecon{nz_str}_nexp.fits"
    rand_maskbit_fn = mock_dir / f"randoms_{tracer}{fakelc_str}{photoz_str}_prerecon{nz_str}_{tracer.lower()}mask.fits.gz"
    rand_maskgrz_fn = mock_dir / f"randoms_{tracer}{fakelc_str}{photoz_str}_prerecon{nz_str}_nexp.fits"
    gal_fn = mock_dir / f"galaxies_{tracer}{fakelc_str}{photoz_str}_prerecon{nz_str}.npz"
    rand_fn = mock_dir / f"randoms_{tracer}{fakelc_str}{photoz_str}_prerecon{nz_str}.npz"
    
    if (not os.path.exists(maskbit_fn)) or (not os.path.exists(maskgrz_fn)) or (not os.path.exists(rand_maskbit_fn)) or (not os.path.exists(rand_maskgrz_fn)):
        shift_str = "--shift" if "base" in sim_name else ""
        os.system("source /global/common/software/desi/desi_environment.sh 23.1")
        if (not os.path.exists(maskbit_fn)):
            os.system(f"srun -N 1 -C cpu -c 256 -t 01:00:00 -q interactive python /global/u1/b/boryanah/repos/desi-examples/bright_star_mask/read_pixel_bitmask.py --tracer {tracer.lower()} --input {gal_fn} --output {maskbit_fn} {shift_str}")
        if (not os.path.exists(maskgrz_fn)):
            os.system(f"srun -N 1 -C cpu -c 256 -t 01:00:00 -q interactive python /global/u1/b/boryanah/repos/desi-examples/bright_star_mask/read_pixel_nexp.py --input {gal_fn} --output {maskgrz_fn} {shift_str}")
        if (not os.path.exists(rand_maskbit_fn)):
            os.system(f"srun -N 1 -C cpu -c 256 -t 02:00:00 -q interactive python /global/u1/b/boryanah/repos/desi-examples/bright_star_mask/read_pixel_bitmask.py --tracer {tracer.lower()} --input {rand_fn} --output {rand_maskbit_fn} {shift_str}")
        if (not os.path.exists(rand_maskgrz_fn)):
            os.system(f"srun -N 1 -C cpu -c 256 -t 02:00:00 -q interactive python /global/u1/b/boryanah/repos/desi-examples/bright_star_mask/read_pixel_nexp.py --input {rand_fn} --output {rand_maskgrz_fn} {shift_str}")
        
    # load the galaxies
    data = np.load(gal_fn)
    RA = data['RA']
    DEC = data['DEC']
    Z_RSD = data['Z_RSD']
    Z = data['Z']
    VEL = data['VEL']
    UNIT_LOS = data['UNIT_LOS']
    POS = data['POS']
    POS_RSD = data['POS_RSD']
    
    # load grz and bit mask
    maskbit = fitsio.read(maskbit_fn)
    maskgrz = fitsio.read(maskgrz_fn)
    mask = (maskgrz["PIXEL_NOBS_G"] > 0) & (maskgrz["PIXEL_NOBS_R"] > 0) & (maskgrz["PIXEL_NOBS_Z"] > 0)
    mask &= maskbit[f"{tracer.lower()}_mask"] == 0
    print(len(mask), len(Z))
    assert len(mask) == len(Z)
    print("mask percentage", np.sum(mask)/len(mask)*100.)
    del maskbit, maskgrz; gc.collect()

    # apply the mask cuts
    RA = RA[mask]
    DEC = DEC[mask]
    Z = Z[mask]
    Z_RSD = Z_RSD[mask]
    POS = POS[mask]
    POS_RSD = POS_RSD[mask]
    VEL = VEL[mask]
    UNIT_LOS = UNIT_LOS[mask]

    # save to file
    np.savez(mock_dir / f"galaxies_{tracer}{fakelc_str}{photoz_str}{mask_str}_prerecon{nz_str}.npz", RA=RA, DEC=DEC, Z_RSD=Z_RSD, Z=Z, VEL=VEL, POS=POS, POS_RSD=POS_RSD, UNIT_LOS=UNIT_LOS)

    # load the randoms
    data = np.load(rand_fn)
    RAND_RA = data['RAND_RA']
    RAND_DEC = data['RAND_DEC']
    RAND_Z = data['RAND_Z']
    RAND_POS = data['RAND_POS']

    # load grz and bit mask
    maskbit = fitsio.read(rand_maskbit_fn)
    maskgrz = fitsio.read(rand_maskgrz_fn)
    mask = (maskgrz["PIXEL_NOBS_G"] > 0) & (maskgrz["PIXEL_NOBS_R"] > 0) & (maskgrz["PIXEL_NOBS_Z"] > 0)
    mask &= maskbit[f"{tracer.lower()}_mask"] == 0
    print(len(mask), len(RAND_Z))
    assert len(mask) == len(RAND_Z)
    del maskbit, maskgrz; gc.collect()
    print("mask percentage", np.sum(mask)/len(mask)*100.)
    
    # apply the mask cuts
    RAND_RA = RAND_RA[mask]
    RAND_DEC = RAND_DEC[mask]
    RAND_Z = RAND_Z[mask]
    RAND_POS = RAND_POS[mask]

    # save to file
    np.savez(mock_dir / f"randoms_{tracer}{fakelc_str}{photoz_str}{mask_str}_prerecon{nz_str}.npz", RAND_RA=RAND_RA, RAND_DEC=RAND_DEC, RAND_Z=RAND_Z, RAND_POS=RAND_POS)
    
class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--sim_name', help='Simulation name', default=DEFAULTS['sim_name'])
    parser.add_argument('--stem', help='Stem file name', default=DEFAULTS['stem'], choices=["DESI_LRG", "DESI_ELG", "DESI_LRG_high_density", "DESI_LRG_bgs", "DESI_ELG_high_density"])
    parser.add_argument('--photoz_error', help='Percentage error on the photometric redshifts', default=DEFAULTS['photoz_error'], type=float)
    parser.add_argument('--nz_type', help='Type of N(z) distribution: Gaussian(mean, 2sigma), Step(min, max), file name', default=DEFAULTS['nz_type'])
    parser.add_argument('--want_fakelc', help='Want to use the fake light cone?', action='store_true')
    args = vars(parser.parse_args())
    main(**args)
