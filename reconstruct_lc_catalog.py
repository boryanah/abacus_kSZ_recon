from pathlib import Path
import os

import numpy as np
import asdf
import argparse

from pyrecon import  utils, IterativeFFTParticleReconstruction, MultiGridReconstruction, IterativeFFTReconstruction
from cosmoprimo.fiducial import Planck2018FullFlatLCDM, AbacusSummit, DESI, TabulatedDESI

DEFAULTS = {}
DEFAULTS['sim_name'] = "AbacusSummit_base_c000_ph002"
DEFAULTS['stem'] = 'DESI_LRG' # 'DESI_ELG'
DEFAULTS['nmesh'] = 512 # 1024
DEFAULTS['sr'] = 10. # Mpc/h
DEFAULTS['rectype'] = "MG"
DEFAULTS['convention'] = "recsym"

"""
Usage:
python reconstruct_lc_catalog.py --sim_name AbacusSummit_base_c000_ph002 --stem DESI_LRG --nmesh 1024 --sr 12.5 --rectype MG --convention recsym
"""

def main(sim_name, stem, nmesh, sr, rectype, convention):
    # how many processes to use for reconstruction: 32, 128 physical cpu per node for cori, perlmutter (hyperthreading doubles)
    ncpu = 128

    # additional specs of the tracer
    extra = '_'.join(stem.split('_')[2:])
    stem = '_'.join(stem.split('_')[:2])
    if extra != '': extra = '_'+extra
    
    # reconstruction parameters
    if rectype == "IFT":
        recfunc = IterativeFFTReconstruction
    elif rectype == "IFTP":
        recfunc = IterativeFFTParticleReconstruction
    elif rectype == "MG":
        recfunc = MultiGridReconstruction

    # directory where mock catalogs are saved
    #mock_dir = Path("/global/cfs/cdirs/desi/users/boryanah/kSZ_recon/") # old
    mock_dir = Path(f"/global/cfs/cdirs/desi/users/boryanah/kSZ_recon/mocks_lc_output_kSZ_recon{extra}/")
    #mock_dir = mock_dir / sim_name / "tmp" # old # remove tmp
    mock_dir = mock_dir / sim_name 
    
    # parameter choices
    if stem == 'DESI_LRG': 
        Mean_z = 0.5
        Delta_z = 0.4
        tracer = "LRG"
        bias = 2.2 # +/- 10%
    elif stem == 'DESI_ELG':
        Mean_z = 0.8
        Delta_z = 0.1
        tracer = "ELG"
        bias = 1.3
    else:
        print("Other tracers not yet implemented"); quit()

    # simulation parameters
    cosmo = DESI()
    ff = cosmo.growth_factor(Mean_z) # Abacus
    H_z = cosmo.hubble_function(Mean_z) # Abacus
    los = 'local'
        
    # load the galaxies
    data = np.load(mock_dir / f"galaxies_{tracer}_prerecon_meanz{Mean_z:.3f}_deltaz{Delta_z:.3f}.npz")
    RA = data['RA']
    DEC = data['DEC']
    Z_RSD = data['Z_RSD']
    Z = data['Z']
    VEL = data['VEL']
    UNIT_LOS = data['UNIT_LOS']

    # load the randoms
    data = np.load(mock_dir / f"randoms_{tracer}_prerecon_meanz{Mean_z:.3f}_deltaz{Delta_z:.3f}.npz")
    RAND_RA = data['RAND_RA']
    RAND_DEC = data['RAND_DEC']
    RAND_Z = data['RAND_Z']

    # directory where the reconstructed mock catalogs are saved
    save_dir = Path("/global/cfs/cdirs/desi/users/boryanah/kSZ_recon/")
    save_recon_dir = Path(save_dir) / "recon" / sim_name / f"z{Mean_z:.3f}"
    os.makedirs(save_recon_dir, exist_ok=True)

    # transform into Cartesian coordinates
    print("max RA and DEC", RA.max(), DEC.max())
    PositionRSD = utils.sky_to_cartesian(cosmo.comoving_radial_distance(Z_RSD), RA, DEC)
    Position = utils.sky_to_cartesian(cosmo.comoving_radial_distance(Z), RA, DEC)
    RandomPosition = utils.sky_to_cartesian(cosmo.comoving_radial_distance(RAND_Z), RAND_RA, RAND_DEC)
    
    # run reconstruction on the mocks w/o RSD
    print('Recon First tracer')
    recon_tracer = recfunc(f=ff, bias=bias, nmesh=nmesh, los=los, positions=Position,
                           nthreads=int(ncpu), fft_engine='fftw', fft_plan='estimate', dtype='f4', wrap=True)
    print('grid set up',flush=True)
    recon_tracer.assign_data(Position)#, dat_cat['WEIGHT'])
    print('data assigned',flush=True)
    recon_tracer.assign_randoms(RandomPosition)#, dat_cat['WEIGHT'])
    print('randoms assigned',flush=True)
    recon_tracer.set_density_contrast(smoothing_radius=sr)
    print('density constrast calculated, now doing recon',flush=True)
    recon_tracer.run()
    print('recon has been run',flush=True)

    # read the displacements in real space
    if rectype == 'IFTP':
        displacements = recon_tracer.read_shifts('data', field='disp')
    else:
        displacements = recon_tracer.read_shifts(Position, field='disp')
    random_displacements = recon_tracer.read_shifts(RandomPosition, field='disp')
    
    # run reconstruction on the mocks w/ RSD
    print('Recon Second tracer')
    recon_tracer = recfunc(f=ff, bias=bias, nmesh=nmesh, los=los, positions=PositionRSD,
                           nthreads=int(ncpu), fft_engine='fftw', fft_plan='estimate', dtype='f4', wrap=True)
    print('grid set up',flush=True)
    recon_tracer.assign_data(PositionRSD)#, dat_cat['WEIGHT'])
    print('data assigned',flush=True)
    recon_tracer.assign_randoms(RandomPosition)#, dat_cat['WEIGHT'])
    print('randoms assigned',flush=True)
    recon_tracer.set_density_contrast(smoothing_radius=sr)
    print('density constrast calculated, now doing recon',flush=True)
    recon_tracer.run()
    print('recon has been run',flush=True)

    # read the displacements in real and redshift space (rsd has the (1+f) factor in the LOS direction)
    if rectype == "IFTP":
        displacements_rsd = recon_tracer.read_shifts('data', field='disp+rsd')
        displacements_rsd_nof = recon_tracer.read_shifts('data', field='disp')
    else:
        displacements_rsd = recon_tracer.read_shifts(PositionRSD, field='disp+rsd')
        displacements_rsd_nof = recon_tracer.read_shifts(PositionRSD, field='disp')
    random_displacements_rsd = recon_tracer.read_shifts(RandomPosition, field='disp+rsd')
    random_displacements_rsd_nof = recon_tracer.read_shifts(RandomPosition, field='disp')

    # save the displacements
    np.savez(save_recon_dir / f"displacements_{tracer}{extra}_postrecon_R{sr:.2f}_b{bias:.1f}_nmesh{nmesh:d}_{convention}_{rectype}_meanz{Mean_z:.3f}_deltaz{Delta_z:.3f}.npz",
             displacements=displacements, displacements_rsd=displacements_rsd, velocities=VEL, unit_los=UNIT_LOS, growth_factor=ff, Hubble_z=H_z,
             random_displacements_rsd=random_displacements_rsd, random_displacements=random_displacements, displacements_rsd_nof=displacements_rsd_nof,
             random_displacements_rsd_nof=random_displacements_rsd_nof)

class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--sim_name', help='Simulation name', default=DEFAULTS['sim_name'])
    parser.add_argument('--stem', help='Stem file name', default=DEFAULTS['stem'], choices=["DESI_LRG", "DESI_ELG", "DESI_LRG_high_density", "DESI_ELG_high_density"])
    parser.add_argument('--nmesh', help='Number of cells per dimension for reconstruction', type=int, default=DEFAULTS['nmesh'])
    parser.add_argument('--sr', help='Smoothing radius', type=float, default=DEFAULTS['sr'])
    parser.add_argument('--rectype', help='Reconstruction type', default=DEFAULTS['rectype'], choices=["IFT", "MG", "IFTP"])
    parser.add_argument('--convention', help='Reconstruction convention', default=DEFAULTS['convention'], choices=["recsym", "reciso"])
    args = vars(parser.parse_args())
    main(**args)
