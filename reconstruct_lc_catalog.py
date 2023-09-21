from pathlib import Path
import os

import numpy as np
import asdf
import argparse

from pyrecon import  utils, IterativeFFTParticleReconstruction, MultiGridReconstruction, IterativeFFTReconstruction
from cosmoprimo.fiducial import Planck2018FullFlatLCDM, AbacusSummit, DESI, TabulatedDESI

from apply_dndz_mask import parse_nztype

DEFAULTS = {}
DEFAULTS['sim_name'] = "AbacusSummit_base_c000_ph002"
DEFAULTS['stem'] = 'DESI_LRG' # 'DESI_ELG'
DEFAULTS['nmesh'] = 1024
DEFAULTS['sr'] = 12.5 # Mpc/h
DEFAULTS['rectype'] = "MG"
DEFAULTS['convention'] = "recsym"
DEFAULTS['nz_type'] = 'Gaussian(0.5, 0.4)'
DEFAULTS['photoz_error'] = 0.

"""
Usage:
python reconstruct_lc_catalog.py --sim_name AbacusSummit_base_c000_ph002 --stem DESI_LRG --nmesh 1024 --sr 12.5 --rectype MG --convention recsym
python reconstruct_lc_catalog.py --sim_name AbacusSummit_huge_c000_ph201 --stem DESI_LRG --nmesh 1024 --sr 12.5 --rectype MG --convention recsym
"""

def get_nz_str(nz_type):    
    # parse the dndz instructions
    nz_dict = parse_nztype(nz_type)
    if nz_dict['Type'] == "Gaussian":
        nz_str = f"_meanz{nz_dict['Mean_z']:.3f}_deltaz{nz_dict['Delta_z']:.3f}"
    elif nz_dict['Type'] == "StepFunction":
        nz_str = f"_minz{nz_dict['Min_z']:.3f}_maxz{nz_dict['Max_z']:.3f}"
    elif nz_dict['Type'] == "FromFile":
        nz_str = f"_{nz_dict['File']}"
    return nz_str

def main(sim_name, stem, stem2, stem3, nmesh, sr, rectype, convention, nz_type, nz_type2, nz_type3, photoz_error, want_fakelc=False, want_mask=False):
    
    # redshift error string
    photoz_str = f"_zerr{photoz_error:.1f}"
    
    # fake light cones
    fakelc_str = "_fakelc" if want_fakelc else ""
    mask_str = "_mask" if want_mask else ""

    # small area (this is kinda stupid)
    if "_small_area" in nz_type:
        nz_type = nz_type.split("_small_area")[0]
        mask_str = "_small_area"
    
    # how many processes to use for reconstruction: 32, 128 physical cpu per node for cori, perlmutter (hyperthreading doubles)
    ncpu = 128

    # initiate tracers
    tracers = []

    # initiate stems
    stems = [stem]
    print(sim_name, nz_type, nz_type2, nz_type3)
    nz_strs = [get_nz_str(nz_type)]
    if stem2 is not None:
        stems.append(stem2)
        nz_strs.append(get_nz_str(nz_type2))
    if stem3 is not None:
        stems.append(stem3)
        nz_strs.append(get_nz_str(nz_type3))
    nz_full_str = ''.join(nz_strs)
        
    # initiate mock dirs
    mock_dirs = []

    # reconstruction parameters
    if rectype == "IFT":
        recfunc = IterativeFFTReconstruction
    elif rectype == "IFTP":
        recfunc = IterativeFFTParticleReconstruction
    elif rectype == "MG":
        recfunc = MultiGridReconstruction

    tracer_extra_str = []
    for stem in stems:
        # additional specs of the tracer
        extra = '_'.join(stem.split('_')[2:])
        stem = '_'.join(stem.split('_')[:2])
        if extra != '': extra = '_'+extra

        # directory where mock catalogs are saved
        mock_dir = Path(f"/global/cfs/cdirs/desi/users/boryanah/kSZ_recon/mocks_lc_output_kSZ_recon{extra}/") / sim_name
        mock_dirs.append(mock_dir)

        # parameter choices
        if stem == 'DESI_LRG': 
            Mean_z = 0.5
            tracer = "LRG"
            tracers.append(tracer)
            bias = 2.2 # +/- 10%
        elif stem == 'DESI_ELG':
            Mean_z = 0.8
            tracer = "ELG"
            tracers.append(tracer)
            bias = 1.3
        else:
            print("Other tracers not yet implemented"); quit()
        tracer_extra_str.append(f"{tracer}{extra}")
    tracer_extra_str = '_'.join(tracer_extra_str)
    print(tracer_extra_str)

    # simulation parameters
    cosmo = DESI() # AbacusSummit
    ff = cosmo.growth_factor(Mean_z)
    H_z = cosmo.hubble_function(Mean_z)
    los = 'local'

    # directory where the reconstructed mock catalogs are saved
    save_dir = Path("/global/cfs/cdirs/desi/users/boryanah/kSZ_recon/")
    save_recon_dir = Path(save_dir) / "recon" / sim_name / f"z{Mean_z:.3f}"
    os.makedirs(save_recon_dir, exist_ok=True)

    # file to save to
    final_fn = save_recon_dir / f"displacements_{tracer_extra_str}{fakelc_str}{photoz_str}{mask_str}_postrecon_R{sr:.2f}_b{bias:.1f}_nmesh{nmesh:d}_{convention}_{rectype}{nz_full_str}.npz"
    #if os.path.exists(final_fn): return
    
    # loop over all tracers
    n_gals = []
    for i, tracer in enumerate(tracers):
        
        # load the galaxies
        data = np.load(mock_dirs[i] / f"galaxies_{tracer}{fakelc_str}{photoz_str}{mask_str}_prerecon{nz_strs[i]}.npz")
        if i == 0:
            RA = data['RA']
            DEC = data['DEC']
            Z_RSD = data['Z_RSD']
            Z = data['Z']
            VEL = data['VEL']
            UNIT_LOS = data['UNIT_LOS']
        else:
            RA = np.hstack((RA, data['RA']))
            DEC = np.hstack((DEC, data['DEC']))
            Z_RSD = np.hstack((Z_RSD, data['Z_RSD']))
            Z = np.hstack((Z, data['Z']))
            VEL = np.vstack((VEL, data['VEL']))
            UNIT_LOS = np.vstack((UNIT_LOS, data['UNIT_LOS']))
        n_gals.append(len(RA))
            
        # load the randoms
        data = np.load(mock_dirs[i] / f"randoms_{tracer}{fakelc_str}{photoz_str}{mask_str}_prerecon{nz_strs[i]}.npz")
        if i == 0:
            RAND_RA = data['RAND_RA']
            RAND_DEC = data['RAND_DEC']
            RAND_Z = data['RAND_Z']
        else:
            RAND_RA = np.hstack((RAND_RA, data['RAND_RA']))
            RAND_DEC = np.hstack((RAND_DEC, data['RAND_DEC']))
            RAND_Z = np.hstack((RAND_Z, data['RAND_Z']))

        print("RA min/max, DEC min/max, Z min/max", RA.min(), RA.max(), DEC.min(), DEC.max(), Z.min(), Z.max())
        print("RAND RA min/max, DEC min/max, Z min/max", RAND_RA.min(), RAND_RA.max(), RAND_DEC.min(), RAND_DEC.max(), RAND_Z.min(), RAND_Z.max())
    n_gals = np.array(n_gals)
    
    # transform into Cartesian coordinates
    PositionRSD = utils.sky_to_cartesian(cosmo.comoving_radial_distance(Z_RSD), RA, DEC)
    Position = utils.sky_to_cartesian(cosmo.comoving_radial_distance(Z), RA, DEC)
    RandomPosition = utils.sky_to_cartesian(cosmo.comoving_radial_distance(RAND_Z), RAND_RA, RAND_DEC)
    
    # run reconstruction on the mocks w/o RSD
    print('Recon First tracer')
    recon_tracer = recfunc(f=ff, bias=bias, nmesh=nmesh, los=los, positions=Position, # used only to define box size if not provided
                           nthreads=int(ncpu), fft_engine='fftw', fft_plan='estimate', dtype='f4', wrap=True) # probably doesn't matter; in principle, False
    print('grid set up', flush=True)
    recon_tracer.assign_data(Position)#, dat_cat['WEIGHT'])
    print('data assigned', flush=True)
    recon_tracer.assign_randoms(RandomPosition)#, dat_cat['WEIGHT'])
    print('randoms assigned', flush=True)
    recon_tracer.set_density_contrast(smoothing_radius=sr)
    print('density constrast calculated, now doing recon',flush=True)
    recon_tracer.run()
    print('recon has been run', flush=True)

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
    print('grid set up', flush=True)
    recon_tracer.assign_data(PositionRSD)#, dat_cat['WEIGHT'])
    print('data assigned', flush=True)
    recon_tracer.assign_randoms(RandomPosition)#, dat_cat['WEIGHT'])
    print('randoms assigned', flush=True)
    recon_tracer.set_density_contrast(smoothing_radius=sr)
    print('density constrast calculated, now doing recon', flush=True)
    recon_tracer.run()
    print('recon has been run', flush=True)

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
    np.savez(final_fn,
             displacements=displacements, displacements_rsd=displacements_rsd, velocities=VEL, unit_los=UNIT_LOS, growth_factor=ff, Hubble_z=H_z,
             random_displacements_rsd=random_displacements_rsd, random_displacements=random_displacements, displacements_rsd_nof=displacements_rsd_nof,
             random_displacements_rsd_nof=random_displacements_rsd_nof, n_gals=n_gals)

class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--sim_name', help='Simulation name', default=DEFAULTS['sim_name'])
    parser.add_argument('--stem', help='Stem file name', default=DEFAULTS['stem'])#, choices=["DESI_LRG", "DESI_ELG", "DESI_LRG_high_density", "DESI_LRG_bgs", "DESI_ELG_high_density"])
    parser.add_argument('--stem2', help='Stem file name 2', default=None)#, choices=["DESI_LRG", "DESI_ELG", "DESI_LRG_high_density", "DESI_LRG_bgs", "DESI_ELG_high_density"])
    parser.add_argument('--stem3', help='Stem file name 3', default=None)#, choices=["DESI_LRG", "DESI_ELG", "DESI_LRG_high_density", "DESI_LRG_bgs", "DESI_ELG_high_density"])
    parser.add_argument('--nmesh', help='Number of cells per dimension for reconstruction', type=int, default=DEFAULTS['nmesh'])
    parser.add_argument('--sr', help='Smoothing radius', type=float, default=DEFAULTS['sr'])
    parser.add_argument('--rectype', help='Reconstruction type', default=DEFAULTS['rectype'], choices=["IFT", "MG", "IFTP"])
    parser.add_argument('--convention', help='Reconstruction convention', default=DEFAULTS['convention'], choices=["recsym", "reciso"])
    parser.add_argument('--photoz_error', help='Percentage error on the photometric redshifts', default=DEFAULTS['photoz_error'], type=float)
    parser.add_argument('--nz_type', help='Type of N(z) distribution: Gaussian(mean, 2sigma), Step(min, max), file name', default=DEFAULTS['nz_type'])
    parser.add_argument('--nz_type2', help='Type of N(z) distribution: Gaussian(mean, 2sigma), Step(min, max), file name', default=None)
    parser.add_argument('--nz_type3', help='Type of N(z) distribution: Gaussian(mean, 2sigma), Step(min, max), file name', default=None)
    parser.add_argument('--want_fakelc', help='Want to use the fake light cone?', action='store_true')
    parser.add_argument('--want_mask', help='Want to apply DESI mask?', action='store_true')
    args = vars(parser.parse_args())
    main(**args)
