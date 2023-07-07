from pathlib import Path
import os

import numpy as np
import asdf
import argparse
from astropy.io import ascii

from abacusnbody.metadata import get_meta
from pyrecon import  utils, IterativeFFTParticleReconstruction, MultiGridReconstruction, IterativeFFTReconstruction
from cosmoprimo.fiducial import Planck2018FullFlatLCDM, AbacusSummit, DESI, TabulatedDESI

DEFAULTS = {}
DEFAULTS['sim_name'] = "AbacusSummit_base_c000_ph002"
DEFAULTS['redshift'] = 0.5 # 0.8
DEFAULTS['tracer'] = "LRG" # "ELG"
DEFAULTS['nmesh'] = 512 # 1024
DEFAULTS['sr'] = 10. # Mpc/h
DEFAULTS['rectype'] = "MG"
DEFAULTS['convention'] = "recsym"

"""
Usage:
python reconstruct_box_catalog.py --sim_name AbacusSummit_base_c000_ph002 --redshift 0.5 --tracer LRG --nmesh 1024 --sr 12.5 --rectype MG --convention recsym

# Notes to self (Boryana):
# If you want redshift-dependent bias, then do `recon_tracer.mesh_delta /= b_z` after `recon.set_density_contrast`, where `mesh_delta` is a 512^3 array, and set bias to 1.
# If you want to input your own delta, then initiate `recfunc()` and get rid of assign_data, assign_randoms and set_density_contrast.
# Instead, do `recon_tracer.mesh_delta[...] = delta_new` or `mesh_delta.value = delta_new` (but apply smoothing to delta_new yourself);
# The only thing that MG needs as input is mesh_delta. Finally, read the shifts as before at the positions of interest.
# Note that IFTP updates the positions internally, which complicates the reading of the shifts
"""

# random seed
np.random.seed(300)

def main(sim_name, redshift, tracer, nmesh, sr, rectype, convention):
    # how many processes to use for reconstruction: 32, 128 physical cpu per node for cori, perlmutter (hyperthreading doubles)
    ncpu = 128

    # additional specs of the tracer
    extra = '_'.join(tracer.split('_')[1:])
    tracer = tracer.split('_')[0]
    if extra != '': extra = '_'+extra
    
    # reconstruction parameters
    if tracer == "LRG":
        bias = 2.2 # +/- 10%
    elif tracer == "ELG":
        bias = 1.3
    else:
        print("Other tracers not yet implemented"); quit()
    if rectype == "IFT":
        recfunc = IterativeFFTReconstruction
    elif rectype == "IFTP":
        recfunc = IterativeFFTParticleReconstruction
    elif rectype == "MG":
        recfunc = MultiGridReconstruction
    
    # simulation parameters
    Lbox = get_meta(sim_name, 0.1)['BoxSize'] # cMpc/h
    cosmo = DESI()
    ff = cosmo.growth_factor(redshift) # Abacus
    H_z = cosmo.hubble_function(redshift) # Abacus
    los = 'z'

    # directory where mock catalogs are saved
    mock_dir = Path(f"/pscratch/sd/b/boryanah/AbacusHOD_scratch/mocks_box_output_kSZ_recon{extra}/")
    #mock_dir = Path("/global/cfs/cdirs/desi/users/boryanah/kSZ_recon/mocks_box_output_test_paper/") # old

    # file names of mocks with and without RSD
    fn_rsd = mock_dir / f"{sim_name}/z{redshift:.3f}/galaxies_rsd/{tracer}s.dat"
    fn = mock_dir / f"{sim_name}/z{redshift:.3f}/galaxies/{tracer}s.dat"

    # directory where the reconstructed mock catalogs are saved
    save_dir = Path("/global/cfs/cdirs/desi/users/boryanah/kSZ_recon/")
    save_recon_dir = Path(save_dir) / "recon" / sim_name / f"z{redshift:.3f}"
    os.makedirs(save_recon_dir, exist_ok=True)

    # read pos and vel w/o RSD (centered at 0)
    f = ascii.read(fn)
    Position = np.vstack((f['x'], f['y'], f['z'])).T
    Velocity = np.vstack((f['vx'], f['vy'], f['vz'])).T

    # read pos w/ RSD (centered at 0)
    f = ascii.read(fn_rsd)
    PositionRSD = np.vstack((f['x'], f['y'], f['z'])).T

    
    # TESTING!!!!!!!!!!!!!!!!!! downsample
    """
    N_target = 0.0005191514900464895 * 2000.**3 #1148610.5 # 2442646.7
    f = N_target/Position.shape[0]
    choice = np.random.rand(Position.shape[0]) < f
    Position = Position[choice]
    Velocity = Velocity[choice]
    PositionRSD = PositionRSD[choice]
    """
    
    # wrap around box so that pos range from [0, Lbox)
    Position %= Lbox
    PositionRSD %= Lbox

    # generate randoms
    rands_fac = 20
    RandomPosition = np.vstack((np.random.rand(rands_fac*Position.shape[0]), np.random.rand(rands_fac*Position.shape[0]), np.random.rand(rands_fac*Position.shape[0])))*Lbox
    RandomPosition = RandomPosition.T

    
    # TESTING
    """
    N_target *= 20
    f = N_target/RandomPosition.shape[0]
    choice = np.random.rand(RandomPosition.shape[0]) < f
    RandomPosition = RandomPosition[choice]
    """
    
    # run reconstruction on the mocks w/o RSD
    print('Recon First tracer')
    recon_tracer = recfunc(f=ff, bias=bias, nmesh=nmesh, los=los, positions=Position, boxsize=Lbox, boxcenter=(Lbox/2, Lbox/2, Lbox/2), # boxcenter not needed when wrap=True
                           nthreads=int(ncpu), fft_engine='fftw', fft_plan='estimate', dtype='f4', wrap=True)
    print('grid set up',flush=True)
    recon_tracer.assign_data(Position)#, dat_cat['WEIGHT'])
    print('data assigned',flush=True)
    recon_tracer.assign_randoms(RandomPosition)#, dat_cat['WEIGHT'])
    print('randoms assigned',flush=True)
    recon_tracer.set_density_contrast(smoothing_radius=sr)
    print('density constrast calculated, now doing recon', flush=True)
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
    recon_tracer = recfunc(f=ff, bias=bias, nmesh=nmesh, los=los, positions=PositionRSD, boxsize=Lbox, boxcenter=(Lbox/2, Lbox/2, Lbox/2),
                           nthreads=int(ncpu), fft_engine='fftw', fft_plan='estimate', dtype='f4', wrap=True)
    print('grid set up',flush=True)
    recon_tracer.assign_data(PositionRSD)#, dat_cat['WEIGHT'])
    print('data assigned',flush=True)
    recon_tracer.assign_randoms(RandomPosition)#, dat_cat['WEIGHT'])
    print('randoms assigned',flush=True)
    recon_tracer.set_density_contrast(smoothing_radius=sr)
    print('density constrast calculated, now doing recon', flush=True)
    recon_tracer.run()
    print('recon has been run',flush=True)

    # read the displacements in real and redshift space (rsd has the (1+f) factor in the LOS direction)
    if rectype == 'IFTP':
        displacements_rsd = recon_tracer.read_shifts('data', field='disp+rsd')
        displacements_rsd_nof = recon_tracer.read_shifts('data', field='disp')
    else:
        displacements_rsd = recon_tracer.read_shifts(PositionRSD, field='disp+rsd')
        displacements_rsd_nof = recon_tracer.read_shifts(PositionRSD, field='disp')
    random_displacements_rsd = recon_tracer.read_shifts(RandomPosition, field='disp+rsd')
    random_displacements_rsd_nof = recon_tracer.read_shifts(RandomPosition, field='disp')

    # save the displacements
    np.savez(save_recon_dir / f"displacements_{tracer}{extra}_postrecon_R{sr:.2f}_b{bias:.1f}_nmesh{nmesh:d}_{convention}_{rectype}_z{redshift:.3f}.npz",
             displacements=displacements, displacements_rsd=displacements_rsd, velocities=Velocity, positions=Position, positions_rsd=PositionRSD,
             growth_factor=ff, Hubble_z=H_z, random_displacements_rsd=random_displacements_rsd, random_displacements=random_displacements,
             random_positions=RandomPosition, displacements_rsd_nof=displacements_rsd_nof, random_displacements_rsd_nof=random_displacements_rsd_nof)

class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--sim_name', help='Simulation name', default=DEFAULTS['sim_name'])
    parser.add_argument('--redshift', help='Redshift', type=float, default=DEFAULTS['redshift'])
    parser.add_argument('--tracer', help='Galaxy type', default=DEFAULTS['tracer'], choices=["LRG", "ELG", "QSO", "LRG_high_density", "LRG_bgs", "ELG_high_density"])
    parser.add_argument('--nmesh', help='Number of cells per dimension for reconstruction', type=int, default=DEFAULTS['nmesh'])
    parser.add_argument('--sr', help='Smoothing radius', type=float, default=DEFAULTS['sr'])
    parser.add_argument('--rectype', help='Reconstruction type', default=DEFAULTS['rectype'], choices=["IFT", "MG", "IFTP"])
    parser.add_argument('--convention', help='Reconstruction convention', default=DEFAULTS['convention'], choices=["recsym", "reciso"])
    args = vars(parser.parse_args())
    main(**args)
