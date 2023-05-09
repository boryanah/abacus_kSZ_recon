from pathlib import Path

import asdf
import argparse
import numpy as np

from pyrecon import  utils,IterativeFFTParticleReconstruction,MultiGridReconstruction,IterativeFFTReconstruction
from cosmoprimo.fiducial import Planck2018FullFlatLCDM, AbacusSummit, DESI, TabulatedDESI

DEFAULTS = {}
DEFAULTS['sim_name'] = "AbacusSummit_base_c000_ph002" # "AbacusSummit_huge_c000_ph201"
DEFAULTS['stem'] = 'DESI_LRG' # 'DESI_ELG'

def main(sim_name, stem):
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
        print("AHHHH you've dealt me a mortal wound"); quit()

    # immutables
    #save_dir = Path("/global/project/projectdirs/desi/users/boryanah/kSZ_recon/")
    save_dir = Path("/global/cfs/cdirs/desi/users/boryanah/kSZ_recon/")

    # load galaxies and randoms
    save_tmp_dir = Path(save_dir) / sim_name / "tmp"
    data = np.load(save_tmp_dir / f"galaxies_{tracer}_prerecon_meanz{Mean_z:.3f}_deltaz{Delta_z:.3f}.npz")
    RA = data['RA']
    DEC = data['DEC']
    Z_RSD = data['Z_RSD']
    Z = data['Z']
    VEL = data['VEL']
    UNIT_LOS = data['UNIT_LOS']
    data = np.load(save_tmp_dir / f"randoms_{tracer}_prerecon_meanz{Mean_z:.3f}_deltaz{Delta_z:.3f}.npz")
    RAND_RA = data['RAND_RA']
    RAND_DEC = data['RAND_DEC']
    RAND_Z = data['RAND_Z']

    # immutable recon params
    rectype = "IFT" #"MG"
    if rectype == "IFT":
        recfunc = IterativeFFTReconstruction
    elif rectype == "MG":
        recfunc = MultiGridReconstruction
    convention = "recsym"
    cosmo = DESI()

    # transform into cartesian # units
    print(RA.max(), DEC.max())
    PositionRSD = utils.sky_to_cartesian(cosmo.comoving_radial_distance(Z_RSD), RA, DEC)
    Position = utils.sky_to_cartesian(cosmo.comoving_radial_distance(Z), RA, DEC)
    RandomPosition = utils.sky_to_cartesian(cosmo.comoving_radial_distance(RAND_Z), RAND_RA, RAND_DEC)

    # recon params
    bias = 2.2 # +/- 10%                   
    nmesh = 512 #1024
    ncpu = 128 #64 # 32 physical cori; 128 physical perlmutter (cpu per node) (hyperthreading is double that)
    sr = 10. # Mpc/h go up to 10
    zeff = Mean_z
    ff = cosmo.growth_factor(zeff) #cosmo.sigma8_z(z=zeff,of='theta_cb')/cosmo.sigma8_z(z=zeff,of='delta_cb') # growth factor
    H_z = cosmo.hubble_function(zeff)
    print("H_z, f(z)", H_z, ff)

    # run reconstruction magic
    print('Recon First tracer')
    recon_tracer = recfunc(f=ff, bias=bias, nmesh=nmesh, los='local', positions=Position,
                           nthreads=int(ncpu), fft_engine='fftw', fft_plan='estimate', dtype='f4', wrap=True)
    print('grid set up',flush=True)
    recon_tracer.assign_data(Position)#,dat_cat['WEIGHT'])
    print('data assigned',flush=True)
    recon_tracer.assign_randoms(RandomPosition)#,dat_cat['WEIGHT'])
    print('randoms assigned',flush=True)
    recon_tracer.set_density_contrast(smoothing_radius=sr)
    print('density constrast calculated, now doing recon',flush=True)
    recon_tracer.run()
    print('recon has been run',flush=True)

    # run reconstruction magic
    print('Recon Second tracer')
    recon_tracer_rsd = recfunc(f=ff, bias=bias, nmesh=nmesh, los='local', positions=PositionRSD,
                           nthreads=int(ncpu), fft_engine='fftw', fft_plan='estimate', dtype='f4', wrap=True)
    print('grid set up',flush=True)
    recon_tracer_rsd.assign_data(PositionRSD)#,dat_cat['WEIGHT'])
    print('data assigned',flush=True)
    recon_tracer_rsd.assign_randoms(RandomPosition)#,dat_cat['WEIGHT'])
    print('randoms assigned',flush=True)
    recon_tracer_rsd.set_density_contrast(smoothing_radius=sr)
    print('density constrast calculated, now doing recon',flush=True)
    recon_tracer_rsd.run()
    print('recon has been run',flush=True)

    """
    # move the dudes (that's how you know you have Psi and not s cause it's D s)
    positions_rec_tracer = {}
    if rectype == 'IFTP':
        positions_rec_tracer['data'] = Position - recon_tracer.read_shifts('data', field='disp+rsd')
    else:
        positions_rec_tracer['data'] = Position - recon_tracer.read_shifts(Position, field='disp+rsd')

    # if recsym
    positions_rec_tracer['randoms'] = RandomPosition - recon_tracer.read_shifts(RandomPosition, field='disp+rsd' if convention == 'recsym' else 'disp')
    """
        
    # save the displacements
    displacements_rsd = recon_tracer_rsd.read_shifts(PositionRSD, field='disp+rsd')
    displacements = recon_tracer.read_shifts(Position, field='disp')
    random_displacements_rsd = recon_tracer_rsd.read_shifts(RandomPosition, field='disp+rsd')
    random_displacements = recon_tracer.read_shifts(RandomPosition, field='disp')
    print(displacements.shape, VEL.shape)
    np.savez(save_tmp_dir / f"displacements_{tracer}_postrecon_meanz{Mean_z:.3f}_deltaz{Delta_z:.3f}.npz", displacements=displacements, displacements_rsd=displacements_rsd, velocities=VEL, unit_los=UNIT_LOS, growth_factor=ff, Hubble_z=H_z, random_displacements_rsd=random_displacements_rsd, random_displacements=random_displacements)

class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--sim_name', help='Simulation name', default=DEFAULTS['sim_name'])
    parser.add_argument('--stem', help='Stem file name', default=DEFAULTS['stem'])

    args = vars(parser.parse_args())
    main(**args)

