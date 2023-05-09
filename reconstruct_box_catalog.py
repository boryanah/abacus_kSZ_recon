# IFFTParticle maybe ask David about the displacements
# how do i directly feed in the density

from pathlib import Path

import asdf
import numpy as np
from astropy.io import ascii

from pyrecon import  utils,IterativeFFTParticleReconstruction,MultiGridReconstruction,IterativeFFTReconstruction
from cosmoprimo.fiducial import Planck2018FullFlatLCDM, AbacusSummit, DESI, TabulatedDESI

# params
redshift = 0.5
sim_name = "AbacusSummit_base_c000_ph002"
Lbox = 2000. # cMpc/h
tracer = "LRG"
#fn_rsd = f"/global/cscratch1/sd/boryanah/abacus_lensing/mocks_box_output_test_paper/z{redshift:.3f}/galaxies_rsd/{tracer}s.dat"
#fn = f"/global/cscratch1/sd/boryanah/abacus_lensing/mocks_box_output_test_paper/z{redshift:.3f}/galaxies/{tracer}s.dat"
fn_rsd = f"/global/cfs/cdirs/desi/users/boryanah/kSZ_recon/mocks_box_output_test_paper/{sim_name}/z{redshift:.3f}/galaxies_rsd/{tracer}s.dat"
fn = f"/global/cfs/cdirs/desi/users/boryanah/kSZ_recon/mocks_box_output_test_paper/{sim_name}/z{redshift:.3f}/galaxies/{tracer}s.dat"
save_dir = Path("/global/cfs/cdirs/desi/users/boryanah/kSZ_recon/")
save_tmp_dir = Path(save_dir) / sim_name / "tmp"

# ask about directly handing the density
# and also different randoms?

# read pos and vel
f = ascii.read(fn)
Position = np.vstack((f['x'], f['y'], f['z'])).T
Velocity = np.vstack((f['vx'], f['vy'], f['vz'])).T
f = ascii.read(fn_rsd)
PositionRSD = np.vstack((f['x'], f['y'], f['z'])).T

print(Position.min())
print(PositionRSD.min())
Position %= Lbox
PositionRSD %= Lbox

# generate randoms
rands_fac = 20
np.random.seed(300)
RandomPosition = np.vstack((np.random.rand(rands_fac*Position.shape[0]), np.random.rand(rands_fac*Position.shape[0]), np.random.rand(rands_fac*Position.shape[0])))*Lbox
RandomPosition = RandomPosition.T

# rec params
recfunc = MultiGridReconstruction #IterativeFFTParticleReconstruction
rectype = "MG"
convention = "recsym"
cosmo = DESI()
los = 'z' # 'local'

# more rec params
bias = 2.2 # +/- 10%                      
nmesh = 512 #1024
ncpu = 64 # 32 physical cori; 128 physical perlmutter (cpu per node) (hyperthreading is double that)
sr = 10 # Mpc/h go up to 10, 15
zeff = redshift
ff = cosmo.growth_factor(zeff) #cosmo.sigma8_z(z=zeff,of='theta_cb')/cosmo.sigma8_z(z=zeff,of='delta_cb') # growth factor
H_z = cosmo.hubble_function(zeff)


print('Recon First tracer')
recon_tracer = recfunc(f=ff, bias=bias, nmesh=nmesh, los=los, positions=Position, boxsize=Lbox, boxcenter=(Lbox/2, Lbox/2, Lbox/2), # boxcenter may not be required bc of wrapping (boxcenter shouldnt matter)
                       nthreads=int(ncpu), fft_engine='fftw', fft_plan='estimate', dtype='f4', wrap=True)
print('grid set up',flush=True)
recon_tracer.assign_data(Position)#,dat_cat['WEIGHT'])
print('data assigned',flush=True)
recon_tracer.assign_randoms(RandomPosition)#,dat_cat['WEIGHT'])
print('randoms assigned',flush=True)
recon_tracer.set_density_contrast(smoothing_radius=sr)
print('density constrast calculated, now doing recon', flush=True)

# recon_tracer.mesh_delta /= b_z # 512^3

# if you want to do delta yourself, then skip assign_data and assign_randoms and get rid of set_density_contrast (but apply smoothing yourself); recon_tracer.mesh_delta[...] = delta_mine or mesh_delta.value; multigrid needs only mesh data; and then just add read_shifts

recon_tracer.run()



print('recon has been run',flush=True)
displacements = recon_tracer.read_shifts(Position, field='disp') # ifftpcle  'data' instead of positions moves it withing but not for the randoms
random_displacements = recon_tracer.read_shifts(RandomPosition, field='disp')

print('Recon Second tracer')
recon_tracer = recfunc(f=ff, bias=bias, nmesh=nmesh, los=los, positions=PositionRSD, boxsize=Lbox, boxcenter=(Lbox/2, Lbox/2, Lbox/2),
                       nthreads=int(ncpu), fft_engine='fftw', fft_plan='estimate', dtype='f4', wrap=True)
print('grid set up',flush=True)
recon_tracer.assign_data(PositionRSD)#,dat_cat['WEIGHT'])
print('data assigned',flush=True)
recon_tracer.assign_randoms(RandomPosition)#,dat_cat['WEIGHT'])
print('randoms assigned',flush=True)
recon_tracer.set_density_contrast(smoothing_radius=sr)
print('density constrast calculated, now doing recon', flush=True)
recon_tracer.run()
print('recon has been run',flush=True)
displacements_rsd = recon_tracer.read_shifts(PositionRSD, field='disp+rsd')
random_displacements_rsd = recon_tracer.read_shifts(RandomPosition, field='disp+rsd')

print(displacements.shape) 
np.savez(save_tmp_dir / f"displacements_{tracer}_postrecon_z{redshift:.3f}.npz", displacements=displacements, displacements_rsd=displacements_rsd, velocities=Velocity, positions=Position, positions_rsd=PositionRSD, growth_factor=ff, Hubble_z=H_z, random_displacements_rsd=random_displacements_rsd, random_displacements=random_displacements, random_positions=RandomPosition)

quit()
positions_rec_tracer = {}
if rectype == 'IFTP':
    positions_rec_tracer['data'] = Position - recon_tracer.read_shifts('data', field='disp+rsd')
else:
    positions_rec_tracer['data'] = Position - recon_tracer.read_shifts(Position, field='disp+rsd')
    
positions_rec_tracer['randoms'] = RandomPosition - recon_tracer.read_shifts(RandomPosition, field='disp+rsd' if convention == 'recsym' else 'disp')
