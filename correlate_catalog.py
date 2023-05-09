from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pycorr import TwoPointCorrelationFunction, TwoPointEstimator, NaturalTwoPointEstimator,project_to_multipoles, project_to_wp #, setup_logging, TwoPointCounter, mpi, utils

from pypower import CatalogMesh, MeshFFTPower, CatalogFFTPower,ArrayMesh,CatalogFFTCorr

# parameters
tracer = "LRG"
Mean_z = 0.5
Delta_z = 0.4
h = 0.6736
a = 1./(1+Mean_z)
box_lc = "box"
mode = "pre"  #"pre" #  "post'
save_dir = Path("/global/cfs/cdirs/desi/users/boryanah/kSZ_recon/")
sim_name = "AbacusSummit_base_c000_ph002"
want_pk = True
Lbox = 2000.

if box_lc == "lc":
    # load galaxies and randoms
    save_tmp_dir = Path(save_dir) / sim_name / "tmp"
    data = np.load(save_tmp_dir / f"galaxies_{tracer}_prerecon_meanz{Mean_z:.3f}_deltaz{Delta_z:.3f}.npz")
    POS = data['POS']
    POS_RSD = data['POS_RSD']
    data = np.load(save_tmp_dir / f"randoms_{tracer}_prerecon_meanz{Mean_z:.3f}_deltaz{Delta_z:.3f}.npz")
    RAND_POS = data['RAND_POS']

    # shift origin
    origin = np.array([-990., -990, -990.])
    POS -= origin
    POS_RSD -= origin
    RAND_POS -= origin

    # load data
    data = np.load(f"{str(save_dir)}/{sim_name}/tmp/displacements_{tracer}_postrecon_meanz{Mean_z:.3f}_deltaz{Delta_z:.3f}.npz")
    print(data.files) # ['displacements', 'velocities', 'growth_factor', 'Hubble_z']
    Psi_dot = data['displacements'] # cMpc/h
    Psi_dot_rsd = data['displacements_rsd'] # cMpc/h
    random_Psi_dot = data['random_displacements'] # cMpc/h
    random_Psi_dot_rsd = data['random_displacements_rsd'] # cMpc/h
    #vel = data['velocities'] # km/s
    #unit_los = data['unit_los']
    #f = data['growth_factor']
    #H = data['Hubble_z'] # km/s/Mpc
else:
    # load data
    data = np.load(f"{str(save_dir)}/{sim_name}/tmp/displacements_{tracer}_postrecon_z{Mean_z:.3f}.npz")
    print(data.files) # ['displacements', 'velocities', 'growth_factor', 'Hubble_z']
    Psi_dot = data['displacements'] # cMpc/h
    Psi_dot_rsd = data['displacements_rsd'] # cMpc/h
    random_Psi_dot = data['random_displacements'] # cMpc/h
    random_Psi_dot_rsd = data['random_displacements_rsd'] # cMpc/h
    POS = data['positions']
    POS_RSD = data['positions_rsd']
    RAND_POS = data['random_positions']
    
positions_rec = {}
if mode == "pre":
    positions_rec['data'] = POS_RSD
else:
    positions_rec['data'] = POS_RSD - Psi_dot_rsd 
    positions_rec['shifted_randoms'] = RAND_POS - random_Psi_dot_rsd
positions_rec['randoms'] = RAND_POS

inds = np.arange(RAND_POS.shape[0], dtype=int)
np.random.shuffle(inds)
positions_rec['randoms'] = positions_rec['randoms'][inds]
if mode == "post":
    positions_rec['shifted_randoms'] = positions_rec['shifted_randoms'][inds]

    
if want_pk:
    nmesh = 512
    
    
    if box_lc == "lc":
        los = 'firstpoint'
        boxcenter = np.array([0., 0., 0.])
        boxsize = None
        wrap = False
    else:
        los = 'z'
        boxcenter = np.array([Lbox/2., Lbox/2., Lbox/2.])
        boxsize = Lbox
        wrap = True
    kedges = np.arange(0.01, 1.0, 0.005)

    if mode == "pre":
        poles_recon = CatalogFFTPower(data_positions1=positions_rec['data'], randoms_positions1=positions_rec['randoms'] if boxsize is None else None,
                                      nmesh=nmesh, resampler='tsc', boxsize=boxsize, boxcenter=boxcenter, interlacing=3, ells=(0,2,4), wrap=wrap,
                                      los=los, edges=kedges, position_type='pos', mpiroot=0)#.poles

    else:
        poles_recon = CatalogFFTPower(data_positions1=positions_rec['data'], shifted_positions1=positions_rec['shifted_randoms'],
                                      randoms_positions1=positions_rec['randoms'] if boxsize is None else None, wrap=wrap,
                                      nmesh=nmesh, resampler='tsc', boxsize=boxsize, boxcenter=boxcenter, interlacing=3, ells=(0,2,4),
                                      los=los, edges=kedges, position_type='pos', mpiroot=0)#.poles
    fn = f"/global/cfs/cdirs/desi/users/boryanah/kSZ_recon/AbacusSummit_base_c000_ph002/tmp/poles_{box_lc}_{mode}.npy"
    poles_recon.save(fn)
    quit()


if mode == "pre":
    size_randoms1 = random_Psi_dot_rsd.shape[0]
    nsplits = 20
    ncpu = 64
    D1D2_cross = None
    result_cross = 0
    edges = (np.linspace(0.01, 200, 201), np.linspace(-1, 1, 201))
    if box_lc == "lc":
        los = 'midpoint'
    else:
        los = 'z'
    
    for isplit in range(nsplits):
        print('Split {:d}/{:d}.'.format(isplit + 1, nsplits))
        sl1 = slice(isplit * size_randoms1 // nsplits, (isplit + 1) * size_randoms1 // nsplits)
        result_cross += TwoPointCorrelationFunction('smu', edges, data_positions1=np.transpose(positions_rec['data']), los=los,
                                                    randoms_positions1=[p2[sl1] for p2 in np.transpose(positions_rec['randoms'])], #positions_rec['randoms'], 
                                                    engine='corrfunc', D1D2=D1D2_cross, nthreads=ncpu, position_type='xyz')
        D1D2_cross = result_cross.D1D2 # D1D2 is passed to next computation, which avoids recomputing it again
    
    """
    result_cross = TwoPointCorrelationFunction('smu', edges, data_positions1=np.transpose(positions_rec['data']), los=los,
                                                randoms_positions1=np.transpose(positions_rec['randoms']),
                                                engine='corrfunc', nthreads=ncpu, position_type='xyz')
    """
else:
    size_randoms1 = random_Psi_dot_rsd.shape[0]
    nsplits = 20
    ncpu = 64
    D1D2_cross = None
    result_cross = 0
    edges = (np.linspace(0.01, 200, 201), np.linspace(-1, 1, 201))
    if box_lc == "lc":
        los = 'midpoint'
    else:
        los = 'z'
    
    for isplit in range(nsplits):
        print('Split {:d}/{:d}.'.format(isplit + 1, nsplits))
        sl1 = slice(isplit * size_randoms1 // nsplits, (isplit + 1) * size_randoms1 // nsplits)
        result_cross += TwoPointCorrelationFunction('smu', edges, data_positions1=np.transpose(positions_rec['data']), los=los,
                                                    randoms_positions1=[p2[sl1] for p2 in np.transpose(positions_rec['randoms'])],
                                                    shifted_positions1=[p2[sl1] for p2 in np.transpose(positions_rec['shifted_randoms'])],
                                                    engine='corrfunc', D1D2=D1D2_cross, nthreads=ncpu, position_type='xyz')
        D1D2_cross = result_cross.D1D2 # D1D2 is passed to next computation, which avoids recomputing it again
    
    """
    result_cross = TwoPointCorrelationFunction('smu', edges, data_positions1=np.transpose(positions_rec['data']), los=los,
                                               randoms_positions1=np.transpose(positions_rec['randoms']),
                                               shifted_positions1=np.transpose(positions_rec['shifted_randoms']),
                                               engine='corrfunc', nthreads=ncpu, position_type='xyz')
    """
fn = f"/global/cfs/cdirs/desi/users/boryanah/kSZ_recon/AbacusSummit_base_c000_ph002/tmp/result_cross_{box_lc}_{mode}.npy"
result_cross.save(fn)

quit()
# velocity in los direction
vel_r = np.sum(vel*unit_los, axis=1)

# compute the rms
vel_rms = np.sqrt(np.mean(vel[:, 0]**2+vel[:, 1]**2+vel[:, 2]**2))/np.sqrt(3.)
vel_r_rms = np.sqrt(np.mean(vel_r**2))
print("vel_rms", vel_rms)
print("vel_r_rms", vel_r_rms)

def get_Psi_dot(Psi_dot, want_rsd=False):
    # divide by 1+f in los direction
    #Psi_dot = (unit_los*Psi_dot)/(1.+f) # real space
    Psi_dot *= a*f*H # km/s/h
    Psi_dot /= h # km/s
    Psi_dot_r = np.sum(Psi_dot*unit_los, axis=1)
    if want_rsd:
        Psi_dot_r /= (1+f) # real space
    N = Psi_dot.shape[0]
    print("number", N)
    assert len(vel_r) == vel.shape[0] == len(Psi_dot_r) == N
    return Psi_dot, Psi_dot_r, N

def print_coeff(Psi_dot, want_rsd=False):
    # compute velocity from displacement
    Psi_dot, Psi_dot_r, N = get_Psi_dot(Psi_dot, want_rsd=want_rsd)

    # compute rms
    Psi_dot_rms = np.sqrt(np.mean(Psi_dot[:, 0]**2+Psi_dot[:, 1]**2+Psi_dot[:, 2]**2))/np.sqrt(3.)
    Psi_dot_r_rms = np.sqrt(np.mean(Psi_dot_r**2))
    print("Psi_dot_rms", Psi_dot_rms)
    print("Psi_dot_r_rms", Psi_dot_r_rms)

    # compute statistics
    print("overall", np.sum((Psi_dot*vel))/N/(3.*vel_rms*Psi_dot_rms))
    print("in each direction", np.sum(Psi_dot*vel, axis=0)/N/(vel_rms*Psi_dot_rms))
    print("in los direction", np.sum(Psi_dot_r*vel_r)/N/(vel_r_rms*Psi_dot_r_rms))

    plt.figure(figsize=(9, 7))
    plt.scatter(vel_r, Psi_dot_r, s=1, alpha=0.1, color='teal')
    if want_rsd:
        plt.savefig("figs/vel_rec_rsd.png")
    else:
        plt.savefig("figs/vel_rec.png")
    plt.close()
print("no rsd")
print_coeff(Psi_dot, want_rsd=False)
print("rsd")
print_coeff(Psi_dot_rsd, want_rsd=True)
