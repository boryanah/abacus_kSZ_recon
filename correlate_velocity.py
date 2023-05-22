from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pycorr import TwoPointCorrelationFunction, TwoPointEstimator, NaturalTwoPointEstimator, project_to_multipoles, project_to_wp #, setup_logging, TwoPointCounter, mpi, utils

from pypower import CatalogMesh, MeshFFTPower, CatalogFFTPower, ArrayMesh, CatalogFFTCorr

from Corrfunc.theory import DD, DDrppi

sys.path.append("/global/homes/b/boryanah/repos/abacusutils")
from abacusnbody.hod.power_spectrum import numba_tsc_3D, calc_power

"""
run with srun
python correlate_velocity.py box 0
python correlate_velocity.py box 1
python correlate_velocity.py lc 0
python correlate_velocity.py lc 1
"""

def get_RRrppi(N1, N2, Lbox, rpbins, pimax, npibins):
    vol_all = np.pi*rpbins**2
    dvol = vol_all[1:]-vol_all[:-1]
    pibins = np.linspace(-pimax, pimax, npibins+1)
    dpi = pibins[1:]-pibins[:-1]
    n2 = N2/Lbox**3
    n2_bin = dvol[:, None]*n2*dpi[None, :]
    pairs = N1*n2_bin
    #pairs *= 2. # wait I think not anymore since we are now doing -pi to pi
    return pairs

# parameters
tracer = "LRG"
Mean_z = 0.5
Delta_z = 0.4
h = 0.6736
a = 1./(1+Mean_z)
box_lc = sys.argv[1] #"box" #"lc"
save_dir = Path("/global/cfs/cdirs/desi/users/boryanah/kSZ_recon/")
sim_name = "AbacusSummit_base_c000_ph002"
Lbox = 2000.
want_pk = int(sys.argv[3]) #False #True
want_rsd = int(sys.argv[2]) #False
rsd_str = "_rsd" if want_rsd else ""

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
    vel = data['velocities'] # km/s
    unit_los = data['unit_los']
else:
    # load data
    data = np.load(f"{str(save_dir)}/{sim_name}/tmp/displacements_{tracer}_postrecon_z{Mean_z:.3f}.npz")
    print(data.files) # ['displacements', 'velocities', 'growth_factor', 'Hubble_z']
    Psi_dot = data['displacements'] # cMpc/h
    Psi_dot_rsd = data['displacements_rsd'] # cMpc/h
    POS = data['positions']
    POS_RSD = data['positions_rsd']
    RAND_POS = data['random_positions']
    vel = data['velocities'] # km/s
    unit_los = np.array([0., 0., 1.])#data['unit_los']

# compute displacements
f = data['growth_factor']
H = data['Hubble_z'] # km/s/Mpc
Psi_dot *= a*f*H/h # km/s
Psi_dot_rsd *= a*f*H/h # km/s
Psi_dot_r = np.sum(Psi_dot*unit_los, axis=1)
Psi_dot_rsd_r = np.sum(Psi_dot_rsd*unit_los, axis=1)
Psi_dot_rsd_r /= (1+f) # real space
vel_r = np.sum(vel*unit_los, axis=1)
print("true rms", np.std(vel_r))
print("recon rms (rsd and no rsd)", np.std(Psi_dot_rsd_r), np.std(Psi_dot_r))

# things to do: why is rsd better than no rsd? how to select one perpendicular vector so you can quote rms for perp; rsd = 1 and 0 difference in theory and practice; also what about the 1+delta

# load the fields
positions_rec = {}
if want_rsd:
    positions_rec['data'] = POS_RSD
    positions_rec['weight1'] = Psi_dot_rsd_r
else:
    positions_rec['data'] = POS
    positions_rec['weight1'] = Psi_dot_r
positions_rec['randoms'] = RAND_POS
positions_rec['weight2'] = vel_r

# galaxy positions
print(POS_RSD.min(), POS_RSD.max(), RAND_POS.min(), RAND_POS.max(), POS.min(), POS.max())

# randomize the randoms
inds = np.arange(RAND_POS.shape[0], dtype=int)
np.random.shuffle(inds)
positions_rec['randoms'] = positions_rec['randoms'][inds]

if want_pk:
    # power params
    nbins_k = 200
    nbins_mu = 50
    k_hMpc_max = 1.
    nmesh = 512
    if box_lc == "lc":
        los = 'firstpoint'
        boxcenter = np.array([0., 0., 0.])
        boxsize = None
        wrap = False
        kedges = np.linspace(0.00, k_hMpc_max, nbins_k+1)
    else:
        los = 'z'
        boxcenter = np.array([Lbox/2., Lbox/2., Lbox/2.])
        boxsize = Lbox
        wrap = True
        kedges = (np.linspace(0.00, k_hMpc_max, nbins_k+1), np.linspace(-1, 1, nbins_mu*2+1))
        
    """
    # compute the galaxy density field and interpolate at the positions of the galaxies
    print("computing tsc")
    dens_gal = np.zeros((nmesh, nmesh, nmesh), dtype=np.float32)
    print("computed tsc")
    numba_tsc_3D(positions_rec['data'], dens_gal, Lbox)
    one_plus_delta = dens_gal/np.mean(dens_gal) # Lehman said 64 for mean
    print(np.mean(dens_gal), positions_rec['data'].shape)
    cell = Lbox/nmesh
    pos_ijk = (np.round(positions_rec['data']/cell)).astype(int)
    print(pos_ijk.max(), pos_ijk.min())
    pos_ijk %= nmesh
    positions_rec['weight1'] /= one_plus_delta[pos_ijk[:, 0], pos_ijk[:, 1], pos_ijk[:, 2]]
    positions_rec['weight2'] /= one_plus_delta[pos_ijk[:, 0], pos_ijk[:, 1], pos_ijk[:, 2]]
    print("changed weights")
    """
    # CatalogMesh
    #mesh = CatalogMesh(data_positions=data['Position'], boxsize=boxsize, nmesh=nmesh, resampler=resampler, interlacing=interlacing, position_type='pos', dtype=dtype).to_mesh()
    # compute one mesh with vz and one with just normal; divide the two and call this mesh_recon and then same for mesh_truth and then cross-correlate with MeshFFTPower
    
    """
    # abacus power parameters
    logk = False
    lbox = Lbox
    paste = "TSC"
    num_cells = nmesh
    compensated = True
    interlaced = False #True
    k_binc, mu_binc, pk3d_recon, N3d, binned_poles, Npoles = calc_power(*positions_rec['data'].T, nbins_k=nbins_k, nbins_mu=nbins_mu, k_hMpc_max=k_hMpc_max, logk=logk, lbox=lbox, paste=paste, num_cells=num_cells, compensated=compensated, interlaced=interlaced, w=positions_rec['weight1'], x2=None, y2=None, z2=None, w2=None, poles=[])
    print("done with recon")
    k_binc, mu_binc, pk3d_cross, N3d, binned_poles, Npoles = calc_power(*positions_rec['data'].T, nbins_k=nbins_k, nbins_mu=nbins_mu, k_hMpc_max=k_hMpc_max, logk=logk, lbox=lbox, paste=paste, num_cells=num_cells, compensated=compensated, interlaced=interlaced, w=positions_rec['weight1'], x2=positions_rec['data'][:, 0], y2=positions_rec['data'][:, 1], z2=positions_rec['data'][:, 2], w2=positions_rec['weight2'], poles=[])
    print("done with cross")
    k_binc, mu_binc, pk3d_truth, N3d, binned_poles, Npoles = calc_power(*positions_rec['data'].T, nbins_k=nbins_k, nbins_mu=nbins_mu, k_hMpc_max=k_hMpc_max, logk=logk, lbox=lbox, paste=paste, num_cells=num_cells, compensated=compensated, interlaced=interlaced, w=positions_rec['weight2'], x2=None, y2=None, z2=None, w2=None, poles=[])
    print("done with truth")
    np.savez(f"/global/cfs/cdirs/desi/users/boryanah/kSZ_recon/AbacusSummit_base_c000_ph002/tmp/vel{rsd_str}_abacus_power_{box_lc}.npz", k_binc=k_binc, mu_binc=mu_binc, pk3d_recon=pk3d_recon, pk3d_cross=pk3d_cross, pk3d_truth=pk3d_truth, N3d=N3d)
    quit()
    """
    
    # compute power spectra
    poles_cross = CatalogFFTPower(data_positions1=positions_rec['data'], data_weights1=positions_rec['weight1'], randoms_positions1=positions_rec['randoms'] if boxsize is None else None,
                                  data_positions2=positions_rec['data'], data_weights2=positions_rec['weight2'], randoms_positions2=positions_rec['randoms'] if boxsize is None else None,
                                  nmesh=nmesh, resampler='tsc', boxsize=boxsize, boxcenter=boxcenter, interlacing=3, ells=(0,2,4), wrap=wrap,
                                  los=los, edges=kedges, position_type='pos', mpiroot=0)
    print("done with cross")
    if box_lc != 'lc':
        poles_cross.wedges.wnorm = (positions_rec['data'].shape[0]/Lbox**3)**2*Lbox**3
    poles_cross.poles.wnorm = (positions_rec['data'].shape[0]/Lbox**3)**2*Lbox**3
    # integral of the density squared; density squared times volume for cubic box (otherwise sum of weights squared divided by volume squared times volume)
    fn = f"/global/cfs/cdirs/desi/users/boryanah/kSZ_recon/AbacusSummit_base_c000_ph002/tmp/vel{rsd_str}_cross_poles_{box_lc}.npy"
    poles_cross.save(fn)

    poles_recon = CatalogFFTPower(data_positions1=positions_rec['data'], data_weights1=positions_rec['weight1'], randoms_positions1=positions_rec['randoms'] if boxsize is None else None,
                                  nmesh=nmesh, resampler='tsc', boxsize=boxsize, boxcenter=boxcenter, interlacing=3, ells=(0,2,4), wrap=wrap,
                                  los=los, edges=kedges, position_type='pos', mpiroot=0)#.poles
    print("done with recon")
    if box_lc != 'lc':
        poles_recon.wedges.wnorm = (positions_rec['data'].shape[0]/Lbox**3)**2*Lbox**3
    poles_recon.poles.wnorm = (positions_rec['data'].shape[0]/Lbox**3)**2*Lbox**3
    poles_recon.poles.shotnoise_nonorm = 0.
    if box_lc != 'lc':
        poles_recon.wedges.shotnoise_nonorm = 0.
    fn = f"/global/cfs/cdirs/desi/users/boryanah/kSZ_recon/AbacusSummit_base_c000_ph002/tmp/vel{rsd_str}_recon_poles_{box_lc}.npy"
    poles_recon.save(fn)

    poles_truth = CatalogFFTPower(data_positions1=positions_rec['data'], data_weights1=positions_rec['weight2'], randoms_positions1=positions_rec['randoms'] if boxsize is None else None,
                                  nmesh=nmesh, resampler='tsc', boxsize=boxsize, boxcenter=boxcenter, interlacing=3, ells=(0,2,4), wrap=wrap,
                                  los=los, edges=kedges, position_type='pos', mpiroot=0)#.poles
    print("done with truth")
    if box_lc != 'lc':
        poles_truth.wedges.wnorm = (positions_rec['data'].shape[0]/Lbox**3)**2*Lbox**3
    poles_truth.poles.wnorm = (positions_rec['data'].shape[0]/Lbox**3)**2*Lbox**3
    poles_truth.poles.shotnoise_nonorm = 0.
    if box_lc != 'lc':
        poles_truth.wedges.shotnoise_nonorm = 0.
    fn = f"/global/cfs/cdirs/desi/users/boryanah/kSZ_recon/AbacusSummit_base_c000_ph002/tmp/vel{rsd_str}_truth_poles_{box_lc}.npy"
    poles_truth.save(fn)

else:

    # bins corr func
    R = 150.
    rbins = np.linspace(0., R, int(R)+1)
    rbinc = (rbins[1:] + rbins[:-1])/2.
    pimax = R
    npibins = int(R)
    
    # corrfunc params
    ncpu = 64
    #edges = (np.linspace(0.01, 200, 201), np.linspace(-1, 1, 201))
    #edges = (np.linspace(1, 201, 101), np.linspace(-50, 50, 101))
    edges = (rbins, np.linspace(-R, R, int(R)+1))
    if box_lc == "lc":
        los = 'midpoint'
        nsplits = 40
        D1D2_cross = None
        D1D2_truth = None
        D1D2_recon = None
        result_cross = 0
        result_truth = 0
        result_recon = 0
        size_randoms1 = positions_rec['randoms'].shape[0]
    else:
        los = 'z'
        boxsize = Lbox

    if box_lc == "lc":
        for isplit in range(nsplits):
            print('Split {:d}/{:d}.'.format(isplit + 1, nsplits))
            sl1 = slice(isplit * size_randoms1 // nsplits, (isplit + 1) * size_randoms1 // nsplits)
            
            """
            result_cross += TwoPointCorrelationFunction('rppi', edges, data_positions1=positions_rec['data'], data_weights1=positions_rec['weight1'], data_positions2=positions_rec['data'], data_weights2=positions_rec['weight2'], los=los, randoms_positions1=[p2[sl1] for p2 in positions_rec['randoms']], randoms_positions2=[p2[sl1] for p2 in positions_rec['randoms']], engine='corrfunc', D1D2=D1D2_cross, nthreads=ncpu, D1D2_weight_type='product_individual', position_type='pos')
            print("done with cross")
            D1D2_cross = result_cross.D1D2

            result_truth += TwoPointCorrelationFunction('rppi', edges, data_positions1=positions_rec['data'], data_weights1=positions_rec['weight2'], los=los, randoms_positions1=[p2[sl1] for p2 in positions_rec['randoms']], randoms_positions2=[p2[sl1] for p2 in positions_rec['randoms']], engine='corrfunc', D1D2=D1D2_truth, nthreads=ncpu, D1D2_weight_type='product_individual', position_type='pos')
            print("done with truth")
            D1D2_truth = result_truth.D1D2

            result_recon += TwoPointCorrelationFunction('rppi', edges, data_positions1=positions_rec['data'], data_weights1=positions_rec['weight1'], los=los, randoms_positions1=[p2[sl1] for p2 in positions_rec['randoms']], randoms_positions2=[p2[sl1] for p2 in positions_rec['randoms']], engine='corrfunc', D1D2=D1D2_recon, nthreads=ncpu, D1D2_weight_type='product_individual', position_type='pos')
            print("done with recon")
            D1D2_recon = result_recon.D1D2
            """
            result_cross += TwoPointCorrelationFunction('rppi', edges, data_positions1=np.transpose(positions_rec['data']), data_weights1=positions_rec['weight1'], data_positions2=np.transpose(positions_rec['data']), data_weights2=positions_rec['weight2'], los=los, randoms_positions1=[p2[sl1] for p2 in np.transpose(positions_rec['randoms'])], randoms_positions2=[p2[sl1] for p2 in np.transpose(positions_rec['randoms'])], engine='corrfunc', D1D2=D1D2_cross, nthreads=ncpu, D1D2_weight_type='product_individual', position_type='xyz')
            print("done with cross")
            D1D2_cross = result_cross.D1D2

            result_truth += TwoPointCorrelationFunction('rppi', edges, data_positions1=np.transpose(positions_rec['data']), data_weights1=positions_rec['weight2'], los=los, randoms_positions1=[p2[sl1] for p2 in np.transpose(positions_rec['randoms'])], randoms_positions2=[p2[sl1] for p2 in np.transpose(positions_rec['randoms'])], engine='corrfunc', D1D2=D1D2_truth, nthreads=ncpu, D1D2_weight_type='product_individual', position_type='xyz')
            print("done with truth")
            D1D2_truth = result_truth.D1D2

            result_recon += TwoPointCorrelationFunction('rppi', edges, data_positions1=np.transpose(positions_rec['data']), data_weights1=positions_rec['weight1'], los=los, randoms_positions1=[p2[sl1] for p2 in np.transpose(positions_rec['randoms'])], randoms_positions2=[p2[sl1] for p2 in np.transpose(positions_rec['randoms'])], engine='corrfunc', D1D2=D1D2_recon, nthreads=ncpu, D1D2_weight_type='product_individual', position_type='xyz')
            print("done with recon")
            D1D2_recon = result_recon.D1D2
            
        N1 = positions_rec['data'].shape[0]
        N2 = positions_rec['data'].shape[0]
        result_cross.D1D2.wnorm = N1*N2
        result_recon.D1D2.wnorm = N1*N2
        result_truth.D1D2.wnorm = N1*N2
        result_cross.run()
        result_recon.run()
        result_truth.run()
    else:
        """
        # compute correlation
        N1 = positions_rec['data'].shape[0]
        N2 = positions_rec['data'].shape[0]
        autocorr = 0
        nthreads = 128
        results = DDrppi(autocorr=autocorr, nthreads=nthreads, pimax=pimax, npibins=npibins, binfile=rbins, X1=positions_rec['data'][:, 0], Y1=positions_rec['data'][:, 1], Z1=positions_rec['data'][:, 2], weights1=positions_rec['weight1'], X2=positions_rec['data'][:, 0], Y2=positions_rec['data'][:, 1], Z2=positions_rec['data'][:, 2], weights2=positions_rec['weight2'], periodic=True, boxsize=Lbox, weight_type="pair_product")
        D1D2 = (results['npairs']*results['weightavg']).astype(float)/(N1*N2)
        D1D2 = D1D2.reshape(len(rbinc), npibins)
        D1D2_now = (results['npairs']).astype(float)/(N1*N2)
        D1D2_now = D1D2_now.reshape(len(rbinc), npibins)
        R1R2 = get_RRrppi(N1, N2, Lbox, rbins, pimax, npibins)/(N1*N2)
        print(R1R2.shape)
        xi = D1D2/R1R2 - 1.
        xi_now = D1D2_now/R1R2 - 1.
        npairs = (results['npairs']).reshape(len(rbinc), npibins)
        print("cross done")
        np.savez(f"data/corr_vel{rsd_str}_cross_{box_lc}.npz", rbinc=rbinc, pimax=pimax, xirppi=xi, xirppi_now=xi_now, DDrppi=D1D2, npairs=npairs)
        
        # compute correlation
        N1 = positions_rec['data'].shape[0]
        N2 = positions_rec['data'].shape[0]
        autocorr = 1
        nthreads = 128
        results = DDrppi(autocorr=autocorr, nthreads=nthreads, pimax=pimax, npibins=npibins, binfile=rbins, X1=positions_rec['data'][:, 0], Y1=positions_rec['data'][:, 1], Z1=positions_rec['data'][:, 2], weights1=positions_rec['weight1'], periodic=True, boxsize=Lbox, weight_type="pair_product")
        D1D2 = (results['npairs']*results['weightavg']).astype(float)/(N1*N2)
        D1D2 = D1D2.reshape(len(rbinc), npibins)
        D1D2_now = (results['npairs']).astype(float)/(N1*N2)
        D1D2_now = D1D2_now.reshape(len(rbinc), npibins)
        R1R2 = get_RRrppi(N1, N2, Lbox, rbins, pimax, npibins)/(N1*N2)
        print(R1R2.shape)
        xi = D1D2/R1R2 - 1.
        xi_now = D1D2_now/R1R2 - 1.
        npairs = (results['npairs']).reshape(len(rbinc), npibins)
        print("recon done")
        np.savez(f"data/corr_vel{rsd_str}_recon_{box_lc}.npz", rbinc=rbinc, pimax=pimax, xirppi=xi, xirppi_now=xi_now, DDrppi=D1D2, npairs=npairs)

        # compute correlation
        N1 = positions_rec['data'].shape[0]
        N2 = positions_rec['data'].shape[0]
        autocorr = 1
        nthreads = 128
        results = DDrppi(autocorr=autocorr, nthreads=nthreads, pimax=pimax, npibins=npibins, binfile=rbins, X1=positions_rec['data'][:, 0], Y1=positions_rec['data'][:, 1], Z1=positions_rec['data'][:, 2], weights1=positions_rec['weight2'], periodic=True, boxsize=Lbox, weight_type="pair_product")
        D1D2 = (results['npairs']*results['weightavg']).astype(float)/(N1*N2)
        D1D2 = D1D2.reshape(len(rbinc), npibins)
        D1D2_now = (results['npairs']).astype(float)/(N1*N2)
        D1D2_now = D1D2_now.reshape(len(rbinc), npibins)
        R1R2 = get_RRrppi(N1, N2, Lbox, rbins, pimax, npibins)/(N1*N2)
        print(R1R2.shape)
        xi = D1D2/R1R2 - 1.
        xi_now = D1D2_now/R1R2 - 1.
        npairs = (results['npairs']).reshape(len(rbinc), npibins)
        print("truth done")
        np.savez(f"data/corr_vel{rsd_str}_truth_{box_lc}.npz", rbinc=rbinc, pimax=pimax, xirppi=xi, xirppi_now=xi_now, DDrppi=D1D2, npairs=npairs)
        quit()
        """

        N1 = positions_rec['data'].shape[0]
        N2 = positions_rec['data'].shape[0]
        
        result_cross = TwoPointCorrelationFunction('rppi', edges, data_positions1=positions_rec['data'], data_weights1=positions_rec['weight1'], data_positions2=positions_rec['data'], data_weights2=positions_rec['weight2'], los=los, randoms_positions1=None, engine='corrfunc', boxsize=boxsize, nthreads=ncpu, D1D2_weight_type='product_individual', position_type='pos')

        result_truth = TwoPointCorrelationFunction('rppi', edges, data_positions1=positions_rec['data'], data_weights1=positions_rec['weight2'], los=los, randoms_positions1=None, engine='corrfunc', boxsize=boxsize, nthreads=ncpu, D1D2_weight_type='product_individual', position_type='pos')

        result_recon = TwoPointCorrelationFunction('rppi', edges, data_positions1=positions_rec['data'], data_weights1=positions_rec['weight1'], los=los, randoms_positions1=None, engine='corrfunc', boxsize=boxsize, nthreads=ncpu, D1D2_weight_type='product_individual', position_type='pos')

        result_cross.D1D2.wnorm = N1*N2
        result_recon.D1D2.wnorm = N1*N2
        result_truth.D1D2.wnorm = N1*N2
        result_cross.run()
        result_recon.run()
        result_truth.run()
        
    fn = f"/global/cfs/cdirs/desi/users/boryanah/kSZ_recon/AbacusSummit_base_c000_ph002/tmp/result_vel{rsd_str}_cross_{box_lc}.npy"
    result_cross.save(fn)

    fn = f"/global/cfs/cdirs/desi/users/boryanah/kSZ_recon/AbacusSummit_base_c000_ph002/tmp/result_vel{rsd_str}_recon_{box_lc}.npy"
    result_recon.save(fn)

    fn = f"/global/cfs/cdirs/desi/users/boryanah/kSZ_recon/AbacusSummit_base_c000_ph002/tmp/result_vel{rsd_str}_truth_{box_lc}.npy"
    result_truth.save(fn)
