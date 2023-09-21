from pathlib import Path
import sys, gc

import numpy as np
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from abacusnbody.metadata import get_meta

from pycorr import TwoPointCorrelationFunction, TwoPointEstimator, NaturalTwoPointEstimator, project_to_multipoles, project_to_wp

from pypower import CatalogMesh, MeshFFTPower, CatalogFFTPower, ArrayMesh, CatalogFFTCorr

from apply_dndz_mask import parse_nztype
from reconstruct_lc_catalog import get_nz_str

"""
run with srun

note that you can't do wedges with lc
# integral of the density squared; density squared times volume for cubic box (otherwise sum of weights squared divided by volume squared times volume)

Usage:
python correlate_velocity.py --sim_name AbacusSummit_base_c000_ph002 --box_lc lc --stem DESI_LRG --nmesh 1024 --sr 12.5 --rectype MG --convention recsym --want_pk --want_rsd --want_mask --nz_type num_dens/main_lrg_pz_dndz_iron_v0.npz
"""

np.random.seed(300)

DEFAULTS = {}
DEFAULTS['sim_name'] = "AbacusSummit_base_c000_ph002"
DEFAULTS['box_lc'] = 'lc' #'box'
DEFAULTS['stem'] = 'DESI_LRG'
DEFAULTS['nmesh'] = 1024
DEFAULTS['sr'] = 12.5 # Mpc/h
DEFAULTS['rectype'] = "MG"
DEFAULTS['convention'] = "recsym"
DEFAULTS['nz_type'] = 'Gaussian(0.5, 0.4)'
DEFAULTS['photoz_error'] = 0.

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

def main(sim_name, box_lc, stem, stem2, stem3, nmesh, sr, rectype, convention, nz_type, nz_type2, nz_type3, photoz_error, want_fakelc=False, want_mask=False, want_pk=False, want_rsd=False):
    want_curb = False #True
    if want_curb:
        curb_zmin = 0.4
        curb_zmax = 0.8
        curb_str = f"_curbzmin{curb_zmin:.1f}_curbzmax{curb_zmax:.1f}"
    else:
        curb_str = ""
        
    # rsd_str
    rsd_str = "_rsd" if want_rsd else ""
    
    # redshift error string
    photoz_str = f"_zerr{photoz_error:.1f}"
    
    # fake light cones
    fakelc_str = "_fakelc" if want_fakelc else ""
    mask_str = "_mask" if want_mask else ""
    
    # it only makes sense to cut the edges in the light cone case
    if want_fakelc:
        assert box_lc == "lc"
    
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

    # get nz_full_str if multitracer
    nz_str = get_nz_str(nz_type)
    nz_strs = [nz_str]
    if nz_type2 is not None:
        nz_strs.append(get_nz_str(nz_type2))
    if nz_type3 is not None:
        nz_strs.append(get_nz_str(nz_type3))
    nz_full_str = ''.join(nz_strs)
    
    # parameters
    h = get_meta(sim_name, 0.1)['H0']/100.
    Lbox = get_meta(sim_name, 0.1)['BoxSize'] # cMpc/h
    a = 1./(1+Mean_z)
    
    # directories
    save_dir = Path("/global/cfs/cdirs/desi/users/boryanah/kSZ_recon/new/")
    save_recon_dir = Path(save_dir) / "recon" / sim_name / f"z{Mean_z:.3f}"
    save_dir_old = Path("/pscratch/sd/b/boryanah/kSZ_recon/")
    save_recon_dir_old = Path(save_dir_old) / "recon" / sim_name / f"z{Mean_z:.3f}"
    mock_dir = Path(f"/global/cfs/cdirs/desi/users/boryanah/kSZ_recon/new/mocks_lc_output_kSZ_recon{extra}/") / sim_name 

    if box_lc == "lc":
        # load galaxies and randoms
        data = np.load(mock_dir / f"galaxies_{tracer}{fakelc_str}{photoz_str}{mask_str}_prerecon{nz_str}.npz")
        POS = data['POS']
        POS_RSD = data['POS_RSD']

        if want_curb:
            Z = data['Z']
        
        data = np.load(mock_dir / f"randoms_{tracer}{fakelc_str}{photoz_str}{mask_str}_prerecon{nz_str}.npz")
        RAND_POS = data['RAND_POS']

        if want_curb:
            RAND_Z = data['RAND_Z']
        
        # shift origin
        origin = np.array([-990., -990, -990.])
        POS -= origin
        POS_RSD -= origin
        RAND_POS -= origin

        # load data
        data = np.load(save_recon_dir / f"displacements_{tracer}{extra}{fakelc_str}{photoz_str}{mask_str}_postrecon_R{sr:.2f}_b{bias:.1f}_nmesh{nmesh:d}_{convention}_{rectype}{nz_full_str}.npz")
            

        try:
            n_gal = data['n_gals'][0]
        except:
            print("n_gals doesn't exist (older file probably)")
            n_gal = data['displacements'].shape[0]
                
        Psi_dot = data['displacements'][:n_gal] # cMpc/h
        Psi_dot_rsd = data['displacements_rsd'][:n_gal] # cMpc/h
        Psi_dot_rsd_nof = data['displacements_rsd_nof'][:n_gal] # cMpc/h
        vel = data['velocities'][:n_gal] # km/s
        unit_los = data['unit_los'][:n_gal]
        unit_perp1 = np.vstack((unit_los[:, 2], unit_los[:, 2], -unit_los[:, 1] -unit_los[:, 0])).T
        unit_perp2 = np.vstack((unit_los[:, 1]*unit_perp1[:, 2] - unit_los[:, 2]*unit_perp1[:, 1],
                                -(unit_los[:, 0]*unit_perp1[:, 2] - unit_los[:, 2]*unit_perp1[:, 0]),
                                unit_los[:, 0]*unit_perp1[:, 1] - unit_los[:, 1]*unit_perp1[:, 0])).T
        unit_perp1 /= np.linalg.norm(unit_perp1, axis=1)[:, None]
        unit_perp2 /= np.linalg.norm(unit_perp2, axis=1)[:, None]
        vel_r = np.sum(vel*unit_los, axis=1)
        vel_p1 = np.sum(vel*unit_perp1, axis=1)
        vel_p2 = np.sum(vel*unit_perp2, axis=1)
        vel = np.vstack((vel_p1, vel_p2, vel_r)).T
        
        
    else:
        want_make_thinner = False
        thin_str = "_thin" if want_make_thinner else ""
        
        # load data
        data = np.load(save_recon_dir_old / f"displacements_{tracer}{extra}_postrecon_R{sr:.2f}_b{bias:.1f}_nmesh{nmesh:d}_{convention}_{rectype}_z{Mean_z:.3f}{thin_str}.npz")
        
        print(data.files) # ['displacements', 'velocities', 'growth_factor', 'Hubble_z']
        Psi_dot = data['displacements'] # cMpc/h
        Psi_dot_rsd = data['displacements_rsd'] # cMpc/h
        Psi_dot_rsd_nof = data['displacements_rsd_nof'] # cMpc/h
        POS = data['positions']
        POS_RSD = data['positions_rsd']
        RAND_POS = data['random_positions']
        vel = data['velocities'] # km/s

        # use halo vel
        mock_dir = Path(f"/pscratch/sd/b/boryanah/AbacusHOD_scratch/mocks_box_output_kSZ_recon{extra}/")
        mock_dir = mock_dir / f"{sim_name}/z{Mean_z:.3f}/"
        vel = np.load(mock_dir / f"galaxies/{tracer}s_halo_vel.npy") # box only

        # TESTING make thinner
        from astropy.io import ascii
        f = ascii.read(mock_dir / f"galaxies/{tracer}s.dat")
        positions = np.vstack((f['x'], f['y'], f['z'])).T
        positions %= Lbox
        
        if want_make_thinner:
            z_max = 830.
            choice = positions[:, 2] < z_max
            vel = vel[choice]
            
        unit_los = np.array([0., 0., 1.])#data['unit_los']
        vel_r = np.sum(vel*unit_los, axis=1)
        
    def get_Psi_dot(Psi, want_rsd=False):
        """
        Internal function for calculating the ZA velocity given the displacement.
        """
        Psi_dot = Psi*a*f*H # km/s/h
        Psi_dot /= h # km/s
        Psi_dot_r = np.sum(Psi_dot*unit_los, axis=1)
        if want_rsd:
            Psi_dot_r /= (1+f) # real space
        if box_lc == "lc":
            Psi_dot_p1 = np.sum(Psi_dot*unit_perp1, axis=1)
            Psi_dot_p2 = np.sum(Psi_dot*unit_perp2, axis=1)            
            Psi_dot = np.vstack((Psi_dot_p1, Psi_dot_p2, Psi_dot_r)).T
        else:
            Psi_dot = np.vstack((Psi_dot[:, 0], Psi_dot[:, 1], Psi_dot_r)).T
        N = Psi_dot.shape[0]
        assert len(vel_r) == vel.shape[0] == len(Psi_dot_r) == N
        return Psi_dot, Psi_dot_r, N
        
    # compute displacements
    f = data['growth_factor']
    H = data['Hubble_z'] # km/s/Mpc

    # turn displacements into velocities
    Psi_dot, Psi_dot_r, N = get_Psi_dot(Psi_dot, want_rsd=False)
    Psi_dot_rsd_nof, Psi_dot_rsd_nof_r, N = get_Psi_dot(Psi_dot_rsd_nof, want_rsd=False)

    if box_lc == "lc" and want_curb:
        choice = (curb_zmin < RAND_Z) & (RAND_Z < curb_zmax)
        RAND_POS = RAND_POS[choice]
        
        choice = (curb_zmin < Z) & (Z < curb_zmax)
        print("number of galaxies", np.sum(choice), len(choice))
        POS_RSD = POS_RSD[choice]
        POS = POS[choice]
        Psi_dot_rsd_nof_r = Psi_dot_rsd_nof_r[choice]
        Psi_dot_r = Psi_dot_r[choice]
        vel_r = vel_r[choice]
        del choice; gc.collect()

    # print the usual r coefficient
    Psi_dot_rsd_nof_r_rms = np.sqrt(np.mean(Psi_dot_rsd_nof_r**2))
    Psi_dot_r_rms = np.sqrt(np.mean(Psi_dot_r**2))
    vel_r_rms = np.sqrt(np.mean(vel_r**2))
    print("LOS R", np.mean(Psi_dot_r*vel_r)/(vel_r_rms*Psi_dot_r_rms))
    print("LOS R (RSD)", np.mean(Psi_dot_rsd_nof_r*vel_r)/(vel_r_rms*Psi_dot_rsd_nof_r_rms))
    
    # load the fields
    positions_rec = {}
    if want_rsd:
        positions_rec['data'] = POS_RSD
        positions_rec['weight1'] = Psi_dot_rsd_nof_r
    else:
        positions_rec['data'] = POS
        positions_rec['weight1'] = Psi_dot_r
    positions_rec['randoms'] = RAND_POS
    positions_rec['weight2'] = vel_r

    # galaxy positions
    print(POS_RSD.min(), POS_RSD.max(), RAND_POS.min(), RAND_POS.max(), POS.min(), POS.max())

    if not want_pk:
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

        # compute power spectra
        poles_cross = CatalogFFTPower(data_positions1=positions_rec['data'], data_weights1=positions_rec['weight1'], randoms_positions1=positions_rec['randoms'] if boxsize is None else None,
                                      data_positions2=positions_rec['data'], data_weights2=positions_rec['weight2'], randoms_positions2=positions_rec['randoms'] if boxsize is None else None,
                                      nmesh=nmesh, resampler='tsc', boxsize=boxsize, boxcenter=boxcenter, interlacing=3, ells=(0,2,4), wrap=wrap,
                                      los=los, edges=kedges, position_type='pos', mpiroot=0)
        print("done with cross")
        if box_lc != 'lc':
            poles_cross.wedges.wnorm = (positions_rec['data'].shape[0]/Lbox**3)**2*Lbox**3
        poles_cross.poles.wnorm = (positions_rec['data'].shape[0]/Lbox**3)**2*Lbox**3
        #poles_cross.save(f"data/vel{rsd_str}_cross_poles_{box_lc}.npy")
        
        poles_recon = CatalogFFTPower(data_positions1=positions_rec['data'], data_weights1=positions_rec['weight1'], randoms_positions1=positions_rec['randoms'] if boxsize is None else None,
                                      nmesh=nmesh, resampler='tsc', boxsize=boxsize, boxcenter=boxcenter, interlacing=3, ells=(0,2,4), wrap=wrap,
                                      los=los, edges=kedges, position_type='pos', mpiroot=0)
        print("done with recon")
        if box_lc != 'lc':
            poles_recon.wedges.wnorm = (positions_rec['data'].shape[0]/Lbox**3)**2*Lbox**3
        poles_recon.poles.wnorm = (positions_rec['data'].shape[0]/Lbox**3)**2*Lbox**3
        poles_recon.poles.shotnoise_nonorm = 0.
        if box_lc != 'lc':
            poles_recon.wedges.shotnoise_nonorm = 0.
        #poles_recon.save(f"data/vel{rsd_str}_recon_poles_{box_lc}.npy")

        poles_truth = CatalogFFTPower(data_positions1=positions_rec['data'], data_weights1=positions_rec['weight2'], randoms_positions1=positions_rec['randoms'] if boxsize is None else None,
                                      nmesh=nmesh, resampler='tsc', boxsize=boxsize, boxcenter=boxcenter, interlacing=3, ells=(0,2,4), wrap=wrap,
                                      los=los, edges=kedges, position_type='pos', mpiroot=0)
        print("done with truth")
        if box_lc != 'lc':
            poles_truth.wedges.wnorm = (positions_rec['data'].shape[0]/Lbox**3)**2*Lbox**3
        poles_truth.poles.wnorm = (positions_rec['data'].shape[0]/Lbox**3)**2*Lbox**3
        poles_truth.poles.shotnoise_nonorm = 0.
        if box_lc != 'lc':
            poles_truth.wedges.shotnoise_nonorm = 0.
        #poles_truth.save(f"data/vel{rsd_str}_truth_poles_{box_lc}.npy")

        # save the power spectra
        k = poles_cross.poles.k
        ells = np.array([0, 2, 4])
        p_cross_ell = np.zeros((len(ells), len(k)))
        p_truth_ell = np.zeros((len(ells), len(k)))
        p_recon_ell = np.zeros((len(ells), len(k)))
        for ill in range(len(ells)):
            p_cross_ell[ill] = poles_cross.poles(ell=ells[ill], complex=False)
            p_truth_ell[ill] = poles_truth.poles(ell=ells[ill], complex=False)
            p_recon_ell[ill] = poles_recon.poles(ell=ells[ill], complex=False)
        r_ell = p_cross_ell/np.sqrt(p_truth_ell*p_recon_ell)
        if box_lc == "lc":
            fn = f"data/power_poles_{tracer}{extra}{fakelc_str}{photoz_str}{mask_str}_postrecon_R{sr:.2f}_b{bias:.1f}_nmesh{nmesh:d}_{convention}_{rectype}{nz_full_str}{rsd_str}{curb_str}.npz"
        else:
            fn = f"data/power_poles_{tracer}{extra}_postrecon_R{sr:.2f}_b{bias:.1f}_nmesh{nmesh:d}_{convention}_{rectype}_z{Mean_z:.3f}{rsd_str}.npz"
        np.savez(fn, k=k, p_cross_ell=p_cross_ell, p_truth_ell=p_truth_ell, p_recon_ell=p_recon_ell, ells=ells)
        
        if box_lc != 'lc':
            muavg = poles_cross.wedges.muavg
            k, Pk_cross = poles_cross.wedges(mu=np.max(muavg), return_k=True, complex=False)
            k, Pk_truth = poles_truth.wedges(mu=np.max(muavg), return_k=True, complex=False)
            k, Pk_recon = poles_recon.wedges(mu=np.max(muavg), return_k=True, complex=False)
            rk = Pk_cross/np.sqrt(Pk_truth*Pk_recon)
            
            np.savez(f"data/power_wedges_{tracer}{extra}_postrecon_R{sr:.2f}_b{bias:.1f}_nmesh{nmesh:d}_{convention}_{rectype}_z{Mean_z:.3f}{thin_str}{rsd_str}.npz", k=k, Pk_cross=Pk_cross, Pk_truth=Pk_truth, Pk_recon=Pk_recon)
            
    else:

        # bins corr func
        R = 150.
        rbins = np.linspace(0., R, int(R)+1)
        rbinc = (rbins[1:] + rbins[:-1])/2.
        pimax = R
        npibins = int(R)

        # corrfunc params
        ncpu = 64
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

class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--sim_name', help='Simulation name', default=DEFAULTS['sim_name'])
    parser.add_argument('--box_lc', help='Cubic box or light cone?', default=DEFAULTS['box_lc'])
    parser.add_argument('--stem', help='Stem file name', default=DEFAULTS['stem'])
    parser.add_argument('--stem2', help='Stem file name 2', default=None)
    parser.add_argument('--stem3', help='Stem file name 3', default=None)
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
    parser.add_argument('--want_rsd', help='Want to use RSD effects?', action='store_true')
    parser.add_argument('--want_pk', help='Want to compute power spectrum (else, correlation function)?', action='store_true')
    args = vars(parser.parse_args())
    main(**args)
