from pathlib import Path
import sys
import os

import numpy as np
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from abacusnbody.metadata import get_meta

# needed only for want_z_evolve
from cosmoprimo.fiducial import Planck2018FullFlatLCDM, AbacusSummit, DESI, TabulatedDESI

from apply_dndz_mask import parse_nztype
from reconstruct_lc_catalog import get_nz_str

DEFAULTS = {}
DEFAULTS['sim_name'] = "AbacusSummit_base_c000_ph002"
DEFAULTS['box_lc'] = 'lc'
DEFAULTS['stem'] = 'DESI_LRG' # 'DESI_ELG'
DEFAULTS['nmesh'] = 1024
DEFAULTS['sr'] = 12.5 # Mpc/h
DEFAULTS['rectype'] = "MG"
DEFAULTS['convention'] = "recsym"
DEFAULTS['nz_type'] = 'Gaussian(0.5, 0.4)'
DEFAULTS['bz_type'] = 'Constant'
DEFAULTS['photoz_error'] = 0.

"""
Usage:
python analyze_reconstruction.py --sim_name AbacusSummit_base_c000_ph002 --box_lc box --stem DESI_LRG --nmesh 512 --sr 10. --rectype MG --convention recsym
python analyze_reconstruction.py --sim_name AbacusSummit_base_c000_ph002 --box_lc box --stem DESI_LRG --nmesh 1024 --sr 12.5 --rectype MG --convention recsym
python analyze_reconstruction.py --sim_name AbacusSummit_base_c000_ph002 --box_lc lc --stem DESI_LRG --nmesh 1024 --sr 12.5 --rectype MG --convention recsym
python analyze_reconstruction.py --sim_name AbacusSummit_huge_c000_ph201 --box_lc lc --stem DESI_LRG --nmesh 1024 --sr 12.5 --rectype MG --convention recsym
"""

def main(sim_name, box_lc, stem, stem2, stem3, nmesh, sr, rectype, convention, nz_type, nz_type2, nz_type3, photoz_error, bz_type, cut_edges=False, want_z_evolve=False, want_fakelc=False, want_mask=False, want_save=False, test_name=""):
    # rename test
    order_str = test_name.split("_")[0] # r01, r02, r03, etc.
    test_name = ' '.join(test_name.split("_")[1:])
    print("test name", test_name)
    
    # redshift error string
    photoz_str = f"_zerr{photoz_error:.1f}"
    
    # fake light cones
    fakelc_str = "_fakelc" if want_fakelc else ""
    mask_str = "_mask" if want_mask else ""

    # small area (this is kinda stupid)
    if "_small_area" in nz_type:
        nz_type, small_str = nz_type.split("_small_area")
        mask_str = "_small_area"+small_str
    
    # it only makes sense to cut the edges in the light cone case
    if cut_edges or want_fakelc:
        assert box_lc == "lc"
        
    # initiate stems
    stems = [stem]
    nz_dict = parse_nztype(nz_type)
    print(sim_name, nz_type, nz_type2, nz_type3)
    nz_str = get_nz_str(nz_type)
    nz_strs = [nz_str]
    if stem2 is not None:
        stems.append(stem2)
        nz_strs.append(get_nz_str(nz_type2))
    if stem3 is not None:
        stems.append(stem3)
        nz_strs.append(get_nz_str(nz_type3))
    nz_full_str = ''.join(nz_strs)

    tracer_extra_str = []
    tracers = []
    for i in range(len(stems)):
        # additional specs of the tracer
        extra_this = '_'.join(stems[i].split('_')[2:])
        stem_this = '_'.join(stems[i].split('_')[:2])
        if extra_this != '': extra_this = '_'+extra_this

        # parameter choices
        if stem_this == 'DESI_LRG': 
            Mean_z = 0.5
            tracer_this = "LRG"
            tracers.append(tracer_this)
            bias = 2.2 # +/- 10%
        elif stem_this == 'DESI_ELG':
            Mean_z = 0.8
            tracer_this = "ELG"
            tracers.append(tracer_this)
            bias = 1.3
        else:
            print("Other tracers not yet implemented"); quit()
        if i == 0: tracer = tracer_this; extra = extra_this
        tracer_extra_str.append(f"{tracer_this}{extra_this}")
    tracer_extra_str = '_'.join(tracer_extra_str)
    print(tracer_extra_str)    

    if bz_type == "Constant":
        bias_str = f"_b{bias:.1f}"
    else:
        assert os.path.exists(bz_type), "Needs to be a file name"
        bias_str = f"_{(bz_type.split('/')[-1]).split('.npz')[0]}"
    
    if "-" in sim_name:
        sim_names = []
        sim_name_base = (sim_name.split('ph')[0])
        ph_lo, ph_hi = (sim_name.split('ph')[-1]).split('-')
        ph_lo, ph_hi = int(ph_lo), int(ph_hi)
        for ph in range(ph_lo, ph_hi+1):
            sim_names.append(f"{sim_name_base}ph{ph:03d}")
        print_stuff = False
    else:
        sim_names = [sim_name]
        print_stuff = True         
    
    # cosmology parameters
    h = get_meta(sim_names[0], 0.1)['H0']/100.
    if nz_dict['Type'] == "Gaussian":
        a = 1./(1+nz_dict['Mean_z'])
    else:
        a = 1./(1+Mean_z)
    
    def get_Psi_dot(Psi, want_rsd=False):
        """
        Internal function for calculating the ZA velocity given the displacement.
        """
        if want_z_evolve:
            Psi_dot = Psi*a[:, None]*f[:, None]*H[:, None] # km/s/h
        else:
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

    def print_coeff(Psi, want_rsd=False):
        """
        Internal function for printing the correlation coefficient given the displacements
        """
        # compute velocity from displacement
        Psi_dot, Psi_dot_r, N = get_Psi_dot(Psi, want_rsd=want_rsd)

        # compute rms
        Psi_dot_rms = np.sqrt(np.mean(Psi_dot[:, 0]**2+Psi_dot[:, 1]**2+Psi_dot[:, 2]**2))/np.sqrt(3.)
        Psi_dot_3d_rms = np.sqrt(np.mean(Psi_dot**2, axis=0))
        Psi_dot_r_rms = np.sqrt(np.mean(Psi_dot_r**2))
        if print_stuff:
            print("1D RMS", Psi_dot_rms)
            print("3D RMS", Psi_dot_3d_rms)
            print("LOS RMS", Psi_dot_r_rms)

        # compute statistics
        if print_stuff:
            print("1D R", np.mean(Psi_dot*vel)/(vel_rms*Psi_dot_rms))
            print("3D R", np.mean(Psi_dot*vel, axis=0)/(vel_3d_rms*Psi_dot_3d_rms))
            print("3D R (pred)", Psi_dot_3d_rms/vel_3d_rms)
            print("LOS R", np.mean(Psi_dot_r*vel_r)/(vel_r_rms*Psi_dot_r_rms))
            print("----------------")

        # plot true vs. reconstructed
        plt.figure(figsize=(9, 7))
        plt.scatter(vel_r, Psi_dot_r, s=1, alpha=0.1, color='teal')
        if want_rsd:
            plt.savefig("figs/vel_rec_rsd.png")
        else:
            plt.savefig("figs/vel_rec.png")
        plt.close()

        return Psi_dot_3d_rms, np.mean(Psi_dot*vel, axis=0)/(vel_3d_rms*Psi_dot_3d_rms)[0]

    # initialize arrays
    n_gals = np.zeros(len(sim_names))
    vel_3d_rms = np.zeros((len(sim_names), 3))
    vrec_3d_rms = np.zeros((len(sim_names), 3))
    vrec_3d_rsd_rms = np.zeros((len(sim_names), 3))
    r_3d = np.zeros((len(sim_names), 3))
    r_3d_rsd = np.zeros((len(sim_names), 3))
    ratio_rand = np.zeros(len(sim_names))
    
    # simulation parameters
    Lbox = get_meta(sim_names[0], 0.1)['BoxSize'] # cMpc/h
    
    for i_sim, sim_name in enumerate(sim_names):
        # directory where the reconstructed mock catalogs are saved
        save_dir = Path("/global/cfs/cdirs/desi/users/boryanah/kSZ_recon/")
        #save_recon_dir_old = Path(save_dir) / "recon" / sim_name / f"z{Mean_z:.3f}"
        save_recon_dir_old = Path("/pscratch/sd/b/boryanah/kSZ_recon/") / "recon" / sim_name / f"z{Mean_z:.3f}" # because we moved to free up space
        save_recon_dir_new = Path(save_dir) / "new" / "recon" / sim_name / f"z{Mean_z:.3f}" # og
        #save_recon_dir_new = Path(save_dir) / "recon" / sim_name / f"z{Mean_z:.3f}" # TESTING!
        
        # load the reconstructed data
        if box_lc == "lc":
            fn_old = save_recon_dir_old / f"displacements_{tracer_extra_str}{fakelc_str}{photoz_str}{mask_str}_postrecon_R{sr:.2f}{bias_str}_nmesh{nmesh:d}_{convention}_{rectype}{nz_full_str}.npz"
            fn_new = save_recon_dir_new / f"displacements_{tracer_extra_str}{fakelc_str}{photoz_str}{mask_str}_postrecon_R{sr:.2f}{bias_str}_nmesh{nmesh:d}_{convention}_{rectype}{nz_full_str}.npz"
        elif box_lc == "box":
            fn_old = save_recon_dir_old / f"displacements_{tracer}{extra}_postrecon_R{sr:.2f}{bias_str}_nmesh{nmesh:d}_{convention}_{rectype}_z{Mean_z:.3f}.npz"
            fn_new = save_recon_dir_new / f"displacements_{tracer}{extra}_postrecon_R{sr:.2f}{bias_str}_nmesh{nmesh:d}_{convention}_{rectype}_z{Mean_z:.3f}.npz"

        missing_new = True
        if os.path.exists(fn_new):
            print(str(fn_new))
            data = np.load(fn_new)
            try:
                n_gal = data['n_gals'][0]
            except:
                print("n_gals doesn't exist (older file probably)")
                n_gal = data['displacements'].shape[0]
            vel = data['velocities'][:n_gal]
            
            # read this file if it does contain displacements (typically we don't so as to save time)
            if not np.isclose(np.sum(data['displacements'][:n_gal]), 0.):
                Psi = data['displacements'][:n_gal] # cMpc/h # galaxies without RSD
                Psi_rsd = data['displacements_rsd'][:n_gal] # cMpc/h # galaxies with RSD
                Psi_rsd_nof = data['displacements_rsd_nof'][:n_gal] # cMpc/h # galaxies with RSD, 1+f divided along LOS direction
                missing_new = False
                
        if missing_new or not os.path.exists(fn_new):
            print(str(fn_old))
            data = np.load(fn_old)
            try:
                n_gal = data['n_gals'][0]
            except:
                print("n_gals doesn't exist (older file probably)")
                n_gal = data['displacements'].shape[0]

            if missing_new:
                # read the displacements, velocities and cosmological parameters
                Psi = data['displacements'][:n_gal] # cMpc/h # galaxies without RSD
                Psi_rsd = data['displacements_rsd'][:n_gal] # cMpc/h # galaxies with RSD
                Psi_rsd_nof = data['displacements_rsd_nof'][:n_gal] # cMpc/h # galaxies with RSD, 1+f divided along LOS direction
                #vel = data['velocities'][:n_gal] # km/s # true velocities
        try:
            n_rand = data['random_displacements'].shape[0]
            ratio_rand[i_sim] = n_rand/n_gal
        except:
            print("no randoms, this must be box")
            assert box_lc == "box"
            
        if box_lc == "box":
            mock_dir = Path(f"/pscratch/sd/b/boryanah/AbacusHOD_scratch/mocks_box_output_kSZ_recon{extra}/")
            mock_dir = mock_dir / f"{sim_name}/z{Mean_z:.3f}/"
            vel = np.load(mock_dir / f"galaxies/{tracer}s_halo_vel.npy") # box only

            
            # TESTING make thinner
            from astropy.io import ascii
            f = ascii.read(mock_dir / f"galaxies/{tracer}s.dat")
            positions = np.vstack((f['x'], f['y'], f['z'])).T
            positions %= Lbox
            z_max = 830.
            choice = positions[:, 2] < z_max
            vel = vel[choice]
            
        assert Psi.shape[0] == vel.shape[0]
        if box_lc == "lc":
            unit_los = data['unit_los'][:n_gal]
        elif box_lc == "box":
            unit_los = np.array([0, 0, 1.])

        # simulation parameters
        if nz_dict['Type'] == "Gaussian":
            cosmo = DESI() # AbacusSummit
            f = cosmo.growth_factor(nz_dict['Mean_z'])
            H = cosmo.hubble_function(nz_dict['Mean_z'])
        else:
            f = data['growth_factor']
            H = data['Hubble_z'] # km/s/Mpc
        
        # cut the edges off
        if cut_edges or want_z_evolve or want_save:

            # load galaxies and randoms
            mock_dir = Path(f"/global/cfs/cdirs/desi/users/boryanah/kSZ_recon/new/mocks_lc_output_kSZ_recon{extra}/")
            mock_dir = mock_dir / sim_name 
            data = np.load(mock_dir / f"galaxies_{tracer}{fakelc_str}{photoz_str}{mask_str}_prerecon{nz_str}.npz")
            Z_RSD = data['Z_RSD']
            Z = data['Z']
            POS = data['POS']
            if want_save:
                RA = data['RA']
                DEC = data['DEC']

            assert len(Z) == vel.shape[0]

            if cut_edges:
                # construct cuts in redshift
                Delta_z_cut = 0.1
                Mean_z_cut = Mean_z
                choice = (Z < Mean_z_cut + Delta_z_cut) & (Z >= Mean_z_cut - Delta_z_cut)

                # construct cuts near the edges of the box
                offset = 50.
                choice &= (POS[:, 0] > -Lbox/2.+offset) & (POS[:, 0] < Lbox/2.-offset)
                choice &= (POS[:, 1] > -Lbox/2.+offset)
                choice &= (POS[:, 2] > -Lbox/2.+offset)
            else:
                choice = np.ones(len(Z), dtype=bool)

            if want_z_evolve: # cosmo
                cosmo = DESI() # Abacus
                f = cosmo.growth_factor(Z)
                a = 1./(1.+Z)
                H = cosmo.hubble_function(Z)

                a = a[choice]
                f = f[choice]
                H = H[choice]

            # apply cuts
            Psi = Psi[choice]
            Psi_rsd = Psi_rsd[choice]
            Psi_rsd_nof = Psi_rsd_nof[choice]
            vel = vel[choice]
            unit_los = unit_los[choice]
            if want_save:
                RA = RA[choice]
                DEC = DEC[choice]
                
        # velocity in los direction
        vel_r = np.sum(vel*unit_los, axis=1)
        if print_stuff:
            print("number of galaxies", len(vel_r))
        n_gals[i_sim] = len(vel_r)

        if box_lc == "lc":
            # perp to los (a, b, c): (c, c, -b-a), (l2 p3 - l3 p2, -(l1 p3 - l3 p1), l1 p2 - l2 p1)
            unit_perp1 = np.vstack((unit_los[:, 2], unit_los[:, 2], -unit_los[:, 1] -unit_los[:, 0])).T
            unit_perp2 = np.vstack((unit_los[:, 1]*unit_perp1[:, 2] - unit_los[:, 2]*unit_perp1[:, 1],
                                  -(unit_los[:, 0]*unit_perp1[:, 2] - unit_los[:, 2]*unit_perp1[:, 0]),
                                    unit_los[:, 0]*unit_perp1[:, 1] - unit_los[:, 1]*unit_perp1[:, 0])).T
            unit_perp1 /= np.linalg.norm(unit_perp1, axis=1)[:, None]
            unit_perp2 /= np.linalg.norm(unit_perp2, axis=1)[:, None]

            # velocities in the transverse direction
            vel_p1 = np.sum(vel*unit_perp1, axis=1)
            vel_p2 = np.sum(vel*unit_perp2, axis=1)
            vel = np.vstack((vel_p1, vel_p2, vel_r)).T

        # compute the rms
        vel_rms = np.sqrt(np.mean(vel[:, 0]**2+vel[:, 1]**2+vel[:, 2]**2))/np.sqrt(3.)
        vel_3d_rms[i_sim] = np.sqrt(np.mean(vel**2, axis=0))
        vel_r_rms = np.sqrt(np.mean(vel_r**2))
        if print_stuff:
            print("TRUE:")
            print("1D RMS", vel_rms)
            print("3D RMS", vel_3d_rms[i_sim])
            print("LOS RMS", vel_r_rms)
            print("")

        if want_save:
            _, Psi_dot_r, _ = get_Psi_dot(Psi, want_rsd=False)
            _, Psi_rsd_nof_dot_r, _ = get_Psi_dot(Psi_rsd_nof, want_rsd=False)
            np.savez(f"data/RA_DEC_v_los{photoz_str}.npz", vel_r=vel_r, RA=RA, DEC=DEC, Psi_dot_r=Psi_dot_r, Psi_rsd_nof_dot_r=Psi_rsd_nof_dot_r, Z=Z)
            return
        
        # print the correlation coefficient
        if print_stuff:
            print("REC NO RSD:")
        vrec_3d_rms[i_sim], r_3d[i_sim] = print_coeff(Psi, want_rsd=False)
        if print_stuff:
            #print("REC RSD (w/ 1+f):") # equivalent to below (checked!)
            pass
        #print_coeff(Psi_rsd, want_rsd=True)
        if print_stuff:
            print("REC RSD (w/o 1+f):")
        vrec_3d_rsd_rms[i_sim], r_3d_rsd[i_sim] = print_coeff(Psi_rsd_nof, want_rsd=False)

    for i_sim in range(len(sim_names)):
        print(i_sim)
        print("R_3D", r_3d[i_sim], n_gals[i_sim], ratio_rand[i_sim])
        print("R_3D_RSD", r_3d_rsd[i_sim], n_gals[i_sim], ratio_rand[i_sim])

    print("--------------------------------------------------")
    print("perp rls", np.mean(r_3d[:, :2]), np.std(r_3d[:, :2]))
    print("perp rsd", np.mean(r_3d_rsd[:, :2]), np.std(r_3d_rsd[:, :2]))
    print("para rls", np.mean(r_3d[:, 2]), np.std(r_3d[:, 2]))
    print("para rsd", np.mean(r_3d_rsd[:, 2]), np.std(r_3d_rsd[:, 2]))
                                                           
    #par_labs = ["Tracer(s)", r"$N(z)$", "Galaxy number", r"$N_{\rm mesh}$", r"$R_{\rm sm}$", r"$\sigma_\perp$", r"$\sigma_{||}$", r"$\sigma^{\rm rec}_\perp$", r"$\sigma^{\rm rec}_{||}$", r"$\sigma^{\rm rec, RSD}_\perp$", r"$\sigma^{\rm rec, RSD}_{||}$", r"$r_\perp$", r"$r_{||}$", r"$r^{\rm RSD}_\perp$", r"$r^{\rm RSD}_{||}$"]
    par_labs = ["Tracer(s)", r"$N(z)$", "Area", r"$\sigma_z/(1+z)$", r"$N_{\rm mesh}$", r"$R_{\rm sm}$", r"$r_\perp$", r"$r_{||}$", r"$r^{\rm RSD}_\perp$", r"$r^{\rm RSD}_{||}$"]

    def get_tracer_tab(stem):
        if stem == "DESI_LRG":
            tracer_tab = "Main LRG"
        elif stem == "DESI_LRG_high_density":
            tracer_tab = "Extended LRG"
        elif stem == "DESI_ELG":
            tracer_tab = "ELG"
        elif stem == "DESI_ELG_uchuu":
            tracer_tab = "ELG"
        elif stem == "DESI_LRG_bgs":
            tracer_tab = "BGS"
        elif stem == "DESI_LRG_uchuu":
            tracer_tab = "BGS"
        return tracer_tab
    tracer_tab = get_tracer_tab(stem)
    if stem2 is not None:
        tracer_tab = tracer_tab.split('Main ')[-1] # smis razkarai Main za da pestish mqsto
        tracer_tab += ", " + get_tracer_tab(stem2)
    if stem3 is not None:
        tracer_tab += ", " + get_tracer_tab(stem3)

    if want_mask: # TODO huge
        area_tab = "DESI NGC"
        if "huge" in sim_names[0]:
            area_tab = "DESI"
    else:
        area_tab = "Octant"
        if "huge" in sim_names[0]:
            area_tab = "Sphere"

    if nz_dict['Type'] == "Gaussian":
        nz_tab = rf"$\mathcal{{N}}({nz_dict['Mean_z']:.1f}, {nz_dict['Delta_z']/2:.1f})$"
    elif nz_dict['Type'] == "StepFunction":
        nz_tab = rf"$\Theta({nz_dict['Min_z']:.1f}, {nz_dict['Max_z']:.1f})$"
    elif nz_dict['Type'] == "FromFile":
        nz_tab = "DESI"
    
    par_vals = []
    par_vals.append(f"{tracer_tab}")
    par_vals.append(f"{nz_tab}")
    par_vals.append(f"{area_tab}")
    par_vals.append(rf"${photoz_error:.1f}$")
    par_vals.append(rf"${nmesh:d}$")
    par_vals.append(rf"${sr:.1f}$")
    par_vals.append(rf"${np.mean(r_3d[:, :2]):.2f} \pm {np.std(r_3d[:, :2]):.3f}$")
    par_vals.append(rf"${np.mean(r_3d[:, 2]):.2f} \pm {np.std(r_3d[:, 2]):.3f}$")
    par_vals.append(rf"${np.mean(r_3d_rsd[:, :2]):.2f} \pm {np.std(r_3d_rsd[:, :2]):.3f}$")
    par_vals.append(rf"${np.mean(r_3d_rsd[:, 2]):.2f} \pm {np.std(r_3d_rsd[:, 2]):.3f}$")
    
    col_str = ' '.join(np.repeat("c", len(par_labs)))
    par_str = ' & '.join(par_labs)
    line = ' & '.join(par_vals)
    line += " \\\\ [0.5ex] \n"

    # create file name
    tracer_fn = '_'.join((''.join(f"{tracer_tab}".split(','))).split(' '))
    area_fn = '_'.join((''.join(f"{area_tab}".split(','))).split(' '))
    if nz_dict['Type'] == "Gaussian":
        nz_fn = rf"meanz{nz_dict['Mean_z']:.1f}_deltaz{nz_dict['Delta_z']:.1f}"
    else:
        nz_fn = nz_tab    
    fn_save = f"table/{order_str}_{tracer_fn}_{area_fn}_{nz_fn}_zerr{photoz_error:.1f}_nmesh{nmesh:d}_sr{sr:.1f}.npz"

    # save file
    np.savez(fn_save,
             test_name=test_name,
             tracer_tab=tracer_tab,
             nz_tab=nz_tab,
             area_tab=area_tab,
             photoz_error=photoz_error,
             nmesh=nmesh,
             sr=sr,
             sigma_perp=np.mean(vel_3d_rms[:, :2]),
             sigma_perp_err=np.std(vel_3d_rms[:, :2]),
             sigma_para=np.mean(vel_3d_rms[:, 2]),
             sigma_para_err=np.std(vel_3d_rms[:, 2]),
             sigma_rec_perp=np.mean(vrec_3d_rms[:, :2]),
             sigma_rec_perp_err=np.std(vrec_3d_rms[:, :2]),
             sigma_rec_para=np.mean(vrec_3d_rms[:, 2]),
             sigma_rec_para_err=np.std(vrec_3d_rms[:, 2]),
             sigma_rec_perp_rsd=np.mean(vrec_3d_rsd_rms[:, :2]),
             sigma_rec_perp_rsd_err=np.std(vrec_3d_rsd_rms[:, :2]),
             sigma_rec_para_rsd=np.mean(vrec_3d_rsd_rms[:, 2]),
             sigma_rec_para_rsd_err=np.std(vrec_3d_rsd_rms[:, 2]),
             r_perp=np.mean(r_3d[:, :2]),
             r_perp_err=np.std(r_3d[:, :2]),
             r_para=np.mean(r_3d[:, 2]),
             r_para_err=np.std(r_3d[:, 2]),
             r_perp_rsd=np.mean(r_3d_rsd[:, :2]),
             r_perp_rsd_err=np.std(r_3d_rsd[:, :2]),
             r_para_rsd=np.mean(r_3d_rsd[:, 2]),
             r_para_rsd_err=np.std(r_3d_rsd[:, 2])
             )

    """
    fn = "table.tex"
    exists = os.path.exists(fn) #open(fn, "r").readlines()[0] == '\begin{table*} \n'
    f = open(fn, "a")
    if not exists:
        f.write("\begin{table*} \n")
        f.write("\begin{center} \n")
        #f.write("\footnotesize \n")
        f.write(f"\\begin{{tabular}}{{ {col_str} }} \n")
        f.write(" \hline\hline \n")
        f.write(f" {par_str} \\\\ [0.5ex] \n")
        f.write(" \hline \n")
    f.write(line)
    if not exists:
        f.write(" \hline \n")
        f.write(" \hline \n")
        f.write("\end{tabular} \n")
        f.write("\end{center} \n")
        f.write("\label{tab:r_coeff} \n")
        f.write("\caption{Table} \n")
        f.write("\end{table*} \n")
    f.close()
    """
        
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
    parser.add_argument('--cut_edges', help='Cut edges of the light cone?', action='store_true')
    parser.add_argument('--photoz_error', help='Percentage error on the photometric redshifts', default=DEFAULTS['photoz_error'], type=float)
    parser.add_argument('--nz_type', help='Type of N(z) distribution: Gaussian(mean, 2sigma), Step(min, max), file name', default=DEFAULTS['nz_type'])
    parser.add_argument('--bz_type', help='Type of b(z) distribution', default=DEFAULTS['bz_type'])
    parser.add_argument('--nz_type2', help='Type of N(z) distribution: Gaussian(mean, 2sigma), Step(min, max), file name', default=None)
    parser.add_argument('--nz_type3', help='Type of N(z) distribution: Gaussian(mean, 2sigma), Step(min, max), file name', default=None)
    parser.add_argument('--want_fakelc', help='Want to use the fake light cone?', action='store_true')
    parser.add_argument('--want_z_evolve', help='Want redshift-evolving faH?', action='store_true')
    parser.add_argument('--want_mask', help='Want to apply DESI mask?', action='store_true')
    parser.add_argument('--want_save', help='Want to save RA, DEC and v_los?', action='store_true')
    parser.add_argument('--test_name', help='Test name', default="")
    args = vars(parser.parse_args())
    main(**args)
