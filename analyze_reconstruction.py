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
DEFAULTS['box_lc'] = 'box'
DEFAULTS['stem'] = 'DESI_LRG' # 'DESI_ELG'
DEFAULTS['nmesh'] = 512 # 1024
DEFAULTS['sr'] = 10. # Mpc/h
DEFAULTS['rectype'] = "MG"
DEFAULTS['convention'] = "recsym"
DEFAULTS['nz_type'] = 'Gaussian(0.5, 0.4)'
DEFAULTS['photoz_error'] = 0.

"""
Usage:
python analyze_reconstruction.py --sim_name AbacusSummit_base_c000_ph002 --box_lc box --stem DESI_LRG --nmesh 1024 --sr 12.5 --rectype MG --convention recsym
python analyze_reconstruction.py --sim_name AbacusSummit_base_c000_ph002 --box_lc lc --stem DESI_LRG --nmesh 1024 --sr 12.5 --rectype MG --convention recsym
python analyze_reconstruction.py --sim_name AbacusSummit_huge_c000_ph201 --box_lc lc --stem DESI_LRG --nmesh 1024 --sr 12.5 --rectype MG --convention recsym
"""

def main(sim_name, box_lc, stem, stem2, stem3, nmesh, sr, rectype, convention, nz_type, nz_type2, nz_type3, photoz_error, cut_edges=False, want_z_evolve=False, want_fakelc=False, want_mask=False, want_save=False):

    # redshift error string
    photoz_str = f"_zerr{photoz_error:.1f}"
    
    # fake light cones
    fakelc_str = "_fakelc" if want_fakelc else ""
    mask_str = "_mask" if want_mask else ""
    
    # it only makes sense to cut the edges in the light cone case
    if cut_edges or want_fakelc:
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
    
    for i_sim, sim_name in enumerate(sim_names):
        # directory where the reconstructed mock catalogs are saved
        save_dir = Path("/global/cfs/cdirs/desi/users/boryanah/kSZ_recon/")
        save_recon_dir = Path(save_dir) / "recon" / sim_name / f"z{Mean_z:.3f}"    

        # load the reconstructed data
        if box_lc == "lc":
            try:
                print(str(save_recon_dir / f"displacements_{tracer}{extra}{fakelc_str}{photoz_str}{mask_str}_postrecon_R{sr:.2f}_b{bias:.1f}_nmesh{nmesh:d}_{convention}_{rectype}{nz_full_str}.npz"))
                data = np.load(save_recon_dir / f"displacements_{tracer}{extra}{fakelc_str}{photoz_str}{mask_str}_postrecon_R{sr:.2f}_b{bias:.1f}_nmesh{nmesh:d}_{convention}_{rectype}{nz_full_str}.npz")
            except:
                print("loading old file", str(save_recon_dir / f"displacements_{tracer}{extra}{fakelc_str}{mask_str}_postrecon_R{sr:.2f}_b{bias:.1f}_nmesh{nmesh:d}_{convention}_{rectype}{nz_full_str}.npz"))
                if np.isclose(photoz_error, 0.): # old nomenclature
                    data = np.load(save_recon_dir / f"displacements_{tracer}{extra}{fakelc_str}{mask_str}_postrecon_R{sr:.2f}_b{bias:.1f}_nmesh{nmesh:d}_{convention}_{rectype}{nz_full_str}.npz")
                else:
                    print("File truly missing"); quit()
        elif box_lc == "box":
            data = np.load(save_recon_dir / f"displacements_{tracer}{extra}_postrecon_R{sr:.2f}_b{bias:.1f}_nmesh{nmesh:d}_{convention}_{rectype}_z{Mean_z:.3f}.npz")

        # get number of galaxies of tracer that matter
        try:
            n_gal = data['n_gals'][0]
        except:
            print("n_gals doesn't exist (older file probably)")
            n_gal = data['displacements'].shape[0]

        # read the displacements, velocities and cosmological parameters
        Psi = data['displacements'][:n_gal] # cMpc/h # galaxies without RSD
        Psi_rsd = data['displacements_rsd'][:n_gal] # cMpc/h # galaxies with RSD
        Psi_rsd_nof = data['displacements_rsd_nof'][:n_gal] # cMpc/h # galaxies with RSD, 1+f divided along LOS direction
        vel = data['velocities'][:n_gal] # km/s # true velocities
        # TESTING!!!!!!!!!!!!!!!!!!!!!
        mock_dir = Path(f"/pscratch/sd/b/boryanah/AbacusHOD_scratch/mocks_box_output_kSZ_recon{extra}/")
        mock_dir = mock_dir / f"{sim_name}/z{Mean_z:.3f}/"
        vel = np.load(mock_dir / f"galaxies/{tracer}s_halo_vel.npy")
        assert Psi.shape[0] == vel.shape[0]
        if box_lc == "lc":
            unit_los = data['unit_los'][:n_gal]
        elif box_lc == "box":
            unit_los = np.array([0, 0, 1.])
        f = data['growth_factor']
        H = data['Hubble_z'] # km/s/Mpc

        # simulation parameters
        Lbox = get_meta(sim_name, 0.1)['BoxSize'] # cMpc/h
        
        # cut the edges off
        if cut_edges or want_z_evolve or want_save:

            # load galaxies and randoms
            mock_dir = Path(f"/global/cfs/cdirs/desi/users/boryanah/kSZ_recon/mocks_lc_output_kSZ_recon{extra}/")
            mock_dir = mock_dir / sim_name 
            data = np.load(mock_dir / f"galaxies_{tracer}{fakelc_str}{photoz_str}{mask_str}_prerecon{nz_str}.npz")
            print(str(mock_dir / f"galaxies_{tracer}{fakelc_str}{photoz_str}{mask_str}_prerecon{nz_str}.npz"))
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
        print(print_coeff(Psi, want_rsd=False))
        vrec_3d_rms[i_sim], r_3d[i_sim] = print_coeff(Psi, want_rsd=False)
        if print_stuff:
            #print("REC RSD (w/ 1+f):") # equivalent to below (checked!)
            pass
        #print_coeff(Psi_rsd, want_rsd=True)
        if print_stuff:
            print("REC RSD (w/o 1+f):")
        vrec_3d_rsd_rms[i_sim], r_3d_rsd[i_sim] = print_coeff(Psi_rsd_nof, want_rsd=False)


    #par_labs = ["Tracer(s)", r"$N(z)$", "Galaxy number", r"$N_{\rm mesh}$", r"$R_{\rm sm}$", r"$\sigma_\perp$", r"$\sigma_{||}$", r"$\sigma^{\rm rec}_\perp$", r"$\sigma^{\rm rec}_{||}$", r"$\sigma^{\rm rec, RSD}_\perp$", r"$\sigma^{\rm rec, RSD}_{||}$", r"$r_\perp$", r"$r_{||}$", r"$r^{\rm RSD}_\perp$", r"$r^{\rm RSD}_{||}$"]
    par_labs = ["Tracer(s)", r"$N(z)$", "Area", r"$\sigma_z/(1+z)$", r"$N_{\rm mesh}$", r"$R_{\rm sm}$", r"$r_\perp$", r"$r_{||}$", r"$r^{\rm RSD}_\perp$", r"$r^{\rm RSD}_{||}$"]

    def get_tracer_tab(stem):
        if stem == "DESI_LRG":
            tracer_tab = "Main LRG"
        elif stem == "DESI_LRG_high_density":
            tracer_tab = "Extended LRG"
        elif stem == "DESI_LRG_bgs":
            tracer_tab = "BGS"
        return tracer_tab
    tracer_tab = get_tracer_tab(stem)
    if stem2 is not None:
        tracer_tab = tracer_tab.split('Main ')[-1]
        tracer_tab += ", " + get_tracer_tab(stem3)
    if stem3 is not None:
        tracer_tab += ", " + get_tracer_tab(stem3)

    if want_mask: # TODO huge
        area_tab = "DESI NGC"
    else:
        area_tab = "Octant"

    nz_tab = r"$\mathcal{N}(0.5, 0.2)$" # TODO step
    nz_dict = parse_nztype(nz_type)
    if nz_dict['Type'] == "Gaussian":
        nz_tab = rf"$\mathcal{{N}}({Mean_z:.1f}, {Delta_z/2:.1f})$"
    elif nz_dict['Type'] == "FromFile":
        nz_tab = "DESI"
    
    par_vals = []
    par_vals.append(f"{tracer_tab}")
    par_vals.append(f"{nz_tab}")
    par_vals.append(f"{area_tab}")
    par_vals.append(rf"${photoz_error:.1f}$")
    par_vals.append(rf"${nmesh:d}$")
    par_vals.append(rf"${sr:.1f}$")
    #par_vals.append(rf"${np.mean(vel_3d_rms[:, :2]):.1f} \pm {np.std(vel_3d_rms[:, :2]):.1f}$")
    #par_vals.append(rf"${np.mean(vel_3d_rms[:, 2]):.1f} \pm {np.std(vel_3d_rms[:, 2]):.1f}$")
    #par_vals.append(rf"${np.mean(vrec_3d_rms[:, :2]):.1f} \pm {np.std(vrec_3d_rms[:, :2]):.1f}$")
    #par_vals.append(rf"${np.mean(vrec_3d_rms[:, 2]):.1f} \pm {np.std(vrec_3d_rms[:, 2]):.1f}$")
    #par_vals.append(rf"${np.mean(vrec_3d_rsd_rms[:, :2]):.1f} \pm {np.std(vrec_3d_rsd_rms[:, :2]):.1f}$")
    #par_vals.append(rf"${np.mean(vrec_3d_rsd_rms[:, 2]):.1f} \pm {np.std(vrec_3d_rsd_rms[:, 2]):.1f}$")
    par_vals.append(rf"${np.mean(r_3d[:, :2]):.2f} \pm {np.std(r_3d[:, :2]):.3f}$")
    par_vals.append(rf"${np.mean(r_3d[:, 2]):.2f} \pm {np.std(r_3d[:, 2]):.3f}$")
    par_vals.append(rf"${np.mean(r_3d_rsd[:, :2]):.2f} \pm {np.std(r_3d_rsd[:, :2]):.3f}$")
    par_vals.append(rf"${np.mean(r_3d_rsd[:, 2]):.2f} \pm {np.std(r_3d_rsd[:, 2]):.3f}$")
    
    col_str = ' '.join(np.repeat("c", len(par_labs)))
    par_str = ' & '.join(par_labs)
    line = ' & '.join(par_vals)
    line += " \\ [0.5ex] \n"


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

    """
    print("\\begin{table*}")
    print("\\begin{center}")
    #print("\\footnotesize")
    print(f"\\begin{{tabular}}{{ {col_str} }} ")
    print(" \\hline\\hline")
    print(f" {par_str} \\\\ [0.5ex] ")
    print(" \\hline")
    print(line)
    print(" \\hline")
    print(" \\hline")
    print("\\end{tabular}")
    print("\\end{center}")
    print("\\label{tab:r_coeff}")
    print("\\caption{Table}")
    print("\\end{table*}")
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
    parser.add_argument('--nz_type2', help='Type of N(z) distribution: Gaussian(mean, 2sigma), Step(min, max), file name', default=None)
    parser.add_argument('--nz_type3', help='Type of N(z) distribution: Gaussian(mean, 2sigma), Step(min, max), file name', default=None)
    parser.add_argument('--want_fakelc', help='Want to use the fake light cone?', action='store_true')
    parser.add_argument('--want_z_evolve', help='Want redshift-evolving faH?', action='store_true')
    parser.add_argument('--want_mask', help='Want to apply DESI mask?', action='store_true')
    parser.add_argument('--want_save', help='Want to save RA, DEC and v_los?', action='store_true')
    args = vars(parser.parse_args())
    main(**args)
