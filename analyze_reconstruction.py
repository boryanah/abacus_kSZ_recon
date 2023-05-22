from pathlib import Path
import sys

import numpy as np
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from abacusnbody.metadata import get_meta

DEFAULTS = {}
DEFAULTS['sim_name'] = "AbacusSummit_base_c000_ph002"
DEFAULTS['box_lc'] = 'box'
DEFAULTS['stem'] = 'DESI_LRG' # 'DESI_ELG'
DEFAULTS['nmesh'] = 512 # 1024
DEFAULTS['sr'] = 10. # Mpc/h
DEFAULTS['rectype'] = "MG"
DEFAULTS['convention'] = "recsym"

"""
Usage:
python analyze_reconstruction.py --sim_name AbacusSummit_base_c000_ph002 --box_lc box --stem DESI_LRG --nmesh 1024 --sr 12.5 --rectype MG --convention recsym
"""

def main(sim_name, box_lc, stem, nmesh, sr, rectype, convention, cut_edges=False):
    # additional specs of the tracer
    extra = '_'.join(stem.split('_')[2:])
    stem = '_'.join(stem.split('_')[:2])
    if extra != '': extra = '_'+extra
        
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

    # directory where the reconstructed mock catalogs are saved
    save_dir = Path("/global/cfs/cdirs/desi/users/boryanah/kSZ_recon/")
    save_recon_dir = Path(save_dir) / "recon" / sim_name / f"z{Mean_z:.3f}"    
    
    # cosmology parameters
    h = get_meta(sim_name, 0.1)['H0']/100.
    a = 1./(1+Mean_z)

    # it only makes sense to cut the edges in the light cone case
    if cut_edges:
        assert box_lc == "lc"

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
        print("1D RMS", Psi_dot_rms)
        print("3D RMS", Psi_dot_3d_rms)
        print("LOS RMS", Psi_dot_r_rms)

        # compute statistics
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

    # load the reconstructed data
    if box_lc == "lc":
        data = np.load(save_recon_dir / f"displacements_{tracer}{extra}_postrecon_R{sr:.2f}_b{bias:.1f}_nmesh{nmesh:d}_{convention}_{rectype}_meanz{Mean_z:.3f}_deltaz{Delta_z:.3f}.npz")
    elif box_lc == "box":
        data = np.load(save_recon_dir / f"displacements_{tracer}{extra}_postrecon_R{sr:.2f}_b{bias:.1f}_nmesh{nmesh:d}_{convention}_{rectype}_z{Mean_z:.3f}.npz")

    # read the displacements, velocities and cosmological parameters
    Psi = data['displacements'] # cMpc/h # galaxies without RSD
    Psi_rsd = data['displacements_rsd'] # cMpc/h # galaxies with RSD
    Psi_rsd_nof = data['displacements_rsd_nof'] # cMpc/h # galaxies with RSD, 1+f divided along LOS direction
    vel = data['velocities'] # km/s # true velocities
    if box_lc == "lc":
        unit_los = data['unit_los']
    elif box_lc == "box":
        unit_los = np.array([0, 0, 1.])
    f = data['growth_factor']
    H = data['Hubble_z'] # km/s/Mpc

    # cut the edges off
    if box_lc == "lc" and cut_edges:
        # simulation parameters
        Lbox = get_meta(sim_name, 0.1)['BoxSize'] # cMpc/h
                
        # load galaxies and randoms
        mock_dir = Path("/global/cfs/cdirs/desi/users/boryanah/kSZ_recon/") # old
        #mock_dir = Path("/global/cfs/cdirs/desi/users/boryanah/kSZ_recon/mocks_lc_output_kSZ_recon{extra}/")
        mock_dir = mock_dir / sim_name / "tmp" # old # remove tmp
        data = np.load(mock_dir / f"galaxies_{tracer}_prerecon_meanz{Mean_z:.3f}_deltaz{Delta_z:.3f}.npz")
        Z_RSD = data['Z_RSD']
        Z = data['Z']
        POS = data['POS']
        assert len(Z) == vel.shape[0]

        # construct cuts in redshift
        Delta_z_cut = 0.1
        choice = (Z < Mean_z + Delta_z_cut) & (Z >= Mean_z - Delta_z_cut)

        # construct cuts near the edges of the box
        offset = 600.
        choice &= (POS[:, 0] > -Lbox/2.+offset) & (POS[:, 0] < Lbox/2.-offset)
        choice &= (POS[:, 1] > -Lbox/2.+offset)
        choice &= (POS[:, 2] > -Lbox/2.+offset)

        # apply cuts
        Psi = Psi[choice]
        Psi_rsd = Psi_rsd[choice]
        Psi_rsd_nof = Psi_rsd_nof[choice]
        vel = vel[choice]
        unit_los = unit_los[choice]

    # velocity in los direction
    vel_r = np.sum(vel*unit_los, axis=1)
    print("number of galaxies", len(vel_r))
    
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
    vel_3d_rms = np.sqrt(np.mean(vel**2, axis=0))
    vel_r_rms = np.sqrt(np.mean(vel_r**2))
    print("TRUE:")
    print("1D RMS", vel_rms)
    print("3D RMS", vel_3d_rms)
    print("LOS RMS", vel_r_rms)
    print("")

    # print the correlation coefficient
    print("REC NO RSD:")
    print_coeff(Psi, want_rsd=False)
    #print("REC RSD (w/ 1+f):") # equivalent to below
    #print_coeff(Psi_rsd, want_rsd=True)
    print("REC RSD (w/o 1+f):")
    print_coeff(Psi_rsd_nof, want_rsd=False)

class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--sim_name', help='Simulation name', default=DEFAULTS['sim_name'])
    parser.add_argument('--box_lc', help='Cubic box or light cone?', default=DEFAULTS['box_lc'])
    parser.add_argument('--stem', help='Stem file name', default=DEFAULTS['stem'])
    parser.add_argument('--nmesh', help='Number of cells per dimension for reconstruction', type=int, default=DEFAULTS['nmesh'])
    parser.add_argument('--sr', help='Smoothing radius', type=float, default=DEFAULTS['sr'])
    parser.add_argument('--rectype', help='Reconstruction type', default=DEFAULTS['rectype'], choices=["IFT", "MG", "IFTP"])
    parser.add_argument('--convention', help='Reconstruction convention', default=DEFAULTS['convention'], choices=["recsym", "reciso"])
    parser.add_argument('--cut_edges', help='Cut edges of the light cone?', action='store_true')
    args = vars(parser.parse_args())
    main(**args)
