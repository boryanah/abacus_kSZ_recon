from pathlib import Path
import os, gc, glob
import sys

import numpy as np
import healpy as hp
import asdf
import argparse
import fitsio

sys.path.append("..")
from prepare_lc_catalog import relate_chi_z

DEFAULTS = {}
DEFAULTS['sim_name'] = "AbacusSummit_base_c000_ph002"
DEFAULTS['stem'] = 'DESI_LRG' # 'DESI_ELG'
DEFAULTS['nz_type'] = 'Gaussian(0.5, 0.4)'
DEFAULTS['photoz_error'] = 0.

"""
Usage:

#python calc_num_dens.py --sim_name AbacusSummit_base_c000_ph002 --stem DESI_LRG_high_density --photoz_error 0. --nz_type 'Gaussian(0.5,0.4)'

python calc_num_dens.py --sim_name AbacusSummit_base_c000_ph002 --stem DESI_LRG --photoz_error 0. --nz_type 'Gaussian(0.5,0.4)'
python calc_num_dens.py --sim_name AbacusSummit_base_c000_ph002 --stem DESI_LRG --photoz_error 0. --nz_type 'Gaussian(0.5,0.2)'
python calc_num_dens.py --sim_name AbacusSummit_base_c000_ph002 --stem DESI_ELG_uchuu --photoz_error 0. --nz_type 'Gaussian(0.8,0.4)'
python calc_num_dens.py --sim_name AbacusSummit_base_c000_ph002 --stem DESI_LRG_uchuu --photoz_error 0. --nz_type 'Gaussian(0.4,0.1)'
python calc_num_dens.py --sim_name AbacusSummit_base_c000_ph002 --stem DESI_LRG --photoz_error 0. --nz_type 'Step(0.4,0.6)'
python calc_num_dens.py --sim_name AbacusSummit_base_c000_ph002 --stem DESI_LRG --photoz_error 0. --nz_type 'Step(0.4,0.8)'
python calc_num_dens.py --sim_name AbacusSummit_base_c000_ph002 --stem DESI_LRG --photoz_error 0. --nz_type 'Step(0.45,0.55)'

python calc_num_dens.py --sim_name AbacusSummit_base_c000_ph002 --stem DESI_LRG --photoz_error 0. --nz_type main_lrg_pz_dndz_iron_v0.npz
python calc_num_dens.py --sim_name AbacusSummit_base_c000_ph002 --stem DESI_LRG_high_density --photoz_error 0. --nz_type extended_lrg_pz_dndz_iron_v0.npz
python calc_num_dens.py --sim_name AbacusSummit_base_c000_ph002 --stem DESI_LRG_uchuu --photoz_error 0. --nz_type BGS_BRIGHT_full_N_nz.npz
python calc_num_dens.py --sim_name AbacusSummit_base_c000_ph002 --stem DESI_ELG_uchuu --photoz_error 0. --nz_type elg_main-800coaddefftime1200-nz-zenodo_perc30.0.npz
"""

def parse_nztype(nz_type):
    """
    Should be one of: Gaussian(mean,2sigma) or Step(min,max) or a direct file name
    """
    nz_dict = {}
    if "gaussian" in nz_type.lower():
        Mean_z, Delta_z = (nz_type.split('(')[-1]).split(')')[0].split(',')
        nz_dict['Type'] = "Gaussian"
        nz_dict['Mean_z'] = float(Mean_z)
        nz_dict['Delta_z'] = float(Delta_z)
    elif "step" in nz_type.lower():
        Min_z, Max_z = (nz_type.split('(')[-1]).split(')')[0].split(',')
        nz_dict['Type'] = "StepFunction"
        nz_dict['Min_z'] = float(Min_z)
        nz_dict['Max_z'] = float(Max_z)
    else:
        assert os.path.exists(nz_type), "if not gaussian or step function, needs to be a NPZ file"
        data = np.load(nz_type)
        z_edges = data['z_edges']
        comov_dens = data['comov_dens']
        nz_dict['Type'] = "FromFile"
        nz_dict['File'] = (nz_type.split('/')[-1]).split('.npz')[0]
        nz_dict['Z_edges'] = z_edges
        nz_dict['Comov_dens'] = comov_dens
    return nz_dict


def get_dndz_mask(RA, DEC, Z, RAND_RA, RAND_DEC, RAND_Z, nz_dict, z_srcs, mask_fns, chi_of_z):
    """
    Returns mask for the galaxies and randoms given their RA, DEC and Z
    """

    if nz_dict['Type'] == "Gaussian":
        # find highest redshift
        z_max = nz_dict['Mean_z'] + nz_dict['Delta_z'] # 2sigma (97.5)
    
        # apply masking
        mask_fn = mask_fns[np.argmax(z_srcs - z_max >= 0.)]
        mask = asdf.open(mask_fn)['data']['mask']
        nside = asdf.open(mask_fn)['header']['HEALPix_nside']
        order = asdf.open(mask_fn)['header']['HEALPix_order']
        nest = True if order == 'NESTED' else False
    
        # mask lenses and randoms
        print("apply mask")
        gal_mask = get_mask_ang(mask, RA, DEC, nest, nside)
        rand_mask = get_mask_ang(mask, RAND_RA, RAND_DEC, nest, nside)
        print("galaxy percentage remaining", np.sum(gal_mask)/len(gal_mask)*100.)
        print("randoms percentage remaining", np.sum(rand_mask)/len(rand_mask)*100.)
        del mask; gc.collect()

        # apply  z cuts
        gal_mask &= get_down_choice(Z, nz_dict['Mean_z'], nz_dict['Delta_z'])
        rand_mask &= get_down_choice(RAND_Z, nz_dict['Mean_z'], nz_dict['Delta_z'])

    elif nz_dict['Type'] == "StepFunction":        
        
        # apply masking
        mask_fn = mask_fns[np.argmax(z_srcs - nz_dict['Max_z'] >= 0.)]
        mask = asdf.open(mask_fn)['data']['mask']
        nside = asdf.open(mask_fn)['header']['HEALPix_nside']
        order = asdf.open(mask_fn)['header']['HEALPix_order']
        nest = True if order == 'NESTED' else False
    
        # mask lenses and randoms
        print("apply mask")
        gal_mask = get_mask_ang(mask, RA, DEC, nest, nside)
        rand_mask = get_mask_ang(mask, RAND_RA, RAND_DEC, nest, nside)
        print("galaxy percentage remaining", np.sum(gal_mask)/len(gal_mask)*100.)
        print("randoms percentage remaining", np.sum(rand_mask)/len(rand_mask)*100.)
        del mask; gc.collect()

        # apply  z cuts
        gal_mask &= (Z > nz_dict['Min_z']) & (Z < nz_dict['Max_z'])
        rand_mask &= (RAND_Z > nz_dict['Min_z']) & (RAND_Z < nz_dict['Max_z'])

    elif nz_dict['Type'] == "FromFile":
        # edges and comoving density
        z_bins = nz_dict['Z_edges']
        comov_dens = nz_dict['Comov_dens']
        z_binc = 0.5 * (z_bins[1:] + z_bins[:-1])

        # ratio of randoms to galaxies
        rand_factor = len(RAND_Z)/len(Z)

    return gal_mask, rand_mask

def main(sim_name, stem, nz_type, photoz_error, want_fakelc=False):

    # parse the dndz instructions
    nz_dict = parse_nztype(nz_type)
    if nz_dict['Type'] == "Gaussian":
        nz_str = f"_meanz{nz_dict['Mean_z']:.3f}_deltaz{nz_dict['Delta_z']:.3f}"
    elif nz_dict['Type'] == "StepFunction":
        nz_str = f"_minz{nz_dict['Min_z']:.3f}_maxz{nz_dict['Max_z']:.3f}"
    elif nz_dict['Type'] == "FromFile":
        nz_str = f"_{nz_dict['File']}"
    
    # redshift error string
    photoz_str = f"_zerr{photoz_error:.1f}"
    
    # fake light cones
    fakelc_str = "_fakelc" if want_fakelc else ""
    
    # additional specs of the tracer
    extra = '_'.join(stem.split('_')[2:])
    stem = '_'.join(stem.split('_')[:2])
    if extra != '': extra = '_'+extra

    # parameter choices
    if stem == 'DESI_LRG':
        min_z = 0.3
        max_z = 1.1
        tracer = "LRG"
    elif stem == 'DESI_ELG':
        min_z = 0.3#0.5
        max_z = 1.1#1.4
        tracer = "ELG"
    else:
        print("Other tracers not yet implemented"); quit()

    # functions relating chi and z
    chi_of_z, z_of_chi = relate_chi_z(sim_name)
    
    # directory where mock catalogs are saved
    mock_dir = Path(f"/global/cfs/cdirs/desi/users/boryanah/kSZ_recon/new/mocks_lc_output_kSZ_recon{extra}/")
    mock_dir = mock_dir / sim_name 

    # read in file names to determine all the available z's
    mask_dir = f"/global/cfs/cdirs/desi/public/cosmosim/AbacusLensing/v1/{sim_name}/"
    mask_fns = sorted(glob.glob(mask_dir+f"mask_0*.asdf"))
    z_srcs = []
    for i in range(len(mask_fns)):
        z_srcs.append(asdf.open(mask_fns[i])['header']['SourceRedshift'])
    z_srcs = np.sort(np.array(z_srcs))
    print("redshift sources = ", z_srcs)

    # save to file
    data = np.load(mock_dir / f"galaxies_{tracer}{fakelc_str}{photoz_str}_prerecon{nz_str}.npz")
    Z = data['Z']
    #RA=RA, DEC=DEC, Z_RSD=Z_RSD, Z=Z, VEL=VEL, POS=POS, POS_RSD=POS_RSD, UNIT_LOS=UNIT_LOS

    # save to file
    #data = np.load(mock_dir / f"randoms_{tracer}{fakelc_str}{photoz_str}_prerecon{nz_str}.npz")
    #RAND_RA, RAND_DEC=RAND_DEC, RAND_Z=RAND_Z, RAND_POS=RAND_POS)

    z_edges = np.linspace(0.1, 2.5, 101)
    dNdz, _ = np.histogram(Z, bins=z_edges)
    # TESTING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    """
    def gaussian(x, mu, sig):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    #z_cent = 0.5*(z_edges[1:] + z_edges[:-1])
    #print(z_cent[(z_cent < 0.8) & (z_cent > 0.3)])
    #print(dNdz[(z_cent < 0.8) & (z_cent > 0.3)])
    #print(gaussian(z_cent[(z_cent < 0.8) & (z_cent > 0.3)], 0.5, 0.2))
    """
    # initialize stuff
    z_bins = np.linspace(0.25, 1.15, 21)
    print(z_bins[1]-z_bins[0])
    z_binc = (z_bins[1:] + z_bins[:-1]) *.5
    num_dens = np.zeros(len(z_binc))

    for i in range(len(z_binc)):
        # volume
        try:
            volume = 4./3.*np.pi*(chi_of_z(z_bins[i+1])**3 - chi_of_z(z_bins[i])**3) # (cMpc/h)^3
        except ValueError:
            print("No volume", z_binc[i])
            volume = 0.
            
        # apply masking
        mask_fn = mask_fns[np.argmin(np.abs(z_srcs - z_binc[i]))]
        mask = asdf.open(mask_fn)['data']['mask']
        nside = asdf.open(mask_fn)['header']['HEALPix_nside']
        order = asdf.open(mask_fn)['header']['HEALPix_order']
        nest = True if order == 'NESTED' else False
        fraction = np.sum(mask)/len(mask)
        del mask; gc.collect()
        print("redshift, volume fraction", z_binc[i], fraction)
            
        # mask for galaxies
        choice = (z_bins[i] < Z) & (Z < z_bins[i+1])
        if np.sum(choice) == 0: print("no galaxies in this redshift bin", z_binc[i]); continue
        V = volume*fraction
        N = np.sum(choice)
        num_dens[i] = N/V
        gc.collect()

    np.savez(f"real{extra}{nz_str}.npz", comov_dens=num_dens, z_edges=z_bins, dNdz=dNdz, z_bins_dNdz=z_edges)
    
class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--sim_name', help='Simulation name', default=DEFAULTS['sim_name'])
    parser.add_argument('--stem', help='Stem file name', default=DEFAULTS['stem'])#, choices=["DESI_LRG", "DESI_ELG", "DESI_LRG_high_density", "DESI_LRG_bgs", "DESI_ELG_high_density"])
    parser.add_argument('--photoz_error', help='Percentage error on the photometric redshifts', default=DEFAULTS['photoz_error'], type=float)
    parser.add_argument('--nz_type', help='Type of N(z) distribution: Gaussian(mean, 2sigma), Step(min, max), file name', default=DEFAULTS['nz_type'])
    parser.add_argument('--want_fakelc', help='Want to use the fake light cone?', action='store_true')
    args = vars(parser.parse_args())
    main(**args)
