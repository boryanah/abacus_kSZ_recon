from pathlib import Path
import os, gc, glob

import numpy as np
import healpy as hp
import asdf
import argparse
import fitsio

from prepare_lc_catalog import relate_chi_z

DEFAULTS = {}
DEFAULTS['sim_name'] = "AbacusSummit_base_c000_ph002"
DEFAULTS['stem'] = 'DESI_LRG' # 'DESI_ELG'
DEFAULTS['nz_type'] = 'Gaussian(0.5, 0.4)'
DEFAULTS['photoz_error'] = 0.

"""
Usage:
python apply_dndz_mask.py --sim_name AbacusSummit_base_c000_ph002 --stem DESI_LRG --photoz_error 2. --nz_type Gaussian(0.5,0.4)
python apply_dndz_mask.py --sim_name AbacusSummit_base_c000_ph002 --stem DESI_LRG --photoz_error 2. --nz_type Step(0.3,0.9)
python apply_dndz_mask.py --sim_name AbacusSummit_base_c000_ph002 --stem DESI_LRG --photoz_error 2. --nz_type num_dens/lrg_fig_1_histograms.npz
"""

np.random.seed(300)

def get_down_choice(Z_l, z_l, Delta_z):
    # apply cuts
    z_edges = np.linspace(0.1, 2.5, 1001)
    z_cent = 0.5*(z_edges[1:] + z_edges[:-1])
    n_tot = len(Z_l)
    n_per = n_tot/len(z_cent)
    dNdz, _ = np.histogram(Z_l, bins=z_edges)
    inds = np.digitize(Z_l, bins=z_edges) - 1
    w_bin = dNdz/n_per
    fac = gaussian(Z_l, z_l, Delta_z/2.)
    fac /= w_bin[inds] # downweight many galaxies in bin
    #fac /= np.max(fac) # normalize so that probabilities don't exceed 1
    print(np.mean(Z_l), np.min(Z_l), np.max(Z_l))
    fac /= np.max(fac[(Z_l > z_l-Delta_z/2.) & (Z_l < z_l+Delta_z/2.)])
    #fac /= np.max(fac[(Z_l > z_l-Delta_z/4.) & (Z_l < z_l+Delta_z/4.)]) # TESTING!!!!# helps with broad Gaussian
    down_choice = np.random.rand(n_tot) < fac
    #down_choice[:] = True
    print("kept fraction (20-30%) = ", np.sum(down_choice)/len(down_choice)*100.)
    return down_choice

def gaussian(x, mu, sig):
    """
    gaussian
    """
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def get_mask_ang(mask, RA, DEC, nest, nside, lonlat=True):
    """
    function that returns a boolean mask given RA and DEC
    """
    ipix = hp.ang2pix(nside, theta=RA, phi=DEC, nest=nest, lonlat=lonlat) # RA, DEC degrees (math convention)
    return mask[ipix] == 1.

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
        nz_dict['Min_z'] = np.min(z_edges)
        nz_dict['Max_z'] = np.max(z_edges)
    return nz_dict


def get_dndz_mask(RA, DEC, Z, RAND_RA, RAND_DEC, RAND_Z, nz_dict, z_srcs, mask_fns, chi_of_z):
    """
    Returns mask for the galaxies and randoms given their RA, DEC and Z
    """

    if nz_dict['Type'] == "Gaussian":
        # find highest redshift
        z_max = nz_dict['Mean_z'] + nz_dict['Delta_z'] # 2sigma (97.5)
    
        # apply masking
        #mask_fn = mask_fns[np.argmax(z_srcs - z_max >= 0.)]
        mask_fn = mask_fns[np.argmax(z_srcs - 0.8 >= 0.)] # TESTING
        if z_max > 0.8:
            print("NOTE THAT WE TAKE THE OCTANT MASK EVEN FOR Z > 0.8 BECAUSE OTHERWISE it messes up both r_perp and r_para (probably needs to be done bin by bin)")
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

        # TESTING remove edges (makes a pretty big difference)
        rand_mask &= (RAND_Z < nz_dict['Mean_z'] + nz_dict['Delta_z']/2.) & (RAND_Z > nz_dict['Mean_z'] - nz_dict['Delta_z']/2.)
        gal_mask &= (Z < nz_dict['Mean_z'] + nz_dict['Delta_z']/2.) & (Z > nz_dict['Mean_z'] - nz_dict['Delta_z']/2.)
        
    elif nz_dict['Type'] == "StepFunction":        
        
        # apply masking
        #mask_fn = mask_fns[np.argmax(z_srcs - nz_dict['Max_z'] >= 0.)] # og
        mask_fn = mask_fns[np.argmax(z_srcs - 0.8 >= 0.)] # TESTING
        if nz_dict['Max_z'] > 0.8:
            print("NOTE THAT WE TAKE THE OCTANT MASK EVEN FOR Z > 0.8 BECAUSE OTHERWISE it messes up both r_perp and r_para (probably needs to be done bin by bin)")
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
        gal_mask &= (Z > nz_dict['Min_z']) & (Z < nz_dict['Max_z']) # og
        # this is to make sure that the random nz follows the galaxies
        z_edges = np.linspace(0.1, 2.5, 1001)
        z_cent = 0.5*(z_edges[1:] + z_edges[:-1])
        dNdz, _ = np.histogram(Z, bins=z_edges)
        n_per = len(RAND_Z)/len(z_cent)
        dNdz_rand, _ = np.histogram(RAND_Z, bins=z_edges)
        inds = np.digitize(RAND_Z, bins=z_edges) - 1
        w_bin = dNdz_rand/n_per # this kinda gives you the weighting you need to get perfectly uniform stuff?
        fac = np.interp(RAND_Z, z_cent, dNdz/np.max(dNdz))
        fac /= w_bin[inds] # downweight many galaxies in bin
        #fac /= np.max(fac[(Z_l > Mean_z-Delta_z/2.) & (Z_l < Mean_z+Delta_z/2.)]) # og
        # TESTING (dosega tova pravim)
        fac /= np.max(fac[(RAND_Z > nz_dict['Min_z']) & (RAND_Z < nz_dict['Max_z'])]) # this is more robust
        rand_mask = np.random.rand(len(RAND_Z)) < fac
        rand_mask &= (RAND_Z > nz_dict['Min_z']) & (RAND_Z < nz_dict['Max_z'])

    elif nz_dict['Type'] == "FromFile":
        # edges and comoving density
        z_bins = nz_dict['Z_edges']
        comov_dens = nz_dict['Comov_dens']
        z_binc = 0.5 * (z_bins[1:] + z_bins[:-1])

        # ratio of randoms to galaxies
        #rand_factor = len(RAND_Z)/len(Z)
        #print("rand_factor", rand_factor)
        """
        # DOSTA PO-OK AMA NEVINAGI IMA DOSTATICHNO MAI
        z_edges = np.linspace(0.1, 2.5, 101)
        z_cent = 0.5*(z_edges[1:] + z_edges[:-1])
        dNdz, _ = np.histogram(Z, bins=z_edges)
        dNdz_rand, _ = np.histogram(RAND_Z, bins=z_edges)
        ratio = (dNdz_rand/dNdz)
        ratio[np.isclose(dNdz, 0.)] = 0.
        rand_factor = np.min(ratio[np.nonzero(ratio)])
        print("rand_factor", rand_factor)
        """
        
        # initialize arrays
        gal_mask = np.zeros(len(Z), dtype=bool)
        rand_mask = np.zeros(len(RAND_Z), dtype=bool)
        for i in range(len(z_binc)):
            # volume
            try:
                volume = 4./3.*np.pi*(chi_of_z(z_bins[i+1])**3 - chi_of_z(z_bins[i])**3) # (cMpc/h)^3
            except ValueError:
                print("No volume", z_binc[i])
                volume = 0.

            # identify galaxies in bin
            choice = (z_bins[i] < Z) & (Z < z_bins[i+1])
            if np.sum(choice) == 0: print("no galaxies in this redshift bin", z_binc[i]); continue
                
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
            N_target = volume*fraction*comov_dens[i]
            N_raw = np.sum(choice)
            keep = N_target/N_raw
            if keep > 1.: N_target = N_raw
            #gal_mask[choice] = np.random.rand(N_raw) < keep
            tmp = gal_mask[choice]
            tmp[np.random.choice(int(np.rint(N_raw)), int(np.rint(N_target)), replace=False)] = True
            gal_mask[choice] = tmp
            N_real = np.sum(gal_mask[choice])
            print("galaxy: N_target, N_raw, keep, N_keep", N_target, N_raw, keep, N_real)

            """
            # NAI-STAROTO
            # mask for randoms
            choice = (z_bins[i] < RAND_Z) & (RAND_Z < z_bins[i+1])
            N_target = N_target*rand_factor
            N_raw = np.sum(choice)
            keep = N_target/N_raw
            #assert keep <= 1.
            #rand_mask[choice] = np.random.rand(N_raw) < keep
            tmp = rand_mask[choice]
            tmp[np.random.choice(int(np.rint(N_raw)), int(np.rint(N_target)), replace=False)] = True
            rand_mask[choice] = tmp
            print("randoms: N_target, N_raw, keep, N_keep", N_target, N_raw, keep, np.sum(rand_mask[choice]))
            print("ratio", np.sum(rand_mask[choice])/N_real)
            gc.collect()
            """

        """
        # PO-NOVO MA MALKO PREEBAVA (TVA E PREDNOTO)
        z_edges = np.linspace(0.1, 2.5, 1001)
        z_cent = 0.5*(z_edges[1:] + z_edges[:-1])
        dNdz, _ = np.histogram(Z[gal_mask], bins=z_edges)
        n_per = len(RAND_Z)/len(z_cent)
        dNdz_rand, _ = np.histogram(RAND_Z, bins=z_edges)
        inds = np.digitize(RAND_Z, bins=z_edges) - 1
        w_bin = dNdz_rand/n_per # this kinda gives you the weighting you need to get perfectly uniform stuff
        fac = np.interp(RAND_Z, z_cent, dNdz/np.max(dNdz))
        fac /= w_bin[inds] # downweight many galaxies in bin
        fac /= np.max(fac[(RAND_Z > nz_dict['Min_z']) & (RAND_Z < nz_dict['Max_z'])]) 
        rand_mask = np.random.rand(len(RAND_Z)) < fac


        """
        # NAI-NOVOTO KOPELE
        z_edges = np.linspace(0.1, 2.5, 101)
        z_cent = 0.5*(z_edges[1:] + z_edges[:-1])
        dNdz, _ = np.histogram(Z[gal_mask], bins=z_edges)
        dNdz_rand, _ = np.histogram(RAND_Z, bins=z_edges)
        ratio = (dNdz_rand/dNdz)
        ratio[np.isclose(dNdz, 0.)] = 0.
        #factor = int(np.round(np.min(ratio[ratio > 0.])))
        factor = (np.min(ratio[ratio > 0.]))
        print("rand_factor", factor)
        dNdz_rand_targ = dNdz*factor
        down = dNdz_rand_targ / dNdz_rand
        inds = np.digitize(RAND_Z, bins=z_edges) - 1
        rand_mask = np.random.rand(len(RAND_Z)) < down[inds]
        
    # TESTING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    rand_mask &= (RAND_Z > 0.35)
    gal_mask &= (Z > 0.35)
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

    # name of the file
    gal_fn = mock_dir / f"galaxies_{tracer}{fakelc_str}_prerecon_minz{min_z:.3f}_maxz{max_z:.3f}.npz"
    rand_fn = mock_dir / f"randoms_{tracer}{fakelc_str}_prerecon_minz{min_z:.3f}_maxz{max_z:.3f}.npz"

    # names of final files
    final_gals_fn = mock_dir / f"galaxies_{tracer}{fakelc_str}{photoz_str}_prerecon{nz_str}.npz"
    final_rand_fn = mock_dir / f"randoms_{tracer}{fakelc_str}{photoz_str}_prerecon{nz_str}.npz"
    #if os.path.exists(final_gals_fn) and os.path.exists(final_rand_fn): return
    
    # read in file names to determine all the available z's
    mask_dir = f"masks/"#f"/global/cfs/cdirs/desi/public/cosmosim/AbacusLensing/v1/{sim_name}/"
    mask_fns = sorted(glob.glob(mask_dir+f"mask_0*.asdf"))
    z_srcs = []
    for i in range(len(mask_fns)):
        z_srcs.append(asdf.open(mask_fns[i])['header']['SourceRedshift'])
    z_srcs = np.sort(np.array(z_srcs))
    print("redshift sources = ", z_srcs)
    
    # load the galaxies
    data = np.load(gal_fn)
    RA = data['RA']
    DEC = data['DEC']
    Z_RSD = data['Z_RSD']
    Z = data['Z']
    VEL = data['VEL']
    UNIT_LOS = data['UNIT_LOS']
    POS = data['POS']
    POS_RSD = data['POS_RSD']

    # load the randoms
    data = np.load(rand_fn)
    RAND_RA = data['RAND_RA']
    RAND_DEC = data['RAND_DEC']
    RAND_Z = data['RAND_Z']
    RAND_POS = data['RAND_POS']

    # get the random factor
    #RAND_Z = np.random.choice(Z, len(RAND_Z))

    # get masks
    gal_mask, rand_mask = get_dndz_mask(RA, DEC, Z, RAND_RA, RAND_DEC, RAND_Z, nz_dict, z_srcs, mask_fns, chi_of_z) # og
    # apply downsampling to rsd positions
    #gal_mask, rand_mask = get_dndz_mask(RA, DEC, Z_RSD, RAND_RA, RAND_DEC, RAND_Z, nz_dict, z_srcs, mask_fns, chi_of_z) # actually seems to make things worse
    
    # apply the mask cuts
    RA = RA[gal_mask]
    DEC = DEC[gal_mask]
    Z = Z[gal_mask]
    Z_RSD = Z_RSD[gal_mask]
    POS = POS[gal_mask]
    POS_RSD = POS_RSD[gal_mask]
    VEL = VEL[gal_mask]
    UNIT_LOS = UNIT_LOS[gal_mask]

    # apply the mask cuts
    RAND_RA = RAND_RA[rand_mask]
    RAND_DEC = RAND_DEC[rand_mask]
    RAND_Z = RAND_Z[rand_mask]
    RAND_POS = RAND_POS[rand_mask]
    
    # TESTING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    z_edges = np.linspace(0.1, 2.5, 101)
    z_cent = 0.5*(z_edges[1:] + z_edges[:-1])
    dNdz, _ = np.histogram(Z, bins=z_edges)
    dNdz_rand, _ = np.histogram(RAND_Z, bins=z_edges)
    ratio = (dNdz_rand/dNdz)
    ratio[np.isclose(dNdz_rand, 0.) & np.isclose(dNdz, 0.)] = 0.
    print("should be constant and large (20-60)", ratio[(z_cent < 0.9) & (z_cent > 0.2)])
    dNdz, _ = np.histogram(Z_RSD, bins=z_edges)
    dNdz_rand, _ = np.histogram(RAND_Z, bins=z_edges)
    ratio = (dNdz_rand/dNdz)
    ratio[np.isclose(dNdz_rand, 0.) & np.isclose(dNdz, 0.)] = 0.
    print("should be constant and large (20-60)", ratio[(z_cent < 0.9) & (z_cent > 0.2)])
    
    # add photometric error
    if not np.isclose(photoz_error, 0.):
        #sigmaz/(1+z) = 0.02
        # ok, let's think about it; so if I give you the N(z), you can 
        if np.isclose(photoz_error, 100.): # special case
            Z_min = Z.min()
            Z_max = Z.max()
            nbins = 100
            from astropy.table import Table
            cat = Table(fitsio.read("/pscratch/sd/r/rongpu/tmp/dr9_lrg_1.1.1_pzbins_20221204_y1_specz.fits"))
            Z_DESI = cat['Z']
            Z_PHOTO_DESI = cat['Z_PHOT_MEDIAN']
            z_edges = np.linspace(Z_min, Z_max, nbins+1)
            for i in range(nbins):
                print(i)
                choice = (z_edges[i] <= Z_DESI) & (z_edges[i+1] > Z_DESI)
                choice_sim = (z_edges[i] <= Z) & (z_edges[i+1] > Z)
                choice_sim_rsd = (z_edges[i] <= Z_RSD) & (z_edges[i+1] > Z_RSD)
                Z[choice_sim] = np.random.choice(Z_PHOTO_DESI[choice], size=np.sum(choice_sim), replace=True)
                Z_RSD[choice_sim_rsd] = np.random.choice(Z_PHOTO_DESI[choice], size=np.sum(choice_sim_rsd), replace=True)
            
            # NOTE THIS ONLY WORKS before geometry
            if Z_max < 0.8001 or "huge" in sim_name:
                RAND_Z = np.random.choice(Z, size=len(RAND_Z), replace=True)
                
        else:
            sigma_Z = photoz_error/100.*(1.+Z)
            sigma_Z_RSD = photoz_error/100.*(1.+Z_RSD)
            Z = Z + np.random.randn(len(Z)) * sigma_Z
            Z_RSD = Z_RSD + np.random.randn(len(Z_RSD)) * sigma_Z_RSD

    if not np.isclose(photoz_error, 0.):
        sigma_Z = photoz_error/100.*(1.+RAND_Z)
        RAND_Z = RAND_Z + np.random.randn(len(RAND_Z)) * sigma_Z
        
    # save to file
    np.savez(final_gals_fn, RA=RA, DEC=DEC, Z_RSD=Z_RSD, Z=Z, VEL=VEL, POS=POS, POS_RSD=POS_RSD, UNIT_LOS=UNIT_LOS)

    # save to file
    np.savez(final_rand_fn, RAND_RA=RAND_RA, RAND_DEC=RAND_DEC, RAND_Z=RAND_Z, RAND_POS=RAND_POS)
    
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
