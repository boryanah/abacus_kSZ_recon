#!/usr/bin/env python3
'''
This is a script for generating HOD mock catalogs.

Usage
-----
$ python ./run_hod.py --help

Before that complete subsampling with:
python -m abacusnbody.hod.prepare_sim --path2config config/cmass_box.yaml
'''

import os
import glob
import time
import sys

import yaml
import numpy as np
import argparse

from abacusnbody.hod.abacus_hod import AbacusHOD

"""
python get_hod.py --path2config config/desi_lc_hod.yaml
python get_hod.py --path2config config/desi_lc_hod_high_density.yaml
python get_hod.py --path2config config/desi_box_hod.yaml
python get_hod.py --path2config config/desi_box_hod_high_density.yaml
"""

DEFAULTS = {}
DEFAULTS['path2config'] = 'config/desi_box_hod.yaml'
#DEFAULTS['path2config'] = 'config/desi_lc_hod.yaml'

# LRG bounds dictionary
LRG_dic = {
    'logM_cut': [0, 12.0, 13.8],
    'logM1': [1, 13.0, 15.5],
    'sigma': [2, -3.0, 1.0], # log scale
    'alpha': [3, 0.3, 1.5],
    'kappa': [4, 0.0, 1.0],
    'alpha_c': [5, 0.0, 0.7],
    'alpha_s': [6, 0.5, 1.5],
    'Bcent': [7, -1, 1],
    'Bsat': [8, -1, 1],
    's': [9, -1, 1],
}

# ELG bounds dictionary
ELG_dic = {
    'Q': [-1, 100., 100.],
    'p_max': [0, 0.01, 1],
    'logM_cut': [1, 11.2, 13],
    'kappa': [2, 0.0, 10.0],
    'sigma': [3, 0.01, 10],
    'logM1': [4, 12, 16],
    'alpha': [5, 0.4, 1.7],
    'gamma': [6, 0.1, 8],
    'alpha_c': [7, 0.0, 1.0],
    'alpha_s': [8, 0.2, 1.8],
    'A_s': [9, 0.1, 10],
    'delta_M1': [10, -5, 0],
    'alpha1': [11, -3, 3],
    'beta': [12, 0, 10],
}


def extract_redshift(fn):
    red = float(fn.split('z')[-1][:5])
    return red

def get_prime(param, z, z1, z2, p1, p2):
    tracer = param[-3:]
    if tracer == "ELG":
        order, p_min, p_max = ELG_dic[param[:-4]]
    elif tracer == "LRG":
        order, p_min, p_max = LRG_dic[param[:-4]]
    a = 1./(1+z)
    a1 = 1./(1+z1)
    a2 = 1./(1+z2)
    pr = (p1-p2)/(a1-a2)
    p = p2 + pr*(a-a2)
    if p < p_min: p = p_min
    if p > p_max: p = p_max
    return p

def main(path2config):

    # load the yaml parameters
    config = yaml.safe_load(open(path2config))
    sim_params = config['sim_params']
    HOD_params = config['HOD_params']
    clustering_params = config['clustering_params']
    
    # additional parameter choices
    want_rsd = HOD_params['want_rsd']
    write_to_disk = HOD_params['write_to_disk']
    bin_params = clustering_params['bin_params']
    rpbins = np.logspace(bin_params['logmin'], bin_params['logmax'], bin_params['nbins'])
    pimax = clustering_params['pimax']
    pi_bin_size = clustering_params['pi_bin_size']
    
    # for what redshifts are subsampled halos and particles available
    cat_lc_dir = os.path.join(sim_params['subsample_dir'], sim_params['sim_name'])
    sim_slices = sorted(glob.glob(os.path.join(cat_lc_dir,'z*')))
    redshifts = [extract_redshift(sim_slices[i]) for i in range(len(sim_slices))]
    print("fyi available redshifts = ", redshifts)
    
    # choose mode
    mode = "interp" #"prime" # "interp"
    
    # evolving HOD parameters
    # LRG
    z_LRG = [0.8, 0.5]
    logM_cut_LRG = [12.65213189, 12.77009669]
    logM1_LRG = [13.99638846, 13.92400013]
    sigma_LRG = [-2.51427767, -2.81441829]
    alpha_LRG = [1.18512723, 1.35525166]
    kappa_LRG = [0.2463739, 0.39624613]
    alpha_c_LRG = [0.16522117, 0.30771436]
    alpha_s_LRG = [0.9429617, 0.91292781]
    Bcent_LRG = [0.11726967,0.06312268]
    Bsat_LRG = [-0.92071821,-0.20589251]
    s_LRG = [0.12369804,0.48072072]

    # ELG
    z_ELG = [1.1, 0.8]
    Q_ELG = [100, 100]
    p_max_ELG = [0.15566863, 0.11575492]
    logM_cut_ELG = [11.65214295, 11.52047975]
    kappa_ELG = [8.9800317, 3.58274643]
    sigma_ELG = [2.4698282, 1.17891817]
    logM1_ELG = [15.31983462, 15.9652601]
    alpha_ELG = [1.29356197, 1.28754672]
    gamma_ELG = [5.07887594, 4.04182441]
    alpha_c_ELG = [0.01732342, 0.17965612]
    alpha_s_ELG = [0.20130412, 0.52305243]
    A_s_ELG = [0.10598401,0.28941215]
    delta_M1_ELG = [-1.69748484,-2.54315379]
    alpha1_ELG = [-2.98799382,-2.59018638]
    beta_ELG = [9.94108137,6.7696066]

    if "high_density" in path2config:
        logM_cut_LRG = np.array(logM_cut_LRG)
        logM1_LRG = np.array(logM1_LRG)
        logM_cut_ELG = np.array(logM_cut_ELG)
        logM1_ELG = np.array(logM1_ELG)
        logM_cut_LRG -= 0.3
        logM1_LRG -= 0.3
        logM_cut_ELG -= 0.3
        logM1_ELG -= 0.3
    
    if mode == "prime":
        #logM_cut(z) = logM_cut(z_pivot) + logM_cut_pr*Delta_a
        Delta_a_LRG = 1./(1+z_LRG[0])-1./(1+z_LRG[1])

        # primes
        logM_cut_pr_LRG = (logM_cut_LRG[0]-logM_cut_LRG[1])/Delta_a_LRG
        logM1_pr_LRG = (logM1_LRG[0]-logM1_LRG[1])/Delta_a_LRG

        # pivots
        z_pivot_LRG = z_LRG[1]
        logM_cut_LRG = logM_cut_LRG[1]
        logM1_LRG = logM1_LRG[1]

        # averages
        sigma_LRG = np.exp((sigma_LRG[0]+sigma_LRG[1])*.5)
        alpha_LRG = (alpha_LRG[0]+alpha_LRG[1])*.5
        kappa_LRG = (kappa_LRG[0]+kappa_LRG[1])*.5
        alpha_c_LRG = (alpha_c_LRG[0]+alpha_c_LRG[1])*.5
        alpha_s_LRG = (alpha_s_LRG[0]+alpha_s_LRG[1])*.5
 
        #logM_cut(z) = logM_cut(z_pivot) + logM_cut_pr*Delta_a
        Delta_a_ELG = 1./(1+z_ELG[0])-1./(1+z_ELG[1])

        # primes
        logM_cut_pr_ELG = (logM_cut_ELG[0]-logM_cut_ELG[1])/Delta_a_ELG
        logM1_pr_ELG = (logM1_ELG[0]-logM1_ELG[1])/Delta_a_ELG

        # pivots
        z_pivot_ELG = z_ELG[1]
        logM_cut_ELG = logM_cut_ELG[1]
        logM1_ELG = logM1_ELG[1]

        # averages
        sigma_ELG = ((sigma_ELG[0]+sigma_ELG[1])*.5)
        alpha_ELG = (alpha_ELG[0]+alpha_ELG[1])*.5
        kappa_ELG = (kappa_ELG[0]+kappa_ELG[1])*.5
        p_max_ELG = (p_max_ELG[0]+p_max_ELG[1])*.5
        Q_ELG = (Q_ELG[0]+Q_ELG[1])*.5
        gamma_ELG = (gamma_ELG[0]+gamma_ELG[1])*.5
        alpha_c_ELG = (alpha_c_ELG[0]+alpha_c_ELG[1])*.5
        alpha_s_ELG = (alpha_s_ELG[0]+alpha_s_ELG[1])*.5

    # requested redshifts
    if 'box' in path2config:
        redshifts = [0.5]
    else:
        redshifts = [0.300, 0.350, 0.400, 0.450, 0.500, 0.575, 0.650, 0.725,  0.800, 0.875, 0.950, 1.025, 1.100, 1.175, 1.250, 1.325, 1.400]
    want_rsds = [True, False]
    print("doing", redshifts)

    phases = np.arange(25, dtype=np.int) # 15 0.875 is the last one
    for k in range(len(phases)):
        # TESTING!!!!!!!!!!!!!!!!!
        #if phases[k] != 2: continue
        if phases[k] <= 10: continue
        
        sim_name = f"AbacusSummit_base_c000_ph{phases[k]:03d}"
        print(sim_name)
        sim_params['sim_name'] = sim_name
        
        for i in range(len(redshifts)):
            
            # this redshift
            redshift = redshifts[i]

            # modify redshift in sim_params
            sim_params['z_mock'] = redshift
            assert HOD_params['write_to_disk'] == write_to_disk == True

            if mode == "interp":
                # primes (zero since we're inputting by hand)
                logM_cut_pr_LRG = 0.
                logM1_pr_LRG = 0.

                # pivots (doesn't matter)
                z_pivot_LRG = z_LRG[1]

                # interp
                logM_cut_LRG_this = get_prime('logM_cut_LRG', redshift, z_LRG[0], z_LRG[1], logM_cut_LRG[0], logM_cut_LRG[1])
                logM1_LRG_this = get_prime('logM1_LRG', redshift, z_LRG[0], z_LRG[1], logM1_LRG[0], logM1_LRG[1])
                sigma_LRG_this = np.exp(get_prime('sigma_LRG', redshift, z_LRG[0], z_LRG[1], sigma_LRG[0], sigma_LRG[1]))
                alpha_LRG_this = get_prime('alpha_LRG', redshift, z_LRG[0], z_LRG[1], alpha_LRG[0], alpha_LRG[1])
                kappa_LRG_this = get_prime('kappa_LRG', redshift, z_LRG[0], z_LRG[1], kappa_LRG[0], kappa_LRG[1])
                alpha_c_LRG_this = get_prime('alpha_c_LRG', redshift, z_LRG[0], z_LRG[1], alpha_c_LRG[0], alpha_c_LRG[1])
                alpha_s_LRG_this = get_prime('alpha_s_LRG', redshift, z_LRG[0], z_LRG[1], alpha_s_LRG[0], alpha_s_LRG[1])
                Bcent_LRG_this = get_prime('Bcent_LRG', redshift, z_LRG[0], z_LRG[1], Bcent_LRG[0], Bcent_LRG[1])
                Bsat_LRG_this = get_prime('Bsat_LRG', redshift, z_LRG[0], z_LRG[1], Bsat_LRG[0], Bsat_LRG[1])
                s_LRG_this = get_prime('s_LRG', redshift, z_LRG[0], z_LRG[1], s_LRG[0], s_LRG[1])

                # primes (zero since we're inputting by hand)
                logM_cut_pr_ELG = 0.
                logM1_pr_ELG = 0.

                # pivots (doesn't matter)
                z_pivot_ELG = z_ELG[1]

                # interp
                logM_cut_ELG_this = get_prime('logM_cut_ELG', redshift, z_ELG[0], z_ELG[1], logM_cut_ELG[0], logM_cut_ELG[1])
                logM1_ELG_this = get_prime('logM1_ELG', redshift, z_ELG[0], z_ELG[1], logM1_ELG[0], logM1_ELG[1])
                sigma_ELG_this = get_prime('sigma_ELG', redshift, z_ELG[0], z_ELG[1], sigma_ELG[0], sigma_ELG[1])
                alpha_ELG_this = get_prime('alpha_ELG', redshift, z_ELG[0], z_ELG[1], alpha_ELG[0], alpha_ELG[1])
                kappa_ELG_this = get_prime('kappa_ELG', redshift, z_ELG[0], z_ELG[1], kappa_ELG[0], kappa_ELG[1])
                p_max_ELG_this = get_prime('p_max_ELG', redshift, z_ELG[0], z_ELG[1], p_max_ELG[0], p_max_ELG[1])
                Q_ELG_this = get_prime('Q_ELG', redshift, z_ELG[0], z_ELG[1], Q_ELG[0], Q_ELG[1])
                gamma_ELG_this = get_prime('gamma_ELG', redshift, z_ELG[0], z_ELG[1], gamma_ELG[0], gamma_ELG[1])
                alpha_c_ELG_this = get_prime('alpha_c_ELG', redshift, z_ELG[0], z_ELG[1], alpha_c_ELG[0], alpha_c_ELG[1])
                alpha_s_ELG_this = get_prime('alpha_s_ELG', redshift, z_ELG[0], z_ELG[1], alpha_s_ELG[0], alpha_s_ELG[1])
                A_s_ELG_this = get_prime('A_s_ELG', redshift, z_ELG[0], z_ELG[1], A_s_ELG[0], A_s_ELG[1])
                delta_M1_ELG_this = get_prime('delta_M1_ELG', redshift, z_ELG[0], z_ELG[1], delta_M1_ELG[0], delta_M1_ELG[1])
                alpha1_ELG_this = get_prime('alpha1_ELG', redshift, z_ELG[0], z_ELG[1], alpha1_ELG[0], alpha1_ELG[1])
                beta_ELG_this = get_prime('beta_ELG', redshift, z_ELG[0], z_ELG[1], beta_ELG[0], beta_ELG[1])

            # modify the HOD params
            HOD_params['LRG_params']['logM_cut'] = logM_cut_LRG_this
            HOD_params['LRG_params']['logM1'] = logM1_LRG_this
            HOD_params['LRG_params']['logM_cut_pr'] = logM_cut_pr_LRG
            HOD_params['LRG_params']['logM1_pr'] = logM1_pr_LRG
            HOD_params['LRG_params']['kappa'] = kappa_LRG_this
            HOD_params['LRG_params']['sigma'] = sigma_LRG_this
            HOD_params['LRG_params']['alpha'] = alpha_LRG_this
            HOD_params['LRG_params']['alpha_c'] = alpha_c_LRG_this
            HOD_params['LRG_params']['alpha_s'] = alpha_s_LRG_this
            HOD_params['LRG_params']['Bcent'] = Bcent_LRG_this
            HOD_params['LRG_params']['Bsat'] = Bsat_LRG_this
            HOD_params['LRG_params']['s'] = s_LRG_this
            HOD_params['LRG_params']['z_pivot'] = z_pivot_LRG

            HOD_params['ELG_params']['logM_cut'] = logM_cut_ELG_this
            HOD_params['ELG_params']['logM1'] = logM1_ELG_this
            HOD_params['ELG_params']['logM_cut_pr'] = logM_cut_pr_ELG
            HOD_params['ELG_params']['logM1_pr'] = logM1_pr_ELG
            HOD_params['ELG_params']['kappa'] = kappa_ELG_this
            HOD_params['ELG_params']['sigma'] = sigma_ELG_this
            HOD_params['ELG_params']['alpha'] = alpha_ELG_this
            HOD_params['ELG_params']['p_max'] = p_max_ELG_this
            HOD_params['ELG_params']['Q'] = Q_ELG_this
            HOD_params['ELG_params']['gamma'] = gamma_ELG_this
            HOD_params['ELG_params']['alpha_c'] = alpha_c_ELG_this
            HOD_params['ELG_params']['alpha_s'] = alpha_s_ELG_this
            HOD_params['ELG_params']['A_s'] = A_s_ELG_this
            HOD_params['ELG_params']['delta_M1'] = delta_M1_ELG_this
            HOD_params['ELG_params']['alpha1'] = alpha1_ELG_this
            HOD_params['ELG_params']['beta'] = beta_ELG_this
            HOD_params['ELG_params']['z_pivot'] = z_pivot_ELG

            # create a new abacushod object
            print(HOD_params['LRG_params'].items())
            print(HOD_params['ELG_params'].items())
            newBall = AbacusHOD(sim_params, HOD_params, clustering_params)
            ngal_dict = newBall.compute_ngal(Nthread = 16)[0]
            N_gal_LRG = ngal_dict['LRG']
            if 'ELG' in ngal_dict.keys():
                N_gal_ELG = ngal_dict['ELG']
                print("calculated number of ELGs = ", N_gal_ELG)
            print("calculated number of LRGs = ", N_gal_LRG)
            
            for j in range(len(want_rsds)):
                want_rsd = want_rsds[j]

                # run the HOD on the current redshift
                start = time.time()
                mock_dict = newBall.run_hod(tracers=newBall.tracers, want_rsd=want_rsd, reseed=None, write_to_disk=write_to_disk, Nthread = 16)
                print("Done redshift ", redshift, "took time ", time.time() - start)

class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass

if __name__ == "__main__":
    # parsing arguments
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--path2config', help='Path to the config file', default=DEFAULTS['path2config'])
    args = vars(parser.parse_args())
    main(**args)
