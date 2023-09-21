
from pathlib import Path
import os
import gc

import numpy as np
import asdf
import argparse
from scipy.interpolate import interp1d

from pyrecon import  utils, IterativeFFTParticleReconstruction, MultiGridReconstruction, IterativeFFTReconstruction
from cosmoprimo.fiducial import Planck2018FullFlatLCDM, AbacusSummit, DESI, TabulatedDESI

from apply_dndz_mask import parse_nztype

DEFAULTS = {}
DEFAULTS['sim_name'] = "AbacusSummit_base_c000_ph002"
DEFAULTS['stem'] = 'DESI_LRG' # 'DESI_ELG'
DEFAULTS['nmesh'] = 1024
DEFAULTS['sr'] = 12.5 # Mpc/h
DEFAULTS['rectype'] = "MG"
DEFAULTS['convention'] = "recsym"
DEFAULTS['nz_type'] = 'Gaussian(0.5, 0.4)'
DEFAULTS['bz_type'] = "Constant"
DEFAULTS['photoz_error'] = 0.

"""
Usage:
python reconstruct_lc_catalog.py --sim_name AbacusSummit_base_c000_ph002 --stem DESI_LRG --nmesh 1024 --sr 12.5 --rectype MG --convention recsym
python reconstruct_lc_catalog.py --sim_name AbacusSummit_huge_c000_ph201 --stem DESI_LRG --nmesh 1024 --sr 12.5 --rectype MG --convention recsym
"""

def get_nz_str(nz_type):    
    # parse the dndz instructions
    nz_dict = parse_nztype(nz_type)
    if nz_dict['Type'] == "Gaussian":
        nz_str = f"_meanz{nz_dict['Mean_z']:.3f}_deltaz{nz_dict['Delta_z']:.3f}"
    elif nz_dict['Type'] == "StepFunction":
        nz_str = f"_minz{nz_dict['Min_z']:.3f}_maxz{nz_dict['Max_z']:.3f}"
    elif nz_dict['Type'] == "FromFile":
        nz_str = f"_{nz_dict['File']}"
    return nz_str

def main(sim_name, stem, stem2, stem3, nmesh, sr, rectype, convention, nz_type, nz_type2, nz_type3, photoz_error, bz_type, want_fakelc=False, want_mask=False):

    # redshift error string
    photoz_str = f"_zerr{photoz_error:.1f}"
    
    # fake light cones
    fakelc_str = "_fakelc" if want_fakelc else ""
    mask_str = "_mask" if want_mask else ""

    # small area (this is kinda stupid)
    if "_small_area" in nz_type:
        nz_type, small_str = nz_type.split("_small_area")
        mask_str = "_small_area"+small_str
    
    # how many processes to use for reconstruction: 32, 128 physical cpu per node for cori, perlmutter (hyperthreading doubles)
    ncpu = 128*2

    # initiate tracers
    tracers = []

    # initiate stems
    stems = [stem]
    print("sim name, nz type", sim_name, nz_type, nz_type2, nz_type3)
    nz_strs = [get_nz_str(nz_type)]
    if stem2 is not None:
        stems.append(stem2)
        nz_strs.append(get_nz_str(nz_type2))
    if stem3 is not None:
        stems.append(stem3)
        nz_strs.append(get_nz_str(nz_type3))
    nz_full_str = ''.join(nz_strs)
        
    # initiate mock dirs
    mock_dirs = []

    # reconstruction parameters
    if rectype == "IFT":
        recfunc = IterativeFFTReconstruction
    elif rectype == "IFTP":
        recfunc = IterativeFFTParticleReconstruction
    elif rectype == "MG":
        recfunc = MultiGridReconstruction

    tracer_extra_str = []
    for stem in stems:
        # additional specs of the tracer
        extra = '_'.join(stem.split('_')[2:])
        stem = '_'.join(stem.split('_')[:2])
        if extra != '': extra = '_'+extra

        # directory where mock catalogs are saved
        mock_dir = Path(f"/global/cfs/cdirs/desi/users/boryanah/kSZ_recon/new/mocks_lc_output_kSZ_recon{extra}/") / sim_name
        mock_dirs.append(mock_dir)

        # parameter choices
        if stem == 'DESI_LRG':
            Mean_z = 0.5
            tracer = "LRG"
            tracers.append(tracer)
            bias = 2.2 # +/- 10%
        elif stem == 'DESI_ELG':
            Mean_z = 0.8
            tracer = "ELG"
            tracers.append(tracer)
            bias = 1.3 # little effect b/n 1.5
        else:
            print("Other tracers not yet implemented"); quit()
        tracer_extra_str.append(f"{tracer}{extra}")
    tracer_extra_str = '_'.join(tracer_extra_str)

    # tbqh the bias of the reconstruction should not be determined by the last of the multi-tracer combination
    if bz_type == "Constant":
        bias_str = f"_b{bias:.1f}"
    else:
        assert os.path.exists(bz_type), "Needs to be a file name"
        data = np.load(bz_type)
        z = data['z'][data['bias'] > 0.]
        bias = data['bias'][data['bias'] > 0.]
        b_of_z = interp1d(z, bias, bounds_error=False, fill_value=(data['bias'][data['bias'] > 0.][0], data['bias'][data['bias'] > 0.][-1]))
        bias_str = f"_{(bz_type.split('/')[-1]).split('.npz')[0]}"
        bias = 1.
        
    # correction?
    if parse_nztype(nz_type)['Type'] == "Gaussian":
        Mean_z = parse_nztype(nz_type)['Mean_z']
    #if "BGS" in nz_type.upper():
    #    Mean_z = 0.3

    # simulation parameters
    cosmo = DESI() # AbacusSummit
    ff = cosmo.growth_factor(Mean_z)
    H_z = cosmo.hubble_function(Mean_z)
    los = 'local'

    # this is just for recording
    if tracer == "LRG": #"BGS" in nz_type.upper():
        Mean_z = 0.5
    elif tracer == "ELG":
        Mean_z = 0.8

    # directory where the reconstructed mock catalogs are saved
    save_dir = Path("/global/cfs/cdirs/desi/users/boryanah/kSZ_recon/new/")
    save_recon_dir = Path(save_dir) / "recon" / sim_name / f"z{Mean_z:.3f}"
    os.makedirs(save_recon_dir, exist_ok=True)

    # file to save to
    final_fn = save_recon_dir / f"displacements_{tracer_extra_str}{fakelc_str}{photoz_str}{mask_str}_postrecon_R{sr:.2f}{bias_str}_nmesh{nmesh:d}_{convention}_{rectype}{nz_full_str}.npz" 
    #if os.path.exists(final_fn): return
    
    # loop over all tracers
    n_gals = []
    for i, tracer in enumerate(tracers):
        
        # load the galaxies
        data = np.load(mock_dirs[i] / f"galaxies_{tracer}{fakelc_str}{photoz_str}{mask_str}_prerecon{nz_strs[i]}.npz")
        N_this = len(data['RA'])
        if i == 0:
            RA = data['RA']
            DEC = data['DEC']
            Z_RSD = data['Z_RSD']
            Z = data['Z']
            VEL = data['VEL']
            UNIT_LOS = data['UNIT_LOS']
        else:
            RA = np.hstack((RA, data['RA']))
            DEC = np.hstack((DEC, data['DEC']))
            Z_RSD = np.hstack((Z_RSD, data['Z_RSD']))
            Z = np.hstack((Z, data['Z']))
            VEL = np.vstack((VEL, data['VEL']))
            UNIT_LOS = np.vstack((UNIT_LOS, data['UNIT_LOS']))
        n_gals.append(len(RA))
        
        # load the randoms
        data = np.load(mock_dirs[i] / f"randoms_{tracer}{fakelc_str}{photoz_str}{mask_str}_prerecon{nz_strs[i]}.npz")
        N_rand_this = len(data['RAND_RA'])
        if i == 0:
            RAND_RA = data['RAND_RA']
            RAND_DEC = data['RAND_DEC']
            RAND_Z = data['RAND_Z']
            if want_fakelc:
                #RAND_Z_RSD = data['RAND_Z_RSD']
                RAND_Z_RSD = data['RAND_Z']
        else:
            RAND_RA = np.hstack((RAND_RA, data['RAND_RA']))
            RAND_DEC = np.hstack((RAND_DEC, data['RAND_DEC']))
            RAND_Z = np.hstack((RAND_Z, data['RAND_Z']))
            if want_fakelc:
                #RAND_Z_RSD = np.hstack((RAND_Z_RSD, data['RAND_Z_RSD']))
                RAND_Z_RSD = np.hstack((RAND_Z_RSD, data['RAND_Z']))
        print("RA min/max, DEC min/max, Z min/max", RA.min(), RA.max(), DEC.min(), DEC.max(), Z.min(), Z.max())
        print("RAND RA min/max, DEC min/max, Z min/max", RAND_RA.min(), RAND_RA.max(), RAND_DEC.min(), RAND_DEC.max(), RAND_Z.min(), RAND_Z.max())

        """
        # TESTING!!!!!!!!!!!!!!!!!!!!!!!
        min_z = 0.2
        max_z = 1.2
        z_edges = np.linspace(0.1, 2.5, 201)
        z_cent = 0.5*(z_edges[1:] + z_edges[:-1])
        dNdz, _ = np.histogram(Z[-N_this:], bins=z_edges)
        dNdz_rand, _ = np.histogram(RAND_Z[-N_rand_this:], bins=z_edges)
        ratio = (dNdz_rand/dNdz)
        ratio[np.isclose(dNdz_rand, 0.) & np.isclose(dNdz, 0.)] = 0.
        print(tracer, "should be constant and large (20-60)", ratio[(z_cent < max_z) & (z_cent > min_z)])
        """
    n_gals = np.array(n_gals)

    testing = False #True
    if testing:
        print("brutal testing")
        from abacusnbody.metadata import get_meta
        from generate_randoms import gen_rand, is_in_lc 
        from apply_dndz_mask import get_down_choice, gaussian, get_mask_ang
        from prepare_lc_catalog import relate_chi_z, read_dat, get_norm, get_ra_dec_chi
        import glob
        
        # blah
        Mean_z = 0.5
        Delta_z = 0.4
        rands_fac = 100
        offset = 10.
        assert nz_type == "Gaussian(0.5,0.4)", f"{nz_type}"

        # read in file names to determine all the available z's
        mask_dir = f"/global/cfs/cdirs/desi/public/cosmosim/AbacusLensing/v1/{sim_name}/"
        mask_fns = sorted(glob.glob(mask_dir+f"mask_0*.asdf"))
        z_srcs = []
        for i in range(len(mask_fns)):
            z_srcs.append(asdf.open(mask_fns[i])['header']['SourceRedshift'])
        z_srcs = np.sort(np.array(z_srcs))
        print("redshift sources = ", z_srcs)
        
        # read from simulation header
        header = get_meta(sim_name, 0.1)
        Lbox = header['BoxSizeHMpc'] # cMpc/h
        mpart = header['ParticleMassHMsun'] # 5.7e10, 2.1e9
        inv_velz2kms = 1./(header['VelZSpace_to_kms']/Lbox)
        origins = np.array(header['LightConeOrigins']).reshape(-1,3)
        origin = origins[0]
        print(f"mpart = {mpart:.2e}")
        
        # functions relating chi and z
        chi_of_z, z_of_chi = relate_chi_z(sim_name)
        gals_min = np.min(chi_of_z(Z))
        gals_max = np.max(chi_of_z(Z))
        
        # generate randoms in L shape
        rands_pos, rands_norm, rands_chis = gen_rand(len(Z), gals_min, gals_max, rands_fac, Lbox, offset, origins)

        # convert the unit vectors into RA and DEC
        rands_RA, rands_DEC, rands_CZ = get_ra_dec_chi(rands_norm, rands_chis)
        del rands_norm, rands_chis; gc.collect()

        # convert chi to redshift
        rands_Z = z_of_chi(rands_CZ)

        # renaming
        RAND_Z = rands_Z
        RAND_RA = rands_RA
        RAND_DEC = rands_DEC
        RAND_POS = rands_pos

        # find highest redshift
        #z_max = np.max(redshifts)
        z_max = Mean_z+Delta_z
        print("number of galaxies", len(Z))

        # apply masking
        mask_fn = mask_fns[np.argmax(z_srcs - z_max >= 0.)]
        mask = asdf.open(mask_fn)['data']['mask']
        nside = asdf.open(mask_fn)['header']['HEALPix_nside']
        order = asdf.open(mask_fn)['header']['HEALPix_order']
        nest = True if order == 'NESTED' else False

        # mask lenses and randoms
        print("applying mask")
        choice = get_mask_ang(mask, RAND_RA, RAND_DEC, nest, nside)
        print("masked fraction rands", np.sum(choice)/len(choice)*100.)
        #RAND_RA, RAND_DEC, RAND_Z, RAND_POS = RAND_RA[choice], RAND_DEC[choice], RAND_Z[choice], RAND_POS[choice]
        del mask; gc.collect()

        # apply redshift selection
        print("apply redshift downsampling")
        choice &= get_down_choice(RAND_Z, Mean_z, Delta_z) 
        RAND_RA, RAND_DEC, RAND_Z, RAND_POS = RAND_RA[choice], RAND_DEC[choice], RAND_Z[choice], RAND_POS[choice]
        print("kept fraction (20-30%) rands", np.sum(choice)/len(choice)*100.)

    
    # TESTING!!!!!!! additional downsampling to make sure N(z) is matched
    if (stem2 is not None) or (stem3 is not None):  # so far helps with rBGS and rELG but not rLRG
        """
        # old-style po-dobre
        z_edges = np.linspace(0.1, 2.5, 1001)
        z_cent = 0.5*(z_edges[1:] + z_edges[:-1])
        dNdz, _ = np.histogram(Z, bins=z_edges)
        n_per = len(RAND_Z)/len(z_cent)
        dNdz_rand, _ = np.histogram(RAND_Z, bins=z_edges)
        inds = np.digitize(RAND_Z, bins=z_edges) - 1
        w_bin = dNdz_rand/n_per # this kinda gives you the weighting you need to get perfectly uniform stuff
        fac = np.interp(RAND_Z, z_cent, dNdz/np.max(dNdz))
        fac /= w_bin[inds] # downweight many galaxies in bin
        nz_dict = {'Min_z': 0.3, 'Max_z': 1.1}
        fac /= np.max(fac[(RAND_Z > nz_dict['Min_z']) & (RAND_Z < nz_dict['Max_z'])])
        choice = np.random.rand(len(RAND_Z)) < fac
        RAND_RA, RAND_DEC, RAND_Z = RAND_RA[choice], RAND_DEC[choice], RAND_Z[choice]
        """
        """
        # NAI-NOVOTO KOPELE (dava poveche rand./gal ratio ama poradi nqkva prichina otslabva tva maina r_LOS
        z_edges = np.linspace(0.1, 2.5, 101)
        z_cent = 0.5*(z_edges[1:] + z_edges[:-1])
        dNdz, _ = np.histogram(Z, bins=z_edges)
        dNdz_rand, _ = np.histogram(RAND_Z, bins=z_edges)
        ratio = (dNdz_rand/dNdz)
        ratio[np.isclose(dNdz, 0.)] = 0.
        #factor = int(np.round(np.min(ratio[ratio > 0.])))
        factor = (np.min(ratio[ratio > 0.]))
        print("rand_factor", factor)
        dNdz_rand_targ = dNdz*factor
        down = dNdz_rand_targ / dNdz_rand
        inds = np.digitize(RAND_Z, bins=z_edges) - 1
        choice = np.random.rand(len(RAND_Z)) < down[inds]
        RAND_RA, RAND_DEC, RAND_Z = RAND_RA[choice], RAND_DEC[choice], RAND_Z[choice]
        # POMISLI dali e pravilno v applydndz che pravish random.rand a ne random.choice
        """
        # tuks TESTING!!!!!!!!!!!!!!!!!!!!! the idea is that this improves the randoms
        sigma_Z = 0.01*(1.+RAND_Z)
        RAND_Z = RAND_Z + np.random.randn(len(RAND_Z)) * sigma_Z
        
        # nai-nai-novoto
        #z_edges = np.linspace(0.1, 2.5, 1001) # edna suvsem malka ideq po-dobre, no po-bavno
        z_edges = np.linspace(0.2, 1.2, 201)
        z_cent = 0.5*(z_edges[1:] + z_edges[:-1])
        dNdz, _ = np.histogram(Z, bins=z_edges)
        dNdz_rand, _ = np.histogram(RAND_Z, bins=z_edges)
        ratio = (dNdz_rand/dNdz)
        ratio[np.isclose(dNdz, 0.)] = 0.
        factor = int(np.round(np.min(ratio[ratio > 0.])))
        print("factor", factor)
        
        rand_mask = np.zeros(len(RAND_Z), dtype=bool)
        for i_z in range(len(z_edges)-1):
            N_target = int(dNdz[i_z])*factor
            N_raw = int(dNdz_rand[i_z])
            if N_target > 0:
                choice = (RAND_Z > z_edges[i_z]) & (RAND_Z <= z_edges[i_z+1])
                tmp = rand_mask[choice]
                if N_raw <= N_target: N_target = N_raw
                tmp[np.random.choice(int(np.rint(N_raw)), int(np.rint(N_target)), replace=False)] = True
                rand_mask[choice] = tmp
        RAND_RA, RAND_DEC, RAND_Z = RAND_RA[rand_mask], RAND_DEC[rand_mask], RAND_Z[rand_mask]

        # tuks TESTING!!!!!!!!!!!!!!!!!!! unnecessary to save probably
        #np.savez(mock_dirs[0] / f"new_randoms_{tracer}{fakelc_str}{photoz_str}{mask_str}_prerecon{nz_strs[i]}.npz", RAND_RA=RAND_RA, RAND_DEC=RAND_DEC, RAND_Z=RAND_Z)

    # TESTING!!!!!!!!!!!!!!!!!!!!!!!!!!!! checking if it worsens things when z_max < 0.8 (found it's the same)
    #RAND_Z = np.random.choice(Z, len(RAND_RA))
        
    # just print stuff TESTING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    RAND_Z_RSD = RAND_Z
    min_z = 0.2
    max_z = 1.2
    z_edges = np.linspace(0.1, 2.5, 101)
    z_cent = 0.5*(z_edges[1:] + z_edges[:-1])
    dNdz, _ = np.histogram(Z, bins=z_edges)
    dNdz_rand, _ = np.histogram(RAND_Z, bins=z_edges)
    ratio = (dNdz_rand/dNdz)
    ratio[np.isclose(dNdz_rand, 0.) & np.isclose(dNdz, 0.)] = 0.
    print("should be constant and large (20-60)", ratio[(z_cent < max_z) & (z_cent > min_z)])
    print("dNdz", dNdz[(z_cent < max_z) & (z_cent > min_z)])
    print("dNdz_rand", dNdz_rand[(z_cent < max_z) & (z_cent > min_z)])
    dNdz, _ = np.histogram(Z_RSD, bins=z_edges)
    dNdz_rand, _ = np.histogram(RAND_Z_RSD, bins=z_edges)
    ratio = (dNdz_rand/dNdz)
    ratio[np.isclose(dNdz_rand, 0.) & np.isclose(dNdz, 0.)] = 0.
    print("should be constant and large (20-60)", ratio[(z_cent < max_z) & (z_cent > min_z)])
    print("dNdz", dNdz[(z_cent < max_z) & (z_cent > min_z)])
    print("dNdz_rand", dNdz_rand[(z_cent < max_z) & (z_cent > min_z)])
    
    # transform into Cartesian coordinates
    PositionRSD = utils.sky_to_cartesian(cosmo.comoving_radial_distance(Z_RSD), RA, DEC)
    Position = utils.sky_to_cartesian(cosmo.comoving_radial_distance(Z), RA, DEC)
    RandomPosition = utils.sky_to_cartesian(cosmo.comoving_radial_distance(RAND_Z), RAND_RA, RAND_DEC)
    if want_fakelc:
        RandomPositionRSD = utils.sky_to_cartesian(cosmo.comoving_radial_distance(RAND_Z_RSD), RAND_RA, RAND_DEC)
    else:
        RandomPositionRSD = RandomPosition
    del RA, DEC
    del RAND_RA, RAND_DEC#, RAND_Z
    gc.collect()
    if bz_type == "Constant":
        del Z, Z_RSD; gc.collect()
    print("Number of galaxies", Position.shape[0])

    """
    # apparently some dudes are zero
    # tuks TESTING!!!!!!!!!!!!!!!!
    Position[np.isnan(Position)] = 0.
    RandomPosition[np.isnan(RandomPosition)] = 0.
    PositionRSD[np.isnan(PositionRSD)] = 0.
    Position[np.isinf(Position)] = 0.
    RandomPosition[np.isinf(RandomPosition)] = 0.
    PositionRSD[np.isinf(PositionRSD)] = 0.
    print("SUPER DUMB", Position.shape)
    """
    
    # run reconstruction on the mocks w/o RSD
    print('Recon First tracer') # TESTING!!!!!
    recon_tracer = recfunc(f=ff, bias=bias, nmesh=nmesh, los=los, positions=Position, # used only to define box size if not provided
                           nthreads=int(ncpu), fft_engine='fftw', fft_plan='estimate', dtype='f4', wrap=True) # probably doesn't matter; in principle, False
    print('grid set up',flush=True)
    if bz_type == "Constant":
        recon_tracer.assign_data(Position, weights=None)
        print('data assigned',flush=True)
        recon_tracer.assign_randoms(RandomPosition)
        print('randoms assigned',flush=True)
        recon_tracer.set_density_contrast(smoothing_radius=sr)
    else:
        # using randoms is a bit better than galaxy
        recon_tracer.assign_data(RandomPosition, weights=b_of_z(RAND_Z))
        print('data assigned',flush=True)
        #del Z; gc.collect()
        mesh_bias = np.zeros_like(recon_tracer.mesh_data)
        mesh_bias[:, :, :] = recon_tracer.mesh_data[:, :, :]
        recon_tracer.mesh_data[:, :, :] = 0. # otherwise adds to mesh
        recon_tracer.assign_data(RandomPosition, weights=None)
        mask = np.isclose(recon_tracer.mesh_data, 0.)
        # obtain b(x)
        mesh_bias[~mask] = mesh_bias[~mask]/recon_tracer.mesh_data[~mask]
        # QKO TESTING!!!!!!! adding is better
        mesh_bias[mask] = np.mean(mesh_bias[~mask]) + 0.6 #  0.3 is better +0.01 is almost as good as mean (-0.01 worse)
        recon_tracer.mesh_data[:, :, :] = 0. # otherwise adds to mesh
        #print("mean bias", np.mean(mesh_bias[~mask]))
        recon_tracer.assign_data(Position, weights=None)
        #mesh_bias[:] = 2.2 # TESTING matches
        recon_tracer.assign_randoms(RandomPosition)
        print('randoms assigned',flush=True)
        # compute delta_g/b
        recon_tracer.set_density_contrast(smoothing_radius=sr) # og
        #recon_tracer.set_density_contrast(smoothing_radius=0.) # TESTING mn zle
        recon_tracer.mesh_delta /= mesh_bias
        #recon_tracer.mesh_delta.smooth_gaussian(sr) 

    print('density constrast calculated, now doing recon',flush=True)
    recon_tracer.run()
    print('recon has been run',flush=True)

    # read the displacements in real space
    if rectype == 'IFTP':
        displacements = recon_tracer.read_shifts('data', field='disp')
    else:
        displacements = recon_tracer.read_shifts(Position, field='disp')
    random_displacements = recon_tracer.read_shifts(RandomPosition, field='disp')
    del Position; gc.collect()

    '''
    # TESTING!!!!!!!!!!!! this is if you only want to run recon once
    displacements_rsd = np.zeros_like(VEL, dtype=np.float32)
    displacements_rsd_nof = np.zeros_like(VEL, dtype=np.float32)
    random_displacements_rsd = np.zeros((len(RAND_Z), 3), dtype=np.float32)
    random_displacements_rsd_nof = np.zeros((len(RAND_Z), 3), dtype=np.float32)
    del RAND_Z; gc.collect()
    # start commenting out here (single quotation) to only do reconstruction once 
    '''
    # run reconstruction on the mocks w/ RSD
    print('Recon Second tracer')
    recon_tracer = recfunc(f=ff, bias=bias, nmesh=nmesh, los=los, positions=PositionRSD,
                           nthreads=int(ncpu), fft_engine='fftw', fft_plan='estimate', dtype='f4', wrap=True)
    print('grid set up',flush=True)


    if bz_type == "Constant":
        recon_tracer.assign_data(PositionRSD, weights=None)
        print('data assigned',flush=True)
        recon_tracer.assign_randoms(RandomPositionRSD)
        print('randoms assigned',flush=True)
        recon_tracer.set_density_contrast(smoothing_radius=sr)
    else:
        recon_tracer.assign_data(RandomPositionRSD, weights=b_of_z(RAND_Z_RSD))
        print('data assigned',flush=True)
        #del Z_RSD; gc.collect()
        mesh_bias = np.zeros_like(recon_tracer.mesh_data)
        mesh_bias[:, :, :] = recon_tracer.mesh_data[:, :, :]
        recon_tracer.mesh_data[:, :, :] = 0. # otherwise adds to mesh
        recon_tracer.assign_data(RandomPositionRSD, weights=None)
        mask = np.isclose(recon_tracer.mesh_data, 0.)
        # obtain b(x)
        mesh_bias[~mask] = mesh_bias[~mask]/recon_tracer.mesh_data[~mask]
        mesh_bias[mask] = np.mean(mesh_bias[~mask]) # fixes size but yields worse stuff?
        recon_tracer.mesh_data[:, :, :] = 0. # otherwise adds to mesh
        #print("mean bias", np.mean(mesh_bias[~mask]))
        recon_tracer.assign_data(PositionRSD)
        recon_tracer.assign_randoms(RandomPositionRSD)
        print('randoms assigned',flush=True)
        # compute delta_g/b
        recon_tracer.set_density_contrast(smoothing_radius=sr) # og
        #recon_tracer.set_density_contrast(smoothing_radius=0.) # TESTING mn zle
        recon_tracer.mesh_delta /= mesh_bias
        #recon_tracer.mesh_delta.smooth_gaussian(sr)
        """
        recon_tracer.assign_data(PositionRSD, weights=b_of_z(Z_RSD))
        print('data assigned',flush=True)
        del Z_RSD; gc.collect()
        mesh_bias = np.zeros_like(recon_tracer.mesh_data)
        mesh_bias[:, :, :] = recon_tracer.mesh_data[:, :, :]
        recon_tracer.mesh_data[:, :, :] = 0. # otherwise adds to mesh
        recon_tracer.assign_data(PositionRSD, weights=None)
        mask = np.isclose(recon_tracer.mesh_data, 0.)
        # obtain b(x)
        mesh_bias[~mask] = mesh_bias[~mask]/recon_tracer.mesh_data[~mask]
        mesh_bias[mask] = np.mean(mesh_bias[~mask]) # fixes size but yields worse stuff?
        recon_tracer.assign_randoms(RandomPositionRSD)
        print('randoms assigned',flush=True)
        # compute delta_g/b
        recon_tracer.set_density_contrast(smoothing_radius=sr) # og
        #recon_tracer.set_density_contrast(smoothing_radius=0.) # TESTING mn zle
        recon_tracer.mesh_delta /= mesh_bias
        #recon_tracer.mesh_delta.smooth_gaussian(sr)
        """
    print('density constrast calculated, now doing recon',flush=True)
    recon_tracer.run()
    print('recon has been run',flush=True)

    # read the displacements in real and redshift space (rsd has the (1+f) factor in the LOS direction)
    if rectype == "IFTP":
        displacements_rsd = recon_tracer.read_shifts('data', field='disp+rsd')
        displacements_rsd_nof = recon_tracer.read_shifts('data', field='disp')
    else:
        displacements_rsd = recon_tracer.read_shifts(PositionRSD, field='disp+rsd')
        displacements_rsd_nof = recon_tracer.read_shifts(PositionRSD, field='disp')
    random_displacements_rsd = recon_tracer.read_shifts(RandomPositionRSD, field='disp+rsd')
    random_displacements_rsd_nof = recon_tracer.read_shifts(RandomPositionRSD, field='disp')

    # end commenting out here to only do reconstruction once 

    """
    displacements = np.zeros_like(VEL, dtype=np.float32)
    displacements_rsd = np.zeros_like(VEL, dtype=np.float32)
    displacements_rsd_nof = np.zeros_like(VEL, dtype=np.float32)
    random_displacements = np.zeros((len(RAND_Z), 3), dtype=np.float32)
    random_displacements_rsd = np.zeros((len(RAND_Z), 3), dtype=np.float32)
    random_displacements_rsd_nof = np.zeros((len(RAND_Z), 3), dtype=np.float32)
    """
    
    # save the displacements
    np.savez(final_fn,
             displacements=displacements, displacements_rsd=displacements_rsd, velocities=VEL, unit_los=UNIT_LOS, growth_factor=ff, Hubble_z=H_z,
             random_displacements_rsd=random_displacements_rsd, random_displacements=random_displacements, displacements_rsd_nof=displacements_rsd_nof,
             random_displacements_rsd_nof=random_displacements_rsd_nof, n_gals=n_gals)

class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--sim_name', help='Simulation name', default=DEFAULTS['sim_name'])
    parser.add_argument('--stem', help='Stem file name', default=DEFAULTS['stem'])#, choices=["DESI_LRG", "DESI_ELG", "DESI_LRG_high_density", "DESI_LRG_bgs", "DESI_ELG_high_density"])
    parser.add_argument('--stem2', help='Stem file name 2', default=None)#, choices=["DESI_LRG", "DESI_ELG", "DESI_LRG_high_density", "DESI_LRG_bgs", "DESI_ELG_high_density"])
    parser.add_argument('--stem3', help='Stem file name 3', default=None)#, choices=["DESI_LRG", "DESI_ELG", "DESI_LRG_high_density", "DESI_LRG_bgs", "DESI_ELG_high_density"])
    parser.add_argument('--nmesh', help='Number of cells per dimension for reconstruction', type=int, default=DEFAULTS['nmesh'])
    parser.add_argument('--sr', help='Smoothing radius', type=float, default=DEFAULTS['sr'])
    parser.add_argument('--rectype', help='Reconstruction type', default=DEFAULTS['rectype'], choices=["IFT", "MG", "IFTP"])
    parser.add_argument('--convention', help='Reconstruction convention', default=DEFAULTS['convention'], choices=["recsym", "reciso"])
    parser.add_argument('--photoz_error', help='Percentage error on the photometric redshifts', default=DEFAULTS['photoz_error'], type=float)
    parser.add_argument('--nz_type', help='Type of N(z) distribution: Gaussian(mean, 2sigma), Step(min, max), file name', default=DEFAULTS['nz_type'])
    parser.add_argument('--bz_type', help='Type of b(z) distribution', default=DEFAULTS['bz_type'])
    parser.add_argument('--nz_type2', help='Type of N(z) distribution: Gaussian(mean, 2sigma), Step(min, max), file name', default=None)
    parser.add_argument('--nz_type3', help='Type of N(z) distribution: Gaussian(mean, 2sigma), Step(min, max), file name', default=None)
    parser.add_argument('--want_fakelc', help='Want to use the fake light cone?', action='store_true')
    parser.add_argument('--want_mask', help='Want to apply DESI mask?', action='store_true')
    args = vars(parser.parse_args())
    main(**args)
