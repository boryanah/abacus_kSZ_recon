import os
import glob

import numpy as np
np.random.seed(300)

def get_vertices_cube(units=0.5, N=3):
    vertices = 2*((np.arange(2**N)[:,None] & (1 << np.arange(N))) > 0) - 1
    return vertices*units

def is_in_cube(x_pos, y_pos, z_pos, verts):
    x_min = np.min(verts[:,0])
    x_max = np.max(verts[:,0])
    y_min = np.min(verts[:,1])
    y_max = np.max(verts[:,1])
    z_min = np.min(verts[:,2])
    z_max = np.max(verts[:,2])

    mask = (x_pos > x_min) & (x_pos <= x_max) & (y_pos > y_min) & (y_pos <= y_max) & (z_pos > z_min) & (z_pos <= z_max)
    return mask

def gen_rand(N, chi_min, chi_max, fac, Lbox, offset, origins):
    # number of randoms to generate
    N_rands = fac*N

    # location of observer
    origins = np.atleast_2d(origins)
    origin = origins[0]
    if origins.shape[0] == 1: # then this has to be the huge box
        assert np.all(np.isclose(origin, 0.)), "this is probably `base` box, so pass all three origins"

    # generate randoms on the unit sphere 
    if origins.shape[0] == 1: # then this has to be the huge box
        costheta = np.random.rand(N_rands)*2.-1.
        phi = np.random.rand(N_rands)*2.*np.pi
    if origins.shape[0] == 3: # then this has to be the base box
        costheta = np.random.rand(N_rands)*1.
        phi = np.random.rand(N_rands)*np.pi/2.
    theta = np.arccos(costheta)
    x_cart = np.sin(theta)*np.cos(phi)
    y_cart = np.sin(theta)*np.sin(phi)
    z_cart = np.cos(theta)
    rands_chis = np.random.rand(N_rands)*(chi_max-chi_min)+chi_min

    # unit vector pointing to each random
    rands_norm = np.vstack((x_cart, y_cart, z_cart)).T
    
    # multiply the unit vectors by comoving distance to observer
    x_cart *= rands_chis
    y_cart *= rands_chis
    z_cart *= rands_chis
    
    # vector between centers of the cubes and origin in Mpc/h (i.e. placing observer at 0, 0, 0)
    box0 = np.array([0., 0., 0.])-origin
    if origins.shape[0] > 1: # not true of only the huge box where the origin is at the center
        assert origins.shape[0] == 3
        assert np.all(origins[1]+np.array([0., 0., Lbox]) == origins[0])
        assert np.all(origins[2]+np.array([0., Lbox, 0.]) == origins[0])
        box1 = np.array([0., 0., Lbox])-origin
        box2 = np.array([0., Lbox, 0.])-origin
    
    # vertices of a cube centered at 0, 0, 0
    vert = get_vertices_cube(units=Lbox/2.)

    # remove edges because this is inherent to the light cone catalogs
    x_vert = vert[:, 0]
    y_vert = vert[:, 1]
    z_vert = vert[:, 2]
    vert[x_vert < 0, 0] += offset
    vert[x_vert > 0, 0] -= offset
    vert[y_vert < 0, 1] += offset
    vert[z_vert < 0, 2] += offset
    if origins.shape[0] == 1: # true of the huge box where the origin is at the center
        vert[y_vert > 0, 1] -= offset
        vert[z_vert > 0, 2] -= offset

    
    # vertices for all three boxes
    vert0 = box0+vert
    if origins.shape[0] > 1 and chi_max >= (Lbox-offset): # not true of only the huge boxes and at low zs for base
        vert1 = box1+vert
        vert2 = box2+vert

    # mask for whether or not the coordinates are within the vertices
    mask0 = is_in_cube(x_cart, y_cart, z_cart, vert0)
    if origins.shape[0] > 1 and chi_max >= (Lbox-offset):
        mask1 = is_in_cube(x_cart, y_cart, z_cart, vert1)
        mask2 = is_in_cube(x_cart, y_cart, z_cart, vert2)
        mask = mask0 | mask1 | mask2
    else:
        mask = mask0
    print("masked randoms = ", np.sum(mask)*100./len(mask))

    rands_pos = np.vstack((x_cart[mask], y_cart[mask], z_cart[mask])).T
    rands_norm = rands_norm[mask]
    rands_chis = rands_chis[mask]
    rands_pos += origin

    return rands_pos, rands_norm, rands_chis
