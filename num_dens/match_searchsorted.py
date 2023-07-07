#!/usr/bin/env python

from numpy import *
from numba import jit
import numpy as np

# og
dtype = np.int32
#dtype = np.int64 # needed for huge box ph201

@jit(nopython=True)
def match(arr1, arr2, arr2_sorted=False, arr2_index=None):
    """
    For each element in arr1 return the index of the element with the
    same value in arr2, or -1 if there is no element with the same value.
    Setting arr2_sorted=True will save some time if arr2 is already sorted
    into ascending order.

    A precomputed sorting index for arr2 can be supplied using the
    arr2_index parameter. This can save time if the routine is called
    repeatedly with the same arr2 but arr2 is not already sorted.

    It is assumed that each element in arr1 only occurs once in arr2.
    """

    #arr1_n = arr1
    #arr2_n = arr2

    # Sort arr2 into ascending order if necessary
    tmp1 = arr1
    if arr2_sorted:
        tmp2 = arr2
    else:
        if arr2_index is None:
            idx = argsort(arr2)
            tmp2 = arr2[idx]
        else:
            # Use supplied sorting index
            idx = arr2_index
            tmp2 = arr2[arr2_index]

    # Find where elements of arr1 are in arr2
    ptr  = searchsorted(tmp2, tmp1)

    # Make sure all elements in ptr are valid indexes into tmp2
    # (any out of range entries won't match so they'll get set to -1
    # in the next bit)
    ptr[ptr>=len(tmp2)] = 0
    ptr[ptr<0]          = 0

    # Return -1 where no match is found
    ptr[where(tmp2[ptr] != tmp1)[0]] = -1
    
    # Put ptr back into original order
    if arr2_sorted:
        ptr = where(ptr>= 0, arange(len(tmp2))[ptr], -1)
    else:
        ind = arange(len(arr2))[idx]
        ptr = where(ptr>= 0, ind[ptr], -1)

    return ptr


@jit(nopython=True)
def match_reduced(arr1, arr2):
    """
    For each element in arr1 return the index of the element with the
    same value in arr2, or -1 if there is no element with the same value.
    It is assumed that arr2 is already sorted into ascending order and
    that each element in arr1 only occurs once in arr2.
    """

    # Find where elements of arr1 are in arr2
    ptr = zeros(len(arr1), dtype=dtype)
    ptr[:] = searchsorted(arr2, arr1)

    # Make sure all elements in ptr are valid indexes into arr2
    # (any out of range entries won't match so they'll get set to -1
    # in the next bit)
    ptr[ptr >= len(arr2)] = 0
    ptr[ptr < 0]          = 0

    # Return -1 where no match is found
    ptr[where(arr2[ptr] != arr1)[0]] = -1
    
    # Put ptr back into original order
    ptr = where(ptr >= 0, arange(len(arr2), dtype=dtype)[ptr], -1)

    return ptr


@jit(nopython=True)
def match_srt(arr1, arr2, cond):
    """
    For each element in arr1 return the index of the element with the
    same value in arr2, or -1 if there is no element with the same value.
    It is assumed that arr2 is already sorted into ascending order and
    that each element in arr1 only occurs once in arr2.
    """

    # Find where elements of arr1 are in arr2
    ptr = zeros(len(arr1), dtype=dtype)
    ptr[:] = searchsorted(arr2, arr1)

    # Make sure all elements in ptr are valid indexes into arr2
    # (any out of range entries won't match so they'll get set to -1
    # in the next bit)
    ptr[ptr >= len(arr2)] = 0
    ptr[ptr < 0]          = 0

    # Return -1 where no match is found
    ptr[where(arr2[ptr] != arr1)[0]] = -1
    
    # Put ptr back into original order
    ptr = where(ptr >= 0, arange(len(arr2), dtype=dtype)[ptr], -1)

    # count the sum of the boolean array
    counter = 0
    for i in range(len(cond)):
        if cond[i] == True:
            counter += 1

    # get the indices of the original mt_pid array satisfying the conditions
    tmp = zeros(counter, dtype=dtype)
    counter = 0
    for i in range(len(cond)):
        if cond[i] == True:
            tmp[counter] = i
            counter += 1
    
    # count how many elements in arr1 have matches in arr2
    counter = 0
    for i in range(len(ptr)):
        if ptr[i] > -1:
            counter += 1

    # record the common indices
    comm1 = zeros(counter, dtype=dtype)
    comm2 = zeros(counter, dtype=dtype)
    counter = 0
    for i in range(len(ptr)):
        if ptr[i] > -1:
            comm1[counter] = tmp[i]
            comm2[counter] = ptr[i]
            counter += 1

    
    return comm1, comm2


@jit(nopython=True)
def match_unsrt(arr1, arr2, arr2_sorted=False, arr2_index=None):
    """
    For each element in arr1 return the index of the element with the
    same value in arr2, or -1 if there is no element with the same value.
    Setting arr2_sorted=True will save some time if arr2 is already sorted
    into ascending order.

    A precomputed sorting index for arr2 can be supplied using the
    arr2_index parameter. This can save time if the routine is called
    repeatedly with the same arr2 but arr2 is not already sorted.

    It is assumed that each element in arr1 only occurs once in arr2.
    """

    # Workaround for a numpy bug (<=1.4): ensure arrays are native endian
    # because searchsorted ignores endian flag
    #if not(arr1.dtype.isnative):
    #    arr1_n = asarray(arr1, dtype=arr1.dtype.newbyteorder("="))
    #else:
    #    arr1_n = arr1
    #if not(arr2.dtype.isnative):
    #    arr2_n = asarray(arr2, dtype=arr2.dtype.newbyteorder("="))
    #else:
    #    arr2_n = arr2
    arr1_n = arr1
    arr2_n = arr2

    # Sort arr2 into ascending order if necessary
    tmp1 = arr1_n
    #if arr2_sorted:
    #   tmp2 = arr2_n
    #    idx = slice(0,len(arr2_n))
    #else:
    if arr2_index is None:
        idx = argsort(arr2_n)
        tmp2 = arr2_n[idx]
    else:
        # Use supplied sorting index
        idx = arr2_index
        tmp2 = arr2_n[arr2_index]

    # Find where elements of arr1 are in arr2
    ptr  = searchsorted(tmp2, tmp1)

    # Make sure all elements in ptr are valid indexes into tmp2
    # (any out of range entries won't match so they'll get set to -1
    # in the next bit)
    ptr[ptr>=len(tmp2)] = 0
    ptr[ptr<0]          = 0

    # Return -1 where no match is found
    ind  = tmp2[ptr] != tmp1
    ptr[ind] = -1

    # Put ptr back into original order
    ind = arange(len(arr2_n))[idx]
    ptr = where(ptr>= 0, ind[ptr], -1)

    return ptr


import numpy as np
import numba

@numba.njit()
def match_halo_pids_to_lc_rvint(nsubsamp, hpid, rvint, lcpid):
    '''
    Given that we have a list of halos in the light cone, match their
    subsample PIDs to light cone particle PIDs in a sorted "zipper".
    Record the RVint of matched particles, and the halo ID, as we go.
    Then, at the end, gather the RVints in halo order by sorting on
    those halo IDs.  Return the RVints, along with the counts per halo.
    
    TODO: this may rearrange the list of subsample PIDs within a halo. Do we
    need to preserve the order, because we save the densest as the first,
    for example?
    
    Parameters
    ----------
    nsubsamp: ndarray of int, shape (H,)
        Number of subsample PIDs per halo, providing the association of
        halo to PID. `sum(nsubsamp)` should equal `N`, the length of `hpid`.
    hpid: ndarray of int, shape (N,)
        An array of halo subsample particle PIDs, indexed by nsubsamp.
    rvint: ndarray of int, shape (M,3)
        An array of pos & vel of the light cone particles, stored as RVint,
        and row-matched to `lcpid`.
    lcpid: ndarray of int, shape (M,)
        Light cone particle PIDs, row-matched to `rvint`.
    
    Returns
    -------
    nmatch: ndarray of int, shape (H,)
        An array like `nsubsamp` that provides the assocation of halo
        to rvint. `sum(nmatch_per_halo)` should equal `L`.
        In a perfect world, if every particle has a match, then this
        array is completely identical to `nsubsamp`.
    hrvint: ndarray of int, shape (L,)
        Array of RVints for the halo particles, in the same order as
        `haloid` (so grouped by halo)
    '''
    
    H = len(nsubsamp)
    N = len(hpid)
    assert nsubsamp.sum() == N
    
    haloids = np.repeat(np.arange(H), nsubsamp)
    pinds = np.arange(N) # B.H.
    
    # put halo particles in pid order
    hord = np.argsort(hpid)
    hpid = hpid[hord]
    haloids = haloids[hord]
    pinds = pinds[hord] # B.H.
    
    # put lc particles in pid order too
    M = len(lcpid)
    lord = np.argsort(lcpid)
    lcpid = lcpid[lord]
    rvint = rvint[lord]
    
    # this will store the results
    # we'll allocate as if we're going to find a match for everything,
    # but will truncate the length to the used amount
    hrvint = np.empty((N,3), dtype=rvint.dtype)
    hrvint_hids = np.empty(N, dtype=haloids.dtype)
    hinds = np.empty(N, dtype=pinds.dtype) # B.H.
    
    j = 0
    nfound = 0  # running total of matches
    for i in range(N):
        this_pid = hpid[i]
        while lcpid[j] < this_pid:
            j += 1
        if lcpid[j] == this_pid:
            # match!
            hrvint[nfound] = rvint[j]
            hrvint_hids[nfound] = haloids[i]
            this_idx = pinds[i] # B.H.
            hinds[nfound] = this_idx  # B.H.
            nfound += 1
            j += 1
    
    # now truncate to the actual length found
    assert nfound <= N
    hrvint = hrvint[:nfound]
    hrvint_hids = hrvint_hids[:nfound]
    hinds = hinds[:nfound] # B.H.
    
    # partition the results on the haloid
    iord = np.argsort(hrvint_hids)
    hrvint = hrvint[iord]
    hrvint_hids = hrvint_hids[iord]  # maybe not needed, but probably makes counting faster
    hinds = hinds[iord] # B.H.
    
    # one more pass to count nmatch per halo
    # TODO: not super optimized, probably doesn't matter
    nmatch = np.zeros(H, dtype=nsubsamp.dtype)
    for i in range(nfound):
        nmatch[hrvint_hids[i]] += 1
        
    assert nmatch.sum() == len(hrvint)
    
    #return nmatch, hrvint # B.H.
    return hinds, nmatch, hrvint
