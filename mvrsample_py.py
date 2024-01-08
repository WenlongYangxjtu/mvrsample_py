"""
    The following code is the mvrsample function in Python. It's slightly different to the MATLAB version as its index starts from 0!

    Consider there is a non-empty square matrix A, and the parameters settings are "bootstrap=True, reps=20, spacing=100, samples=20, burnin=10000".

    To turn it on, simply call

            ranks,rho,F,prho,pi,piu = mvrsample_py(A,bootstrap=True,reps=20,spacing=100,samples=20,burnin=10000)

    At the end of code, we take the calculation of a 3X3 mmatrix as an example.
                                                                                """

import sys
import time
import math
import random
import numpy as np
import pandas as pd
from collections import Counter


def mvrsample_py(A, **kwargs):
    """
        Inputs:
            A: a non-empty square matrix;
            **kwargs can include the following parameters:
                names: names of the nodes;
                bootstrap: boostrap the edges? e.g. bootstrap = True;
                rotate: size of a rotation; e.g. rotate = 2;
                reps: number of repititions of sampler to run; e.g. nreps = 5;
                samples: number of samples to store; e.g. samples = 100;
                spacing: number of samples to store; e.g. spacing = 100;
                burnin: burn-in steps before sampling; e.g. burnin = 10000;
                quiet: display messages? e.g. quiet = False;

        Returns:
            rank: a N x 2 matrix whose rows give the (index, mean pi score) for each node in A, in descending order of mean pi, across repetitions.
            rho: a scalar that gives the fraction of edges in A that violate the ranks ordering.
            F: a N x N matrix that is equivalent to A under the ranking given by ranks
            prho: a 1 x N vector containing the rho values for each rep
            pi: a reps x N matrix containing the pi vectors for each rep
            piu: a reps x N matrix containing the std.dev of pi for each rep

        """

    names = None;  # names of the nodes
    bstrap = None;  # boostrap the edges?
    nr = None;  # size of a rotation
    nreps = None;  # number of repititions of sampler to run
    n_samp = None;  # number of samples to store
    s_rate = None;  # take sample every s_rate steps
    Tb = None;  # burn-in steps before sampling
    quiet = None;  # default: display messages

    # parse command-line parameters; trap for bad input
    for i, j in kwargs.items():
        if i == 'names':
            names = kwargs['names']
        elif i == 'rotate':
            nr = kwargs['rotate']
        elif i == 'bootstrap':
            bstrap = kwargs['bootstrap']
        elif i == 'reps':
            nreps = kwargs['reps']
        elif i == 'spacing':
            s_rate = kwargs['spacing']
        elif i == 'samples':
            n_samp = kwargs['samples']
        elif i == 'burnin':
            Tb = kwargs['burnin']
        elif i == 'quiet':
            quiet = kwargs['quiet']
        else:
            print('(MVRSAMPLE) Ignoring invalid argument: ', i)

    if not ((type(A) == np.ndarray) and (len(A.shape) == 2) and (A.shape[0] == A.shape[1])):
        print('(MVRSAMPLE) Error: input ''A'' must be a non-empty square matrix; bailing out.\n')
        sys.exit('Please check your data!')
    if not ((np.all(A >= 0)) and (sum(sum(np.floor(A) != A)) == 0)):
        print('(MVRSAMPLE) Error: input ''A'' elements must be natural numbers; bailing out.\n')
        sys.exit('Please check your data!')
    if not ((names == None) or ((type(names) in [list, tuple, set, np.ndarray]) and (len(names) == A.shape[0]))):
        print('(MVRSAMPLE) Error: ''names'' argument must be same length as size(A,1); using default.\n')
        names = None
    if not ((nr == None) or ((type(nr) in [int, float]) and (nr >= 2) and (np.floor(nr) == nr))):
        print('(MVRSAMPLE) Error: ''rotate'' argument must be a positive integer > 1; using default = 2.\n')
        nr = None
    if not ((nreps == None) or ((type(nreps) in [int, float]) and (nreps >= 1) and (np.floor(nreps) == nreps))):
        print('(MVRSAMPLE) Error: ''reps'' argument must be a positive integer > 0; using default = 1.\n')
        nr = None
    if not ((n_samp == None) or ((type(n_samp) in [int, float]) and (n_samp >= 1) and (np.floor(n_samp) == n_samp))):
        print('(MVRSAMPLE) Error: ''samples'' argument must be a positive integer > 0; using default = n.\n')
        nr = None
    if not ((s_rate == None) or ((type(s_rate) in [int, float]) and (s_rate >= 1) and (np.floor(s_rate) == s_rate))):
        print('(MVRSAMPLE) Error: ''spacing'' argument must be a positive integer > 0; using default = n.\n')
        nr = None
    if not ((Tb == None) or ((type(Tb) in [int, float]) and (Tb >= 1) and (np.floor(Tb) == Tb))):
        print('(MVRSAMPLE) Error: ''burnin'' argument must be a positive integer > 0; using default = n^2.\n')
        nr = None

    A = A.astype(int)
    # basic network statistics
    n = A.shape[0]
    m = sum(sum(A))

    # default settings
    names = list(range(0, len(A))) if names == None else names  # rotation size = pairwise swaps
    nr = 2 if nr == None else nr  # names = indices
    bstrap = False if bstrap == None else bstrap  # bootstrap = false
    nreps = 1 if nreps == None else nreps  # 1 repetition of sampler
    s_rate = n if s_rate == None else s_rate  # n steps per sampled state
    n_samp = n if n_samp == None else n_samp  # n samples stored
    Tb = n ** 2 if Tb == None else Tb  # n^2 steps for burn-in
    quiet = False if quiet == None else quiet  # display messages

    def Bootstrap(A, n, m):
        indices = np.nonzero(A)
        row_indices, column_indices, values = indices[0], indices[1], A[indices]
        X = sorted(zip(row_indices, column_indices, values), key=lambda i: i[1])
        Y = [[(i[0], i[1])] * i[2] for i in X]
        Y = [j for i in Y for j in i]
        randomnum = [math.floor(random.random() * m) for i in Y]
        sample_indices = [Y[i] for i in randomnum]
        B = np.zeros((n, n))
        for a, b in dict(Counter(sample_indices)).items():
            B[a[0], a[1]] = b
        return B

    def Rank(alist, reverse=True):
        a = np.zeros(len(alist))
        b = [(i, j) for i, j in enumerate(alist)]
        c = sorted(b, key=lambda i: i[1], reverse=reverse)
        Index = [i[0] for i in c]
        for i in range(0, len(alist)):
            a[Index[i]] = i
        a = [int(i) for i in a]
        return a

    def adjacency_matrix(n, h, B):
        F = np.zeros((n, n))
        for p in range(0, n):
            for q in range(p, n):
                F[h[p], h[q]] = F[h[p], h[q]] + B[p, q]
                if p != q:
                    F[h[q], h[p]] = F[h[q], h[p]] + B[q, p]
        return F

    tstr = ['off', 'on'];
    if quiet == False:
        print('Minimum violation ranking sampler\n')
        print('   Copyright 2015 Aaron Clauset\n')
        print('   Warning: This can be a very slow calculation; please be patient.\n')
        print('   nodes, n = {}\n   edges, m = {}\n   reps     = {}\n'.format(n, m, nreps))
        print('   bootstrap of edges      = {}\n'.format(tstr[bstrap]))
        print('   number of nodes to swap = {}\n'.format(nr))
        print('   steps between samples   = {}\n'.format(s_rate))
        print('   target sample count     = {}\n'.format(n_samp))

    tic = time.time()  # start the clock
    prho = np.zeros(nreps)  # output: fraction of edges that violate MVR (by rep)
    pi = np.zeros((n, nreps))  # output: mean of ranks across MVR samples (by rep)
    piu = np.zeros((n, nreps))  # output: std of ranks across MVR samples (by rep)

    for ijk in range(0, nreps):

        # 1. if necessary, bootstrap the set of edges by sampling them
        #    uniformly at random with replacement. turning this feature off
        #    will reduce the sampler's ability to accurately estimate the
        #    uncertainty of the MVR score.
        if bstrap == True:
            # 1a. bootstrap the edges
            B = Bootstrap(A, n, m)  # adjacency matrix, bootstrapped
        else:
            # 1b. don't bootstrap the edges
            B = A.copy()

            # 2a. initialize the ranking out-degree, in decreasing order
        kout = [sum(row) for row in B]  # get the out-degrees
        h = Rank(kout, reverse=True)  # sort them and get the ranking list

        # 2b. initialize the MVR score
        F = adjacency_matrix(n, h, B)  # the reordered adjacency matrix
        score = sum(sum(np.triu(F, k=0))) - sum(sum(np.tril(F, k=-1)))

        if quiet == False:
            toc1 = time.time()
            interval1 = toc1 - tic
            print(
                '[rep={}][t={:>4s}] violations = {} ({:.2f}%)\tconverging: {}\t({:.2f}m done)\n'.format(
                    ijk + 1, str(1), m - score,100 * (1 - score / m), Tb, interval1 / 60))
        maxs = score  # the best score so far

        # 2c. initialize the zero-temperature MCMC sampler
        rs = np.zeros((n, n_samp))  # stored samples
        k = 0  # index of sample
        T = Tb + n_samp * s_rate  #
        f_stop = 0
        cnt = 0
        t = 1

        # 3. Run the zero-temperature MCMC to sample the minimum violation
        # rankings. The proposal step of the MCMC chooses a uniformly random
        # group of vertices of size r and then tries rotating them to create
        # a new ordering. If that ordering is no worse than the current
        # ordering, it accepts the move (Metropolis-Hastings rule) and
        # repeats. Otherwise, it discards that proposal, and chooses a new
        # one. The MCMC runs for Tb "burn in" steps before beginning the
        # sampling of MVRs. Some information is written to stdout as the
        # MCMC progresses.

        while True:
            t = t + 1
            # 3a. check stopping criteria
            if t > T:
                f_stop = 1
            if f_stop > 0:
                break
            # 3b. choose r positions to swap
            h2 = h.copy()
            s = 1 + math.ceil((nr - 1) * random.random())
            pr = np.zeros(s, dtype=int)
            unique = set([h2[i] for i in pr])
            while len(unique) < len(pr):
                pr = np.random.randint(0, n, size=s)
                unique = set([h2[i] for i in pr])
            # 3c. "rotate" them
            rotation = [h2[pr[-1]]] + [h2[i] for i in pr[0:-1]]
            for u, v in enumerate(pr):
                h2[v] = rotation[u]
            # 3d. tabulate proposed block matrix
            F2 = adjacency_matrix(n, h2, B)
            # 3e. compute F2's score
            snew = sum(sum(np.triu(F2, k=0))) - sum(sum(np.tril(F2, k=-1)))
            if snew >= maxs:
                # if new maximum
                if snew > maxs:
                    maxs = snew
                    score = snew
                    if (quiet == False) and (t >= Tb):
                        print(
                            '[rep={}][t={}] violations = {} ({:.1f}%)   found a better MVR; restarting sampling\n'.format(
                                ijk + 1, t, m - score, 100 * (1 - score / m)))
                    if t > Tb:
                        k, cnt, t = 0, 0, Tb + 1  # reset sampling
                cnt = cnt + 1  # increment neutral counter
                h = h2.copy()  # store new ordering
                F = F2.copy()  # store new ordered adjancecy matrix
            if (t > Tb) and (t % (math.ceil(s_rate)) == 0):
                rs[:, k] = h  # store sample
                k = k + 1  # count number of samples
                cnt = 0  # reset neutral counter

            # 3f. update the user on the progress (stdout)
            if (t % 1000) == 0:
                if quiet == False:
                    if t <= Tb:
                        toc2 = time.time()
                        interval2 = toc2 - tic
                        print(
                            '[rep={}][t={}] violations = {} ({:.1f}%)   converging: {}   ({:.2f}m done | {:.2f}m to go)\n'.format(
                                ijk + 1, t, m - score, 100 * (1 - score / m), Tb - t, interval2 / 60,
                                ((T * nreps) / (t + ijk * t - 1)) * (interval2 / 60)))
                    else:
                        print(
                            '[rep={}][t={}] violations = {} ({:.1f}%)   samples: {} ({:.1f}%)   ({:.2f}m done | {:.2f}m to go)\n'.format(
                                ijk + 1, t, m - score, 100 * (1 - score / m), k, 100 * k / n_samp, interval2 / 60,
                                ((T * nreps) / (t + ijk * t - 1)) * (interval2 / 60)))

                # write mean ranks for the top-50 (so far)
                if (t > Tb) and (k > 1) and (n >= 50):
                    ranks = np.zeros((n, 3))
                    rs_mean = np.mean(rs[:, 0:k - 1], axis=1)
                    rs_std = np.std(rs[:, 0:k - 1], axis=1, ddof=1)
                    ranks[:, 0] = np.arange(0, n)
                    ranks[:, 1] = rs_mean
                    ranks[:, 2] = rs_std
                    ranks = np.array(sorted(ranks, key=lambda i: i[1]))
                    for kik in range(0, 50):
                        grab = ranks[kik, 0]
                        if quiet == False:
                            print('{:.2f} ({:.2f})  {}\n'.format(ranks[kik, 1], ranks[kik, 2], names[int(grab)]))

            # 3g. recheck stopping criteria
            if t > T:
                f_stop = 1
            if f_stop > 0:
                break

        # store the results of this rep
        prho[ijk] = (m - score) / m
        pi[:, ijk] = np.mean(rs, axis=1)
        piu[:, ijk] = np.std(rs, axis=1, ddof=1)

    # compute the mean results and return them
    ranks = np.zeros((n, 2))
    pi_mean = np.mean(pi, axis=1)
    ranks[:, 0] = np.arange(0, n)
    ranks[:, 1] = pi_mean
    ranks = np.array(sorted(ranks, key=lambda i: i[1]))
    h = Rank(pi_mean, reverse=False)
    F = adjacency_matrix(n, h, A)  # the reordered adjacency matrix

    # fraction of edges that violate the ranking
    rho = sum(sum(np.tril(F, k=-1))) / m

    return ranks, rho, F, prho, pi, piu

# An example
# input mvrsample_input.txt file, which contains a 3X3 matrix like:
# 1,2,1
# 0,3,1
# 1,0,1
A = np.loadtxt(r"C:\matrix_eg.txt")

# Consider the following parameters settings "bootstrap=True,reps=20,spacing=100,samples=20,burnin=10000" and run the function code
ranks,rho,F,prho,pi,piu = mvrsample_py(A,bootstrap=True,reps=20,spacing=100,samples=20,burnin=10000)

#save the returns to txt file respectively.
np.savetxt(r"C:\ranks.txt", ranks)
np.savetxt(r"C:\F.txt", F)
np.savetxt(r"C:\prho.txt", prho)
np.savetxt(r"C:\pi.txt", pi)
np.savetxt(r"C:\piu.txt", piu)