#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Try using rolling window when computing standard deviation on trimmed data.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def compute_rolled_std(s, roll=5, trim_fraction=0.2):
    s_roll = []
    s_delta = s - np.median(s, axis=1)[:, None]
    for i in range(roll):
        s_roll.append(s_delta[roll+i : -roll+i, :])
    s_roll = np.hstack(s_roll)
    s_roll_trim = stats.trimboth(s_roll, trim_fraction, axis=1)
    
    s_std_roll = np.std(s_roll_trim, axis=1)
    return s_std_roll


def compute_rolled_mad(s, roll=5):
    s_roll = []
    s_delta = s - np.median(s, axis=1)[:, None]
    for i in range(roll):
        s_roll.append(s_delta[roll+i : -roll+i, :])
    s_roll = np.hstack(s_roll)
    
    s_mad_roll = stats.median_abs_deviation(s_roll, axis=1)
    return s_mad_roll


def compute_stds(n=5, p=1000, trim_fraction=0.2, hist_sample=False):
    np.random.seed(1)
    s = np.random.randn(p, n)
    s_std= np.std(s, axis=1)
    
    s_trim = stats.trimboth(s, trim_fraction, axis=1)
    s_std_trim = np.std(s_trim, axis=1)
    
    s_std_roll = compute_rolled_std(s, roll=5, trim_fraction=trim_fraction)
    s_mad_roll = compute_rolled_mad(s, roll=5)

    if hist_sample:
        plt.figure(2, clear=True)
        plt.hist(s[0, :], alpha=0.5)
        plt.hist(s_trim[0, :], alpha=0.5)
    
    return s_std, s_std_trim, s_std_roll, s_mad_roll

def compute_means(n=5, p=1000, trim_fraction=0.2, hist_sample=False):
    np.random.seed(1)
    s = np.random.randn(p, n)
    s_mean = np.mean(s, axis=1)
    
    s_trim = stats.trimboth(s, trim_fraction, axis=1)
    s_mean_trim = np.mean(s_trim, axis=1)
    return s_mean, s_mean_trim

#plt.figure(1, clear=True)

#plt.plot(s_std, s_std_trim, 'k.')
#
#
#plt.figure(1, clear=True)
#plt.hist(s_std, alpha=0.5)
#plt.hist(s_std_trim, alpha=0.5)
#
#s_std, s_std_trim = compute_stds(n=100)
#plt.hist(s_std, alpha=0.5)
#plt.hist(s_std_trim, alpha=0.5)

s_std, s_std_trim, s_std_roll, s_mad_roll = compute_stds(n=50)
s_mean, s_mean_trim = compute_means(n=5)

plt.figure(1, clear=True)
plt.subplot(221)
_, _bins, _ = plt.hist(s_mean, 25, alpha=0.5);
plt.hist(s_mean_trim, bins=_bins, alpha=0.5);

plt.subplot(222)
plt.plot(s_mean, s_mean_trim, 'k.')
plt.axis('square')

plt.subplot(223)
_, _bins, _ = plt.hist(s_std, 25, alpha=0.5);
plt.hist(s_std_trim, alpha=0.5, label="std_trim");
plt.hist(s_std_roll, alpha=0.5, label="std_trim_roll");

#_, _bins, _ = plt.hist(1/s_std, 25, alpha=0.5);
#plt.hist(1/s_std_trim, bins=_bins, alpha=0.5);

plt.subplot(224)
plt.plot(s_std, s_std_trim, 'k.')
plt.plot(s_std[5:-5], s_std_roll, 'g.')
plt.plot(s_std[5:-5], s_mad_roll, 'b.')
plt.plot([0, 2], [0, 2], 'r-')
plt.axis('square')

#%%
#s_std, s_std_trim = compute_stds(n=100, hist_sample=True)
#plt.figure(1, clear=False)
plt.figure(1, clear=True)
plt.plot(s_std, s_std_trim, 'k.', alpha=0.5)
# plt.plot(s_std[5:-5], s_std_roll, 'g.', alpha=0.5)
plt.plot(s_std[5:-5], s_mad_roll, 'b.', alpha=0.5)
plt.plot([0, 2], [0, 2], 'r-')
plt.axis('square')
plt.axhline(0.6745)

#%%
np.random.seed(1)


def test_one_sample(n=3, p=100, trim_fraction=0.2):


    s = np.random.randn(p, n)
    
#    s_mean = np.mean(s, axis=1)
#    s_std= np.std(s, axis=1)
#    ksi = (s - s_mean[:, None]) / s_std[:, None]
    
#    s_trim = stats.trimboth(s, trim_fraction, axis=1)
#    s_mean_trim = np.mean(s_trim, axis=1)
#    s_std_trim = np.std(s_trim, axis=1)
#    ksi = (s - s_mean_trim[:, None]) / s_std_trim[:, None]
    
    roll = 10
    # s_std_roll = compute_rolled_std(s, roll=roll, trim_fraction=trim_fraction)
    # ksi = (s - np.median(s, axis=1)[:, None])[roll:-roll, :] / s_std_roll[:, None]
    s_mad_roll = compute_rolled_mad(s, roll=roll)
    ksi = (s - np.median(s, axis=1)[:, None])[roll:-roll, :] / s_mad_roll[:, None]
    
#    print(s_trim.shape, s.shape)
    

    chisq = 1/p * np.sum(ksi**2, axis=0)
#    chisq = 1/p * np.sum(np.abs(ksi), axis=0)
    
    return chisq


def test_many_samples(n=3, p=100, N=100):
    chisqs = []
    for i in range(N):
        chisq = test_one_sample(n=n, p=p)
        chisqs.append(chisq)
        
    return np.hstack(chisqs)
        

def test_many_samples_with_hist(n=3, p=100, N=100, thresh=50):
    chisq = test_many_samples(n=n, p=p, N=N)
    print(chisq.min())
    plt.hist(chisq[chisq<thresh], density=True, alpha=0.5, label=f"n={n}, p={p}, N={N}");

#chisq_005 = test_many_samples(n=5, N=1000)
#chisq_020 = test_many_samples(n=20, N=300)
#chisq_050 = test_many_samples(n=50, N=60)

plt.figure(1, clear=True)
test_many_samples_with_hist(n=5, N=1000)
#test_many_samples_with_hist(n=8, N=1000)
#test_many_samples_with_hist(n=10, N=1000)
#test_many_samples_with_hist(n=20, N=1000)
test_many_samples_with_hist(n=50, N=1000)
plt.title("$\chi^2$-values histogram")
plt.legend()
