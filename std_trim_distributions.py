# -*- coding: utf-8 -*-
"""
Plot distributions of chi_square for various sample sizes with and without
trimming the data. 
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

#%%
np.random.seed(1)

# function to generate random data and compute stds

def compute_stds(n=50, p=100, trim_fraction=0.2, hist_sample=False):
    s = np.random.randn(p, n)
    s_std = np.std(s, axis=1)
    s_mad = stats.median_abs_deviation(s, axis=1)
    
    s_trim = stats.trimboth(s, trim_fraction, axis=1)
    s_std_trim = np.std(s_trim, axis=1)
    print(s.shape, s_trim.shape)
    
    if hist_sample:
        plt.figure(2, clear=True)
        plt.hist(s[0, :], alpha=0.5, label="original data")
        plt.hist(s_trim[0, :], alpha=0.5, label="trimmed data")
        plt.legend()
    
    return s_std, s_std_trim, s_mad


# compute_stds(hist_sample=True)

#%%

# compare trimmed std vs non-trimmed std

small_n = 20
large_n = 100

s_std, s_std_trim, _ = compute_stds(n=small_n)
plt.figure(1, clear=True)
plt.plot(s_std, s_std_trim, 'k.', label=f"n = {small_n}")

s_std, s_std_trim, _ = compute_stds(n=large_n)
plt.figure(1, clear=False)
plt.plot(s_std, s_std_trim, 'b.', label=f"n = {large_n}")
plt.plot([0, 2], [0, 2], 'r-')

plt.axis('square')
plt.xlabel("std")
plt.ylabel("trim_std")
plt.legend()


#%%
np.random.seed(1)


def test_one_sample(n=3, p=100, trim_fraction=0.2):


    s = np.random.randn(p, n)
    
#    s_mean = np.mean(s, axis=1)
#    s_std= np.std(s, axis=1)
#    ksi = (s - s_mean[:, None]) / s_std[:, None]
    
    s_trim = stats.trimboth(s, trim_fraction, axis=1)
    s_mean_trim = np.mean(s_trim, axis=1)
    s_std_trim = np.std(s_trim, axis=1)
    ksi = (s - s_mean_trim[:, None]) / s_std_trim[:, None]

    chisq = 1/p * np.sum(ksi**2, axis=0)
#    chisq = 1/p * np.sum(np.abs(ksi), axis=0)
    
    return chisq


def test_one_sample_modified(n=3, p=100):
    s = np.random.randn(p, n)
    # s_mad = stats.median_abs_deviation(s, axis=1)
    # s_mad = np.median(np.abs(s - np.median(s, axis=1)[:, None]), axis=1)
    s_mad = np.median(np.abs(s - np.median(s, axis=1)[:, None]).ravel()) / 0.67449
    # s_mad_med = np.median(s_mad)
    s_mean = np.mean(s, axis=1)
    s_median = np.median(s, axis=1)
    ksi = (s - s_median[:, None]) / s_mad
    
    mod_chisq = 1/p * np.sum(ksi**2, axis=0)
    
    return mod_chisq
    

def test_many_samples(n=3, p=100, N=100):
    chisqs = []
    for i in range(N):
        chisq = test_one_sample_modified(n=n, p=p)
        chisqs.append(chisq)
        
    return np.hstack(chisqs)
        

def test_many_samples_with_hist(n=3, p=100, N=1000, thresh=50):
    chisq = test_many_samples(n=n, p=p, N=N)
    plt.hist(chisq[chisq<thresh], density=True, alpha=0.5, label=f"n={n}, p={p}, N={N}");

#chisq_005 = test_many_samples(n=5, N=1000)
#chisq_020 = test_many_samples(n=20, N=300)
#chisq_050 = test_many_samples(n=50, N=60)

plt.figure(1, clear=True)
test_many_samples_with_hist(n=5)
# test_many_samples_with_hist(n=8)
test_many_samples_with_hist(n=10)
# test_many_samples_with_hist(n=20)
test_many_samples_with_hist(n=50)
plt.title("$\chi^2$-values histogram")
plt.legend()


#plt.hist(chisq_005[chisq_005<100], density=True, alpha=0.5);
#plt.hist(chisq_020[chisq_020<100], density=True, alpha=0.5);
#plt.hist(chisq_050[chisq_050<100], density=True, alpha=0.5);