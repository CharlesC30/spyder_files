#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 20:07:09 2023

@author: charles
"""
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from lmfit import Parameters, minimize, fit_report

#%%
# stolen from xas repo
def compute_shift_between_spectra_alt(energy, mu, energy_ref, mu_ref, e0=8333, de=50):
    roi_mask = (energy_ref > (e0 - de / 2)) & (energy_ref < (e0 + de / 2))
    energy_ref_roi = energy_ref[roi_mask]
    mu_ref_roi = mu_ref[roi_mask]
    
    energy_ref_roi_norm = (energy_ref_roi - energy_ref_roi.min()) / (energy_ref_roi.max() - energy_ref_roi.min())
    
    def interpolated_spectrum(pars):
        e_shift = pars.valuesdict()['e_shift']
        x = np.interp(energy_ref_roi, energy - e_shift, mu)
        basis = np.vstack((np.ones(x.shape), x, energy_ref_roi_norm, energy_ref_roi_norm**2, energy_ref_roi_norm**3)).T
        c, _, _, _ = np.linalg.lstsq(basis, mu_ref_roi, rcond=-1)
        return basis @ c

    def residuals(pars):
        return (interpolated_spectrum(pars) - mu_ref_roi)

    pars = Parameters()
    pars.add('e_shift', value=0)
    out = minimize(residuals, pars)
    e_shift = out.params['e_shift'].value
    mu_fit = interpolated_spectrum(out.params)
    return e_shift, energy_ref_roi, mu_fit

#%%
def gaussian_matrix(t_in, t_out, sigma):
    # sigma = fwhm / 2.355
    ksi = (t_in[None, :] - t_out[:, None]) / sigma
    bla = np.exp( -0.5 * ksi**2)
    bla = bla / np.sum(bla, axis=1)[:, None] # !!!!!!!!!!!!!!!!!!!
    return bla

# load the data
with open("/home/charles/Desktop/test_data/test_data_2023_01_27.pkl", "rb") as f:
    test_data = pickle.load(f)
    
uid_list = list(test_data.keys())

def get_df_with_mus(uid_idx):
    uid = uid_list[uid_idx]
    test_df = pd.DataFrame(test_data[uid]["data"])
    
    test_df["mut"] = -np.log(test_df["it"] / test_df["i0"])
    test_df["muf"] = test_df["iff"] / test_df["i0"]
    test_df["mur"] = -np.log(test_df["ir"] / test_df["it"])
    return test_df

df1 = get_df_with_mus(5000)
df2 = get_df_with_mus(5000)
plt.plot(df1.energy, df1.mur)
plt.plot(df2.energy, df2.mur)

#%%
# e_shift, _, _ = compute_shift_between_spectra(df2.energy, df2.mur, df1.energy, df1.mur)

# interpolate high energy res for convolution
cs1 = CubicSpline(df1.energy, df1.mur)
cs2 = CubicSpline(df2.energy, df2.mur)
new_grid = np.arange(df2.energy.min(), df2.energy.max(), 0.1)
interp_spectrum1 = cs1(new_grid)
interp_spectrum2 = cs2(new_grid)

def compute_conv_sigma(energy, mu, mu_ref):
    def smoothed_spectrum(pars):
        sigma = pars.valuesdict()['sigma']
        conv_matrix = gaussian_matrix(energy, energy, sigma=sigma)
        return conv_matrix @ mu_ref
        
    def residuals(pars):
        return smoothed_spectrum(pars) - mu
    
    pars = Parameters()
    pars.add("sigma", value=1)
    out = minimize(residuals, pars)
    sigma = out.params["sigma"].value
    mu_fit = smoothed_spectrum(out.params)
    return sigma, mu_fit

# sigma, _ = compute_conv_sigma(new_grid, interp_spectrum2, interp_spectrum1)

#%%

def conv_spectrum(energy_in, energy_out, mu_in, sigma):
    conv_matrix = gaussian_matrix(energy_in, energy_out, sigma=sigma)
    return conv_matrix @ mu_in


def add_energy_points(energy_arr, sigma):
    five_sigma = sigma * np.arange(-5, 6)
    res_arr = (five_sigma[None, :] + energy_arr[:, None]).ravel()
    return res_arr[5:-5]


def compute_energy_offset_and_broadening(energy, mu, energy_ref, mu_ref, e0=8333, de=200):
    
    cs = CubicSpline(energy_ref, mu_ref)
    
    roi_mask = (energy > (e0 - de / 2)) & (energy < (e0 + de / 2))
    energy_roi = energy[roi_mask]
    mu_roi = mu[roi_mask]
    
    energy_roi_norm = (energy_roi - energy_roi.min()) / (energy_roi.max() - energy_roi.min())
    
    def get_mu_fit(pars):
        shift = pars.valuesdict()['shift']
        sigma = pars.valuesdict()['sigma']
        if sigma > 0.02:
            fine_grid_energy_ref = np.arange(energy_ref.min(), energy_ref.max(), sigma/10)
        else:
            fine_grid_energy_ref = add_energy_points(energy_ref, sigma)
        fine_grid_mu_ref = cs(fine_grid_energy_ref)
        # print(sigma)
        mu_ref_conv = conv_spectrum(
            fine_grid_energy_ref - shift, energy_roi, fine_grid_mu_ref, sigma=sigma
            )
        
        basis = np.vstack(
            (mu_ref_conv, np.ones(energy_roi.shape), energy_roi_norm)
            ).T
        c, _, _, _ = np.linalg.lstsq(basis, mu_roi, rcond=-1)
        return basis @ c
    
    def residuals(pars):
        return get_mu_fit(pars) - mu_roi
    
    shift_guess = compute_shift_between_spectra_alt(energy_ref, mu_ref, energy, mu, e0=e0, de=de)[0]
    print(shift_guess)
    pars = Parameters()
    pars.add("sigma", value=0.01, min=0)
    pars.add("shift", value=shift_guess)
    out = minimize(residuals, pars)
    sigma = out.params["sigma"].value
    shift = out.params["shift"].value
    print(fit_report(out))
    return shift, sigma, energy_roi, get_mu_fit(pars)

shift, sigma, energy_fit, mu_fit = compute_energy_offset_and_broadening(df2.energy, df2.mur, df1.energy, df1.mur)
plt.xlim(energy_fit.min(), energy_fit.max())
# plt.plot(df1.energy, df1.mur, label="mu_ref")
plt.plot(df2.energy, df2.mur, label="mu_target")
plt.plot(energy_fit, mu_fit, label="mu_fit")
plt.legend()
plt.show()
#%%
# testing conv_spectrum
e0=8333 
de=50

energy = df2.energy
mu = df2.mur
energy_ref = df1.energy
mu_ref = df1.mur

cs = CubicSpline(energy_ref, mu_ref)
fine_grid_energy_ref = np.arange(energy_ref.min(), energy_ref.max(), 0.05)
fine_grid_mu_ref = cs(fine_grid_energy_ref)

roi_mask = (energy > (e0 - de / 2)) & (energy < (e0 + de / 2))
energy_roi = energy[roi_mask]
mu_roi = mu[roi_mask]

shift=5
sigma=2
mu_ref_conv = conv_spectrum(
            fine_grid_energy_ref - shift, energy_roi, fine_grid_mu_ref, sigma=sigma
            )
plt.xlim(energy_roi.min(), energy_roi.max())
plt.plot(energy_ref, mu_ref)
plt.plot(energy_roi, mu_roi)
plt.plot(energy_roi, mu_ref_conv)

