#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 14:00:39 2023

@author: charles
"""

import xraydb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.linalg import toeplitz
from scipy.interpolate import CubicSpline
import pickle
#%%
# load energy grids
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
df2 = get_df_with_mus(-1)

e0=8333 
de=50

energy = df2.energy
mu = df2.mur
energy_ref = df1.energy
mu_ref = df1.mur

cs = CubicSpline(energy_ref, mu_ref)
fine_grid_energy_ref = np.arange(energy_ref.min(), energy_ref.max(), 0.005)
fine_grid_mu_ref = cs(fine_grid_energy_ref)

roi_mask = (energy > (e0 - de / 2)) & (energy < (e0 + de / 2))
energy_roi = energy[roi_mask]
mu_roi = mu[roi_mask]
#%%
dw_si111 = xraydb.darwin_width(10000, 'Si', (1, 1, 1))

fig, ax = plt.subplots(constrained_layout=True)

norm_real_irf = dw_si111.intensity**2 / np.sum(dw_si111.intensity**2)
ax.plot(dw_si111.denergy, norm_real_irf)

# irf_cs = CubicSpline(dw_si111.denergy[::-1], norm_real_irf[::-1], bc_type='periodic')

t_in = fine_grid_energy_ref
t_out = energy_roi
first_row_grid = t_in - t_out[0]

dw_energy = dw_si111.denergy[::-1]
# new_grid = np.arange(dw_energy.min() - 50, dw_energy.max() + 50, 0.005)
first_row = np.interp(first_row_grid, dw_si111.denergy[::-1], norm_real_irf[::-1])
plt.plot(first_row_grid, first_row)




# plt.plot(first_row)

# conv_matrix = toeplitz(norm_real_irf, r=np.zeros(100)).transpose()
# plt.imshow(conv_matrix)









