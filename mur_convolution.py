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

with open("/home/charles/Desktop/test_data/test_data_2023_01_27.pkl", "rb") as f:
    test_data = pickle.load(f)
    
uid = list(test_data.keys())[5000]
test_df = pd.DataFrame(test_data[uid]["data"])

energy = test_df.energy
mur = -np.log(test_df["ir"] / test_df["it"])

cs = CubicSpline(energy, mur)
new_grid = np.arange(energy.min(), energy.max(), 0.1)
interp_spectrum = cs(new_grid)
                     
plt.plot(new_grid, interp_spectrum)
plt.show()

def gaussian_matrix(t_in, t_out, sigma):
    # sigma = fwhm / 2.355
    ksi = (t_in[None, :] - t_out[:, None]) / sigma
    bla = np.exp( -0.5 * ksi**2)
    bla = bla / np.sum(bla, axis=1)[:, None] # !!!!!!!!!!!!!!!!!!!
    return bla

conv_matrix = gaussian_matrix(new_grid, new_grid, sigma=2)

res = conv_matrix @ interp_spectrum
plt.plot(new_grid, res, 'r')