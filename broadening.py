#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 16:15:26 2023

@author: charles
"""

from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def test_muf_spectrum():
    dfs = pd.read_pickle("/home/charles/Desktop/test_data/no_outlier_muf.pkl")
    df = dfs[3821]
    return df["energy"], df["muf"]
    

def step_func(t, loc):
    return np.where(t < loc, 0, 1)


def gaussian(x, amp, cen, sigma, bkg):
    """1-d gaussian: gaussian(x, amp, cen, sigma, bkg)"""
    return (bkg + amp * np.exp(-(x-cen)**2 / (2*sigma**2)))


def normalized_gaussian(x, cen, sigma, bkg=0):
    return (0.3989 / (sigma)) * gaussian(x, 1, cen=cen, sigma=sigma, bkg=bkg)


def main():
    
    time = np.arange(-5, 15, 0.01)
    dt = time[1] - time[0]
    
    IRF = normalized_gaussian(time, cen=0, sigma=1, bkg=0) # instrument response function
    inst_signal = step_func(time, 5)
    
    exp_signal = np.convolve(inst_signal, IRF) * dt
    # exp_scaled = exp_signal / exp_signal.max()
    
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(time, IRF, label='IRF')
    ax[0].plot(time, inst_signal, label='$x_{inst}$')
    ax[0].legend()
    # ax[1].plot(exp_signal, label='$x_{exp}$')
    ax[1].plot(time, exp_signal[len(time) // 2: 3*len(time) // 2], label='$x_{exp}$')
    ax[1].legend()
    plt.show()
    return

if __name__ == "__main__":
    main()
    
    
    