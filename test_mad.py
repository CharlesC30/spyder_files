#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 12:52:01 2023

@author: charles
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

p = 100
n = 5

for _ in range(1000):
    s = np.random.randn(p, n)
    scipy_mad = stats.median_abs_deviation(s, axis=1)
    test_mad = np.median(np.abs(s - np.median(s, axis=1)[:, None]), axis=1)
    np.testing.assert_equal(test_mad, scipy_mad)
