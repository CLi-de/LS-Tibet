#!/usr/bin/env pytho      
# -*- coding: utf-8 -*-
# @Author  : CHEN Li
# @Time    : 2024/12/9 18:44
# @File    : utils.py
# @annotation

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# # x = np.arange(0, -1, 12)
# grid_f = np.loadtxt('./data_sup/grid_samples_static.csv', dtype=str, delimiter=",", encoding='UTF-8')
# samples_f = grid_f[1:, :-2].astype(np.float32)

def cal_measure(pred, y_test):
    TP = ((pred == 1) * (y_test == 1)).astype(int).sum()
    FP = ((pred == 1) * (y_test == 0)).astype(int).sum()
    FN = ((pred == 0) * (y_test == 1)).astype(int).sum()
    TN = ((pred == 0) * (y_test == 0)).astype(int).sum()
    # statistical measure
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F_measures = 2 * Precision * Recall / (Precision + Recall)
    print('Precision: %f' % Precision, '\nRecall: %f' % Recall, '\nF_measures: %f' % F_measures)
