#!/usr/bin/env pytho      
# -*- coding: utf-8 -*-
# @Author  : CHEN Li
# @Time    : 2025/4/9 9:29
# @File    : Xgboost_.py
# @annotation

"""
Overall performance evaluation by XGB
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score

import xgboost
import shap

from utils import cal_measure

def Xgboost_(x_train, y_train, x_test, y_test, f_names, savefig_name):
    """predict and test"""
    # print('start Xgboost evaluation...')
    model = xgboost.XGBClassifier().fit(x_train, y_train)
    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)
    # 训练精度
    print('train_Accuracy: %f' % accuracy_score(y_train, pred_train))
    # 测试精度
    print('test_Accuracy: %f' % accuracy_score(y_test, pred_test))
    # pred1 = clf2.predict_proba() # 预测类别概率
    cal_measure(pred_test, y_test)
    # kappa_value = cohen_kappa_score(pred_test, y_test)
    # print('Cohen_Kappa: %f' % kappa_value)

    # SHAP
    print('SHAP...')
    # SHAP_(model.predict_proba, x_train, x_test, f_names)
    shap.initjs()
    # SHAP demo are using dataframe instead of nparray
    x_train = pd.DataFrame(x_train)  # 将numpy的array数组x_test转为dataframe格式。
    x_test = pd.DataFrame(x_test)
    x_train.columns = f_names  # 添加特征名称
    x_test.columns = f_names

    explainer = shap.Explainer(model)
    shap_values = explainer(x_train[:250])

    def font_setting(plt, xlabel=None):
        font1 = {'family': 'Times New Roman',
                 'weight': 'normal',
                 'size': 14,
                 }
        font2 = {'family': 'Times New Roman',
                 'weight': 'normal',
                 'size': 14,
                 }
        plt.yticks(fontsize=10, font=font1)
        plt.xlabel(xlabel, fontdict=font2)

    def save_pic(savename, xlabel=None):
        font_setting(plt, xlabel)
        plt.tight_layout()  # keep labels within frame
        plt.savefig(savename)
        plt.close()

    '''success'''
    # # waterfall
    # shap.plots.waterfall(shap_values[0], max_display=15, show=False)
    # save_pic('tmp/waterfall' + savefig_name + '.pdf',
    #          'Contribution of various LIFs to output in a single sample')

    # bar
    shap.plots.bar(shap_values, max_display=15, show=False)
    save_pic('tmp/bar' + savefig_name + '.pdf', 'LIF importance')

    # violin
    shap.summary_plot(shap_values, max_display=15, show=False, plot_type='violin')
    save_pic('tmp/violin' + savefig_name + '.pdf', 'SHAP values')

    # scatter
    shap.plots.scatter(shap_values, show=False, color='blue')
    # save_pic('tmp/scatter' + savefig_name + '.pdf')
    # font_setting(plt, xlabel)
    plt.tight_layout()  # keep labels within frame
    plt.savefig('tmp/scatter' + savefig_name + '.pdf')
    plt.close()
    # heatmap
    shap.plots.heatmap(shap_values, max_display=15, show=False)
    save_pic('tmp/heatmap' + savefig_name + '.pdf', 'Non-landslide/Landslide samples')

    '''failures'''
    # # force
    # shap.plots.force(shap_values[0], show=False)
    # font_setting(plt)
    # plt.tight_layout()
    # plt.savefig('tmp/force' + savefig_name + '.pdf')
    # plt.close()
    #
    # # forces
    # shap.plots.force(shap_values)
    # shap.plots.force(shap_values[0], show=False)
    # font_setting(plt)
    # plt.tight_layout()  #
    # plt.savefig('tmp/forces' + savefig_name + '.pdf')
    # plt.close()

    # shap.plots.scatter(shap_values[:, "RM"], color=shap_values)

def feature_normalization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma, mu, sigma


if __name__ == "__main__":
    """input data"""
    # nonlandslide samples
    n_data = np.loadtxt('./nonlandslides.csv', dtype=str, delimiter=",", encoding='UTF-8')
    n_samples = n_data[1:, :].astype(np.float32)
    f_names = n_samples[0, :-1].astype(str)

    # rainfall landslide samples
    p_rainfall_data = np.loadtxt('./landsides_rainfall.csv', dtype=str, delimiter=",", encoding='UTF-8')
    p_rainfall_samples = p_rainfall_data[1:, :].astype(np.float32)
    np.random.shuffle(n_samples)
    n_rainfall_samples = n_samples[:3000, :]
    # normalization
    rainfall_samples = np.vstack((p_rainfall_samples, n_rainfall_samples))
    samples_f, mean, std = feature_normalization(rainfall_samples[:, :-1])
    rainfall_samples = np.hstack((samples_f, rainfall_samples[:, -1].reshape(-1, 1)))

    # seismic landslide samples
    p_seismic_data = np.loadtxt('./landslides_seismic.csv', dtype=str, delimiter=",", encoding='UTF-8')
    p_seismic_samples = p_seismic_data[1:, :].astype(np.float32)
    np.random.shuffle(n_samples)
    n_seismic_samples = n_samples[:700, :]
    # normalization
    seismic_samples = np.vstack((p_seismic_samples, n_seismic_samples))
    samples_f, mean, std = feature_normalization(seismic_samples[:, :-1])
    seismic_samples = np.hstack((samples_f, seismic_samples[:, -1].reshape(-1, 1)))

    # slow-moving landslide samples
    p_SM_data = np.loadtxt('./landslides_SM.csv', dtype=str, delimiter=",", encoding='UTF-8')
    p_SM_samples = p_SM_data[1:, :-2].astype(np.float32)
    np.random.shuffle(n_samples)
    n_SM_samples = n_samples[:500, :]
    # normalization
    SM_samples = np.vstack((p_SM_samples, n_SM_samples))
    samples_f, mean, std = feature_normalization(SM_samples[:, :-1])
    SM_samples = np.hstack((samples_f, SM_samples[:, -1].reshape(-1, 1)))
    """dataset generation"""
    x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(rainfall_samples[:, :-1], rainfall_samples[:, -1],
                                                                test_size=0.2, shuffle=True)
    x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split(seismic_samples[:, :-1], seismic_samples[:, -1],
                                                                test_size=0.2, shuffle=True)
    x_train_3, x_test_3, y_train_3, y_test_3 = train_test_split(SM_samples[:, :-1], SM_samples[:, -1],
                                                                test_size=0.2, shuffle=True)
    """XGboost analysis"""
    Xgboost_(x_train_1, y_train_1, x_test_1, y_test_1, f_names, 'XGB_1')
    Xgboost_(x_train_2, y_train_2, x_test_2, y_test_2, f_names, 'XGB_2')
    Xgboost_(x_train_3, y_train_3, x_test_3, y_test_3, f_names, 'XGB_3')

    print('done Xgboost model explanation! \n')