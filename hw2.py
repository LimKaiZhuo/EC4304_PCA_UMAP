import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy, math
import time, itertools
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.linalg import eigh
from openpyxl.utils.dataframe import dataframe_to_rows
import openpyxl
import statsmodels.api as sm
from own_package.others import print_array_to_excel
import umap

def generate_error(std, numel):
    return np.random.normal(loc=0, scale=std, size=numel)

def selector(case):
    if case == 1:
        mean_store = []
        var_store = []
        acf_store = []
        phi_range = [0.2,0.5]
        var_range = [0.1,0.3,0.5,0.7,1]
        values_store = list(itertools.product(phi_range, var_range))
        for phi, var in values_store:
            mean_y_store = []
            var_y_store = []
            acf_single_store = []
            for k in range(5):
                numel = 100000
                phi=phi
                var = var
                e = generate_error(math.sqrt(var), numel)
                y = [1,1]
                for i in range(numel-1):
                    t = i+1
                    y.append(phi*e[t-1]*y[-2]+e[t])
                mean_y_store.append(np.mean(y))
                var_y_store.append(np.var(y))
                acf_single_store.append(sm.tsa.acf(x=np.array(y)))
            mean_store.append(np.mean(np.array(mean_y_store)))
            var_store.append(np.mean(np.array(var_y_store)))
            acf_store.append(np.mean(np.array(acf_single_store), axis=0))

        wb = openpyxl.Workbook()
        ws = wb.active

        df = pd.DataFrame(data=np.concatenate((np.array(values_store),
                                               np.array(mean_store)[...,None],
                                               np.array(var_store)[...,None],
                                               np.array(acf_store)), axis=1), columns= ['phi', 'e var', 'mean Y', 'var Y'] + list(range(len(acf_store[0]))))
        rows = dataframe_to_rows(df)
        for r_idx, row in enumerate(rows, 1):
            for c_idx, value in enumerate(row, 1):
                ws.cell(row=r_idx + 1, column=c_idx, value=value)
        wb.save('./hw2.xlsx')

        '''
        plt.figure(figsize=(30,10))
        plt.plot(y)
        plt.plot([], [], ' ', label='mean = {}'.format(mean_y))
        plt.plot([], [], ' ', label='var = {}'.format(var_y))
        plt.title('phi = {} , e_var = {}'.format(phi, var))
        plt.legend(loc='upper left')
        plt.savefig('./hw2/phi={},var={}.png'.format(phi, var), bbox_inches='tight')
        plt.close()
        
        acf = sm.tsa.acf(x=np.array(y))
        x = list(range(len(acf)))
        plt.figure(figsize=(30,10))
        plt.plot(x, acf)
        plt.title('ACF: phi = {} , e_var = {}'.format(phi, var))
        plt.plot([], [], ' ', label='{:.3e} , {:.3e} , {:.3e} , {:.3e}'.format(acf[0], acf[1], acf[2] ,acf[3]))
        plt.legend(loc='upper left')
        plt.savefig('./hw2/acf.png', bbox_inches='tight')
        plt.close()
        '''
selector(1)