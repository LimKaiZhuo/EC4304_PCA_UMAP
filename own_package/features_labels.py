import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy, math
import time
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl import Workbook


def read_excel_data(excel_dir):
    features = pd.read_excel(excel_dir, sheet_name='features')
    features_names = features.columns.values.tolist()
    features = features.values

    labels = pd.read_excel(excel_dir, sheet_name='labels')
    labels_names = labels.columns.values.tolist()
    labels = labels.values.astype(np.float)

    time_stamp = np.copy(features[:, 0])
    features = np.copy(features[:, 1:]).astype(np.float)

    return features, labels, time_stamp, features_names, labels_names


class Fl_master():
    def __init__(self, features, labels, time_stamp, features_names, labels_names):
        self.features_o = features  # 2D ndarray. The _o stands for original dataset for features and labels.
        self.labels_o = labels  # 2D ndarray
        self.time_stamp = time_stamp  # 1D ndarray
        self.nobs = np.shape(features)[0]  # Scalar
        self.time_idx = np.arange(self.nobs)  # 1D ndarray

        all_x = np.concatenate((features, labels), axis=1)
        nan_store = np.where(np.isnan(all_x))
        col_mean = np.nanmean(all_x, axis=0)
        all_x[nan_store] = np.take(col_mean, nan_store[1])
        all_x = all_x
        all_x_scaler = StandardScaler()
        all_x_scaler.fit(all_x)
        all_x_norm = all_x_scaler.transform(all_x)
        all_x_norm = self.iterated_em(all_x_norm=all_x_norm, nan_store=nan_store, pca_p=10, max_iter=10000, tol=1e-3)
        all_x_final = all_x_scaler.inverse_transform(all_x_norm)

        wb = Workbook('./results.xlsx')
        wb.create_sheet('iter_em results')
        ws = wb['iter_em results']

        df = pd.DataFrame(data=np.concatenate((self.time_stamp[..., None], all_x_final), axis=1), columns = features_names + labels_names)

        for r in dataframe_to_rows(df, index=True, header=True):
            ws.append(r)

        wb.save('./results.xlsx')

        pass

    def iterated_em(self, all_x_norm, nan_store, pca_p, max_iter, tol):
        n_nan = len(nan_store[0])
        diff = np.zeros(n_nan) + 1000
        iter = 0
        while iter < max_iter:
            if max(diff) < tol:
                print('Tolerance met with maximum error = {}\n Number of iterations taken = {}'.format(max(diff), iter))
                break
            pca = PCA(n_components=pca_p)
            f = pca.fit_transform(all_x_norm)
            all_x_norm_1 = pca.inverse_transform(f)
            diff = all_x_norm[nan_store] - all_x_norm_1[nan_store]
            all_x_norm[nan_store] = all_x_norm_1[nan_store]
            iter +=1
        else:
            raise ValueError('During iterated EM method, maximum iteration of {} without meeting tolerance requirement. Current maximum error = {}'.format(
                iter, max(diff)))
        return all_x_norm
