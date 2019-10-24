import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy, math
import time
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from openpyxl.utils.dataframe import dataframe_to_rows
import openpyxl
import statsmodels.api as sm

from own_package.models import DFM_PCA


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


class Fl_master:
    def __init__(self, x, y, time_stamp, features_names, labels_names, time_idx=None):
        self.x = x  # 2D ndarray. The _o stands for original dataset for features and labels.
        self.y = y  # 2D ndarray
        self.yo = y  # 2D ndarray of y original without cumulative changes
        self.time_stamp = time_stamp  # 1D ndarray
        self.nobs, self.N = np.shape(x)  # Scalar X scalar
        if time_idx is not None:
            self.time_idx = time_idx
        else:
            self.time_idx = np.arange(self.nobs)  # 1D ndarray
        self.features_names = features_names
        self.labels_names = labels_names

        pass

    def iterated_em(self, features, labels, pca_p, max_iter, tol, excel_dir):
        all_x = np.concatenate((features, labels), axis=1)
        nan_store = np.where(np.isnan(all_x))
        col_mean = np.nanmean(all_x, axis=0)
        all_x[nan_store] = np.take(col_mean, nan_store[1])

        n_nan = len(nan_store[0])
        diff = np.zeros(n_nan) + 1000
        iter = 0
        while iter < max_iter:
            if max(diff) < tol:
                print('Tolerance met with maximum percentage error = {}%\n Number of iterations taken = {}'.format(
                    max(diff), iter))
                break
            all_x_scaler = StandardScaler()
            all_x_scaler.fit(all_x)
            all_x_norm = all_x_scaler.transform(all_x)

            factors, loadings_T = self.pca_factor_estimation(x=all_x_norm, r=pca_p)

            all_x_norm_1 = factors @ loadings_T
            all_x_1 = all_x_scaler.inverse_transform(all_x_norm_1)

            diff = (all_x_1[nan_store] - all_x[nan_store]) / all_x[nan_store] * 100  # Tolerance is in percentage error
            all_x[nan_store] = all_x_1[nan_store]
            iter += 1
        else:
            raise ValueError(
                'During iterated EM method, maximum iteration of {} without meeting tolerance requirement.'
                ' Current maximum percentage error = {}%'.format(iter, max(diff)))

        wb = openpyxl.load_workbook(excel_dir)
        wb.create_sheet('iter_em results')
        sheet_name = wb.sheetnames[-1]
        ws = wb[sheet_name]

        df = pd.DataFrame(data=np.concatenate((self.time_stamp[..., None], all_x), axis=1),
                          columns=self.features_names + self.labels_names)

        for r in dataframe_to_rows(df, index=False, header=True):
            ws.append(r)

        wb.save(excel_dir)

        return all_x

    def pca_factor_estimation(self, x, r):
        pca = PCA(n_components=r)
        pca.fit_transform(x.T)
        factors = pca.components_.T * math.sqrt(self.nobs)
        loadings_T = (factors.T @ x / self.nobs)
        return factors, loadings_T

    def ic_value(self, x, factors, loadings_T):
        x_hat = factors @ loadings_T
        v = 1 / (self.nobs * self.N) * np.sum((x - x_hat) ** 2)
        k = np.shape(factors)[1]

        # Using g2 penalty.
        c_nt = min(self.nobs, self.N)
        return math.log(v) + k * ((self.nobs + self.N) / (self.nobs * self.N)) * math.log(c_nt ** 2)

    def percentage_split(self, percentage):
        idx_split = round(self.nobs * (1-percentage))
        return (self.x[:idx_split, :], self.x[idx_split:, :]), \
               (self.yo[:idx_split, :], self.yo[idx_split:, :]), \
               (self.y[:idx_split, :], self.y[idx_split:, :]), \
               (self.time_stamp[:idx_split], self.time_stamp[idx_split:]), \
               (self.time_idx[:idx_split], self.time_idx[idx_split:]), \
               (idx_split, self.nobs - idx_split)

    def pca_umap_prepare_data_matrix(self, factors, yo, y, h, m, p):
        '''

        :param factors: 2D ndarray of factors. observation x dimension = N x r
        :param y: 2D ndarray of y observation values. N x 1
        :param l: 2D ndarray of target y labels. Either same as y or cumulative difference vector of (N-h) x 1
        :param h: h steps ahead
        :param m: number of factor lags
        :param p: number of AR lags on Y
        :return: [ff, fy, l] = [factors data matrix, y data matrix, labels for target y h step ahead]
        '''
        a = max(m, p)
        T = np.shape(factors)[0]

        y = y[-T+h+a-1:, :]
        ff = factors[a-1:T-h, :]
        fy = yo[a-1:T-h, :]

        x_idx = self.time_idx[a-1:T-h]
        y_idx = self.time_idx[-T+h+a-1:]

        if m >= 2:
            for idx in range(2, m+1):
                ff = np.concatenate((ff, factors[a-idx+1-1:T-h-idx+1, :]), axis=1)

        if p >= 2:
            for idx in range(2, p+1):
                fy = np.concatenate((fy, y[a-idx+1-1:T-h-idx+1, :]), axis=1)

        return ff, fy, y, x_idx, y_idx

    def prepare_data_vector(self, f, yo, m, p):
        v = f[-1,:][None,...]

        if m >= 2:
            for idx in range(m-1):
                v = np.concatenate((v, f[-idx+2,:][None,...]),axis=1)

        if p >= 2:
            for idx in range(p-1):
                v = np.concatenate((v, yo[-idx+2,:][None,...]),axis=1)

        return v

    def prepare_validation_data_h_step_ahead(self, x_v, yo_v, y_v):
        pass


    def pca_expanding_window(self, h, m, p, r, factor_model, x_t, yo_t, y_t, x_v, yo_v, y_v):
        y_hat_store = []
        e_hat_store = []
        n_val = np.shape(x_v)[0]

        # Training on train set first
        f_t, _ = factor_model(x=x_t, r=r)
        f_LM_t, yo_LM_t, y_t, x_idx_t, y_idx_t = self.pca_umap_prepare_data_matrix(f_t, yo_t, y_t, h, m, p)
        ols_model = sm.OLS(endog=y_t, exog=np.concatenate((f_LM_t, yo_LM_t),axis=1))

        results_t = ols_model.fit(cov_type='HC0')

        for idx, (x_1, yo_1, y_1) in enumerate(zip(x_v.tolist(), yo_v.tolist(), y_v.tolist())):
            y_1_hat = ols_model.predict(exog=self.prepare_data_vector(f=f_t, yo=y_t, m=m, p=p))
            e_1_hat = y_1 - y_1_hat

            x_t = np.concatenate((x_t, np.array(x_1)[..., None]), axis=0)
            yo_t = np.concatenate((yo_t, np.array(yo_1)[..., None]), axis=0)
            y_t = np.concatenate((y_t, np.array(y_1)[..., None]), axis=0)

            f_t, _ = factor_model(x=x_t, r=r)
            f_LM_t, yo_LM_t, y_t, x_idx_t, y_idx_t = self.pca_umap_prepare_data_matrix(f_t, yo_t, y_t, h, m, p)
            ols_model = sm.OLS(endog=y_t, exog=np.concatenate((f_LM_t, yo_LM_t), axis=1))

            model.train(ft, lt)

            y_hat_store.append(y_1_hat)
            e_hat_store.append(e_1_hat)



class Fl_pca(Fl_master):
    def __init__(self, val_split, yo, **kwargs):
        super(Fl_pca, self).__init__(**kwargs)
        self.yo = yo
        self.val_split = val_split
        (self.x_t, self.x_v), (self.yo_t, self.yo_v), (self.y_t, self.y_v),\
        (self.ts_t, self.ts_v), (self.tidx_t, self.tidx_v), (self.nobs_t, self.nobs_v) = self.percentage_split(0.2)

        '''
        r = self.pca_k_selection(lower_k=5, upper_k=30)
        '''
        r = 15

        pass

    def pca_k_selection(self, lower_k=5, upper_k=100):

        ic_store = []

        # Training phase. First get best k value for factor estimation using IC criteria
        k_store = list(range(lower_k, upper_k+1))
        for k in k_store:
            factors, loadings_T = self.pca_factor_estimation(x=self.ft, r=k)
            ic_store.append(self.ic_value(x=self.ft, factors=factors, loadings_T=loadings_T))

        r = k_store[np.argmin(ic_store)]

        print('Training results: Optimal factor dimension r = {}'.format(r))

        return r

