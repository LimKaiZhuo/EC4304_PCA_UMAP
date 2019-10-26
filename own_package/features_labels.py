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


def read_excel_data(excel_dir):
    features = pd.read_excel(excel_dir, sheet_name='features')
    features_names = features.columns.values.tolist()
    features = features.values

    labels = pd.read_excel(excel_dir, sheet_name='labels')
    labels_names = [labels.columns.values.tolist()[0]]
    label_type = labels.values[1, 0]
    labels = labels.values[:, 0][..., None].astype(np.float)

    time_stamp = np.copy(features[:, 0])
    features = np.copy(features[:, 1:]).astype(np.float)

    return features, labels, time_stamp, features_names, labels_names, label_type


def read_excel_dataloader(excel_dir):
    xls = pd.ExcelFile(excel_dir)
    df = pd.read_excel(xls, 'x')
    x_names = df.columns.values
    x = df.values

    df = pd.read_excel(xls, 'yo')
    yo_names = df.columns.values
    yo = df.values

    df = pd.read_excel(xls, 'y')
    y_names = df.columns.values
    y = df.values

    df = pd.read_excel(xls, 'time_stamp')
    time_stamp = df.values

    return (x, x_names, yo, yo_names, y, y_names, time_stamp)


class Fl_master:
    def __init__(self, x, yo, time_stamp, features_names, labels_names, y=None, y_names=None, time_idx=None):
        self.x = x  # 2D ndarray. The _o stands for original dataset for features and labels.
        if y is None:
            self.y = yo  # 2D ndarray
            self.y_names = None
        else:
            self.y = y
            self.y_names = y_names
        self.yo = yo  # 2D ndarray of y original without cumulative changes
        self.time_stamp = time_stamp  # 1D ndarray
        self.nobs, self.N = np.shape(x)  # Scalar X scalar
        if time_idx is not None:
            self.time_idx = time_idx
        else:
            self.time_idx = np.arange(self.nobs)  # 1D ndarray
        self.features_names = features_names
        self.labels_names = labels_names

        pass

    def y_h_step_transformation(self, h_steps):
        y_store = []

        yo = self.yo.flatten()
        for h in h_steps:
            if type == 5:
                y_transformed = 1200 / h * np.log(yo[h:] / yo[:-h])
                y_transformed = [np.nan] * h + y_transformed.tolist()
                y_store.append(y_transformed)
            elif type == 6:
                y_transformed = np.array(yo)[2:] - 2 * np.array(yo)[1:-1] + np.array(yo)[:-2]
                y_transformed = [np.nan, np.nan] + y_transformed.tolist()
                y_store.append(y_transformed)
            else:
                raise KeyError('Label type is not 5 or 6')

        self.yo = self.yo * 1200

        pass

    def iterated_em(self, features, labels, pca_p, max_iter, tol, excel_dir):
        all_x = np.concatenate((features, labels), axis=1)
        nan_store = np.where(np.isnan(all_x))
        col_mean = np.nanmean(all_x, axis=0)
        all_x[nan_store] = np.take(col_mean, nan_store[1])

        n_nan = len(nan_store[0])
        diff = 1000
        iter = 0
        while iter < max_iter:
            if diff < tol:
                print('Tolerance met with maximum percentage error = {}%\n Number of iterations taken = {}'.format(
                    diff, iter))
                break
            all_x_scaler = StandardScaler()
            all_x_scaler.fit(all_x)
            all_x_norm = all_x_scaler.transform(all_x)

            factors, loadings_T = self.pca_factor_estimation(x=all_x_norm, r=pca_p, x_transformed_already=True)

            all_x_norm_1 = factors @ loadings_T
            all_x_1 = all_x_scaler.inverse_transform(all_x_norm_1)

            try:
                # diff = max((all_x_1[nan_store] - all_x[nan_store]) / all_x[nan_store] * 100)  # Tolerance is in percentage error
                diff = np.max((factors - factors_old) / factors_old * 100)
                factors_old = factors
            except:
                diff = 1000
                factors_old = factors
            all_x[nan_store] = all_x_1[nan_store]
            iter += 1
        else:
            raise ValueError(
                'During iterated EM method, maximum iteration of {} without meeting tolerance requirement.'
                ' Current maximum percentage error = {}%'.format(iter, diff))

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

    def pca_factor_estimation(self, x, r, x_transformed_already=False):
        if not x_transformed_already:
            x_scaler = StandardScaler()
            x_scaler.fit(x)
            x = x_scaler.transform(x)
        '''
        pca = PCA(n_components=r)

        pca.fit_transform(x.T)
        factors = pca.components_.T * math.sqrt(self.nobs)
        loadings_T = (factors.T @ x / self.nobs)
        '''
        '''
        pca.fit_transform(x)
        loadings = pca.components_.T * math.sqrt(self.N)
        factors = x @ loadings / self.N
        '''

        w, v = eigh(x.T @ x)
        loadings = np.fliplr(v[:, -r:])
        loadings = loadings * math.sqrt(self.N)
        factors = x @ loadings / self.N
        loadings_T = loadings.T

        '''
        w, v = eigh(x @ x.T)
        factors = np.fliplr(v[:, -r:]) * math.sqrt(self.nobs)
        loadings_T = factors.T @ x / self.nobs
        '''

        return factors, loadings_T

    def umap_factor_estimation(self, x, r, x_transformed_already=False):
        if not x_transformed_already:
            x_scaler = StandardScaler()
            x_scaler.fit(x)
            x = x_scaler.transform(x)
        reducer = umap.UMAP(a=1.576943460405378, angular_rp_forest=False,
                            b=0.8950608781227859, init='spectral',
                            local_connectivity=1.0, metric='euclidean', metric_kwds={},
                            min_dist=0.1, n_components=r, n_epochs=None, n_neighbors=15,
                            negative_sample_rate=5, random_state=42, set_op_mix_ratio=1.0,
                            spread=1.0, target_metric='l2', target_metric_kwds={},
                            transform_queue_size=4.0, transform_seed=42, verbose=False)
        reducer.fit(x)
        factors = reducer.transform(x)

        return factors, None

    def ic_value(self, x, factors, loadings_T, x_transformed_already=False):
        if not x_transformed_already:
            x_scaler = StandardScaler()
            x_scaler.fit(x)
            x = x_scaler.transform(x)
        x_hat = factors @ loadings_T
        v = 1 / (self.nobs * self.N) * np.sum((x - x_hat) ** 2)
        k = np.shape(factors)[1]

        # Using g2 penalty.
        c_nt = min(self.nobs, self.N)
        return math.log(v) + k * ((self.nobs + self.N) / (self.nobs * self.N)) * math.log(c_nt)

    def percentage_split(self, percentage):
        idx_split = round(self.nobs * (1 - percentage))
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

        y = y[-T + h + a - 1:, :]
        ff = factors[a - 1:T - h, :]
        fy = yo[a - 1:T - h, :]

        x_idx = self.time_idx[a - 1:T - h]
        y_idx = self.time_idx[-T + h + a - 1:]

        if m >= 2:
            for idx in range(2, m + 1):
                ff = np.concatenate((ff, factors[a - idx + 1 - 1:T - h - idx + 1, :]), axis=1)

        if p >= 2:
            for idx in range(2, p + 1):
                fy = np.concatenate((fy, yo[a - idx + 1 - 1:T - h - idx + 1, :]), axis=1)

        return ff, fy, y, x_idx, y_idx

    def prepare_data_vector(self, f, yo, m, p):
        v = f[-1, :][None, ...]

        if m >= 2:
            for idx in range(2, m + 1):
                v = np.concatenate((v, f[-idx, :][None, ...]), axis=1)

        if p >= 1:
            for idx in range(1, p + 1):
                v = np.concatenate((v, yo[-idx, :][None, ...]), axis=1)

        return v

    def prepare_validation_data_h_step_ahead(self, x_v, yo_v, y_v, h):
        return x_v[:-h, :], yo_v[:-h, :], y_v[h:]

    def pls_expanding_window(self, h, m, p, r, factor_model, x_t, yo_t, y_t, x_v, yo_v, y_v):
        y_hat_store = []
        e_hat_store = []

        x_v, yo_v, y_v = self.prepare_validation_data_h_step_ahead(x_v, yo_v, y_v, h)
        n_val = np.shape(x_v)[0]

        # Training on train set first
        f_t, _ = factor_model(x=x_t, r=r)
        f_LM_t, yo_LM_t, y_t, x_idx_t, y_idx_t = self.pca_umap_prepare_data_matrix(f_t, yo_t, y_t, h, m, p)
        ols_model = sm.OLS(endog=y_t, exog=sm.add_constant(np.concatenate((f_LM_t, yo_LM_t), axis=1)))

        ols_model = ols_model.fit()
        results_t = copy.deepcopy(ols_model)

        for idx, (x_1, yo_1, y_1) in enumerate(zip(x_v.tolist(), yo_v.tolist(), y_v.tolist())):
            v = self.prepare_data_vector(f=f_t, yo=yo_t, m=m, p=p)
            y_1_hat = ols_model.predict(exog=np.concatenate((np.ones((1, 1)), v), axis=1))
            e_1_hat = y_1 - y_1_hat

            y_hat_store.append(y_1_hat.item())
            e_hat_store.append(e_1_hat.item())

            x_t = np.concatenate((x_t, np.array(x_1)[None, ...]), axis=0)
            yo_t = np.concatenate((yo_t, np.array(yo_1)[None, ...]), axis=0)
            y_t = np.concatenate((y_t, np.array(y_1)[None, ...]), axis=0)

            if idx + 1 == n_val:
                break  # since last iteration, no need to waste time re-estimating model

            f_t, _ = factor_model(x=x_t, r=r)
            f_LM_t, yo_LM_t, y_t, x_idx_t, y_idx_t = self.pca_umap_prepare_data_matrix(f_t, yo_t, y_t, h, m, p)
            ols_model = sm.OLS(endog=y_t, exog=sm.add_constant(np.concatenate((f_LM_t, yo_LM_t), axis=1)))
            ols_model = ols_model.fit()

        return y_hat_store, e_hat_store, math.sqrt(np.mean(np.array(e_hat_store) ** 2)), results_t

    def aic_bic_for_pca_umap(self, h, m, p, r, h_max, m_max, p_max, factor_model, x, yo, y):
        f, _ = factor_model(x=x, r=r)
        f_LM, yo_LM, y, _, _ = self.pca_umap_prepare_data_matrix(f, yo, y, h, m, p)
        T = np.shape(f)[0]
        a_max = max(m_max, p_max)
        if h < h_max or m < m_max or p < p_max:
            f_LM = f_LM[-T + h_max + a_max - 1:, :]
            yo_LM = yo_LM[-T + h_max + a_max - 1:, :]
            y = y[-T + h_max + a_max - 1:, :]
            pass

        ols_model = sm.OLS(endog=y, exog=sm.add_constant(np.concatenate((f_LM, yo_LM), axis=1)))
        ols_model = ols_model.fit()
        return ols_model.aic, ols_model.bic

    def ar_prepare_data_matrix(self, yo, y, h, p):
        '''

        :param factors: 2D ndarray of factors. observation x dimension = N x r
        :param y: 2D ndarray of y observation values. N x 1
        :param l: 2D ndarray of target y labels. Either same as y or cumulative difference vector of (N-h) x 1
        :param h: h steps ahead
        :param m: number of factor lags
        :param p: number of AR lags on Y
        :return: [ff, fy, l] = [factors data matrix, y data matrix, labels for target y h step ahead]
        '''
        a = p
        T = np.shape(yo)[0]

        y = y[-T + h + a - 1:, :]
        fy = yo[a - 1:T - h, :]

        y_idx = self.time_idx[-T + h + a - 1:]

        if p >= 2:
            for idx in range(2, p + 1):
                fy = np.concatenate((fy, yo[a - idx + 1 - 1:T - h - idx + 1, :]), axis=1)

        return fy, y, y_idx

    def ar_prepare_data_vector(self, yo, p):
        v = yo[-1, :][None, ...]

        if p >= 2:
            for idx in range(2, p + 1):
                v = np.concatenate((v, yo[-idx, :][None, ...]), axis=1)

        return v

    def ar_prepare_validation_data_h_step_ahead(self, yo_v, y_v, h):
        return yo_v[:-h, :], y_v[h:]

    def ar_pls_expanding_window(self, h, p, r, yo_t, y_t, yo_v, y_v):
        y_hat_store = []
        e_hat_store = []

        yo_v, y_v = self.ar_prepare_validation_data_h_step_ahead(yo_v, y_v, h)
        n_val = np.shape(yo_v)[0]

        # Training on train set first
        yo_LM_t, y_t, y_idx_t = self.ar_prepare_data_matrix(yo_t, y_t, h, p)
        ols_model = sm.OLS(endog=y_t, exog=sm.add_constant(yo_LM_t))

        ols_model = ols_model.fit()
        results_t = copy.deepcopy(ols_model)

        for idx, (yo_1, y_1) in enumerate(zip(yo_v.tolist(), y_v.tolist())):
            v = self.ar_prepare_data_vector(yo=yo_t, p=p)
            y_1_hat = ols_model.predict(exog=np.concatenate((np.ones((1, 1)), v), axis=1))
            e_1_hat = y_1 - y_1_hat

            y_hat_store.append(y_1_hat.item())
            e_hat_store.append(e_1_hat.item())

            yo_t = np.concatenate((yo_t, np.array(yo_1)[None, ...]), axis=0)
            y_t = np.concatenate((y_t, np.array(y_1)[None, ...]), axis=0)

            if idx + 1 == n_val:
                break  # since last iteration, no need to waste time re-estimating model

            yo_LM_t, y_t, y_idx_t = self.ar_prepare_data_matrix(yo_t, y_t, h, p)
            ols_model = sm.OLS(endog=y_t, exog=sm.add_constant(yo_LM_t))
            ols_model = ols_model.fit()

        return y_hat_store, e_hat_store, math.sqrt(np.mean(np.array(e_hat_store) ** 2)), results_t

    def aic_bic_for_ar(self, h, p, r, h_max, p_max, yo, y):
        yo_LM, y, _ = self.ar_prepare_data_matrix(yo, y, h, p)
        T = np.shape(yo)[0]
        a_max = p_max
        if h < h_max or p < p_max:
            yo_LM = yo_LM[-T + h_max + a_max - 1:, :]
            y = y[-T + h_max + a_max - 1:, :]
            pass

        ols_model = sm.OLS(endog=y, exog=sm.add_constant(yo_LM))
        ols_model = ols_model.fit()
        return ols_model.aic, ols_model.bic


class Fl_pca(Fl_master):
    def __init__(self, val_split, y, **kwargs):
        super(Fl_pca, self).__init__(**kwargs)
        self.val_split = val_split
        self.y = y
        (self.x_t, self.x_v), (self.yo_t, self.yo_v), (self.y_t, self.y_v), \
        (self.ts_t, self.ts_v), (self.tidx_t, self.tidx_v), (self.nobs_t, self.nobs_v) = self.percentage_split(0.2)

        '''
        r = self.pca_k_selection(lower_k=5, upper_k=30)
        '''
        r = 15

        pass

    def pca_k_selection(self, lower_k=5, upper_k=100):
        ic_store = []
        x = self.x
        x_scaler = StandardScaler()
        x_scaler.fit(x)
        x = x_scaler.transform(x)

        # Training phase. First get best k value for factor estimation using IC criteria
        k_store = list(range(lower_k, upper_k + 1))
        for k in k_store:
            factors, loadings_T = self.pca_factor_estimation(x=x, r=k, x_transformed_already=True)
            ic_store.append(self.ic_value(x=x, factors=factors, loadings_T=loadings_T, x_transformed_already=True))

        r = k_store[np.argmin(ic_store)]

        print('Training results: Optimal factor dimension r = {}'.format(r))

        return r, ic_store

    def hparam_selection(self, model, type, bounds_m, bounds_p, h, h_idx, h_max, r, results_dir):
        m_store = list(range(bounds_m[0], bounds_m[1] + 1))
        p_store = list(range(bounds_p[0], bounds_p[1] + 1))
        hparams_store = list(itertools.product(m_store, p_store))
        rmse_store = []
        aic_t_store = []
        bic_t_store = []
        y_hat_store = []

        m_max = bounds_m[1]
        p_max = bounds_p[1]

        if model == 'PCA' or model == 'UMAP':
            if model == 'PCA':
                factor_model = self.pca_factor_estimation
            else:
                factor_model = self.umap_factor_estimation

            if type == 'PLS':
                for m, p in hparams_store:
                    y_hat, _, rmse, results_t = self.pls_expanding_window(h=h, m=m, p=p, r=r,
                                                                          factor_model=factor_model,
                                                                          x_t=self.x_t, yo_t=self.yo_t,
                                                                          y_t=self.y_t[:, h_idx][..., None],
                                                                          x_v=self.x_v, yo_v=self.yo_v,
                                                                          y_v=self.y_v[:, h_idx][..., None])

                    rmse_store.append(rmse)
                    aic_t_store.append(results_t.aic)
                    bic_t_store.append(results_t.bic)
                    y_hat_store.append(y_hat)
                df = pd.DataFrame(data=np.concatenate((np.array(hparams_store),
                                                       np.array(rmse_store)[..., None],
                                                       np.array(aic_t_store)[..., None],
                                                       np.array(bic_t_store)[..., None],
                                                       np.array(y_hat_store)), axis=1),
                                  columns=['m', 'p', 'Val RMSE', 'AIC_t', 'BIC_t'] + self.y_v[h:, h_idx].flatten().tolist())


            elif type == 'AIC_BIC':
                for m, p in hparams_store:
                    aic, bic = self.aic_bic_for_pca_umap(h=h, m=m, p=p, r=r, h_max=h_max, m_max=m_max, p_max=p_max,
                                                         factor_model=factor_model,
                                                         x=self.x, yo=self.yo, y=self.y)
                    rmse_store.append(-1)
                    aic_t_store.append(aic)
                    bic_t_store.append(bic)
                df = pd.DataFrame(data=np.concatenate((np.array(hparams_store),
                                                       np.array(rmse_store)[..., None],
                                                       np.array(aic_t_store)[..., None],
                                                       np.array(bic_t_store)[..., None]), axis=1),
                                  columns=['m', 'p', 'Val RMSE', 'AIC_t', 'BIC_t'])

        elif model == 'AR':
            if type == 'PLS':
                for m, p in hparams_store:
                    y_hat, _, rmse, results_t = self.ar_pls_expanding_window(h=h, p=p, r=r, yo_t=self.yo_t,
                                                                             y_t=self.y_t[:, h_idx][..., None],
                                                                             yo_v=self.yo_v,
                                                                             y_v=self.y_v[:, h_idx][..., None])

                    rmse_store.append(rmse)
                    aic_t_store.append(results_t.aic)
                    bic_t_store.append(results_t.bic)
                    y_hat_store.append(y_hat)
                df = pd.DataFrame(data=np.concatenate((np.array(hparams_store),
                                                       np.array(rmse_store)[..., None],
                                                       np.array(aic_t_store)[..., None],
                                                       np.array(bic_t_store)[..., None],
                                                       np.array(y_hat_store)), axis=1),
                                  columns=['m', 'p', 'Val RMSE', 'AIC_t', 'BIC_t'] + self.y_v[h:, h_idx].flatten().tolist())


            elif type == 'AIC_BIC':
                for m, p in hparams_store:
                    aic, bic = self.aic_bic_for_ar(h=h, p=p, r=r, h_max=h_max,  p_max=p_max,yo=self.yo, y=self.y)
                    rmse_store.append(-1)
                    aic_t_store.append(aic)
                    bic_t_store.append(bic)
                df = pd.DataFrame(data=np.concatenate((np.array(hparams_store),
                                                       np.array(rmse_store)[..., None],
                                                       np.array(aic_t_store)[..., None],
                                                       np.array(bic_t_store)[..., None]), axis=1),
                                  columns=['m', 'p', 'Val RMSE', 'AIC_t', 'BIC_t'])
        else:
            raise KeyError('Factor model selected is not available.')

        return df
