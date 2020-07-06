import numpy as np
import pandas as pd
import copy, math, pickle
import time, itertools
import concurrent.futures
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from scipy.linalg import eigh
from openpyxl.utils.dataframe import dataframe_to_rows
import openpyxl
import umap
import statsmodels.api as sm

from own_package.others import create_results_directory


def read_excel_data(excel_dir):
    features = pd.read_excel(excel_dir, sheet_name='features')
    features_names = features.columns.values.tolist()
    features = features.values

    labels = pd.read_excel(excel_dir, sheet_name='labels')
    labels_names = [labels.columns.values.tolist()[0]]
    try:
        label_type = labels.values[1, 0]
    except:
        label_type = None
    labels = labels.values[:, 0][..., None].astype(np.float)

    time_stamp = np.copy(features[:, 0])
    features = np.copy(features[:, 1:]).astype(np.float)

    return features, labels, time_stamp, features_names, labels_names, label_type


def read_excel_dataloader(excel_dir):
    xls = pd.ExcelFile(excel_dir)
    df = pd.read_excel(xls, 'x', index_col=0)
    x_names = df.columns.values
    x = df.values

    df = pd.read_excel(xls, 'yo', index_col=0)
    yo_names = df.columns.values
    yo = df.values

    df = pd.read_excel(xls, 'y', index_col=0)
    y_names = df.columns.values
    y = df.values

    time_stamp = df.index.values

    return (x, x_names, yo, yo_names, y, y_names, time_stamp)


def hparam_selection(fl, model, type, bounds_m, bounds_p, h, h_idx, h_max, r, results_dir, extension=False,
                     rolling=False, **kwargs):
    if extension:
        m_init = list(range(bounds_m[0], bounds_m[1] + 1))
        p_init = list(range(bounds_p[0], bounds_p[1] + 1))

        h_param_init = list(itertools.product(m_init, p_init))

        m_cancel = list(range(1, 3 + 1))
        p_cancel = list(range(1, 6 + 1))

        h_param_cancel = list(itertools.product(m_cancel, p_cancel))

        hparams_store = [x for x in h_param_init if x not in h_param_cancel]
    else:
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
            factor_model = fl.pca_factor_estimation
        else:
            factor_model = fl.umap_factor_estimation

        if type == 'PLS':
            data_store_save_dir = create_results_directory('{}/{}_h{}'.format(results_dir, model, h))
            for m, p in hparams_store:
                y_hat, _, rmse, results_t = fl.pls_expanding_window(h=h, m=m, p=p, r=r,
                                                                    factor_model=factor_model,
                                                                    x_t=fl.x_t, yo_t=fl.yo_t,
                                                                    y_t=fl.y_t[:, h_idx][..., None],
                                                                    x_v=fl.x_v, yo_v=fl.yo_v,
                                                                    y_v=fl.y_v[:, h_idx][..., None],
                                                                    rolling=rolling,
                                                                    save_dir=data_store_save_dir,
                                                                    save_name=model,
                                                                    )

                rmse_store.append(rmse)
                aic_t_store.append(results_t.aic)
                bic_t_store.append(results_t.bic)
                y_hat_store.append(y_hat)
            df = pd.DataFrame(data=np.concatenate((np.array(hparams_store),
                                                   np.array(rmse_store)[..., None],
                                                   np.array(aic_t_store)[..., None],
                                                   np.array(bic_t_store)[..., None],
                                                   np.array(y_hat_store)), axis=1),
                              columns=['m', 'p', 'Val RMSE', 'AIC_t', 'BIC_t'] + fl.y_v[:,
                                                                                 h_idx].flatten().tolist())


        elif type == 'AIC_BIC':
            for m, p in hparams_store:
                aic, bic = fl.aic_bic_for_pca_umap(h=h, m=m, p=p, r=r, h_max=h_max, m_max=m_max, p_max=p_max,
                                                   factor_model=factor_model,
                                                   x=fl.x, yo=fl.yo, y=fl.y[:, h_idx][..., None])
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
            data_store_save_dir = create_results_directory('{}/{}_h{}'.format(results_dir, model, h))
            for m, p in hparams_store:
                y_hat, _, rmse, results_t = fl.ar_pls_expanding_window(h=h, p=p, r=r, yo_t=fl.yo_t,
                                                                       y_t=fl.y_t[:, h_idx][..., None],
                                                                       yo_v=fl.yo_v,
                                                                       y_v=fl.y_v[:, h_idx][..., None],
                                                                       rolling=rolling,
                                                                       save_dir=data_store_save_dir,
                                                                       save_name=model,
                                                                       )

                rmse_store.append(rmse)
                aic_t_store.append(results_t.aic)
                bic_t_store.append(results_t.bic)
                y_hat_store.append(y_hat)
            df = pd.DataFrame(data=np.concatenate((np.array(hparams_store),
                                                   np.array(rmse_store)[..., None],
                                                   np.array(aic_t_store)[..., None],
                                                   np.array(bic_t_store)[..., None],
                                                   np.array(y_hat_store)), axis=1),
                              columns=['m', 'p', 'Val RMSE', 'AIC_t', 'BIC_t'] + fl.y_v[:, h_idx].flatten().tolist())


        elif type == 'AIC_BIC':
            for m, p in hparams_store:
                aic, bic = fl.aic_bic_for_ar(h=h, p=p, r=r, h_max=h_max, p_max=p_max, yo=fl.yo,
                                             y=fl.y[:, h_idx][..., None])
                rmse_store.append(-1)
                aic_t_store.append(aic)
                bic_t_store.append(bic)
            df = pd.DataFrame(data=np.concatenate((np.array(hparams_store),
                                                   np.array(rmse_store)[..., None],
                                                   np.array(aic_t_store)[..., None],
                                                   np.array(bic_t_store)[..., None]), axis=1),
                              columns=['m', 'p', 'Val RMSE', 'AIC_t', 'BIC_t'])
    elif model in ['CW{}'.format(i) for i in range(1, 9)] + ['CWd{}'.format(i) for i in range(1, 9)] + [
        'XGB{}'.format(i) for i in range(1, 9)]:
        z_type = int(model[-1])
        cw_model_class = kwargs['cw_model_class']
        cw_hparams = kwargs['cw_hparams']
        if type == 'PLS':
            for m, p in hparams_store:
                data_store_save_dir = create_results_directory('{}/{}_h{}'.format(results_dir, model, h))
                y_hat, _, rmse = fl.multicore_pls_expanding_window(h=h, p=p, m=m, r=r,
                                                                   cw_model_class=cw_model_class,
                                                                   cw_hparams=cw_hparams,
                                                                   x_t=fl.x_t,
                                                                   x_v=fl.x_v,
                                                                   yo_t=fl.yo_t,
                                                                   y_t=fl.y_t[:, h_idx][..., None],
                                                                   yo_v=fl.yo_v,
                                                                   y_v=fl.y_v[:, h_idx][..., None],
                                                                   rolling=rolling,
                                                                   z_type=z_type,
                                                                   save_dir=data_store_save_dir,
                                                                   save_name=model)

                rmse_store.append(rmse)
                y_hat_store.append(y_hat)

            aic_t_store = [np.nan] * len(rmse_store)
            bic_t_store = [np.nan] * len(rmse_store)
            df = pd.DataFrame(data=np.concatenate((np.array(hparams_store),
                                                   np.array(rmse_store)[..., None],
                                                   np.array(aic_t_store)[..., None],
                                                   np.array(bic_t_store)[..., None],
                                                   np.array(y_hat_store)), axis=1),
                              columns=['m', 'p', 'Val RMSE', 'AIC_t', 'BIC_t'] + fl.y_v[:, h_idx].flatten().tolist())


        elif type == 'AIC_BIC':
            raise KeyError('CW model selected. AIC_BIC method not available for CW model.')
    else:
        raise KeyError('Factor model selected is not available.')

    return df


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

    '''
    NOT USED
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
    '''

    def pca_factor_estimation(self, x, r, x_transformed_already=False):
        if not x_transformed_already:
            x_scaler = StandardScaler()
            x_scaler.fit(x)
            x = x_scaler.transform(x)

        w, v = eigh(x.T @ x)
        loadings = np.fliplr(v[:, -r:])
        loadings = loadings * math.sqrt(self.N)
        factors = x @ loadings / self.N
        loadings_T = loadings.T

        return factors, loadings_T

    def iterated_em(self, features, labels, pca_p, max_iter, tol, excel_dir):
        # all_x = np.concatenate((features, labels), axis=1)
        all_x = features
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
                          columns=self.features_names)

        for r in dataframe_to_rows(df, index=False, header=True):
            ws.append(r)

        wb.save(excel_dir)
        return all_x

    def percentage_split(self, percentage):
        idx_split = round(self.nobs * (1 - percentage))
        return (self.x[:idx_split, :], self.x[idx_split:, :]), \
               (self.yo[:idx_split, :], self.yo[idx_split:, :]), \
               (self.y[:idx_split, :], self.y[idx_split:, :]), \
               (self.time_stamp[:idx_split], self.time_stamp[idx_split:]), \
               (self.time_idx[:idx_split], self.time_idx[idx_split:]), \
               (idx_split, self.nobs - idx_split)


class Fl_pca(Fl_master):
    def __init__(self, val_split, y, **kwargs):
        super(Fl_pca, self).__init__(**kwargs)
        self.val_split = val_split
        self.y = y
        (self.x_t, self.x_v), (self.yo_t, self.yo_v), (self.y_t, self.y_v), \
        (self.ts_t, self.ts_v), (self.tidx_t, self.tidx_v), (self.nobs_t, self.nobs_v) = self.percentage_split(0.2)

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

    '''
    NOT IN USE
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
        if h == 1:
            return x_v, yo_v, y_v
        else:
            return x_v[:-(h-1), :], yo_v[:-(h-1), :], y_v[h-1:]
    '''

    def pls_expanding_window(self, h, m, p, r, factor_model, x_t, yo_t, y_t, x_v, yo_v, y_v,
                             rolling=False, save_dir=None, save_name=None):
        y_hat_store = []
        e_hat_store = []

        # x_v, yo_v, y_v = self.prepare_validation_data_h_step_ahead(x_v, yo_v, y_v, h)
        n_val = np.shape(x_v)[0]

        # Training on train set first
        f_t, _ = factor_model(x=x_t, r=r)
        f_LM_t, yo_LM_t, y_RO_t, x_idx_t, y_idx_t = self.pca_umap_prepare_data_matrix(f_t, yo_t, y_t, h, m, p)
        ols_model = sm.OLS(endog=y_RO_t, exog=sm.add_constant(np.concatenate((f_LM_t, yo_LM_t), axis=1)))

        ols_model = ols_model.fit()
        results_t = copy.deepcopy(ols_model)

        for idx, (x_1, yo_1, y_1) in enumerate(zip(x_v.tolist(), yo_v.tolist(), y_v.tolist())):
            x_t = np.concatenate((x_t, np.array(x_1)[None, ...]), axis=0)
            yo_t = np.concatenate((yo_t, np.array(yo_1)[None, ...]), axis=0)
            y_t = np.concatenate((y_t, np.array(y_1)[None, ...]), axis=0)
            f_t = np.concatenate((f_t, np.zeros((1, f_t.shape[1]))), axis=0)

            # f_t, _ = factor_model(x=x_t, r=r) SHOULD NOT RE-ESTIMATE FACTORS BEFORE FORECASTING
            f_LM_t, yo_LM_t, y_RO_t, x_idx_t, y_idx_t = self.pca_umap_prepare_data_matrix(f_t, yo_t, y_t, h, m, p)
            exog = np.concatenate((f_LM_t, yo_LM_t), axis=1)[-1, :][None, ...]

            y_1_hat = ols_model.predict(exog=np.concatenate((np.ones((1, 1)), exog), axis=1))
            e_1_hat = y_1 - y_1_hat

            y_hat_store.append(y_1_hat.item())
            e_hat_store.append(e_1_hat.item())

            if save_dir:
                try:
                    data_store.append({'y_hat': y_1_hat.item(),
                                       'e_hat': e_1_hat.item(),
                                       'aic': ols_model.aic,
                                       'bic': ols_model.bic})
                except:
                    data_store = [{'y_hat': y_1_hat.item(),
                                   'e_hat': e_1_hat.item(),
                                   'aic': ols_model.aic,
                                   'bic': ols_model.bic}]

            if idx + 1 == n_val:
                break  # since last iteration, no need to waste time re-estimating model

            if rolling:
                # Drop the first observation in the matrix for rolling window
                x_t = x_t[1:, :]
                yo_t = yo_t[1:, :]
                y_t = y_t[1:, :]
                f_t, _ = factor_model(x=x_t, r=r)
                f_LM_t, yo_LM_t, y_RO_t, x_idx_t, y_idx_t = self.pca_umap_prepare_data_matrix(f_t, yo_t, y_t, h, m, p)

            ols_model = sm.OLS(endog=y_RO_t, exog=sm.add_constant(np.concatenate((f_LM_t, yo_LM_t), axis=1)))
            ols_model = ols_model.fit()

        if save_dir:
            with open('{}/{}_m{}_p{}_h{}.pkl'.format(save_dir, save_name, m, p, h), "wb") as file:
                pickle.dump(data_store, file)

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


class Fl_ar(Fl_master):
    def __init__(self, val_split, y, **kwargs):
        super(Fl_ar, self).__init__(**kwargs)
        self.val_split = val_split
        self.y = y
        (self.x_t, self.x_v), (self.yo_t, self.yo_v), (self.y_t, self.y_v), \
        (self.ts_t, self.ts_v), (self.tidx_t, self.tidx_v), (self.nobs_t, self.nobs_v) = self.percentage_split(0.2)

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

    '''
    NOT IN USE
    def ar_prepare_data_vector(self, yo, p):
        v = yo[-1, :][None, ...]

        if p >= 2:
            for idx in range(2, p + 1):
                v = np.concatenate((v, yo[-idx, :][None, ...]), axis=1)

        return v

    def ar_prepare_validation_data_h_step_ahead(self, yo_v, y_v, h):
        if h == 1:
            return yo_v, y_v
        else:
            return yo_v[:-(h-1), :], y_v[h-1:]
    '''

    def ar_pls_expanding_window(self, h, p, r, yo_t, y_t, yo_v, y_v, rolling=False,
                                save_dir=None, save_name=None):
        y_hat_store = []
        e_hat_store = []

        # yo_v, y_v = self.ar_prepare_validation_data_h_step_ahead(yo_v, y_v, h)
        n_val = np.shape(yo_v)[0]

        # Training on train set first.
        # yo = information set for original y. As new information comes in, append to this matrix.
        # y = information set for transformed y to be forecasted.
        # yo_LM_t = original y lag matrix for training dataset
        # y_RO_t = y regression output for training dataset
        yo_LM_t, y_RO_t, y_idx_t = self.ar_prepare_data_matrix(yo_t, y_t, h, p)
        ols_model = sm.OLS(endog=y_RO_t, exog=sm.add_constant(yo_LM_t))

        ols_model = ols_model.fit()
        results_t = copy.deepcopy(ols_model)

        for idx, (yo_1, y_1) in enumerate(zip(yo_v.tolist(), y_v.tolist())):
            yo_t = np.concatenate((yo_t, np.array(yo_1)[None, ...]), axis=0)
            y_t = np.concatenate((y_t, np.array(y_1)[None, ...]), axis=0)

            yo_LM_t, y_RO_t, y_idx_t = self.ar_prepare_data_matrix(yo_t, y_t, h, p)
            exog = yo_LM_t[-1, :][None, ...]

            y_1_hat = ols_model.predict(exog=np.concatenate((np.ones((1, 1)), exog), axis=1))
            e_1_hat = y_1 - y_1_hat

            y_hat_store.append(y_1_hat.item())
            e_hat_store.append(e_1_hat.item())

            if save_dir:
                try:
                    data_store.append({'y_hat': y_1_hat.item(),
                                       'e_hat': e_1_hat.item(),
                                       'aic': ols_model.aic,
                                       'bic': ols_model.bic})
                except:
                    data_store = [{'y_hat': y_1_hat.item(),
                                   'e_hat': e_1_hat.item(),
                                   'aic': ols_model.aic,
                                   'bic': ols_model.bic}]

            if idx + 1 == n_val:
                data_store[-1]['y_check'] = [y_v[0], y_v[-1]]
                break  # since last iteration, no need to waste time re-estimating model

            if rolling:
                yo_t = yo_t[1:, :]
                y_t = y_t[1:, :]
                yo_LM_t = yo_LM_t[1:, :]
                y_RO_t = y_RO_t[1:, :]

            # yo_LM_t, y_t, y_idx_t = self.ar_prepare_data_matrix(yo_t, y_t, h, p)
            ols_model = sm.OLS(endog=y_RO_t, exog=sm.add_constant(yo_LM_t))
            ols_model = ols_model.fit()

        if save_dir:
            with open('{}/{}_p{}_h{}.pkl'.format(save_dir, save_name, p, h), "wb") as file:
                pickle.dump(data_store, file)

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


class Fl_cw(Fl_master):
    def __init__(self, val_split, y, **kwargs):
        super().__init__(**kwargs)
        self.val_split = val_split
        self.y = y
        (self.x_t, self.x_v), (self.yo_t, self.yo_v), (self.y_t, self.y_v), \
        (self.ts_t, self.ts_v), (self.tidx_t, self.tidx_v), (self.nobs_t, self.nobs_v) = self.percentage_split(0.2)

    def prepare_data_matrix(self, x, yo, y, h, m, p, z_type):
        '''

        :param x: 2D ndarray of x. observation x dimension = N x r
        :param y: 2D ndarray of y observation values. N x 1
        :param l: 2D ndarray of target y labels. Either same as y or cumulative difference vector of (N-h) x 1
        :param h: h steps ahead
        :param m: number of factor lags
        :param p: number of AR lags on Y
        :return: [ff, fy, l] = [factors data matrix, y data matrix, labels for target y h step ahead]
        '''
        if z_type == 1:
            z = x
        elif z_type == 2:
            z = np.concatenate((x, x ** 2), axis=1)
        elif z_type == 3:
            z, _ = self.pca_factor_estimation(x=x, r=8)
        elif z_type == 4:
            z, _ = self.pca_factor_estimation(x=x, r=8)
            z = np.concatenate((z, z ** 2), axis=1)
        elif z_type == 5:
            s = pd.DataFrame(x)
            z = np.array(pd.concat([s, s.shift(), s.shift(2), s.shift(3), s.shift(4)], axis=1))[4:, :]
            yo = yo[4:, :]
            y = y[4:, :]
            z, _ = self.pca_factor_estimation(x=np.concatenate((z, z ** 2), axis=1), r=8)
        elif z_type == 6:
            s = pd.DataFrame(x)
            z = np.array(pd.concat([s, s.shift(), s.shift(2), s.shift(3), s.shift(4)], axis=1))[4:, :]
            yo = yo[4:, :]
            y = y[4:, :]
            z, _ = self.pca_factor_estimation(x=np.concatenate((z, z ** 2), axis=1), r=8)
            z = np.concatenate((z, z ** 2), axis=1)
        elif z_type == 7:
            z, _ = self.pca_factor_estimation(x=np.concatenate((x, x ** 2), axis=1), r=8)
        elif z_type == 8:
            z, _ = self.pca_factor_estimation(x=np.concatenate((x, x ** 2), axis=1), r=8)
            z = np.concatenate((z, z ** 2), axis=1)

        a = max(m, p)
        T = np.shape(z)[0]

        y = y[-T + h + a - 1:, :]
        ff = z[a - 1:T - h, :]
        fy = yo[a - 1:T - h, :]

        if m >= 2:
            for idx in range(2, m + 1):
                ff = np.concatenate((ff, z[a - idx + 1 - 1:T - h - idx + 1, :]), axis=1)

        if p >= 2:
            for idx in range(2, p + 1):
                fy = np.concatenate((fy, yo[a - idx + 1 - 1:T - h - idx + 1, :]), axis=1)

        return sm.add_constant(np.concatenate((ff, fy), axis=1)), y

    def pls_expanding_window(self, h, m, p, cw_model_class, cw_hparams, x_t, yo_t, y_t, x_v, yo_v, y_v, z_type,
                             rolling=False,
                             save_dir=None, save_name=None):
        y_hat_store = []
        e_hat_store = []

        n_val = np.shape(x_v)[0]

        # Training on train set first
        z_matrix, y_vec = self.prepare_data_matrix(x_t, yo_t, y_t, h, m, p, z_type)
        cw_model = cw_model_class(z_matrix=z_matrix, y_vec=y_vec, hparams=cw_hparams)
        cw_model.fit()

        for idx, (x_1, yo_1, y_1) in enumerate(zip(x_v.tolist(), yo_v.tolist(), y_v.tolist())):
            x_t = np.concatenate((x_t, np.array(x_1)[None, ...]), axis=0)
            yo_t = np.concatenate((yo_t, np.array(yo_1)[None, ...]), axis=0)
            y_t = np.concatenate((y_t, np.array(y_1)[None, ...]), axis=0)

            z_matrix, y_vec = self.prepare_data_matrix(x_t, yo_t, y_t, h, m, p, z_type)
            exog = z_matrix[-1:, :]

            y_1_hat = cw_model.predict(exog=exog)
            e_1_hat = y_1[0] - y_1_hat.item()

            y_hat_store.append(y_1_hat.item())
            e_hat_store.append(e_1_hat)

            if save_dir:
                end = time.time()
                if (idx) % 5 == 0:
                    try:
                        print('Time taken for 5 steps CW PLS is {}'.format(end - start))
                        with open('{}/{}_h{}_{}.pkl'.format(save_dir, save_name, h, idx), "wb") as file:
                            pickle.dump(data_store, file)
                    except:
                        pass
                    data_store = []
                data_store.append(cw_model.return_data_dict())
                start = time.time()

            if idx + 1 == n_val:
                break  # since last iteration, no need to waste time re-estimating model

            if rolling:
                # Drop the first observation in the matrix for rolling window
                x_t = x_t[1:, :]
                yo_t = yo_t[1:, :]
                y_t = y_t[1:, :]
                z_matrix, y_vec = self.prepare_data_matrix(x_t, yo_t, y_t, h, m, p, z_type)

            cw_model = cw_model_class(z_matrix=z_matrix, y_vec=y_vec, hparams=cw_hparams)
            cw_model.fit()

        if idx % 5 != 0:
            with open('{}/{}_h{}_{}.pkl'.format(save_dir, save_name, h, idx), "wb") as file:
                pickle.dump(data_store, file)

        return y_hat_store, e_hat_store, math.sqrt(np.mean(np.array(e_hat_store) ** 2))

    def one_step_eval(self, arg, extra, idx):
        x, yo, y = arg
        cw_model_class, cw_hparams, h, m, p, r, z_type, save_dir = extra
        z_matrix, y_vec = self.prepare_data_matrix(x[:-1, :], yo[:-1, :], y[:-1, :], h, m, p, z_type)
        cw_model = cw_model_class(z_matrix=z_matrix, y_vec=y_vec, hparams=cw_hparams, r=r)
        if save_dir:
            plot_name = f'{save_dir}/{idx}'
        else:
            plot_name = None
        cw_model.fit(plot_name=plot_name)

        z_matrix, y_vec = self.prepare_data_matrix(x, yo, y, h, m, p, z_type)
        exog = z_matrix[-1:, :]

        y_1_hat = cw_model.predict(exog=exog).item()
        e_1_hat = y[-1].item() - y_1_hat


        if save_dir:
            t1 = time.perf_counter()
            e_1_hat_store = np.cumsum(cw_model.bhat_new_store.toarray(), axis=0) @ exog.squeeze() - y[-1].item()
            plt.plot(e_1_hat_store)
            plt.axvline(cw_model.m_star, linestyle='--')
            plt.savefig(f'{save_dir}/test_{idx}.png')
            plt.close()
            t2 = time.perf_counter()
            print(t2 - t1)

        return y_1_hat, e_1_hat, cw_model.return_data_dict()

    def multicore_pls_expanding_window(self, h, m, p, r, cw_model_class, cw_hparams, x_t, yo_t, y_t, x_v, yo_v, y_v,
                                       z_type,
                                       save_dir, save_name,
                                       rolling=False):
        if z_type < 3:
            # Only dataset 3,4,5,6,7,8 has factors
            r = None

        if rolling:
            args = [
                [np.concatenate((t[idx:, :], v[:idx + 1, :]), axis=0) for t, v in [[x_t, x_v], [yo_t, yo_v], [y_t, y_v]]
                 ] for idx in range(x_v.shape[0])]
        else:
            args = [[np.concatenate((t, v[:idx + 1, :]), axis=0) for t, v in [[x_t, x_v], [yo_t, yo_v], [y_t, y_v]]
                     ] for idx in range(x_v.shape[0])]

        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executer:
            results = executer.map(self.one_step_eval, args,
                                   itertools.repeat((cw_model_class, cw_hparams, h, m, p, r, z_type,
                                                     None)),
                                   itertools.count())

        y_hat_store, e_hat_store, cw_data_store = zip(*list(results))
        hparams = cw_hparams.copy()
        hparams = hparams.update([('h', h),
                           ('m', m),
                           ('p', p),
                           ('r', r),
                           ('z_type', z_type)])
        cw_data_store[0]['hparams'] = hparams
        with open('{}/{}_h{}_all.pkl'.format(save_dir, save_name, h), "wb") as file:
            pickle.dump(cw_data_store, file)

        return y_hat_store, e_hat_store, math.sqrt(np.mean(np.array(e_hat_store) ** 2))


class Fl_xgb(Fl_cw):
    def one_step_eval(self, arg, extra, idx):
        x, yo, y = arg
        cw_model_class, cw_hparams, h, m, p, r, z_type, save_dir = extra
        z_matrix, y_vec = self.prepare_data_matrix(x[:-1, :], yo[:-1, :], y[:-1, :], h, m, p, z_type)
        cw_model = cw_model_class(z_matrix=z_matrix, y_vec=y_vec, hparams=cw_hparams, r=r)
        if save_dir:
            plot_name = f'{save_dir}/{idx}'
        else:
            plot_name = None


        z_matrix, y_vec = self.prepare_data_matrix(x, yo, y, h, m, p, z_type)
        exog = z_matrix[-1:, :]

        cw_model.fit(deval=xgb.DMatrix(data=exog, label=y[[-1]]),plot_name=plot_name)

        y_1_hat = cw_model.predict(exog=exog).item()
        e_1_hat = y[-1].item() - y_1_hat

        if save_dir:
            t1 = time.perf_counter()
            e_1_hat_store = np.cumsum(cw_model.bhat_new_store.toarray(), axis=0) @ exog.squeeze() - y[-1].item()
            plt.plot(e_1_hat_store)
            plt.axvline(cw_model.m_star, linestyle='--')
            plt.savefig(f'{save_dir}/test_{idx}.png')
            plt.close()
            t2 = time.perf_counter()
            print(t2 - t1)

        return y_1_hat, e_1_hat, cw_model.return_data_dict()


class Fl_cw_data_store(Fl_cw):
    def predict(self, b_hat_store, exog):
        return np.cumsum(b_hat_store.toarray(), axis=0) @ exog.squeeze()

    def pls_expanding_window(self, h, m, p, data_store, x_t, yo_t, y_t, x_v, yo_v, y_v, z_type, save_dir,
                             rolling=False):
        y_hat_store = []
        e_hat_store = []
        n_val = np.shape(x_v)[0]

        plot_idx = 20

        for idx, (x_1, yo_1, y_1) in enumerate(zip(x_v.tolist(), yo_v.tolist(), y_v.tolist())):
            x_t = np.concatenate((x_t, np.array(x_1)[None, ...]), axis=0)
            yo_t = np.concatenate((yo_t, np.array(yo_1)[None, ...]), axis=0)
            y_t = np.concatenate((y_t, np.array(y_1)[None, ...]), axis=0)

            z_matrix, y_vec = self.prepare_data_matrix(x_t, yo_t, y_t, h, m, p, z_type)
            exog = z_matrix[-1:, :]
            cum_bhat_store = np.cumsum(data_store[idx]['bhat_new_store'].toarray(), axis=0)
            y_1_hat = np.array(cum_bhat_store @ exog.squeeze())
            e_1_hat = y_1[0] - y_1_hat

            y_hat_store.append(y_1_hat)
            e_hat_store.append(e_1_hat)

            if idx % 20 == 0:
                # plot stuff
                train_error = np.mean((z_matrix @ cum_bhat_store.T - y_vec) ** 2, axis=0)
                plt.plot(train_error)
                plt.axvline(data_store[idx]['m_star'], linestyle='--', color='r')
                plt.ylim([min(train_error[:data_store[idx]['m_star'] + 2]),
                          max(train_error[:data_store[idx]['m_star'] + 2])])
                plt.savefig(f'{save_dir}/trainerror_{idx}.png')
                plt.close()

                plt.plot(e_1_hat)
                plt.axvline(data_store[idx]['m_star'], linestyle='--', color='r')
                plt.ylim([min(e_1_hat[:data_store[idx]['m_star'] + 2]), max(e_1_hat[:data_store[idx]['m_star'] + 2])])
                plt.savefig(f'{save_dir}/testerror_{idx}.png')
                plt.close()

            if idx + 1 == n_val:
                break  # since last iteration, no need to waste time re-estimating model

            if rolling:
                # Drop the first observation in the matrix for rolling window
                x_t = x_t[1:, :]
                yo_t = yo_t[1:, :]
                y_t = y_t[1:, :]

        return y_hat_store, e_hat_store, None
