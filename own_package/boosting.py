import numpy as np
import pandas as pd
import statsmodels.api as sm
import operator
import openpyxl
import concurrent.futures
import multiprocessing as mp
import itertools, time
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import cross_val_score
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from scipy.sparse import csr_matrix

from own_package.others import print_df_to_excel, create_excel_file


class Boost():
    def __init__(self, z_matrix, y_vec):
        self.z_matrix = sm.add_constant(z_matrix)
        if len(y_vec.shape) != 2:
            raise TypeError('y_vec needs to be a Tx1 array.')
        self.y_vec = y_vec
        self.T, self.N = self.z_matrix.shape
        self.fitted = False

    def fit_single_predictor(self, endog, i):
        '''
        # slower by 40sec vs 30sec
        params = np.linalg.pinv(self.z_matrix[:, [i]]).dot(endog)
        # 61sec params = sum(endog.T@self.z_matrix[:, [i]])/sum(self.z_matrix[:, [i]]**2)[...,None]
        y_hat = self.z_matrix[:, [i]]@params
        ssr = sum((y_hat-endog)**2)
        #print(f'ssr: {ssr-ols_model.ssr}, params : {params-ols_model.params}, yhat :{sum(y_hat-ols_model.fittedvalues[..., None])}')
        return params, ssr, y_hat
        '''
        ols_model = sm.OLS(endog=endog, exog=self.z_matrix[:, i][..., None]).fit()
        return ols_model.params, ols_model.ssr, ols_model.fittedvalues[..., None]


'''
def fit_single_predictor(z_matrix, endog, i):
    #print(f'task {i}')
    ols_model = sm.OLS(endog=endog, exog=z_matrix[:, i][..., None]).fit()
    return ols_model.params, ols_model.ssr, ols_model.fittedvalues[..., None]
'''


class ComponentwiseL2Boost(Boost):
    def __init__(self, z_matrix, y_vec, hparams, r):
        super().__init__(z_matrix=z_matrix, y_vec=y_vec)
        self.r = r
        self.hparams = hparams
        self.m_max = hparams['m_max']
        self.learning_rate = hparams['learning_rate']
        if hparams['ic_mode'] == 'aic':
            self.ic_calculation_value = 2
            self.ic_cn_value = 2
        elif hparams['ic_mode'] == 'bic':
            self.ic_calculation_value = np.log(self.T)
            self.ic_cn_value = np.log(self.N)

    def get_aic_bic(self, phi, b_matrix):
        if self.r:
            return np.log(np.sum((self.y_vec - phi) ** 2)) + self.ic_calculation_value * np.trace(
                b_matrix) / self.T + self.r * self.ic_cn_value/self.N
        else:
            return np.log(np.sum((self.y_vec - phi) ** 2)) + self.ic_calculation_value * np.trace(b_matrix) / self.T

    def fit(self):
        y_vec = self.y_vec
        z_matrix = self.z_matrix

        bhat_vec = np.zeros(self.N)  # +1 for constant term
        bhat_vec[0] = np.mean(y_vec)
        bhat_new_store = [bhat_vec]

        b_matrix = np.ones((self.T, 1))
        b_matrix = b_matrix @ b_matrix.T / self.T

        phi = np.array([bhat_vec[0]] * self.T)[..., None]

        i_best_store = [0]
        ic_store = [self.get_aic_bic(phi=phi, b_matrix=b_matrix)]
        # t1 = time.perf_counter()
        for m in range(self.m_max):
            '''
            ssr_best = 1e20
            i_best = -1
            b_best = -1
            g_best = -1

            for i in range(self.N):
                b_i, ssr_i, g_i = self.fit_single_predictor(endog=y_vec - phi, i=i)
                if ssr_i < ssr_best:
                    ssr_best = ssr_i
                    i_best = i
                    g_best = g_i
                    b_best = b_i

           #with concurrent.futures.ProcessPoolExecutor() as executer:
           #    results = executer.map(fit_single_predictor, itertools.repeat(self.z_matrix), itertools.repeat(y_vec - phi), list(range(self.N)))
            '''

            endog = y_vec - phi
            params_store = np.sum(endog * z_matrix, axis=0) / np.sum(z_matrix ** 2, axis=0)
            endog_hat = z_matrix * params_store
            ssr = np.sum((endog_hat - endog) ** 2, axis=0)
            i_best = np.argmin(ssr)
            g_best = endog_hat[:, [i_best]]
            b_best = params_store[i_best]

            bhat_new = np.zeros(self.N)
            bhat_new[i_best] = b_best * self.learning_rate  # index +1 since first entry is for constant term
            bhat_new_store.append(bhat_new)
            i_best_store.append(i_best)

            phi = phi + self.learning_rate * g_best

            p_matrix = z_matrix[:, i_best][..., None]
            p_matrix = p_matrix @ p_matrix.T / (p_matrix.T @ p_matrix)
            b_matrix = b_matrix + self.learning_rate * p_matrix @ (np.eye(self.T) - b_matrix)
            ic_store.append(self.get_aic_bic(phi=phi, b_matrix=b_matrix))

        # t2 = time.perf_counter()
        # print(f'{t2 - t1}')
        self.bhat_new_store = csr_matrix(bhat_new_store)
        self.i_best_store = i_best_store

        self.ic_store = ic_store
        self.m_star, self.ic_optimal = min(enumerate(ic_store), key=operator.itemgetter(1))
        self.bhat = sum(bhat_new_store[:self.m_star + 1])[..., None]
        self.params = self.bhat.squeeze()

        i_star_counts = [0] * self.N
        for i in self.i_best_store[:self.m_star + 1]:
            i_star_counts[i] += 1
        self.i_star_frac = np.array(i_star_counts) / self.m_star
        self.fitted = True
        self.resid = self.y_vec - self.z_matrix @ self.bhat
        self.ssr = (self.resid.T @ self.resid).item()
        pass

    def predict(self, exog):
        return exog @ self.bhat

    def return_data_dict(self):
        return {'T': self.T,
                'N': self.N,
                'bhat_new_store': self.bhat_new_store,
                'i_best_store': self.i_best_store,
                'ic_store': self.ic_store,
                'm_star': self.m_star,
                'ic_optimal': self.ic_optimal,
                'bhat': self.bhat,
                'params': self.params,
                'i_star_frac': self.i_star_frac,
                'resid': self.resid,
                'ssr': self.ssr,
                }


class ComponentwiseL2BoostDropout(ComponentwiseL2Boost):
    def fit(self):
        y_vec = self.y_vec
        z_matrix = self.z_matrix

        bhat_vec = np.zeros(self.N)  # +1 for constant term
        bhat_vec[0] = np.mean(y_vec)
        bhat_new_store = np.array([bhat_vec])

        b_matrix = np.ones((self.T, 1))
        b_matrix = b_matrix @ b_matrix.T / self.T

        phi = np.array([bhat_vec[0]] * self.T)[..., None]
        vg_store = np.array([phi])

        i_best_store = [0]
        ic_store = [self.get_aic_bic(phi=phi, b_matrix=b_matrix)]
        np.random.seed(int(np.ceil(abs(y_vec[-1, 0]) * 1e4)))

        eye = np.eye(self.T)
        for m in range(self.m_max):
            # form drop vector
            drop_vec = np.random.choice(a=[0, 1], p=[1 - self.hparams['dropout'], self.hparams['dropout']],
                                        size=(m + 1,)).astype(bool)
            if sum(drop_vec) == 0:
                # if happen to drop nothing, randomly select one base learner to drop
                drop_vec[np.random.randint(0, m + 1, 1)] = 1

            # dropped_count = sum(drop_vec)
            dropped_phi = sum(vg_store[drop_vec])
            phi_d = phi - dropped_phi

            '''
            t1 = time.time()
            ssr_best = 1e20
            i_best = -1
            b_best = -1
            g_best = -1
            for i in range(self.N):
                b_i, ssr_i, g_i = self.fit_single_predictor(endog=y_vec - phi_d / (1 - self.hparams['dropout']), i=i)
                if ssr_i < ssr_best:
                    ssr_best = ssr_i
                    i_best = i
                    g_best = g_i
                    b_best = b_i
            t2 = time.time()
            '''
            endog = y_vec - phi_d / (1 - self.hparams['dropout'])
            params_store = np.sum(endog * z_matrix, axis=0) / np.sum(z_matrix ** 2, axis=0)
            endog_hat = z_matrix * params_store
            ssr = np.sum((endog_hat - endog) ** 2, axis=0)
            i_best = np.argmin(ssr)
            g_best = endog_hat[:, [i_best]]
            b_best = params_store[i_best]

            '''
            start = time.time()
            for _ in range(100):
                ssr_best = 1e20
                i_best = -1
                b_best = -1
                g_best = -1
                #p_store = []
                #ssr_store = []
                for i in range(self.N):
                    b_i, ssr_i, g_i = self.fit_single_predictor(endog=y_vec - phi_d / (1 - self.hparams['dropout']), i=i)
                    #p_store.append(b_i)
                    #ssr_store.append(ssr_i)
                    if ssr_i < ssr_best:
                        ssr_best = ssr_i
                        i_best = i
                        g_best = g_i
                        b_best = b_i
            end = time.time()
            print(f'One iter time {end-start}')
            print(f'One iter time {end-start}, {i_best-i_best2}, {max((g_best-g_best2)**2)}, {b_best-b_best2},'
                  f' {max((np.array(p_store).squeeze()-params_store)**2)}, {max((ssr-np.array(ssr_store))**2)}')
            '''

            bhat_new = np.zeros(self.N)
            bhat_new[i_best] = b_best * self.learning_rate

            bhat_new_store = np.concatenate((bhat_new_store, (bhat_new[None, ...])), axis=0)
            i_best_store.append(i_best)

            phi = phi_d + dropped_phi + self.learning_rate * g_best
            vg_store = np.concatenate((vg_store, (self.learning_rate * g_best)[None, ...]), axis=0)

            # phi = phi_d + dropped_phi*dropped_count/(dropped_count+1) + self.learning_rate * g_best/dropped_count
            # vg_store[drop_vec] = vg_store[drop_vec]*dropped_count/(dropped_count+1)
            # vg_store = np.concatenate((vg_store, (self.learning_rate * g_best/dropped_count)[None,...]), axis=0)

            p_matrix = z_matrix[:, [i_best]]
            p_matrix = p_matrix @ p_matrix.T / (p_matrix.T @ p_matrix)
            b_matrix = b_matrix + self.learning_rate * p_matrix @ (eye - b_matrix)
            ic_store.append(self.get_aic_bic(phi=phi, b_matrix=b_matrix))

        self.bhat_new_store = csr_matrix(bhat_new_store)
        self.i_best_store = i_best_store

        self.ic_store = ic_store
        self.m_star, self.ic_optimal = min(enumerate(ic_store), key=operator.itemgetter(1))
        # self.m_star = 150 #####################################
        self.bhat = sum(bhat_new_store[:self.m_star + 1])[..., None]
        self.params = self.bhat.squeeze()

        i_star_counts = [0] * self.N
        for i in self.i_best_store[:self.m_star + 1]:
            i_star_counts[i] += 1
        self.i_star_frac = np.array(i_star_counts) / self.m_star
        self.fitted = True
        self.resid = self.y_vec - self.z_matrix @ self.bhat
        self.ssr = (self.resid.T @ self.resid).item()
        pass


class SMwrapper(BaseEstimator, RegressorMixin):
    def __init__(self, sm_class, alpha):
        self.sm_class = sm_class
        self.alpha = alpha
        self.model = None
        self.result = None

    def fit(self, x, y):
        self.model = self.sm_class(exog=x, endog=y)
        self.result = self.model.fit_regularized(L1_wt=1, alpha=self.alpha)

    def predict(self, X):
        return self.result.predict(X)


def run_testing():
    n_total = 100
    t_train = 20
    t_test = 100
    simulation_runs = 20
    df_store = []

    def func(z):
        return 1 + 5 * z[:, [0]] + 2 * z[:, [1]] + z[:, [2]] + np.random.normal(0, 2, (z.shape[0], 1))

    for _ in range(simulation_runs):
        z = np.random.normal(0, 1, (t_train, n_total))
        y = func(z)
        z_test = np.random.normal(0, 1, (t_test, n_total))
        y_test = func(z_test)

        hparams = {'m_max': 600, 'learning_rate': 0.1, 'ic_mode': 'aic', 'dropout': 0.5}
        cwl2d_01 = ComponentwiseL2BoostDropout(z_matrix=z, y_vec=y, hparams=hparams)
        cwl2d_01.fit()
        yhat_L2d_01 = cwl2d_01.predict(exog=sm.add_constant(z_test))
        ssr_L2d_01 = sum((y_test - yhat_L2d_01) ** 2)

        hparams = {'m_max': 500, 'learning_rate': 0.3, 'ic_mode': 'aic', 'dropout': 0.5}
        cwl2d_03 = ComponentwiseL2BoostDropout(z_matrix=z, y_vec=y, hparams=hparams)
        cwl2d_03.fit()
        yhat_L2d_03 = cwl2d_03.predict(exog=sm.add_constant(z_test))
        ssr_L2d_03 = sum((y_test - yhat_L2d_03) ** 2)

        # lasso 10CV
        space = [Real(low=-10, high=1, name='alpha')]

        @use_named_args(space)
        def fitness(**params):
            return -np.mean(cross_val_score(SMwrapper(sm.OLS, 10 ** params['alpha']), sm.add_constant(z), y,
                                            cv=10,
                                            scoring='neg_mean_squared_error'))

        results = gp_minimize(func=fitness,
                              dimensions=space,
                              acq_func='EI',  # Expected Improvement.
                              n_calls=20,
                              verbose=False)

        alpha = results.x[0]  # in lg10
        lasso = sm.OLS(endog=y, exog=sm.add_constant(z)).fit_regularized(L1_wt=1, alpha=10 ** alpha)
        yhat_lasso = lasso.predict(sm.add_constant(z_test))[..., None]
        ssr_lasso = sum((y_test - yhat_lasso) ** 2)

        hparams = {'m_max': 2000, 'learning_rate': 0.1, 'ic_mode': 'aic'}
        cwl2 = ComponentwiseL2Boost(z_matrix=z, y_vec=y, hparams=hparams)
        cwl2.fit()

        yhat_L2 = cwl2.predict(exog=sm.add_constant(z_test))
        ssr_L2 = sum((y_test - yhat_L2) ** 2)

        hparams = {'m_max': 2000, 'learning_rate': 0.3, 'ic_mode': 'aic'}
        cwl2_03 = ComponentwiseL2Boost(z_matrix=z, y_vec=y, hparams=hparams)
        cwl2_03.fit()
        yhat_L2_03 = cwl2_03.predict(exog=sm.add_constant(z_test))
        ssr_L2_03 = sum((y_test - yhat_L2_03) ** 2)

        ols = sm.OLS(endog=y, exog=sm.add_constant(z)).fit()
        yhat_ols = ols.predict(sm.add_constant(z_test))[..., None]
        ssr_ols = sum((y_test - yhat_ols) ** 2)

        results_store = {'n_total': n_total,
                         'T_train': t_train,
                         'T_test': t_test,
                         'Simulation Runs': simulation_runs,
                         'OLS MSE': ssr_ols / t_test,
                         'Lasso MSE': ssr_lasso / t_test,
                         'CW0.1 MSE': ssr_L2 / t_test,
                         'CW0.3 MSE': ssr_L2_03 / t_test,
                         'CWd0.1 MSE': ssr_L2d_01 / t_test,
                         'CWd0.3 MSE': ssr_L2d_03 / t_test,
                         'lasso_alpha': 10 ** alpha,
                         'm_star0.1': cwl2.m_star,
                         'm_star0.3': cwl2_03.m_star,
                         'm_star0.1_d': cwl2d_01.m_star,
                         'm_star0.3_d': cwl2d_03.m_star,
                         'predictor': np.arange(n_total + 1),
                         'True params': [1, 5, 2, 1] + [0] * (n_total - 3),
                         'ols params': ols.params,
                         'Lasso params': lasso.params,
                         'CWL2 params0.1': cwl2.params,
                         'CWL2 params0.3': cwl2_03.params,
                         'CWL2d params0.1': cwl2d_01.params,
                         'CWL2d params0.3': cwl2d_03.params,
                         'ic_count0.1': cwl2.i_star_frac,
                         'ic_count0.3': cwl2_03.i_star_frac,
                         'ic_count0.1_d': cwl2d_01.i_star_frac,
                         'ic_count0.3_d': cwl2d_03.i_star_frac,
                         }

        df_store.append(pd.DataFrame({k: pd.Series(v) for k, v in results_store.items()}))

    df = pd.concat(objs=df_store).groupby(level=0).mean()
    excel_name = './results/test_comparision.xlsx'
    excel_name = create_excel_file(excel_name)
    wb = openpyxl.load_workbook(excel_name)
    ws = wb[wb.sheetnames[-1]]
    print_df_to_excel(df=df, ws=ws)
    wb.save(excel_name)
