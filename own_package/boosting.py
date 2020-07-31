import numpy as np
import pandas as pd
import xgboost as xgb
import statsmodels.api as sm
import operator
import openpyxl, shap
import concurrent.futures
import multiprocessing as mp
import itertools, time
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import cross_val_score
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

from own_package.others import print_df_to_excel, create_excel_file, create_results_directory


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
            self.ic_selection = 0
        elif hparams['ic_mode'] == 'bic':
            self.ic_selection = 1

        self.ic_calculation_value = [2, np.log(self.T)]
        self.ic_cn_value = [2, np.log(self.N)]

    def get_aic_bic(self, phi, b_matrix):
        if self.r:
            return [np.log(np.sum((self.y_vec - phi) ** 2)) + self.ic_calculation_value[0] * np.trace(
                b_matrix) / self.T + self.r * self.ic_cn_value[0] / self.N,
                    np.log(np.sum((self.y_vec - phi) ** 2)) + self.ic_calculation_value[1] * np.trace(
                        b_matrix) / self.T + self.r * self.ic_cn_value[1] / self.N
                    ]
        else:
            return [
                np.log(np.sum((self.y_vec - phi) ** 2)) + self.ic_calculation_value[0] * np.trace(b_matrix) / self.T,
                np.log(np.sum((self.y_vec - phi) ** 2)) + self.ic_calculation_value[1] * np.trace(b_matrix) / self.T]

    def fit(self, plot_name=None):
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
        self.m_star, self.ic_optimal = min(enumerate([ic[self.ic_selection] for ic in ic_store]),
                                           key=operator.itemgetter(1))
        self.bhat = sum(bhat_new_store[:self.m_star + 1])[..., None]
        self.params = self.bhat.squeeze()

        i_star_counts = [0] * self.N
        for i in self.i_best_store[:self.m_star + 1]:
            i_star_counts[i] += 1
        self.i_star_frac = np.array(i_star_counts) / self.m_star
        self.fitted = True
        self.resid = self.y_vec - self.z_matrix @ self.bhat
        self.ssr = (self.resid.T @ self.resid).item()

        if plot_name:
            plt.plot(np.mean((z_matrix @ np.cumsum(np.array(bhat_new_store), axis=0).T - y_vec) ** 2, axis=0))
            plt.xlabel('m iterations')
            plt.ylabel('Training MSE')
            plt.axvline(self.m_star, linestyle='--')
            plt.savefig(f'{plot_name}_trainingcurve.png')
            plt.close()

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
                # 'bhat': self.bhat,
                # 'params': self.params,
                'i_star_frac': self.i_star_frac,
                'resid': self.resid,
                'ssr': self.ssr,
                }


class ComponentwiseL2BoostDropout(ComponentwiseL2Boost):
    def fit(self, plot_name=None):
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

        r_store = []
        eye = np.eye(self.T)

        best_bhat_store = bhat_new_store
        best_ic = ic_store[0][0] + 10
        for m in range(self.m_max):
            # form drop vector
            if m>-1:
                drop_vec = np.random.choice(a=[0, 1], p=[1 - self.hparams['dropout'], self.hparams['dropout']],
                                            size=(m + 1,)).astype(bool)
                if sum(drop_vec) == 0:
                    # if happen to drop nothing, randomly select one base learner to drop
                    drop_vec[np.random.randint(0, m + 1, 1)] = 1

                # drop_vec = np.concatenate(
                #     [np.random.choice(a=[0, 1], p=[1 - self.hparams['dropout'], self.hparams['dropout']],
                #                       size=(10,)).astype(bool) for _ in range(int(np.ceil((m + 1) / 10)))])[:m + 1]

                dropped_count = sum(drop_vec)
                dropped_phi = sum(vg_store[drop_vec])
                phi_d = phi - dropped_phi
            else:
                phi_d=phi

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
            #if dropped_count / len(drop_vec) == 1:
            #    endog = y_vec
            #else:
            #    endog = y_vec - phi_d / (1 - dropped_count / len(drop_vec))
            endog = y_vec - phi_d
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
            i_best_store.append(i_best)

            # bhat_new_store = np.concatenate((bhat_new_store, (bhat_new[None, ...])), axis=0)
            if m>-1:
                bhat_new_store[drop_vec] = bhat_new_store[drop_vec] * dropped_count / (dropped_count + 1)
                bhat_new_store = np.concatenate((bhat_new_store, (bhat_new[None, ...]) / (dropped_count + 1)), axis=0)
                phi = phi_d + dropped_phi * dropped_count / (dropped_count + 1) + self.learning_rate * g_best / (
                        dropped_count + 1)
                vg_store[drop_vec] = vg_store[drop_vec] * dropped_count / (dropped_count + 1)
                vg_store = np.concatenate((vg_store, (self.learning_rate * g_best / (dropped_count + 1))[None, ...]),
                                          axis=0)
            else:
                bhat_new_store = np.concatenate((bhat_new_store, (bhat_new[None, ...])), axis=0)
                phi = phi_d + self.learning_rate * g_best
                vg_store = np.concatenate((vg_store, (self.learning_rate * g_best)[None, ...]), axis=0)



            p_matrix = z_matrix[:, [i_best]]
            p_matrix = p_matrix @ p_matrix.T / (p_matrix.T @ p_matrix)
            b_matrix = b_matrix + self.learning_rate * p_matrix @ (eye - b_matrix)
            ic_store.append(self.get_aic_bic(phi=phi, b_matrix=b_matrix))
            r_store.append(np.sum((self.y_vec - phi) ** 2))

            if m<=10:
                add = 10
            else:
                add = 0

            if ic_store[-1][0] + add < best_ic:
                best_bhat_store = bhat_new_store

        self.bhat_new_store = csr_matrix(bhat_new_store)
        self.i_best_store = i_best_store

        self.ic_store = ic_store
        addition = [ic_store[0][0]*0.05]*10 + [0]*(len(ic_store)-10)
        self.m_star, self.ic_optimal = min(enumerate([ic[self.ic_selection]+add for ic,add in zip(ic_store,addition)]),
                                           key=operator.itemgetter(1))
        # self.m_star = 150 #####################################
        #self.bhat = sum(bhat_new_store[:self.m_star + 1])[..., None]
        self.bhat = np.sum(best_bhat_store, axis=0)[..., None]
        self.params = self.bhat.squeeze()

        i_star_counts = [0] * self.N
        for i in self.i_best_store[:self.m_star + 1]:
            i_star_counts[i] += 1
        self.i_star_frac = np.array(i_star_counts) / self.m_star
        self.fitted = True
        self.resid = self.y_vec - self.z_matrix @ self.bhat
        self.ssr = (self.resid.T @ self.resid).item()

        if plot_name:
            plt.plot(np.mean((z_matrix @ np.cumsum(np.array(bhat_new_store), axis=0).T - y_vec) ** 2, axis=0))
            plt.xlabel('m iterations')
            plt.ylabel('Training MSE')
            plt.axvline(self.m_star, linestyle='--')
            plt.savefig(f'{plot_name}_trainingcurve.png')
            plt.close()





class Xgboost(Boost):
    def __init__(self, z_matrix, y_vec, hparams, r):
        super().__init__(z_matrix=z_matrix, y_vec=y_vec)
        self.r = r
        self.hparams = hparams

        self.ic_calculation_value = [2, np.log(self.T)]
        self.ic_cn_value = [2, np.log(self.N)]

    def fit(self, deval, ehat_eval=None, plot_name=None, feature_names=None):
        self.progress = dict()
        try:
            if self.hparams['adap_gamma']:
                # Adaptive trees
                weights = np.exp(-10**self.hparams['adap_gamma']*(1-(np.arange(self.T)+1)/self.T))
            else:
                weights = None
        except KeyError:
            weights = None
        dtrain = xgb.DMatrix(self.z_matrix, label=self.y_vec, weight=weights)
        self.model = xgb.train(self.hparams, dtrain=dtrain, num_boost_round=self.hparams['num_boost_round'],
                               early_stopping_rounds=self.hparams['early_stopping_rounds'],
                               feval=ehat_eval,
                               evals=[(dtrain, 'train'), (deval, 'h_step_ahead')],
                               evals_result=self.progress,
                               verbose_eval=False)

        if plot_name:
            plt.plot(self.progress['train']['rmse'], label='train')
            plt.plot(self.progress['h_step_ahead']['rmse'], label='h step ahead')
            plt.xlabel('m iterations')
            plt.ylabel('RMSE')
            plt.savefig(f'{plot_name}_trainingcurve.png')
            plt.close()

        self.feature_names = feature_names

    def predict(self, exog, best_ntree_limit=None):
        #if not best_ntree_limit:
        #    best_ntree_limit = 0  # Use all trees
        return self.model.predict(xgb.DMatrix(exog), ntree_limit=best_ntree_limit)

    def return_data_dict(self):
        if self.feature_names:
            self.model.feature_names = self.feature_names
            self.feature_score = self.model.get_score(importance_type='gain')
            model = self.model.save_raw()[4:]
            def myfun(self=None):
                return model
            self.model.save_raw = myfun
            explainer = shap.TreeExplainer(self.model)
            shap_values = csr_matrix(explainer.shap_values(self.z_matrix))
            return {'feature_score': self.feature_score,
                    'progress': self.progress,
                    'best_ntree_limit': self.model.best_ntree_limit,
                    'feature_names': self.feature_names,
                    'shap_values': shap_values}
        else:  # No feature score
            return {'progress': self.progress,
                    'best_ntree_limit': self.model.best_ntree_limit}


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
    plt.rcParams["font.family"] = "Times New Roman"
    results_dir = create_results_directory('./results/simulation')
    n_total = 10
    t_train = 20
    t_test = 100
    simulation_runs = 20
    df_store = []

    def func(z):
        return 1 + 5 * z[:, [0]] + 2 * z[:, [1]] + z[:, [2]] + np.random.normal(0, 2, (z.shape[0], 1))

    def plot(cw, name):
        plt.plot(np.mean((sm.add_constant(z_test) @ np.cumsum(np.array(cw.bhat_new_store.toarray()),
                                                              axis=0).T - y_test) ** 2, axis=0)[5:])
        plt.xlabel('m iterations')
        plt.ylabel('Test MSE')
        plt.axvline(cw.m_star, linestyle='--')
        plt.savefig(f'{results_dir}/{name}.png')
        plt.close()
        final = min(cw.m_star + 25, cw.bhat_new_store.shape[0])
        plt.plot(np.mean((sm.add_constant(z_test) @ np.cumsum(np.array(cw.bhat_new_store.toarray()),
                                                              axis=0).T - y_test) ** 2, axis=0)[5:final])
        plt.xlabel('m iterations')
        plt.ylabel('Test MSE')
        plt.axvline(cw.m_star, linestyle='--')
        plt.savefig(f'{results_dir}/{name}_zoomed.png')
        plt.close()

    def cw_run(cw, hparams, store, idx, name):
        cw = cw(z_matrix=z, y_vec=y, hparams=hparams, r=None)
        if idx == 0:
            cw.fit(plot_name=f'{results_dir}/{name}')
        else:
            cw.fit()
        yhat = cw.predict(exog=sm.add_constant(z_test))
        ssr = sum((y_test - yhat) ** 2)
        store.append([(f'{name} MSE', ssr / t_test),
                      (f'{name} m_star', cw.m_star),
                      (f'{name} params', cw.params),
                      (f'{name} i frac', cw.i_star_frac)])
        if idx == 0:
            plot(cw, name)

    for idx in range(simulation_runs):
        z = np.random.normal(0, 1, (t_train, n_total))
        y = func(z)
        z_test = np.random.normal(0, 1, (t_test, n_total))
        y_test = func(z_test)

        ols = sm.OLS(endog=y, exog=sm.add_constant(z)).fit()
        yhat_ols = ols.predict(sm.add_constant(z_test))[..., None]
        ssr_ols = sum((y_test - yhat_ols) ** 2)

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

        results_store = {'n_total': n_total,
                         'T_train': t_train,
                         'T_test': t_test,
                         'Simulation Runs': simulation_runs,
                         'OLS MSE': ssr_ols / t_test,
                         'Lasso MSE': ssr_lasso / t_test,
                         'lasso_alpha': 10 ** alpha,
                         'predictor': np.arange(n_total + 1),
                         'True params': [1, 5, 2, 1] + [0] * (n_total - 3),
                         'ols params': ols.params,
                         'Lasso params': lasso.params,
                         }

        store = []

        hparams = {'m_max': 500, 'learning_rate': 0.1, 'ic_mode': 'aic', 'dropout': 0.5}
        cw_run(cw=ComponentwiseL2BoostDropout, hparams=hparams, store=store, idx=idx, name='cwd01_50')

        hparams = {'m_max': 500, 'learning_rate': 0.3, 'ic_mode': 'aic', 'dropout': 0.5}
        cw_run(cw=ComponentwiseL2BoostDropout, hparams=hparams, store=store, idx=idx, name='cwd03_50')

        hparams = {'m_max': 2000, 'learning_rate': 0.1, 'ic_mode': 'aic'}
        cw_run(cw=ComponentwiseL2Boost, hparams=hparams, store=store, idx=idx, name='cw01')

        hparams = {'m_max': 2000, 'learning_rate': 0.3, 'ic_mode': 'aic'}
        cw_run(cw=ComponentwiseL2Boost, hparams=hparams, store=store, idx=idx, name='cw03')

        hparams = {'m_max': 500, 'learning_rate': 0.1, 'ic_mode': 'aic', 'dropout': 0.5}
        cw_run(cw=ComponentwiseL2BoostDropout, hparams=hparams, store=store, idx=idx, name='cwd01_50')

        hparams = {'m_max': 500, 'learning_rate': 0.3, 'ic_mode': 'aic', 'dropout': 0.5}
        cw_run(cw=ComponentwiseL2BoostDropout, hparams=hparams, store=store, idx=idx, name='cwd03_50')

        store = list(zip(*store))
        for item in store:
            results_store.update(item)

        df_store.append(pd.DataFrame({k: pd.Series(v) for k, v in results_store.items()}))

    df = pd.concat(objs=df_store).groupby(level=0).mean()
    excel_name = f'{results_dir}/test_comparision.xlsx'
    excel_name = create_excel_file(excel_name)
    wb = openpyxl.load_workbook(excel_name)
    ws = wb[wb.sheetnames[-1]]
    print_df_to_excel(df=df, ws=ws)
    wb.save(excel_name)
