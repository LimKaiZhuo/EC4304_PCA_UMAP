import numpy as np
from scipy.linalg import fractional_matrix_power
import pandas as pd
import statsmodels.api as sm
import openpyxl
import matplotlib.pyplot as plt
from own_package.boosting import Xgboost
from own_package.analysis import LocalLevel
from own_package.others import print_df_to_excel, create_excel_file
import pickle, time, itertools, shap

first_est_date = '1970:1'


def forecast_error(y_predicted, y_true):
    return 'ehat', np.sum(y_true.get_label() - y_predicted)


def poos_experiment(fl_master, fl, est_dates, z_type, h, h_idx, m_max, p_max,
                    model_mode,
                    save_dir,
                    **kwargs):
    est_dates = est_dates + [fl_master.time_stamp[-1]]

    data_store = []
    for idx, (est_date, next_tune_date) in enumerate(zip(est_dates[:-1], est_dates[1:])):
        t1 = time.perf_counter()
        (x_est, _), (yo_est, _), (y_est, _), (ts_est, _), _, _ = fl_master.date_split(est_date)
        (x_tt, _), (yo_tt, _), (y_tt, _), (ts_tt, _), _, _ = fl_master.date_split(next_tune_date, date_start=est_date)
        if model_mode == 'xgb':
            hparams_df = fl.xgb_hparam_opt(x=x_est, yo=yo_est, y=y_est[:, [h_idx]], h=h,
                                           m_max=m_max, p_max=p_max,
                                           z_type=z_type,
                                           hparam_opt_params=kwargs['hparam_opt_params'],
                                           default_hparams=kwargs['default_hparams'],
                                           results_dir=None,
                                           model_name=None)

            # wb = openpyxl.Workbook()
            # ws = wb[wb.sheetnames[-1]]
            # print_df_to_excel(df=hparams_df, ws=ws)
            # wb.save('./results/poos/hparams.xlsx')
            # print('done')

            hparams = {**kwargs['default_hparams'], **hparams_df.iloc[0, :].to_dict()}
            hparams['early_stopping_rounds'] = None
            hparams['m'] = int(hparams['m'])
            hparams['max_depth'] = int(hparams['max_depth'])
            hparams['ehat_eval'] = forecast_error
            _, _, _, poos_data_store = fl.pls_expanding_window(h=h, p=hparams['m'] * 2, m=hparams['m'], r=8,
                                                               cw_model_class=Xgboost,
                                                               cw_hparams=hparams,
                                                               x_t=x_est,
                                                               x_v=x_tt,
                                                               yo_t=yo_est,
                                                               y_t=y_est[:, h_idx][..., None],
                                                               yo_v=yo_tt,
                                                               y_v=y_tt[:, h_idx][..., None],
                                                               rolling=False,
                                                               z_type=z_type,
                                                               save_dir=None,
                                                               save_name=None)
            data_store.append({'est_date': est_date,
                               'next_tune_date': next_tune_date,
                               'hparams_df': hparams_df,
                               'poos_data_store': poos_data_store},
                              )
            with open('{}/poos_h{}.pkl'.format(save_dir, h), "wb") as file:
                pickle.dump(data_store, file)

        elif model_mode == 'pca':
            hparams_df = fl.pca_hparam_opt(x=x_est, yo=yo_est, y=y_est[:, [h_idx]], h=h,
                                           m_max=m_max, p_max=p_max, r=8)
            hparams = hparams_df.iloc[0, :]
            yhat, ehat, _, _ = fl.pls_expanding_window(h=h, p=int(hparams['p']), m=int(hparams['m']), r=8,
                                                       factor_model=fl.pca_factor_estimation,
                                                       x_t=x_est,
                                                       x_v=x_tt,
                                                       yo_t=yo_est,
                                                       y_t=y_est[:, h_idx][..., None],
                                                       yo_v=yo_tt,
                                                       y_v=y_tt[:, h_idx][..., None],
                                                       rolling=False,
                                                       save_dir=None,
                                                       save_name=None)
            data_store.append({'est_date': est_date,
                               'next_tune_date': next_tune_date,
                               'hparams_df': hparams_df,
                               'ehat': ehat,
                               'yhat': yhat},
                              )
        elif model_mode == 'ar':
            hparams_df = fl.ar_hparam_opt(yo=yo_est, y=y_est[:, [h_idx]], h=h, p_max=p_max,)
            hparams = hparams_df.iloc[0, :]
            yhat, ehat, _, _ = fl.ar_pls_expanding_window(h=h, p=int(hparams['p']),
                                                          yo_t=yo_est,
                                                          y_t=y_est[:, h_idx][..., None],
                                                          yo_v=yo_tt,
                                                          y_v=y_tt[:, h_idx][..., None], rolling=False,
                                                          save_dir=None, save_name=None)
            data_store.append({'est_date': est_date,
                               'next_tune_date': next_tune_date,
                               'hparams_df': hparams_df,
                               'ehat': ehat,
                               'yhat': yhat},
                              )

        t2 = time.perf_counter()
        print(f'Completed {est_date} for h={h}. Time taken {t2 - t1}')

    if model_mode == 'ar4':
        _, (yo_est, yo_tt), (y_est, y_tt), (ts_est, ts_tt), _, _ = fl_master.date_split(est_dates[0])
        _, e_hat_store, _, _ = fl.ar_pls_expanding_window(h=h, p=4,
                                                          yo_t=yo_est,
                                                          y_t=y_est[:, h_idx][..., None],
                                                          yo_v=yo_tt,
                                                          y_v=y_tt[:, h_idx][..., None], rolling=False,
                                                          save_dir=None, save_name=None)

        idx = fl_master.time_stamp.index(first_est_date)
        y = fl_master.y[idx:, h_idx]
        ts = fl_master.time_stamp[idx:]

        data_df = pd.DataFrame.from_dict({'time_stamp': ts,
                                          f'y_{h}': y,
                                          'ar4_ehat': e_hat_store, }).set_index('time_stamp')
        hparam_df = pd.DataFrame(data=np.array([4] * len(est_dates[:-1]))[..., None], columns=['p'])
        with open(f'{save_dir}/poos_{model_mode}_h{h}_analysis_results.pkl', "wb") as file:
            pickle.dump({'data_df': data_df, 'hparam_df': hparam_df}, file)

    elif model_mode in ['pca', 'ar']:
        hparam_df = pd.concat([data['hparams_df'].iloc[0, :] for data in data_store], axis=1).T

        idx = fl_master.time_stamp.index(first_est_date)
        y = fl_master.y[idx:, h_idx]
        ts = fl_master.time_stamp[idx:]

        data_df = pd.DataFrame.from_dict({'time_stamp': ts,
                                          f'y_{h}': y,
                                          f'{model_mode}_ehat': [x for data in data_store for x in data['ehat']],
                                          }).set_index('time_stamp')
        with open(f'{save_dir}/poos_{model_mode}_h{h}_analysis_results.pkl', "wb") as file:
            pickle.dump({'data_df': data_df, 'hparam_df': hparam_df}, file)


def poos_analysis(fl_master, h, h_idx, model_mode, results_dir, save_dir):
    def calculate_sequence_ntree(block_ae, hparam_ntree):
        best_idx_store = list(np.argmin(block_ae, axis=1))
        predicted_idx = [hparam_ntree] * 5
        current_window = best_idx_store[:5]
        for y in best_idx_store[5:]:
            model = LocalLevel(current_window)
            res = model.fit(disp=False)
            predicted_idx.append(int(res.forecast()))
            current_window.append(y)
        return best_idx_store, predicted_idx

    def calculate_horizon_rmse_ae(block_ae, block_ehat, sequence_ntree):
        return np.average([ae[idx] ** 2 for ae, idx in zip(block_ae, sequence_ntree)]) ** 0.5, \
               [ehat[idx] for ehat, idx in zip(block_ehat, sequence_ntree)]

    with open(save_dir, 'rb') as handle:
        data_store = pickle.load(handle)

    '''
    h = 24
    with open(f'./results/poos/poos_DPC_xgba/poos_h{h}a.pkl', 'rb') as handle:
    a = pickle.load(handle)
    
    with open(f'./results/poos/poos_DPC_xgba/poos_h{h}b.pkl', 'rb') as handle:
    b = pickle.load(handle)
    
    with open(f'./results/poos/poos_DPC_xgba/poos_h{h}c.pkl', 'rb') as handle:
    c = pickle.load(handle)
    
    with open('./results/poos/poos_DPC_xgba/poos_h12.pkl', "wb") as file:
    pickle.dump(data, file)
    '''

    idx = fl_master.time_stamp.index(first_est_date)
    y = fl_master.y[idx:, h_idx]
    ts = fl_master.time_stamp[idx:]

    if model_mode == 'xgb':
        index_products = list(itertools.product(fl_master.features_names, list(range(24)))) + [('y', idx) for idx in
                                                                                               range(48)]

        index_products = [('y', str(idx)) for idx in range(3)]

        score_df = pd.DataFrame(index=pd.MultiIndex.from_tuples(index_products, names=['Variable', 'Lag'])).T

        ae_store = []
        ehat_store = []
        optimal_ntree_store = []
        predicted_ntree_store = []
        hparam_ntree_store = []
        block_store = []
        for idx, data in enumerate(data_store):
            for single_step in data['poos_data_store']:
                scores = {(k.partition('_L')[0], k.partition('_L')[-1]): v for k, v in
                          single_step['feature_score'].items()}
                score_df = score_df.append(scores, ignore_index=True)

            block_ae = [single_step['progress']['h_step_ahead']['rmse'] for single_step in data['poos_data_store']]
            block_store.extend([idx] * len(block_ae))
            block_ehat = [single_step['progress']['h_step_ahead']['ehat'] for single_step in data['poos_data_store']]
            ae_store.extend(block_ae)
            ehat_store.extend(block_ehat)
            hparam_ntree = min(
                int(round(data['hparams_df'].iloc[0, data['hparams_df'].columns.tolist().index('m iters')] / 0.85)),
                600) - 1
            optimal_ntree, predicted_ntree = calculate_sequence_ntree(block_ae=block_ae,
                                                                      hparam_ntree=hparam_ntree)
            optimal_ntree_store.extend(optimal_ntree)
            predicted_ntree_store.extend(predicted_ntree)
            hparam_ntree_store.extend([hparam_ntree] * len(optimal_ntree))

        # Full SSM prediction
        _, ssm_full_ntree_store = calculate_sequence_ntree(block_ae=ae_store,
                                                           hparam_ntree=min(int(round(data_store[0]['hparams_df'].iloc[
                                                                                          0, data_store[0][
                                                                                              'hparams_df'].columns.tolist().
                                                                                      index('m iters')] / 0.85)),
                                                                            600) - 1)

        # Horizons RMSE calculation
        oracle_rmse, oracle_ehat = calculate_horizon_rmse_ae(ae_store, ehat_store, optimal_ntree_store)
        max_iter_rmse, max_iter_ehat = calculate_horizon_rmse_ae(ae_store, ehat_store, [-1] * len(ae_store))
        hparam_rmse, hparam_ehat = calculate_horizon_rmse_ae(ae_store, ehat_store, hparam_ntree_store)
        ssm_rmse, ssm_ehat = calculate_horizon_rmse_ae(ae_store, ehat_store, predicted_ntree_store)
        ssm_full_rmse, ssm_full_ehat = calculate_horizon_rmse_ae(ae_store, ehat_store, ssm_full_ntree_store)

        df = pd.DataFrame.from_dict({'time_stamp': ts,
                                     'hparam_block': block_store,
                                     f'y_{h}': y,
                                     'oracle_ehat': oracle_ehat,
                                     'ssm_ehat': ssm_ehat,
                                     'ssm_full_ehat': ssm_full_ehat,
                                     'hparam_ehat': hparam_ehat,
                                     'max_iter_ehat': max_iter_ehat,
                                     'oracle_ntree': optimal_ntree_store,
                                     'ssm_ntree': predicted_ntree_store,
                                     'ssm_full_ntree': ssm_full_ntree_store,
                                     'hparam_ntree': hparam_ntree_store}).set_index('time_stamp')

        # df.plot(y=['oracle_ehat', 'ssm_ehat', 'ssm_full_ehat', 'hparam_ehat', 'max_iter_ehat'], use_index=True)

        rmse_df = pd.DataFrame.from_dict({'oracle_rmse': [oracle_rmse],
                                          'ssm_rmse': [ssm_rmse],
                                          'ssm_full_rmse': [ssm_full_rmse],
                                          'hparam_rmse': [hparam_rmse],
                                          'max_iter_rmse': [max_iter_rmse]})

        hparam_df = [data['hparams_df'].iloc[0, :] for data in data_store]
        hparam_df = pd.concat(hparam_df, axis=1).T

        with open(f'{results_dir}/poos_{model_mode}_h{h}_analysis_results.pkl', "wb") as file:
            pickle.dump({'data_df': df, 'rmse_df': rmse_df, 'hparam_df': hparam_df}, file)


def poos_processed_data_analysis(results_dir, save_dir_store, h_store, model_mode):
    first_est_date = '1970:1'
    est_dates = [f'{x}:12' for x in range(1969, 2020, 5)[1:-1]]
    df_store = []
    for save_dir in save_dir_store:
        with open(save_dir, 'rb') as handle:
            data_store = pickle.load(handle)

        data_df = data_store['data_df']
        ehat_df = data_df[[x for x in data_df.columns.values if '_ehat' in x]]
        ehat_df.columns = [name.replace('ehat', 'rmse') for name in ehat_df.columns.values]
        time_stamps = data_df.index.tolist()
        est_dates_idx = [0] + [time_stamps.index(est) + 1 for est in est_dates] + [-1]

        rmse_store = []
        for start, end in zip(est_dates_idx[:-1], est_dates_idx[1:]):
            rmse_store.append((ehat_df.iloc[start:end, :] ** 2).mean(axis=0) ** 0.5)

        start_store = ['1970:1', '1970:1', '1983:1', '2007:1', '2019:12']  # Inclusive
        end_store = ['2020:6', '1983:1', '2007:1', '2020:6', '2020:6']  # Exclusive of those dates

        for start, end in zip(start_store, end_store):
            start = time_stamps.index(start)
            try:
                end = time_stamps.index(end)
            except ValueError:
                end = -1  # Means is longer than last date ==> take last entry
            rmse_store.append((ehat_df.iloc[start:end, :] ** 2).mean(axis=0) ** 0.5)

        df = pd.concat((pd.concat(rmse_store, axis=1).T, data_store['hparam_df'].reset_index()), axis=1)
        df['start_dates'] = [first_est_date] + est_dates + start_store
        df['end_dates'] = est_dates + [time_stamps[-1]] + end_store
        df_store.append(df)

    excel_name = create_excel_file(f'{results_dir}/poos_analysis_{model_mode}.xlsx')
    wb = openpyxl.Workbook()
    for df, h in zip(df_store, h_store):
        wb.create_sheet(f'h{h}')
        ws = wb[f'h{h}']
        print_df_to_excel(df=df, ws=ws)

    wb.save(excel_name)


def poos_model_evaluation(ar_store, pca_store, xgb_store, results_dir):
    def xl_test(x):
        n = x.shape[0]
        u = x - x.mean()
        sigma = (x**2).mean()
        v = u**2 - sigma
        z = np.concatenate((u[None,:],v[None,:]), axis=0)
        ar1_model = sm.tsa.ARMA(u, (1,0)).fit(disp=False)
        rho = abs(ar1_model.params[1])
        q = int(np.round((1.5*n)**(1/3)*(2*rho/(1-rho**2))**(2/3)))

        sum_uu = 0
        sum_uv = 0
        sum_vu = 0
        sum_vv = 0
        for h in range(1, q+1):
            g_uu = 1/n*np.sum(u[h:]*u[:-h])
            g_uv = 1/n*np.sum(u[h:]*v[:-h])
            g_vu = 1/n*np.sum(v[h:]*u[:-h])
            g_vv = 1/n*np.sum(v[h:]*v[:-h])
            sum_uu += (1-h/(q+1))*g_uu
            sum_uv += (1 - h / (q + 1)) * g_uv
            sum_vu += (1 - h / (q + 1)) * g_vu
            sum_vv += (1 - h / (q + 1)) * g_vv

        g_uu_0 = 1 / n * np.sum(u * u)
        g_uv_0 = 1 / n * np.sum(u * v)
        g_vv_0 = 1 / n * np.sum(v * v)

        w_u = g_uu_0+2*sum_uu
        w_v = g_vv_0+2*sum_vv
        w_uv = g_uv_0 + sum_uv + sum_vu
        if w_uv<0:
            print('w_uv is negative')
        omega = np.array([[w_u, abs(w_uv)**0.5], [abs(w_uv)**0.5, w_v]])
        omega_neghalf = fractional_matrix_power(omega, -0.5)

        sequence = np.empty_like(z)
        z_cumsum = np.cumsum(z, axis=1)
        for i in range(z.shape[1]):
            sequence[:, i ] = omega_neghalf@z_cumsum[:,i]

        c = 1/n**0.5 * np.max(np.linalg.norm(sequence, ord=1, axis=0))
        if c<=1.89:
            p = 0.1
        elif c<=2.07:
            p=0.05
        elif c<=2.4:
            p=0.01
        else:
            p=0
        return p

    h_store = 1, 3, 6, 12, 24
    results_store = {}
    results_xl_store = {}
    for h, ar, pca, xgb in zip(h_store, ar_store, pca_store, xgb_store):
        with open(ar, 'rb') as handle:
            ar_data = pickle.load(handle)

        with open(pca, 'rb') as handle:
            pca_data = pickle.load(handle)

        with open(xgb, 'rb') as handle:
            xgb_data = pickle.load(handle)

        model_data = pd.concat([pca_data['data_df'], xgb_data['data_df']], axis=1)

        d_vectors = (ar_data['data_df']['ar_ehat'].values ** 2)[..., None] - \
                    model_data[[x for x in model_data.columns.values if '_ehat' in x]].values ** 2


        p_values = np.apply_along_axis(lambda x: sm.tsa.stattools.kpss(x),
                                       axis=0, arr=d_vectors)[1, :]

        p_xl_values = np.apply_along_axis(lambda x: xl_test(x),
                                       axis=0, arr=d_vectors)

        results_store[f'h{h}'] = p_values
        results_xl_store[f'h{h}'] = p_xl_values

        # Plotting
        plt.close()
        d_vectors = pd.DataFrame(d_vectors, columns=[x.replace('_ehat', '_d') for x in model_data.columns.values if '_ehat' in x])
        d_vectors.plot(subplots=True, layout=(3,2))
        plt.savefig(f'{results_dir}/h{h}_d_plot.png')

    results_df = pd.DataFrame.from_dict(results_store, orient='index',
                                        columns=[x.replace('_ehat', '_KPSS') for x in model_data.columns.values if '_ehat' in x])

    results_xldf = pd.DataFrame.from_dict(results_xl_store, orient='index',
                                        columns=[x.replace('_ehat', '_XL')  for x in model_data.columns.values if '_ehat' in x])

    results_df = pd.concat((results_df, results_xldf), axis=1)

    wb = openpyxl.Workbook()
    ws = wb[wb.sheetnames[-1]]
    print_df_to_excel(df=results_df, ws=ws)
    wb.save(f'{results_dir}/test_summary.xlsx')



class Shap_data:
    def __init__(self, shap_matrix, feature_names):
        df = pd.DataFrame(shap_matrix, columns=feature_names).iloc[:, 1:].T  # Remove first column of constants
        # Convert to multi index. Split name and lags
        df.index = pd.MultiIndex.from_tuples(df.index.str.split('_L').tolist())
        self.feature_names = feature_names
        self.df = df.T  # Change back to multi-column
        self.grouped_df = self.df.abs().sum(level=0, axis=1)
        self.shap_abs = self.df.abs().sum(axis=0)
        self.grouped_shap_abs = self.grouped_df.sum(axis=0)

    def summary_plot(self, grouped=False, plot_type='bar'):
        if grouped:
            shap.summary_plot(self.grouped_df, feature_names=self.grouped_df.columns.values, plot_type=plot_type)
        else:
            shap.summary_plot(self.df, feature_names=self.df.columns.values, plot_type=plot_type)


def poos_shap(fl_master, fl, xgb_store, results_dir):
    excel_name = create_excel_file(f'{results_dir}/poos_shap.xlsx')
    wb = openpyxl.load_workbook(excel_name)

    idx = fl_master.time_stamp.index(first_est_date)
    ts = fl_master.time_stamp[idx:]

    for xgb_dir,h  in zip(xgb_store, [1,3,6,12,24]):
        with open(xgb_dir, 'rb') as handle:
            xgb_data = pickle.load(handle)

        shap_abs = []
        grouped_shap_abs = []

        for block in xgb_data:
            feature_names = block['poos_data_store'][0]['feature_names']
            for step in block['poos_data_store']:
                shap_data = Shap_data(shap_matrix=step['shap_values'].toarray(), feature_names=feature_names)
                shap_abs.append(shap_data.shap_abs)
                grouped_shap_abs.append(shap_data.grouped_shap_abs)

        shap_abs = pd.concat(shap_abs, axis=1).T
        shap_abs.index = ts
        gshap_abs = pd.concat(grouped_shap_abs, axis=1).T
        gshap_abs.index=ts

        def get_top_columns_per_row(df, n_top):
            ranked_matrix = np.argsort(-df.values, axis=1)[:, :n_top]
            return pd.DataFrame(df.columns.values[ranked_matrix],
                                index=df.index,
                                columns=[f'Rank {idx + 1}' for idx in range(n_top)]), \
                   pd.DataFrame(df.values[np.repeat(np.arange(df.shape[0])[:, None], n_top, axis=1), ranked_matrix],
                                index=df.index,
                                columns=[f'Rank {idx + 1}' for idx in range(n_top)])

        top_shap_names, top_shap_values = get_top_columns_per_row(df=shap_abs, n_top=10)
        top_gshap_names, top_gshap_values = get_top_columns_per_row(df=gshap_abs, n_top=10)

        #gshap_abs_norm = gshap_abs.div(gshap_abs.max(axis=1), axis=0)
        #gshap_abs.plot(y=np.unique(top_gshap_names.values))

        def print_df(sheet_name, df, h):
            wb.create_sheet(f'h{h}_{sheet_name}')
            ws = wb[f'h{h}_{sheet_name}']
            print_df_to_excel(ws=ws, df=df)

        print_df('shap_names', df=top_shap_names.applymap(str), h=h)
        print_df('shap_values', df=top_shap_values, h=h)
        print_df('gshap_names', df=top_gshap_names.applymap(str), h=h)
        print_df('gshap_values', df=top_gshap_values, h=h)

        wb.save(excel_name)



