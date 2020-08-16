import numpy as np
from scipy.linalg import fractional_matrix_power
import pandas as pd
import statsmodels.api as sm
import openpyxl
import matplotlib.pyplot as plt
from arch.bootstrap import MCS, SPA
from own_package.boosting import Xgboost
from own_package.analysis import LocalLevel
from own_package.others import print_df_to_excel, create_excel_file
import pickle, time, itertools, shap

first_est_date = '1970:1'


def forecast_error(y_predicted, y_true):
    return 'ehat', np.sum(y_true.get_label() - y_predicted)


def poos_experiment(fl_master, fl, est_dates, z_type, h, h_idx, m_max, p_max, first_est_date,
                    model_mode, save_dir,
                    **kwargs):
    est_dates = est_dates + [fl_master.time_stamp[-1]]

    data_store = []

    if model_mode == 'xgb_with_hparam':
        with open(f'{kwargs["hparam_save_dir"]}/poos_h{h}.pkl', 'rb') as handle:
            hparam_data_store = pickle.load(handle)


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
                                                               rolling=True,
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
        elif model_mode == 'xgb_with_hparam':
            hparams_df = hparam_data_store[idx]['hparams_df']
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
                                                               rolling=True,
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
                                                       rolling=True,
                                                       save_dir=None,
                                                       save_name=None)
            data_store.append({'est_date': est_date,
                               'next_tune_date': next_tune_date,
                               'hparams_df': hparams_df,
                               'ehat': ehat,
                               'yhat': yhat},
                              )
        elif model_mode == 'ar':
            hparams_df = fl.ar_hparam_opt(yo=yo_est, y=y_est[:, [h_idx]], h=h, p_max=p_max, )
            hparams = hparams_df.iloc[0, :]
            yhat, ehat, _, _ = fl.ar_pls_expanding_window(h=h, p=int(hparams['p']),
                                                          yo_t=yo_est,
                                                          y_t=y_est[:, h_idx][..., None],
                                                          yo_v=yo_tt,
                                                          y_v=y_tt[:, h_idx][..., None], rolling=True,
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
                                                          y_v=y_tt[:, h_idx][..., None], rolling=True,
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


def poos_analysis(fl_master, h, h_idx, model_mode, est_mode, results_dir, save_dir, first_est_date):
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

    def calculate_hparam_ntree(hparam_df):
        if est_mode == 'rep_holdout':
            div_factor = 0.85
        elif est_mode == 'rfcv':
            div_factor = 0.8
        return min(int(round(hparam_df.iloc[0, hparam_df.columns.tolist().index('m iters')] / div_factor)), 600) - 1

    with open(save_dir, 'rb') as handle:
        data_store = pickle.load(handle)

    idx = fl_master.time_stamp.index(first_est_date)
    y = fl_master.y[idx:, h_idx]
    ts = fl_master.time_stamp[idx:]

    if model_mode == 'xgb':
        index_products = [('y', str(idx)) for idx in range(3)]

        score_df = pd.DataFrame(index=pd.MultiIndex.from_tuples(index_products, names=['Variable', 'Lag'])).T

        ae_store = []
        ehat_store = []
        optimal_ntree_store = []
        predicted_ntree_store = []
        rw_ntree_store = []
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
            hparam_ntree = calculate_hparam_ntree(data['hparams_df'])
            optimal_ntree, predicted_ntree = calculate_sequence_ntree(block_ae=block_ae,
                                                                      hparam_ntree=hparam_ntree)
            optimal_ntree_store.extend(optimal_ntree)
            rw_ntree_store.extend([hparam_ntree]*5+optimal_ntree[4:-1])
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
        rw_rmse, rw_ehat = calculate_horizon_rmse_ae(ae_store, ehat_store, rw_ntree_store)

        df = pd.DataFrame.from_dict({'time_stamp': ts,
                                     'hparam_block': block_store,
                                     f'y_{h}': y,
                                     'oracle_ehat': oracle_ehat,
                                     'ssm_ehat': ssm_ehat,
                                     'ssm_full_ehat': ssm_full_ehat,
                                     'rw_ehat': rw_ehat,
                                     'hparam_ehat': hparam_ehat,
                                     'max_iter_ehat': max_iter_ehat,
                                     'oracle_ntree': optimal_ntree_store,
                                     'ssm_ntree': predicted_ntree_store,
                                     'ssm_full_ntree': ssm_full_ntree_store,
                                     'rw_ntree':rw_ntree_store,
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


def poos_analysis_combining_xgb(h, results_dir, poos_post_dir_store):
    yhat_df_store = []
    hparam_df_store = []
    for poos_post_dir in poos_post_dir_store:
        with open(f'{poos_post_dir}/poos_xgb_h{h}_analysis_results.pkl', 'rb') as handle:
            data = pickle.load(handle)
        df = data['data_df']
        yhat_df = df[[x for x in df.columns if 'y_' in x]].values - df[[x for x in df.columns if '_ehat' in x]]
        yhat_df_store.append(yhat_df)
        hparam_df_store.append(data['hparam_df'].reset_index(drop=True))

    y = df[[x for x in df.columns if 'y_' in x]].values
    data_df = pd.concat(yhat_df_store).groupby(level=0).mean().reindex(df.index)
    data_df = y - data_df
    hparam_df = pd.concat(hparam_df_store).groupby(level=0).mean()

    with open(f'{results_dir}/poos_xgb_h{h}_analysis_results.pkl', "wb") as file:
        pickle.dump({'data_df': data_df, 'hparam_df':hparam_df}, file)


def poos_xgb_plotting_m(h, results_dir):
    with open(f'{results_dir}/poos_xgb_h{h}_analysis_results.pkl', 'rb') as handle:
        data = pickle.load(handle)
    df = data['data_df']
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.margins(x=0.01, y=0.01)
    df.plot(kind='line', y=['oracle_ntree', 'ssm_ntree'], ax=ax, marker='o', ms=1.5)
    ax.set_ylabel('ntrees')
    fig.tight_layout()
    fig.savefig(f'{results_dir}/ntrees_vs_time_h{h}.png', transparent=False, dpi=300, bbox_inches="tight")

    with open(f'{results_dir}/poos_h{h}.pkl', 'rb') as handle:
        data = pickle.load(handle)
    ehat_store = np.array([step['progress']['h_step_ahead']['rmse'] for block in data for step in block['poos_data_store']])
    ehat_store = (ehat_store-ehat_store.min(axis=1)[:,None])/(ehat_store.max(axis=1)[:,None]-ehat_store.min(axis=1)[:,None])
    ehat_df = pd.DataFrame(ehat_store.T, columns=df.index.values)
    fig, ax = plt.subplots()
    ax.margins(x=0.01, y=0.01)
    y_store = ['1970:1', '1983:1', '2007:1', '2020:3']
    ehat_df.plot(kind='line', y=y_store, ax=ax)
    ax.set_ylabel('|ehat|')
    ax.set_xlabel('ntrees')
    ax.legend(loc='right')
    for y in y_store:
        ax.scatter(df.loc[y]['ssm_ntree'], ehat_df[y].iloc[int(df.loc[y]['ssm_ntree'])], marker='x', s=100)
    fig.tight_layout()
    fig.savefig(f'{results_dir}/ehat_vs_ntrees_h{h}.png', transparent=False, dpi=300, bbox_inches="tight")



def poos_processed_data_analysis(results_dir, save_dir_store, h_store, model_mode, nber_excel_dir):
    first_est_date = '1970:1'
    # Load NBER dates
    nber = pd.read_excel(nber_excel_dir, sheet_name='main')
    peaks = pd.DatetimeIndex(pd.to_datetime(nber['Peak month']))[1:]  # First month is NaN
    troughs = pd.DatetimeIndex(pd.to_datetime(nber['Trough month']))[1:]
    dates = pd.date_range('1970-01-01', '2020-12-01', freq='MS')
    nber_df = pd.DataFrame(data=[-1]*len(dates), index=dates, columns=['business_cycle'])
    for idx,(d1, d2) in enumerate(zip(peaks, troughs)):
        # recession ==> 0
        nber_df.loc[d1:d2-pd.offsets.DateOffset(1, 'months')] = 0
    for idx,(d1,d2) in enumerate(zip(troughs[:-1], peaks[1:])):
        # expansionary ==> 1
        nber_df.loc[d1:d2-pd.offsets.DateOffset(1, 'months')] = 1
    nber_df.index = [f'{int(year)}:{int(month)}' for year, month in zip(nber_df.index.year.values, nber_df.index.month.values)]

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

        start_store = ['1970:1','1970:1', '1970:1', '1983:1', '2007:1', '2019:12']  # Inclusive
        end_store = ['2020:1','2020:12', '1983:1', '2007:1', '2020:1', '2020:12']  # Exclusive of those dates

        for start, end in zip(start_store, end_store):
            start = time_stamps.index(start)
            try:
                end = time_stamps.index(end)
                rmse_store.append((ehat_df.iloc[start:end, :] ** 2).mean(axis=0) ** 0.5)
            except ValueError:
                # Means end is the last date
                rmse_store.append((ehat_df.iloc[start:, :] ** 2).mean(axis=0) ** 0.5)

        # NBER Expansionary and Recessionary
        rmse_store.append((ehat_df.loc[nber_df['business_cycle']==1, :] ** 2).mean(axis=0) ** 0.5)
        rmse_store.append((ehat_df.loc[nber_df['business_cycle'] == 0, :] ** 2).mean(axis=0) ** 0.5)

        # Combining data into dataframe
        df = pd.concat((pd.concat(rmse_store, axis=1).T, data_store['hparam_df'].reset_index()), axis=1)
        df['start_dates'] = [first_est_date] + est_dates + start_store + ['Expansionary', 'Recessionary']
        df['end_dates'] = est_dates + [time_stamps[-1]] + end_store + ['', '']
        df_store.append(df)

    excel_name = create_excel_file(f'{results_dir}/poos_analysis_{model_mode}.xlsx')
    wb = openpyxl.Workbook()
    for df, h in zip(df_store, h_store):
        wb.create_sheet(f'h{h}')
        ws = wb[f'h{h}']
        print_df_to_excel(df=df, ws=ws)

    wb.save(excel_name)


def poos_model_evaluation(fl_master, ar_store, pca_store, xgb_store, results_dir, blocked_dates, blocks,
                          first_est_date):
    def xl_test(x):
        n = x.shape[0]
        u = x - x.mean()
        sigma = (x ** 2).mean()
        v = u ** 2 - sigma
        z = np.concatenate((u[None, :], v[None, :]), axis=0)
        ar1_model = sm.tsa.ARMA(u, (1, 0)).fit(disp=False)
        rho = abs(ar1_model.params[1])
        q = int(np.round((1.5 * n) ** (1 / 3) * (2 * rho / (1 - rho ** 2)) ** (2 / 3)))

        sum_uu = 0
        sum_uv = 0
        sum_vu = 0
        sum_vv = 0
        for h in range(1, q + 1):
            g_uu = 1 / n * np.sum(u[h:] * u[:-h])
            g_uv = 1 / n * np.sum(u[h:] * v[:-h])
            g_vu = 1 / n * np.sum(v[h:] * u[:-h])
            g_vv = 1 / n * np.sum(v[h:] * v[:-h])
            sum_uu += (1 - h / (q + 1)) * g_uu
            sum_uv += (1 - h / (q + 1)) * g_uv
            sum_vu += (1 - h / (q + 1)) * g_vu
            sum_vv += (1 - h / (q + 1)) * g_vv

        g_uu_0 = 1 / n * np.sum(u * u)
        g_uv_0 = 1 / n * np.sum(u * v)
        g_vv_0 = 1 / n * np.sum(v * v)

        w_u = g_uu_0 + 2 * sum_uu
        w_v = g_vv_0 + 2 * sum_vv
        w_uv = g_uv_0 + sum_uv + sum_vu
        if w_uv < 0:
            print('w_uv is negative')
        omega = np.array([[w_u, abs(w_uv) ** 0.5], [abs(w_uv) ** 0.5, w_v]])
        omega_neghalf = fractional_matrix_power(omega, -0.5)

        sequence = np.empty_like(z)
        z_cumsum = np.cumsum(z, axis=1)
        for i in range(z.shape[1]):
            sequence[:, i] = omega_neghalf @ z_cumsum[:, i]

        c = 1 / n ** 0.5 * np.max(np.linalg.norm(sequence, ord=1, axis=0))
        if c <= 1.89:
            p = 0.1
        elif c <= 2.07:
            p = 0.05
        elif c <= 2.4:
            p = 0.01
        else:
            p = 0
        return p

    def remove_lr_blocks(df, idx, v, slice_mode='number'):
        if slice_mode == 'number':
            return df.drop(df.index[idx - v:idx + v + 1])
        elif slice_mode == 'index':
            idx = np.where(df.index == idx)[0][0]
            return df.drop(df.index[idx - v:idx + v + 1])
        else:
            raise KeyError('slice_mode should be number or index.')

    def compute_mcs(e2_df, drop_cols=None):
        if drop_cols:
            e2_df = e2_df.drop(drop_cols, axis=1)
        mcs = MCS(e2_df.copy(), size=0.1)
        try:
            mcs.compute()
            return mcs.pvalues.T.squeeze()
        except IndexError:
            # Means it is the 1970:1 block where ssm and ssm full is exactly the same.
            # So need to remove ssm full
            blank = pd.Series()
            if (e2_df['ssm_full_ehat^2'] == e2_df['ssm_ehat^2']).all():
                e2_df = e2_df.drop('ssm_full_ehat^2', axis=1)
                blank['ssm_full_ehat^2'] = 0
            if (e2_df['hparam_ehat^2'] == e2_df['max_iter_ehat^2']).all():
                e2_df = e2_df.drop('max_iter_ehat^2', axis=1)
                blank['max_iter_ehat^2'] = 0
            mcs = MCS(e2_df.copy(), size=0.1)
            mcs.compute()
            series = mcs.pvalues.T.squeeze()
            series = pd.concat((series, blank))
            return series

    def compute_spa(e2_df):
        series = {}
        for col in e2_df.columns.values:
            spa = SPA(e2_df[col], e2_df.drop(col, axis=1))
            spa.seed(42)
            spa.compute()
            series[col] = spa.pvalues['consistent']
        return pd.Series(series)

    def plotting_mcs(df, save_dir):
        df = df.copy()
        index = df.index.values
        blocks = list(range(10)) * 5
        index = [[x.partition('_')[0], block + 1] for block, x in zip(blocks, index)]
        df.index = pd.MultiIndex.from_arrays(np.array(index).T.tolist())
        f, a = plt.subplots(3, 2, constrained_layout=True)
        df.xs('h1').plot(ax=a[0, 0], legend=False)
        df.xs('h3').plot(ax=a[0, 1], legend=False)
        df.xs('h6').plot(ax=a[1, 0], legend=False)
        df.xs('h12').plot(ax=a[1, 1], legend=False)
        df.xs('h24').plot(ax=a[2, 0], legend=False)
        a[0, 0].set_title('h = 1')
        a[0, 1].set_title('h = 3')
        a[1, 0].set_title('h = 6')
        a[1, 1].set_title('h = 12')
        a[2, 0].set_title('h = 24')
        f.delaxes(a[2, 1])

        axLine, axLabel = f.axes[0].get_legend_handles_labels()
        lines = axLine
        labels = axLabel
        f.legend(lines, labels, loc='lower right')
        plt.savefig(save_dir, bbox_inches='tight')

    idx = fl_master.time_stamp.index(first_est_date)
    ts = fl_master.time_stamp[idx:]

    h_store = 1, 3, 6, 12, 24
    results_store = {}
    mcs_store = {}
    mcs_store_no_oracle = {}
    spa_store = {'no_oracle': {},
                 'selected': {}}
    if blocks:
        est_dates = [f'{x}:12' for x in range(1969, 2020, 5)[1:-1]]
        est_dates_idx = [0] + [ts.index(est) + 1 for est in est_dates] + [-1]

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

            losses_data = pd.concat([ar_data['data_df'], pca_data['data_df'], xgb_data['data_df']], axis=1)
            losses_data = losses_data[[x for x in losses_data.columns.values if '_ehat' in x]] ** 2
            losses_data.columns = [f'{x}^2' for x in losses_data.columns]

            for start, end in zip(est_dates_idx[:-1], est_dates_idx[1:]):
                p_values = np.apply_along_axis(lambda x: sm.tsa.stattools.kpss(x),
                                               axis=0, arr=d_vectors[start:end])[1, :]
                #p_xl_values = np.apply_along_axis(lambda x: xl_test(x),
                #                                  axis=0, arr=d_vectors[start:end])
                results_store[f'h{h}_{ts[start]}~{ts[end]}'] = np.concatenate((p_values, p_values))

                series = compute_mcs(e2_df=losses_data.iloc[start:end].copy(), drop_cols=None)
                series_no_oracle = compute_mcs(e2_df=losses_data.iloc[start:end].copy(), drop_cols=['oracle_ehat^2'])
                mcs_store[f'h{h}_{ts[start]}~{ts[end]}'] = series
                mcs_store_no_oracle[f'h{h}_{ts[start]}~{ts[end]}'] = series_no_oracle

                for k, _ in spa_store.items():
                    if k == 'no_oracle':
                        spa_store[k][f'h{h}_{ts[start]}~{ts[end]}'] = compute_spa(
                            losses_data.iloc[start:end].drop('oracle_ehat^2', axis=1))
                    elif k == 'selected':
                        spa_store[k][f'h{h}_{ts[start]}~{ts[end]}'] = compute_spa(
                            losses_data.iloc[start:end].drop(
                                ['oracle_ehat^2', 'ssm_full_ehat^2', 'hparam_ehat^2', 'max_iter_ehat^2'], axis=1))

    else:
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

            # Plotting
            plt.close()
            d_vectors = pd.DataFrame(d_vectors, columns=[x.replace('_ehat', '_d') for x in model_data.columns.values if
                                                         '_ehat' in x], index=ts)
            d_vectors.plot(subplots=True, layout=(3, 2))
            plt.savefig(f'{results_dir}/h{h}_d_plot.png')

            v = 12
            for dates in blocked_dates:
                d_vectors = remove_lr_blocks(d_vectors, idx=dates, v=v, slice_mode='index')

            d_vectors.plot(subplots=True, layout=(3, 2))
            plt.savefig(f'{results_dir}/h{h}_d_removed_plot.png')

            d_vectors = d_vectors.values

            p_values_adj = np.apply_along_axis(lambda x: sm.tsa.stattools.kpss(x),
                                               axis=0, arr=d_vectors)[1, :]

            p_xl_values_adj = np.apply_along_axis(lambda x: xl_test(x),
                                                  axis=0, arr=d_vectors)

            results_store[f'h{h}'] = [f'{x:.3f} / {y:.3f}' for x, y in zip(p_values, p_values_adj)] + [
                f'{x:.3f} / {y:.3f}' for x, y in zip(p_xl_values, p_xl_values_adj)]

    results_df = pd.DataFrame.from_dict(results_store, orient='index',
                                        columns=[x.replace('_ehat', type) for type in ['_KPSS', '_XL'] for x in
                                                 model_data.columns.values if '_ehat' in x])

    wb = openpyxl.Workbook()
    ws = wb[wb.sheetnames[-1]]
    print_df_to_excel(df=results_df, ws=ws)

    if blocks:
        wb.create_sheet('MCS')
        ws = wb['MCS']
        results_df = pd.DataFrame(mcs_store).T
        results_df = results_df[losses_data.columns.values]
        results_df.columns = [x.replace('_ehat^2', type) for type in ['_MCS'] for x in
                              losses_data.columns.values if '_ehat^2' in x]
        plotting_mcs(results_df, save_dir=f'{results_dir}/MCS.png')
        mean = results_df.mean()
        count10 = results_df[results_df<0.1].count()
        results_df.loc['mean'] = mean
        results_df.loc['<10%'] = count10
        print_df_to_excel(df=results_df, ws=ws)

        wb.create_sheet('MCS_no_oracle')
        ws = wb['MCS_no_oracle']
        results_df = pd.DataFrame(mcs_store_no_oracle).T
        results_df = results_df.reindex(columns=losses_data.columns.values, fill_value=-1)
        results_df.columns = [x.replace('_ehat^2', type) for type in ['_MCS'] for x in
                              losses_data.columns.values if '_ehat^2' in x]
        results_df.drop('oracle_MCS', axis=1, inplace=True)
        plotting_mcs(results_df, save_dir=f'{results_dir}/MCS_no_oracle.png')
        plotting_mcs(results_df[['ar_MCS', 'pca_MCS', 'ssm_MCS']], save_dir=f'{results_dir}/MCS_selected.png')
        plotting_mcs(results_df[['ssm_MCS', 'ssm_full_MCS', 'hparam_MCS', 'max_iter_MCS']],
                     save_dir=f'{results_dir}/MCS_xgb.png')
        mean = results_df.mean()
        count10 = results_df[results_df<0.1].count()
        results_df.loc['mean'] = mean
        results_df.loc['<10%'] = count10
        print_df_to_excel(df=results_df, ws=ws)

        for k, v in spa_store.items():
            wb.create_sheet(f'SPA_{k}')
            ws = wb[f'SPA_{k}']
            results_df = pd.DataFrame(v).T
            plotting_mcs(results_df, save_dir=f'{results_dir}/SPA_{k}.png')
            mean = results_df.mean()
            count10 = results_df[results_df < 0.1].count()
            results_df.loc['mean'] = mean
            results_df.loc['<10%'] = count10
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


def poos_shap(fl_master, fl, xgb_store, first_est_date, results_dir):
    excel_name = create_excel_file(f'{results_dir}/poos_shap.xlsx')
    wb = openpyxl.load_workbook(excel_name)

    idx = fl_master.time_stamp.index(first_est_date)
    ts = fl_master.time_stamp[idx:]

    for xgb_dir, h in zip(xgb_store, [1, 3, 6, 12, 24]):
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
        gshap_abs.index = ts

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

        # gshap_abs_norm = gshap_abs.div(gshap_abs.max(axis=1), axis=0)
        # gshap_abs.plot(y=np.unique(top_gshap_names.values))

        def print_df(sheet_name, df, h):
            wb.create_sheet(f'h{h}_{sheet_name}')
            ws = wb[f'h{h}_{sheet_name}']
            print_df_to_excel(ws=ws, df=df)

        print_df('shap_names', df=top_shap_names.applymap(str), h=h)
        print_df('shap_values', df=top_shap_values, h=h)
        print_df('gshap_names', df=top_gshap_names.applymap(str), h=h)
        print_df('gshap_values', df=top_gshap_values, h=h)

        wb.save(excel_name)
