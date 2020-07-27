import numpy as np
import pandas as pd
import openpyxl
import matplotlib.pyplot as plt
from own_package.boosting import Xgboost
from own_package.analysis import LocalLevel
from own_package.others import print_df_to_excel
import pickle, time


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

            #wb = openpyxl.Workbook()
            #ws = wb[wb.sheetnames[-1]]
            #print_df_to_excel(df=hparams_df, ws=ws)
            #wb.save('./results/poos/hparams.xlsx')
            #print('done')

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
        t2 = time.perf_counter()
        print(f'Completed {est_date} for h={h}. Time taken {t2 - t1}')


def poos_analysis(fl_master, h, h_idx, model_mode, results_dir, save_dir):
    first_est_date = '1970:1'

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

    if model_mode == 'xgb':
        ae_store = []
        ehat_store = []
        optimal_ntree_store = []
        predicted_ntree_store = []
        hparam_ntree_store = []
        block_store = []
        for idx, data in enumerate(data_store):
            block_ae = [single_step['progress']['h_step_ahead']['rmse'] for single_step in data['poos_data_store']]
            block_store.extend([idx]*len(block_ae))
            block_ehat = [single_step['progress']['h_step_ahead']['ehat'] for single_step in data['poos_data_store']]
            ae_store.extend(block_ae)
            ehat_store.extend(block_ehat)
            optimal_ntree, \
            predicted_ntree = calculate_sequence_ntree(block_ae=block_ae,
                                                       hparam_ntree=int(
                                                           round(data['hparams_df'].
                                                                 iloc[0, data['hparams_df'].columns.tolist().index(
                                                               'm iters')] / 0.85
                                                                 )))
            optimal_ntree_store.extend(optimal_ntree)
            predicted_ntree_store.extend(predicted_ntree)
            hparam_ntree_store.extend([int(round(data['hparams_df'].iloc[
                                                     0, data['hparams_df'].columns.tolist().index('m iters')] / 0.85
                                                 ))] * len(optimal_ntree))

        # Full SSM prediction
        _, ssm_full_ntree_store = calculate_sequence_ntree(block_ae=ae_store,
                                                        hparam_ntree=int(round(data_store[0]['hparams_df'].iloc[
                                                                                   0, data_store[0][
                                                                                       'hparams_df'].columns.tolist().
                                                                               index('m iters')] / 0.85)))

        # Horizons RMSE calculation
        oracle_rmse, oracle_ehat = calculate_horizon_rmse_ae(ae_store, ehat_store, optimal_ntree_store)
        max_iter_rmse, max_iter_ehat = calculate_horizon_rmse_ae(ae_store, ehat_store, [-1] * len(ae_store))
        hparam_rmse, hparam_ehat = calculate_horizon_rmse_ae(ae_store, ehat_store, hparam_ntree_store)
        ssm_rmse, ssm_ehat = calculate_horizon_rmse_ae(ae_store, ehat_store, predicted_ntree_store)
        ssm_full_rmse, ssm_full_ehat = calculate_horizon_rmse_ae(ae_store, ehat_store, ssm_full_ntree_store)


        idx = fl_master.time_stamp.index(first_est_date)
        y = fl_master.y[idx:, h_idx]
        ts = fl_master.time_stamp[idx:]

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
                                     'ssm_full_ntree':ssm_full_ntree_store,
                                     'hparam_ntree':hparam_ntree_store}).set_index('time_stamp')

        # df.plot(y=['oracle_ehat', 'ssm_ehat', 'ssm_full_ehat', 'hparam_ehat', 'max_iter_ehat'], use_index=True)

        rmse_df = pd.DataFrame.from_dict({'oracle_rmse': [oracle_rmse],
                                          'ssm_rmse': [ssm_rmse],
                                          'ssm_full_rmse': [ssm_full_rmse],
                                          'hparam_rmse': [hparam_rmse],
                                          'max_iter_rmse': [max_iter_rmse]})


        hparam_df = [data['hparams_df'].iloc[0,:] for data in data_store]
        hparam_df = pd.concat(optimal_hparam_df, axis=1).T

        with open(f'{results_dir}/poos_{model_mode}_h{h}_analysis_results.pkl', "wb") as file:
            pickle.dump({'data_df': df, 'rmse_df': rmse_df, 'hparam_df':hparam_df}, file)


def poos_processed_data_analysis(save_dir):
    first_est_date = '1970:1'
    est_dates = [f'{x}:12' for x in range(1969, 2020, 5)[1:-1]]

    with open(save_dir, 'rb') as handle:
        data_store = pickle.load(handle)

    data_df = data_store['data_df']
    ehat_df = data_df[[x for x in data_df.columns.values if '_ehat' in x]]
    ehat_df.columns = [name.replace('ehat', 'rmse') for name in ehat_df.columns.values]
    time_stamps = data_df.index.tolist()
    est_dates_idx = [0]+[time_stamps.index(est)+1 for est in est_dates] + [-1]

    rmse_store = []
    for start, end in zip(est_dates_idx[:-1], est_dates_idx[1:]):
        rmse_store.append((ehat_df.iloc[start:end,:]**2).mean(axis=0)**0.5)

    start_store = ['1970:1', '1983:1', '2007:1', '2019:12']  # Inclusive
    end_store = ['1983:1', '2007:1', '2020:6', '2020:6']  # Exclusive of those dates

    for start, end in zip(start_store, end_store):
        start = time_stamps.index(start)
        try:
            end = time_stamps.index(end)
        except ValueError:
            end = -1  # Means is longer than last date ==> take last entry
        rmse_store.append((ehat_df.iloc[start:end,:]**2).mean(axis=0)**0.5)

    df = pd.concat((pd.concat(rmse_store, axis=1).T, data_store['hparam_df'].reset_index()), axis=1)
    df['start_dates'] = est_dates_idx[:-1]

    print('hi')

