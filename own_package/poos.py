import numpy as np
import pandas as pd
from own_package.boosting import Xgboost
import pickle, time

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
            hparams = {**kwargs['default_hparams'], **hparams_df.iloc[0, :].to_dict()}
            hparams['early_stopping_rounds'] = None
            hparams['m']=int(hparams['m'])
            hparams['max_depth'] = int(hparams['max_depth'])
            _, _, _, poos_data_store = fl.pls_expanding_window(h=h, p=hparams['m']*2, m=hparams['m'], r=8,
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
        print(f'Completed {est_date} for h={h}. Time taken {t2-t1}')


def poos_analysis(save_dir):
    with open(save_dir, 'rb') as handle:
        data_store=pickle.load(handle)

    print(f'Loaded {save_dir}')

