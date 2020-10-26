from own_package.poos import poos_experiment, poos_analysis
from own_package.others import create_results_directory
from own_package.features_labels import read_excel_data, read_excel_dataloader, Fl_master, Fl_pca, Fl_ar, \
    Fl_cw, Fl_xgb, hparam_selection




def selector(case, **kwargs):
    if case == 1:
        # Run poos experiment
        var_name = kwargs['var_name']
        excel_dir = kwargs['excel_dir']
        results_dir = create_results_directory('./results/poos/{}'.format(var_name))
        output = read_excel_dataloader(excel_dir=excel_dir)
        fl_master = Fl_master(x=output[0], features_names=output[1],
                              yo=output[2], labels_names=output[3],
                              y=output[4], y_names=output[5],
                              time_stamp=output[6])
        fl_xgb = Fl_xgb(val_split=None, x=None, yo=None, y=None,
                        time_stamp=None, time_idx=None,
                        features_names=fl_master.features_names, labels_names=fl_master.labels_names,
                        y_names=fl_master.y_names)

        first_est_date = '1970:1'

        model_mode = 'xgb'
        if model_mode == 'xgb' or model_mode == 'xgb_with_hparam':
            default_hparams = {'seed': 42,
                               'booster': 'gbtree',
                               'learning_rate': 0.1,
                               'objective': 'reg:squarederror',
                               'verbosity': 0,
                               'subsample': 1,
                               'num_boost_round': 600,
                               'early_stopping_rounds': 100,
                               'ehat_eval': None,
                               # 'eval_metric':['rmse'],
                               # DART params
                               'rate_drop': 0.2,
                               'skip_drop': 0.5,
                               # params that will vary
                               'm': 6,
                               'p': 12,
                               'max_depth': 1,
                               'colsample_bytree': 0.5,
                               }
            hparam_opt_params = {'hparam_mode': 'bo', 'n_calls': 150, 'n_random_starts': 100,
                                 'val_mode': 'rep_holdout',
                                 'n_blocks': 5, 'cut_point': 0.95,
                                 'variables': {'max_depth': {'type': 'Integer', 'lower': 1, 'upper': 6},
                                               'colsample_bytree': {'type': 'Real', 'lower': 0.5, 'upper': 1},
                                               'm': {'type': 'Integer', 'lower': 1, 'upper': 24},
                                               # 'p': {'type': 'Integer', 'lower': 1, 'upper': 48},
                                               'adap_gamma': {'type': 'Real', 'lower': -2, 'upper': 1.5}
                                               },
                                 }
        elif model_mode == 'rf':
            default_hparams = {'seed': 42,
                               'booster': 'gbtree',
                               'learning_rate': 1,
                               'objective': 'reg:squarederror',
                               'verbosity': 0,
                               'subsample': 1,
                               'num_boost_round': 1,
                               'early_stopping_rounds': None,
                               'ehat_eval': None,
                               # 'eval_metric':['rmse'],
                               # params that will vary
                               'm': 6,
                               'p': 12,
                               'max_depth': 1,
                               'colsample_bytree': 1,
                               }
            hparam_opt_params = {'hparam_mode': 'bo', 'n_calls': 200, 'n_random_starts': 150,
                                 'val_mode': 'rfcv',
                                 'n_blocks': 5, 'cut_point': 0.95,
                                 'variables': {'max_depth': {'type': 'Integer', 'lower': 1, 'upper': 6},
                                               'subsample': {'type': 'Real', 'lower': 0.5, 'upper': 1},
                                               'colsample_bytree': {'type': 'Real', 'lower': 0.5, 'upper': 1},
                                               'm': {'type': 'Integer', 'lower': 1, 'upper': 24},
                                               # 'p': {'type': 'Integer', 'lower': 1, 'upper': 48},
                                               'num_parallel_tree': {'type': 'Integer', 'lower': 1, 'upper': 1200}
                                               },
                                 }
        else:
            default_hparams=None
            hparam_opt_params = None

        hparam_save_dir = './results/poos/poos_IND_xgbar'
        est_dates = [f'{x}:12' for x in range(1969, 2020, 5)[:-1]]
        #poos_experiment(fl_master=fl_master, fl=fl_xgb, est_dates=est_dates, z_type=1, h=1, h_idx=0,
        #                m_max=12, p_max=24, model_mode=model_mode, save_dir=results_dir, first_est_date=first_est_date,
        #                default_hparams=default_hparams, hparam_opt_params=hparam_opt_params,
        #                hparam_save_dir=hparam_save_dir
        #                )
        poos_experiment(fl_master=fl_master, fl=fl_xgb, est_dates=est_dates, z_type=1, h=3, h_idx=1,
                        m_max=12, p_max=24, model_mode=model_mode, save_dir=results_dir,first_est_date=first_est_date,
                        default_hparams=default_hparams, hparam_opt_params=hparam_opt_params,
                        hparam_save_dir=hparam_save_dir
                        )
        poos_experiment(fl_master=fl_master, fl=fl_xgb, est_dates=est_dates, z_type=1, h=6, h_idx=2,
                        m_max=12, p_max=24, model_mode=model_mode, save_dir=results_dir,first_est_date=first_est_date,
                        default_hparams=default_hparams, hparam_opt_params=hparam_opt_params,
                        hparam_save_dir=hparam_save_dir
                        )
        poos_experiment(fl_master=fl_master, fl=fl_xgb, est_dates=est_dates, z_type=1, h=12, h_idx=3,
                        m_max=12, p_max=24, model_mode=model_mode, save_dir=results_dir,first_est_date=first_est_date,
                        default_hparams=default_hparams, hparam_opt_params=hparam_opt_params,
                        hparam_save_dir=hparam_save_dir
                        )
        poos_experiment(fl_master=fl_master, fl=fl_xgb, est_dates=est_dates, z_type=1, h=24, h_idx=4,
                        m_max=12, p_max=24, model_mode=model_mode, save_dir=results_dir,first_est_date=first_est_date,
                        default_hparams=default_hparams, hparam_opt_params=hparam_opt_params,
                        hparam_save_dir=hparam_save_dir
                        )

    elif case == 2:
        excel_dir = kwargs['excel_dir']
        output = read_excel_dataloader(excel_dir=excel_dir)
        fl_master = Fl_master(x=output[0], features_names=output[1],
                              yo=output[2], labels_names=output[3],
                              y=output[4], y_names=output[5],
                              time_stamp=output[6])
        poos_analysis(fl_master=fl_master,model_mode='xgb',save_dir='./results/poos/poos_IND_15/poos_h1.pkl')


if __name__ == '__main__':
    selector(case=1, excel_dir='./excel/dataset_0720/INDPRO_data_loader.xlsx', var_name='poos_IND_xgba_rh_s42')
