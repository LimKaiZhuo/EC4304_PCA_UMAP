from own_package.poos import poos_analysis, poos_processed_data_analysis, poos_experiment, poos_model_evaluation, \
    poos_shap, poos_analysis_combining_xgb, poos_xgb_plotting_m
from own_package.others import create_results_directory
from own_package.features_labels import read_excel_data, read_excel_dataloader, Fl_master, Fl_pca, Fl_ar, \
    Fl_cw, Fl_xgb, hparam_selection


def selector(case, **kwargs):
    if case == 1:
        # Run poos experiment
        var_name = kwargs['var_name']
        excel_dir = kwargs['excel_dir']
        results_dir = create_results_directory('./results/expt1/{}'.format(var_name))
        output = read_excel_dataloader(excel_dir=excel_dir)
        fl_master = Fl_master(x=output[0], features_names=output[1],
                              yo=output[2], labels_names=output[3],
                              y=output[4], y_names=output[5],
                              time_stamp=output[6])
        fl_xgb = Fl_xgb(val_split=None, x=None, yo=None, y=None,
                        time_stamp=None, time_idx=None,
                        features_names=fl_master.features_names, labels_names=fl_master.labels_names,
                        y_names=fl_master.y_names)
        first_est_date = '2005:1'
        est_dates = ['2004:12']
        model_mode = 'rf'
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
            hparam_opt_params = {'hparam_mode': 'bo', 'n_calls': 200, 'n_random_starts': 150,
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
                                 'val_mode': 'rep_holdout',
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
        poos_experiment(fl_master=fl_master, fl=fl_xgb, est_dates=est_dates, z_type=1, h=1, h_idx=0,
                        m_max=12, p_max=24, model_mode=model_mode, save_dir=results_dir, first_est_date=first_est_date,
                        default_hparams=default_hparams, hparam_opt_params=hparam_opt_params,
                        hparam_save_dir=hparam_save_dir
                        )
        poos_experiment(fl_master=fl_master, fl=fl_xgb, est_dates=est_dates, z_type=1, h=3, h_idx=1,
                        m_max=12, p_max=24, model_mode=model_mode, save_dir=results_dir, first_est_date=first_est_date,
                        default_hparams=default_hparams, hparam_opt_params=hparam_opt_params,
                        hparam_save_dir=hparam_save_dir
                        )
        # poos_experiment(fl_master=fl_master, fl=fl_xgb, est_dates=est_dates, z_type=1, h=6, h_idx=2,
        #                m_max=12, p_max=24, model_mode=model_mode, save_dir=results_dir,first_est_date=first_est_date,
        #                default_hparams=default_hparams, hparam_opt_params=hparam_opt_params,
        #                hparam_save_dir=hparam_save_dir
        #                )
        # poos_experiment(fl_master=fl_master, fl=fl_xgb, est_dates=est_dates, z_type=1, h=12, h_idx=3,
        #                m_max=12, p_max=24, model_mode=model_mode, save_dir=results_dir,first_est_date=first_est_date,
        #                default_hparams=default_hparams, hparam_opt_params=hparam_opt_params,
        #                hparam_save_dir=hparam_save_dir
        #                )
        # poos_experiment(fl_master=fl_master, fl=fl_xgb, est_dates=est_dates, z_type=1, h=24, h_idx=4,
        #                m_max=12, p_max=24, model_mode=model_mode, save_dir=results_dir,first_est_date=first_est_date,
        #                default_hparams=default_hparams, hparam_opt_params=hparam_opt_params,
        #                hparam_save_dir=hparam_save_dir
        #                )
    elif case == 2:
        excel_dir = './excel/dataset_0720/INDPRO_data_loader.xlsx'
        output = read_excel_dataloader(excel_dir=excel_dir)
        fl_master = Fl_master(x=output[0], features_names=output[1],
                              yo=output[2], labels_names=output[3],
                              y=output[4], y_names=output[5],
                              time_stamp=output[6])
        first_est_date = '2005:1'
        h_store = [1, 3, 6, 12, 24]
        h_idx_store = [0, 1, 2, 3, 4]
        for h, h_idx in zip(h_store, h_idx_store):
            poos_analysis(fl_master=fl_master, h=h, h_idx=h_idx, model_mode='xgb', est_mode='rfcv',
                          results_dir='./results/expt1/poos_IND_xgba_rh_s42', first_est_date=first_est_date,
                          save_dir=f'./results/expt1//poos_IND_xgba_rh_s42/poos_h{h}.pkl')
    elif case == 3:
        model_name = 'xgba_rh_s42'
        var_name = 'IND'
        poos_type = 'expt1'
        first_est_date = '2005:1'
        est_dates = ['2004:12']
        poos_processed_data_analysis(
            save_dir_store=[f'./results/{poos_type}/poos_{var_name}_{model_name}/poos_xgb_h1_analysis_results.pkl',
                            f'./results/{poos_type}/poos_{var_name}_{model_name}/poos_xgb_h3_analysis_results.pkl',
                            f'./results/{poos_type}/poos_{var_name}_{model_name}/poos_xgb_h6_analysis_results.pkl',
                            f'./results/{poos_type}/poos_{var_name}_{model_name}/poos_xgb_h12_analysis_results.pkl',
                            f'./results/{poos_type}/poos_{var_name}_{model_name}/poos_xgb_h24_analysis_results.pkl',
                            ],
            h_store=['1',
                     '3',
                     '6',
                     '12',
                     '24',
                     ],
            results_dir=f'./results/{poos_type}/poos_{var_name}_{model_name}', model_mode=model_name,
            nber_excel_dir='./excel/NBER_062020.xlsx',
            est_dates=est_dates, first_est_date=first_est_date)
    elif case == 4:
        # Run poos experiment for ar4 or pca
        var_name = kwargs['var_name']
        excel_dir = kwargs['excel_dir']
        results_dir = create_results_directory('./results/expt1/{}'.format(var_name))
        output = read_excel_dataloader(excel_dir=excel_dir)
        fl_master = Fl_master(x=output[0], features_names=output[1],
                              yo=output[2], labels_names=output[3],
                              y=output[4], y_names=output[5],
                              time_stamp=output[6])
        fl = Fl_ar(val_split=None, x=None, yo=None, y=None,
                   time_stamp=None, time_idx=None,
                   features_names=fl_master.features_names, labels_names=fl_master.labels_names,
                   y_names=fl_master.y_names)

        first_est_date = '2005:1'
        est_dates = ['2004:12']

        model_mode = 'ar'
        poos_experiment(fl_master=fl_master, fl=fl, est_dates=est_dates, z_type=1, h=1, h_idx=0,
                        m_max=3, p_max=12, model_mode=model_mode, save_dir=results_dir, first_est_date=first_est_date,
                        )
        poos_experiment(fl_master=fl_master, fl=fl, est_dates=est_dates, z_type=1, h=3, h_idx=1,
                        m_max=3, p_max=12, model_mode=model_mode, save_dir=results_dir, first_est_date=first_est_date,
                        )
        poos_experiment(fl_master=fl_master, fl=fl, est_dates=est_dates, z_type=1, h=6, h_idx=2,
                        m_max=3, p_max=12, model_mode=model_mode, save_dir=results_dir, first_est_date=first_est_date,
                        )
        poos_experiment(fl_master=fl_master, fl=fl, est_dates=est_dates, z_type=1, h=12, h_idx=3,
                        m_max=3, p_max=12, model_mode=model_mode, save_dir=results_dir, first_est_date=first_est_date,
                        )
        poos_experiment(fl_master=fl_master, fl=fl, est_dates=est_dates, z_type=1, h=24, h_idx=4,
                        m_max=3, p_max=12, model_mode=model_mode, save_dir=results_dir, first_est_date=first_est_date,
                        )

        poos_processed_data_analysis(
            save_dir_store=[f'{results_dir}/poos_{model_mode}_h1_analysis_results.pkl',
                            f'{results_dir}/poos_{model_mode}_h3_analysis_results.pkl',
                            f'{results_dir}/poos_{model_mode}_h6_analysis_results.pkl',
                            f'{results_dir}/poos_{model_mode}_h12_analysis_results.pkl',
                            f'{results_dir}/poos_{model_mode}_h24_analysis_results.pkl',
                            ],
            h_store=['1',
                     '3',
                     '6',
                     '12',
                     '24',
                     ],
            results_dir=results_dir, model_mode=model_mode, nber_excel_dir='./excel/NBER_062020.xlsx',
            est_dates=est_dates, first_est_date=first_est_date)

        model_mode = 'pca'
        results_dir = create_results_directory('./results/expt1/{}'.format(var_name))
        fl = Fl_pca(val_split=None, x=None, yo=None, y=None,
                    time_stamp=None, time_idx=None,
                    features_names=fl_master.features_names, labels_names=fl_master.labels_names,
                    y_names=fl_master.y_names)
        poos_experiment(fl_master=fl_master, fl=fl, est_dates=est_dates, z_type=1, h=1, h_idx=0,
                        m_max=3, p_max=12, model_mode=model_mode, save_dir=results_dir, first_est_date=first_est_date,
                        )
        poos_experiment(fl_master=fl_master, fl=fl, est_dates=est_dates, z_type=1, h=3, h_idx=1,
                        m_max=3, p_max=12, model_mode=model_mode, save_dir=results_dir, first_est_date=first_est_date,
                        )
        poos_experiment(fl_master=fl_master, fl=fl, est_dates=est_dates, z_type=1, h=6, h_idx=2,
                        m_max=3, p_max=12, model_mode=model_mode, save_dir=results_dir, first_est_date=first_est_date,
                        )
        poos_experiment(fl_master=fl_master, fl=fl, est_dates=est_dates, z_type=1, h=12, h_idx=3,
                        m_max=3, p_max=12, model_mode=model_mode, save_dir=results_dir, first_est_date=first_est_date,
                        )
        poos_experiment(fl_master=fl_master, fl=fl, est_dates=est_dates, z_type=1, h=24, h_idx=4,
                        m_max=3, p_max=12, model_mode=model_mode, save_dir=results_dir, first_est_date=first_est_date,
                        )

        poos_processed_data_analysis(
            save_dir_store=[f'{results_dir}/poos_{model_mode}_h1_analysis_results.pkl',
                            f'{results_dir}/poos_{model_mode}_h3_analysis_results.pkl',
                            f'{results_dir}/poos_{model_mode}_h6_analysis_results.pkl',
                            f'{results_dir}/poos_{model_mode}_h12_analysis_results.pkl',
                            f'{results_dir}/poos_{model_mode}_h24_analysis_results.pkl',
                            ],
            h_store=['1',
                     '3',
                     '6',
                     '12',
                     '24',
                     ],
            results_dir=results_dir, model_mode=model_mode, nber_excel_dir='./excel/NBER_062020.xlsx',
            est_dates=est_dates, first_est_date=first_est_date)


if __name__ == '__main__':
    selector(case=4, excel_dir='./excel/dataset_0720/INDPRO_data_loader.xlsx', var_name='poos_INDPRO_ar')
