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

        est_dates = [f'{x}:12' for x in range(1969, 2020, 5)[:-1]]

        default_hparams = {'seed': 42,
                           'booster': 'gbtree',
                           'learning_rate': 0.1,
                           'objective': 'reg:squarederror',
                           'verbosity': 0,
                           'subsample': 1,
                           'num_boost_round': 600,
                           'early_stopping_rounds': 100,
                           # DART params
                           'rate_drop': 0.2,
                           'skip_drop': 0.5,
                           # params that will vary
                           'm': 7,
                           'p': 14,
                           'max_depth': 1,
                           'colsample_bytree': 0.5,
                           }

        hparam_opt_params = {'hparam_mode': 'bo', 'n_calls': 70, 'n_random_starts': 50,
                             'val_mode': 'rep_holdout',
                             'n_blocks': 2, 'cut_point': 0.97,
                             'variables': {'max_depth': {'type': 'Integer', 'lower': 1, 'upper': 6},
                                           'colsample_bytree': {'type': 'Real', 'lower': 0.5, 'upper': 1},
                                           'm': {'type': 'Integer', 'lower': 1, 'upper': 24},
                                           # 'p': {'type': 'Integer', 'lower': 1, 'upper': 48},
                                           # 'gamma': {'type': 'Real', 'lower': 0.01, 'upper': 30}
                                           },
                             }
        poos_experiment(fl_master=fl_master, fl=fl_xgb, est_dates=est_dates, z_type=1, h=1, h_idx=0,
                        m_max=12, p_max=24, model_mode='xgb', save_dir=results_dir,
                        default_hparams=default_hparams, hparam_opt_params=hparam_opt_params
                        )
        poos_experiment(fl_master=fl_master, fl=fl_xgb, est_dates=est_dates, z_type=1, h=3, h_idx=1,
                        m_max=12, p_max=24, model_mode='xgb', save_dir=results_dir,
                        default_hparams=default_hparams, hparam_opt_params=hparam_opt_params
                        )
        poos_experiment(fl_master=fl_master, fl=fl_xgb, est_dates=est_dates, z_type=1, h=6, h_idx=2,
                        m_max=12, p_max=24, model_mode='xgb', save_dir=results_dir,
                        default_hparams=default_hparams, hparam_opt_params=hparam_opt_params
                        )
    elif case == 2:
        poos_analysis('./results/poos/poos_IND_test_26/poos_h1.pkl')


if __name__ == '__main__':
    selector(case=2, excel_dir='./excel/dataset2/INDPRO_data_loader.xlsx', var_name='poos_IND_test')
