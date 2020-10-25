from own_package.poos import poos_analysis, poos_processed_data_analysis, poos_experiment, poos_model_evaluation, \
    poos_shap, poos_analysis_combining_xgb, poos_xgb_plotting_m, combine_poos_excel_results
from own_package.others import create_results_directory, create_id_dict, create_id_store
from own_package.features_labels import read_excel_data, read_excel_dataloader, Fl_master, Fl_pca, Fl_ar, \
    Fl_cw, Fl_xgb, hparam_selection
from own_package.postprocess import difference_to_levels, plot_forecasts, combine_poos_analysis


def selector(case, **kwargs):
    if case == 1:
        pass
    elif case == 2:
        '''
        For xgb models. After finishing POOS experiments, run this to convert raw data in poos_hXX.pkl into forecasted
        ehats. Performs online learning of optimal ntrees. Save results in another pickle object which is a dict of 
        results.
        '''
        excel_dir = './excel/dataset_0720/CPIA1_data_loader.xlsx'
        output = read_excel_dataloader(excel_dir=excel_dir)
        fl_master = Fl_master(x=output[0], features_names=output[1],
                              yo=output[2], labels_names=output[3],
                              y=output[4], y_names=output[5],
                              time_stamp=output[6])
        first_est_date = '2005:1'
        id = create_id_dict(var_name='CPIA1',
                            h=[1, 3, 6, 12, 24],
                            est='rh',
                            model='xgb',
                            model_name='xgbag0',
                            expt='expt1',
                            combined_name=None,
                            seed=42, )
        h_store = [1, 3, 6, 12, 24]
        h_idx_store = [0, 1, 2, 3, 4]
        for h, h_idx in zip(h_store, h_idx_store):
            poos_analysis(fl_master=fl_master, h=h, h_idx=h_idx, model_mode=id['model'], est_mode=id['est'],
                          results_dir=id['results_dir'],
                          first_est_date=first_est_date,
                          save_dir=f'{id["results_dir"]}/poos_h{h}.pkl')
    elif case == 3:
        '''
        Read pickle dict of results into excel.
        '''
        try:
            id = kwargs['id']
            levels = kwargs['levels']
            combi = None
        except KeyError:
            id = create_id_dict(var_name='IND',
                                h=[1, 3, 6, 12, 24],
                                est='rh',
                                model='xgb',
                                model_name='xgba',
                                expt='expt1',
                                combined_name=None,
                                seed=42,
                                results_dir='./results/expt1/model_combination_CPIA1')
            levels = False
            combi = [['rw', 'll*ln'],
                     ['rw', 'llt*ln'],
                     ['rw', 'll*ln', 'llt*ln'], ]
        if levels:
            levels = '_levels'
        else:
            levels = ''
        first_est_date = '2005:1'
        est_dates = ['2004:12']

        poos_processed_data_analysis(
            save_dir_store=[f'{id["results_dir"]}/poos_{id["model"]}_h1_analysis_results{levels}.pkl',
                            f'{id["results_dir"]}/poos_{id["model"]}_h3_analysis_results{levels}.pkl',
                            f'{id["results_dir"]}/poos_{id["model"]}_h6_analysis_results{levels}.pkl',
                            f'{id["results_dir"]}/poos_{id["model"]}_h12_analysis_results{levels}.pkl',
                            f'{id["results_dir"]}/poos_{id["model"]}_h24_analysis_results{levels}.pkl',
                            ],
            h_store=['1',
                     '3',
                     '6',
                     '12',
                     '24',
                     ],
            results_dir=id['results_dir'],
            model_full_name=id['model_full_name'] + levels, model=id['model'],
            nber_excel_dir='./excel/NBER_062020.xlsx',
            est_dates=est_dates, first_est_date=first_est_date,
            combinations=combi)
    elif case == 3.2:
        # Plot information about m iteration errors for xgb. Uses the post processed of poos_h{}.pkl.
        h_store = [1, 3, 6, 12, 24]
        h_idx_store = [0, 1, 2, 3, 4]
        results_dir = './results/expt1/poos_CPIA1_xgba_rh_s42'
        for h, h_idx in zip(h_store, h_idx_store):
            poos_xgb_plotting_m(h=h, results_dir=results_dir, ssm_modes=['ll*ln'])
    elif case == 3.3:
        '''
        Convert pickled dict of results from I(1) or I(2) based on rawdata_excel and the varname into levels.
        Then print the levels results into excel.
        '''
        rawdata_excel = './excel/2020-07.xlsx'  # _I1 for CPIA1 only! Remove that for IND and CPIA
        id = create_id_dict(var_name='CPIA',
                            h=[1, 3, 6, 12, 24],
                            est=None,
                            model='pca',
                            model_name='pca',
                            expt='expt1',
                            seed=42)

        first_est_date = '2005:1'
        est_dates = ['2004:12']
        difference_to_levels(varname='CPIAUCSL',
                             save_dir_store=[f'{id["results_dir"]}/poos_{id["model"]}_h1_analysis_results.pkl',
                                             f'{id["results_dir"]}/poos_{id["model"]}_h3_analysis_results.pkl',
                                             f'{id["results_dir"]}/poos_{id["model"]}_h6_analysis_results.pkl',
                                             f'{id["results_dir"]}/poos_{id["model"]}_h12_analysis_results.pkl',
                                             f'{id["results_dir"]}/poos_{id["model"]}_h24_analysis_results.pkl',
                                             ],
                             h_store=['1', '3', '6', '12', '24', ],
                             rawdata_excel=rawdata_excel, first_est_date=first_est_date)

        levels = '_i1'
        poos_processed_data_analysis(
            save_dir_store=[f'{id["results_dir"]}/poos_{id["model"]}_h1_analysis_results{levels}.pkl',
                            f'{id["results_dir"]}/poos_{id["model"]}_h3_analysis_results{levels}.pkl',
                            f'{id["results_dir"]}/poos_{id["model"]}_h6_analysis_results{levels}.pkl',
                            f'{id["results_dir"]}/poos_{id["model"]}_h12_analysis_results{levels}.pkl',
                            f'{id["results_dir"]}/poos_{id["model"]}_h24_analysis_results{levels}.pkl',
                            ],
            h_store=['1',
                     '3',
                     '6',
                     '12',
                     '24',
                     ],
            results_dir=id['results_dir'],
            model_full_name=id['model_full_name'] + levels, model=id['model'],
            nber_excel_dir='./excel/NBER_062020.xlsx',
            est_dates=est_dates, first_est_date=first_est_date,
            combinations=[['rw', 'll*ln'],
                          ['rw', 'llt*ln'],
                          ['rw', 'll*ln', 'llt*ln'], ])
    elif case == 4:
        # Settings
        var_name = 'IND'  # CPIA or CPIA1 or IND
        full_var_name = 'INDPRO'  # CPIAUCSL or INDPRO
        excel_dir = './excel/dataset_0720/INDPRO_data_loader.xlsx'
        rawdata_excel = './excel/2020-07.xlsx'  # _I1 for CPIA1 only! Remove that for IND and CPIA

        # Run poos experiment for rw or ar or pca
        def run_basic(model_mode, fl_master, results_dir):
            if model_mode == 'ar' or model_mode == 'rw':
                fl = Fl_ar(val_split=None, x=None, yo=None, y=None,
                           time_stamp=None, time_idx=None,
                           features_names=fl_master.features_names, labels_names=fl_master.labels_names,
                           y_names=fl_master.y_names)
            elif model_mode == 'pca':
                fl = Fl_pca(val_split=None, x=None, yo=None, y=None,
                            time_stamp=None, time_idx=None,
                            features_names=fl_master.features_names, labels_names=fl_master.labels_names,
                            y_names=fl_master.y_names)
            else:
                raise KeyError('Model mode must be rw, ar or pca')

            first_est_date = '2005:1'
            est_dates = ['2004:12']

            poos_experiment(fl_master=fl_master, fl=fl, est_dates=est_dates, z_type=1, h=1, h_idx=0,
                            m_max=3, p_max=12, model_mode=model_mode, save_dir=results_dir,
                            first_est_date=first_est_date,
                            )
            poos_experiment(fl_master=fl_master, fl=fl, est_dates=est_dates, z_type=1, h=3, h_idx=1,
                            m_max=3, p_max=12, model_mode=model_mode, save_dir=results_dir,
                            first_est_date=first_est_date,
                            )
            poos_experiment(fl_master=fl_master, fl=fl, est_dates=est_dates, z_type=1, h=6, h_idx=2,
                            m_max=3, p_max=12, model_mode=model_mode, save_dir=results_dir,
                            first_est_date=first_est_date,
                            )
            poos_experiment(fl_master=fl_master, fl=fl, est_dates=est_dates, z_type=1, h=12, h_idx=3,
                            m_max=3, p_max=12, model_mode=model_mode, save_dir=results_dir,
                            first_est_date=first_est_date,
                            )
            poos_experiment(fl_master=fl_master, fl=fl, est_dates=est_dates, z_type=1, h=24, h_idx=4,
                            m_max=3, p_max=12, model_mode=model_mode, save_dir=results_dir,
                            first_est_date=first_est_date,
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
                results_dir=results_dir, model_full_name=model_mode, model=model_mode,
                nber_excel_dir='./excel/NBER_062020.xlsx', est_dates=est_dates, first_est_date=first_est_date)

            difference_to_levels(varname=full_var_name,
                                 save_dir_store=[f'{results_dir}/poos_{model_mode}_h1_analysis_results.pkl',
                                                 f'{results_dir}/poos_{model_mode}_h3_analysis_results.pkl',
                                                 f'{results_dir}/poos_{model_mode}_h6_analysis_results.pkl',
                                                 f'{results_dir}/poos_{model_mode}_h12_analysis_results.pkl',
                                                 f'{results_dir}/poos_{model_mode}_h24_analysis_results.pkl',
                                                 ],
                                 h_store=['1', '3', '6', '12', '24', ],
                                 rawdata_excel=rawdata_excel, first_est_date=first_est_date)

            levels = '_levels'
            poos_processed_data_analysis(
                save_dir_store=[f'{results_dir}/poos_{model_mode}_h1_analysis_results{levels}.pkl',
                                f'{results_dir}/poos_{model_mode}_h3_analysis_results{levels}.pkl',
                                f'{results_dir}/poos_{model_mode}_h6_analysis_results{levels}.pkl',
                                f'{results_dir}/poos_{model_mode}_h12_analysis_results{levels}.pkl',
                                f'{results_dir}/poos_{model_mode}_h24_analysis_results{levels}.pkl',
                                ],
                h_store=['1',
                         '3',
                         '6',
                         '12',
                         '24',
                         ],
                results_dir=results_dir, model_full_name=model_mode + levels, model=model_mode,
                nber_excel_dir='./excel/NBER_062020.xlsx', est_dates=est_dates, first_est_date=first_est_date)

            try:
                levels = '_i1'
                poos_processed_data_analysis(
                    save_dir_store=[f'{results_dir}/poos_{model_mode}_h1_analysis_results{levels}.pkl',
                                    f'{results_dir}/poos_{model_mode}_h3_analysis_results{levels}.pkl',
                                    f'{results_dir}/poos_{model_mode}_h6_analysis_results{levels}.pkl',
                                    f'{results_dir}/poos_{model_mode}_h12_analysis_results{levels}.pkl',
                                    f'{results_dir}/poos_{model_mode}_h24_analysis_results{levels}.pkl',
                                    ],
                    h_store=['1',
                             '3',
                             '6',
                             '12',
                             '24',
                             ],
                    results_dir=results_dir, model_full_name=model_mode + levels, model=model_mode,
                    nber_excel_dir='./excel/NBER_062020.xlsx', est_dates=est_dates, first_est_date=first_est_date)
            except:
                pass

        # Load data from excel to fl_master
        output = read_excel_dataloader(excel_dir=excel_dir)
        fl_master = Fl_master(x=output[0], features_names=output[1],
                              yo=output[2], labels_names=output[3],
                              y=output[4], y_names=output[5],
                              time_stamp=output[6])
        # Model Runs
        model_mode = 'rw'
        results_dir = create_results_directory(f'./results/expt1/poos_{var_name}_{model_mode}')
        run_basic(model_mode=model_mode, results_dir=results_dir, fl_master=fl_master)
        model_mode = 'ar'
        results_dir = create_results_directory(f'./results/expt1/poos_{var_name}_{model_mode}')
        run_basic(model_mode=model_mode, results_dir=results_dir, fl_master=fl_master)
        model_mode = 'pca'
        results_dir = create_results_directory(f'./results/expt1/poos_{var_name}_{model_mode}')
        run_basic(model_mode=model_mode, results_dir=results_dir, fl_master=fl_master)
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
            results_dir=results_dir, model_full_name=model_mode, model=model_mode,
            nber_excel_dir='./excel/NBER_062020.xlsx',
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
            results_dir=results_dir, model_full_name=model_mode, model=model_mode,
            nber_excel_dir='./excel/NBER_062020.xlsx',
            est_dates=est_dates, first_est_date=first_est_date)
    elif case == 5:
        # Forecast evaluation DM
        h_store = [1, 3, 6, 12, 24]

        output = read_excel_dataloader(excel_dir='./excel/dataset_0720/INDPRO_data_loader.xlsx')
        fl_master = Fl_master(x=output[0], features_names=output[1],
                              yo=output[2], labels_names=output[3],
                              y=output[4], y_names=output[5],
                              time_stamp=output[6])

        first_est_date = '2005:1'
        est_dates = ['2004:12', ]
        var_store = {'CPIA': ['2008:11'],
                     'PAY': ['1975:3'],
                     'CMR': ['1975:1', '1980:6', '2009:2'],
                     'DPC': ['2020:3'],
                     'IND': ['1975:1', '1980:6', '2009:2'],
                     }
        var_store = {'CPIA1': []}
        var_name1 = 'CPIA'

        for var_name, dates in var_store.items():
            results_dir = create_results_directory(f'./results/expt1/model_eval_{var_name}')
            trad_store = [x for x in zip(
                [f'./results/expt1/poos_{var_name}_rw/poos_rw_h{h}_analysis_results.pkl' for h in h_store],
                [f'./results/expt1/poos_{var_name}_ar/poos_ar_h{h}_analysis_results.pkl' for h in h_store],
                [f'./results/expt1/poos_{var_name}_pca/poos_pca_h{h}_analysis_results.pkl' for h in h_store],
                [f'./results/expt1/poos_{var_name1}_rw/poos_rw_h{h}_analysis_results_i1.pkl' for h in h_store],
                [f'./results/expt1/poos_{var_name1}_ar/poos_ar_h{h}_analysis_results_i1.pkl' for h in h_store],
                [f'./results/expt1/poos_{var_name1}_pca/poos_pca_h{h}_analysis_results_i1.pkl' for h in h_store],
            )]
            xgb_store = [x for x in zip(
                [f'./results/expt1/poos_{var_name}_xgba_rh_s42/poos_xgb_h{h}_analysis_results.pkl' for h in h_store],
                [f'./results/expt1/poos_{var_name}_xgba_rfcv_s42/poos_xgb_h{h}_analysis_results.pkl' for h in h_store],
                [f'./results/expt1/poos_{var_name}_rf_rh_s42/poos_rf_h{h}_analysis_results.pkl' for h in h_store],
                [f'./results/expt1/poos_{var_name}_rf_rfcv_s42/poos_rf_h{h}_analysis_results.pkl' for h in h_store],
                [f'./results/expt1/poos_{var_name1}_xgba_rh_s42/poos_xgb_h{h}_analysis_results_i1.pkl' for h in
                 h_store],
                [f'./results/expt1/poos_{var_name1}_xgba_rfcv_s42/poos_xgb_h{h}_analysis_results_i1.pkl' for h in
                 h_store],
                [f'./results/expt1/poos_{var_name1}_rf_rh_s42/poos_rf_h{h}_analysis_results_i1.pkl' for h in h_store],
                [f'./results/expt1/poos_{var_name1}_rf_rfcv_s42/poos_rf_h{h}_analysis_results_i1.pkl' for h in h_store],
            )]
            poos_model_evaluation(fl_master=fl_master,
                                  trad_stores=trad_store,
                                  xgb_stores=xgb_store,
                                  results_dir=results_dir,
                                  first_est_date=first_est_date,
                                  est_dates=est_dates,
                                  combinations=[['rw', 'll*ln'],
                                                ['rw', 'llt*ln'],
                                                ['rw', 'll*ln', 'llt*ln'],
                                                ]
                                  )
    elif case == 6:
        # Combine rmse excel together into 1 excel file
        var_name = 'IND'
        id1 = create_id_dict(var_name=var_name,
                             h=[1, 3, 6, 12, 24],
                             est='rh',
                             model='xgb',
                             model_name='xgba',
                             expt='expt1',
                             seed=42)
        id2 = create_id_dict(var_name=var_name,
                             h=[1, 3, 6, 12, 24],
                             est='rfcv',
                             model='xgb',
                             model_name='xgba',
                             expt='expt1',
                             seed=42)
        id3 = create_id_dict(var_name=var_name,
                             h=[1, 3, 6, 12, 24],
                             est='rh',
                             model='rf',
                             model_name='rf',
                             expt='expt1',
                             seed=42)
        id4 = create_id_dict(var_name=var_name,
                             h=[1, 3, 6, 12, 24],
                             est='rfcv',
                             model='rf',
                             model_name='rf',
                             expt='expt1',
                             seed=42)
        id5 = create_id_dict(var_name=var_name,
                             h=[1, 3, 6, 12, 24],
                             est='rh',
                             model='xgb',
                             model_name='xgba',
                             expt='expt1',
                             seed=100)
        id6 = create_id_dict(var_name=var_name,
                             h=[1, 3, 6, 12, 24],
                             est='rh',
                             model='xgb',
                             model_name='xgba',
                             expt='expt1',
                             seed=200)
        id7 = create_id_dict(var_name=var_name,
                             h=[1, 3, 6, 12, 24],
                             est='rh',
                             model='xgb',
                             model_name='xgba',
                             expt='expt1',
                             seed=300)
        id8 = create_id_dict(var_name=var_name,
                             h=[1, 3, 6, 12, 24],
                             est='rh',
                             model='xgb',
                             model_name='xgbahp',
                             expt='expt1',
                             seed=100)
        id9 = create_id_dict(var_name=var_name,
                             h=[1, 3, 6, 12, 24],
                             est='rh',
                             model='xgb',
                             model_name='xgbahp',
                             expt='expt1',
                             seed=200)
        id10 = create_id_dict(var_name=var_name,
                              h=[1, 3, 6, 12, 24],
                              est='rh',
                              model='xgb',
                              model_name='xgbahp',
                              expt='expt1',
                              seed=300)
        id11 = create_id_dict(var_name=var_name,
                              h=[1, 3, 6, 12, 24],
                              combined_name='c&xgba', est='rh', seed='c',
                              expt='expt1')
        id12 = create_id_dict(var_name=var_name,
                              h=[1, 3, 6, 12, 24],
                              combined_name='c&xgbahp', est='rh', seed='c',
                              expt='expt1')
        id_store = [id1, id2, id3, id4, ]  # id5, id6, id7, id8, id9, id10, id11, id12]  # ]
        levels = False

        if levels:
            levels = '_i1'
        else:
            levels = ''
        excel_store = [f'./results/expt1/poos_{var_name}_rw/poos_analysis_rw{levels}.xlsx',
                       f'./results/expt1/poos_{var_name}_ar/poos_analysis_ar{levels}.xlsx',
                       f'./results/expt1/poos_{var_name}_pca/poos_analysis_pca{levels}.xlsx', ] + \
                      [f'{x["results_dir"]}/poos_analysis_{x["model_full_name"]}{levels}.xlsx' for x in id_store]
        name_store = ['rw', 'ar', 'pca'] + [f'{x["model_name"]}({x["est"]}, s{x["seed"]})' for x in id_store]
        combine_poos_excel_results(excel_store=excel_store, results_dir=f'./results/expt1/{var_name}',
                                   name_store=name_store,
                                   selected_xgba=['oracle', 'hparam', 'rw', 'll', 'll*ln', 'rw+ll*ln'])
    elif case == 6.1:
        # Combining hXX analysis results pkl from different models into one via simple averaging
        combined_name = 'c1'
        var_name = 'CPIA1'
        seed = [42, 42, 42,
                42, 42, 42, 42,
                100, 200, 300, ]
        est = [None, None, None,
               'rh', 'rfcv', 'rh', 'rfcv',
               'rh', 'rh', 'rh', ]
        model = ['rw', 'ar', 'pca',
                 'xgb','xgb','rf', 'rf',
                 'xgb', 'xgb', 'xgb', ]
        model_name = ['rw', 'ar', 'pca',
                      'xgba', 'xgba', 'rf', 'rf',
                      'xgba', 'xgba', 'xgba', ]
        levels = False
        # est = 'rh'

        id_store = create_id_store(var_name=var_name, expt_type='expt1',
                                   est=est, model=model, model_name=model_name, seed=seed)

        results_dir = create_results_directory(f'./results/expt1/model_combination_{var_name}_{combined_name}')
        combine_poos_analysis(results_dir=results_dir, id_store=id_store,
                              levels=levels, combined_name=combined_name)

        selector(case=3, levels=levels, id=create_id_dict(var_name=var_name,
                                                          h=[1, 3, 6, 12, 24],
                                                          combined_name=combined_name,
                                                          expt='expt1',
                                                          est=est))

    elif case == 6.2:
        # Plotting levels forecast.
        var_name = 'CPIA'
        est = [None, None, 'rh', 'rfcv', 'rh', 'rfcv']
        model = ['ar', 'pca', 'xgb', 'xgb', 'rf', 'rf']
        model_name = ['ar', 'pca', 'xgba', 'xgba', 'rf', 'rf']
        seed = [None, None, 42, 42, 42, 42]
        levels = True

        id_store = create_id_store(var_name=var_name, expt_type='expt1',
                                   est=est, model=model, model_name=model_name, seed=seed)
        if levels:
            levels = '_levels'
        else:
            levels = ''
        save_dir_store = [[f'{id["results_dir"]}/poos_{id["model"]}_h1_analysis_results{levels}.pkl',
                           f'{id["results_dir"]}/poos_{id["model"]}_h3_analysis_results{levels}.pkl',
                           f'{id["results_dir"]}/poos_{id["model"]}_h6_analysis_results{levels}.pkl',
                           f'{id["results_dir"]}/poos_{id["model"]}_h12_analysis_results{levels}.pkl',
                           f'{id["results_dir"]}/poos_{id["model"]}_h24_analysis_results{levels}.pkl',
                           ] for id in id_store]
        save_dir_store = [list(x) for x in zip(*save_dir_store)]
        plot_forecasts(save_dir_store=save_dir_store, results_dir=f'./results/expt1/forecast_plots_{var_name}',
                       model_names=model_name, est_store=est,
                       h_store=id_store[0]['h'])
    elif case == 7:
        # SHAP plots
        h_store = [1, 3, 6, 12, 24]
        var_name = 'CPIA1'
        excel_dir = './excel/dataset_0720/CPIA1_data_loader.xlsx'
        feature_info_dir = './excel/feature_info.xlsx'
        id1 = create_id_dict(var_name=var_name,
                             h=[1, 3, 6, 12, 24],
                             est='rh',
                             model='xgb',
                             model_name='xgba',
                             expt='expt1',
                             seed=42)
        id2 = create_id_dict(var_name=var_name,
                             h=[1, 3, 6, 12, 24],
                             est='rh',
                             model='xgb',
                             model_name='xgba',
                             expt='expt1',
                             seed=100)
        id3 = create_id_dict(var_name=var_name,
                             h=[1, 3, 6, 12, 24],
                             est='rh',
                             model='xgb',
                             model_name='xgba',
                             expt='expt1',
                             seed=200)
        id4 = create_id_dict(var_name=var_name,
                             h=[1, 3, 6, 12, 24],
                             est='rh',
                             model='xgb',
                             model_name='xgba',
                             expt='expt1',
                             seed=300)
        results_dir = create_results_directory('./results/expt1/shap_{}'.format(var_name))
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
        other_xgb_store = [{h: f'{id_["results_dir"]}/poos_h{h}.pkl' for h in h_store} for id_ in [id2, id3, id4]]
        poos_shap(fl_master=fl_master, fl=fl_xgb,
                  xgb_store=[f'{id1["results_dir"]}/poos_h{h}.pkl' for h in h_store],
                  other_xgb_store=other_xgb_store,
                  first_est_date=first_est_date,
                  results_dir=results_dir,
                  feature_info_dir=feature_info_dir)


if __name__ == '__main__':
    # selector(case=2, excel_dir='./excel/dataset_0720/CPIA1_data_loader.xlsx', var_name='poos_CPIA1_ar')
    # selector(case=3, excel_dir='./excel/dataset_0720/CPIA1_data_loader.xlsx', var_name='poos_CPIA1_ar')
    # selector(case=3.3, excel_dir='./excel/dataset_0720/CPIA1_data_loader.xlsx', var_name='poos_CPIA1_ar')
    #selector(case=3, levels=False, id=create_id_dict(var_name='CPIA1',
    #                                                  h=[1, 3, 6, 12, 24],
    #                                                  combined_name='c1',
    #                                                  expt='expt1',
    #                                                  est='rh'))
    selector(case=4, excel_dir='./excel/dataset_0720/INDPRO_data_loader.xlsx', var_name='poos_IND_ar')
