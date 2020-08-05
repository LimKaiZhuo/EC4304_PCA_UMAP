from own_package.boosting import run_testing
from own_package.poos import poos_analysis, poos_processed_data_analysis, poos_experiment, poos_model_evaluation, \
    poos_shap
from own_package.features_labels import read_excel_dataloader, Fl_master, Fl_ar, Fl_pca, Fl_xgb
from own_package.others import create_results_directory


class A(object):  # deriving from 'object' declares A as a 'new-style-class'
    def foo(self):
        print('foo')


class B(A):
    def doo(self):
        super().foo()  # calls 'A.foo()'
        print('du')


def selector(case, **kwargs):
    if case == 1:
        run_testing()
    elif case == 2:
        excel_dir = kwargs['excel_dir']
        output = read_excel_dataloader(excel_dir=excel_dir)
        fl_master = Fl_master(x=output[0], features_names=output[1],
                              yo=output[2], labels_names=output[3],
                              y=output[4], y_names=output[5],
                              time_stamp=output[6])
        h_store = [1, 3, 6, 12, 24]
        h_idx_store = [0, 1, 2, 3, 4]
        for h, h_idx in zip(h_store, h_idx_store):
            poos_analysis(fl_master=fl_master, h=h, h_idx=h_idx, model_mode='xgb',
                          results_dir='./results/poos/poos_CMR_xgba',
                          save_dir=f'./results/poos/poos_CMR_xgba/poos_h{h}.pkl')
    elif case == 3:
        model_name = 'pca'
        var_name = 'CMR'
        poos_processed_data_analysis(
            save_dir_store=[f'./results/poos/poos_{var_name}_{model_name}/poos_{model_name}_h1_analysis_results.pkl',
                            f'./results/poos/poos_{var_name}_{model_name}/poos_{model_name}_h3_analysis_results.pkl',
                            f'./results/poos/poos_{var_name}_{model_name}/poos_{model_name}_h6_analysis_results.pkl',
                            f'./results/poos/poos_{var_name}_{model_name}/poos_{model_name}_h12_analysis_results.pkl',
                            f'./results/poos/poos_{var_name}_{model_name}/poos_{model_name}_h24_analysis_results.pkl',
                            ],
            h_store=['1',
                     '3',
                     '6',
                     '12',
                     '24',
                     ],
            results_dir=f'./results/poos/poos_{var_name}_{model_name}',
            model_mode=model_name)
    elif case == 4:
        # Run poos experiment for ar4 or pca
        var_name = kwargs['var_name']
        excel_dir = kwargs['excel_dir']
        results_dir = create_results_directory('./results/poos/{}'.format(var_name))
        output = read_excel_dataloader(excel_dir=excel_dir)
        fl_master = Fl_master(x=output[0], features_names=output[1],
                              yo=output[2], labels_names=output[3],
                              y=output[4], y_names=output[5],
                              time_stamp=output[6])
        fl = Fl_ar(val_split=None, x=None, yo=None, y=None,
                   time_stamp=None, time_idx=None,
                   features_names=fl_master.features_names, labels_names=fl_master.labels_names,
                   y_names=fl_master.y_names)

        est_dates = [f'{x}:12' for x in range(1969, 2020, 5)[:-1]]

        model_mode = 'ar'
        poos_experiment(fl_master=fl_master, fl=fl, est_dates=est_dates, z_type=1, h=1, h_idx=0,
                        m_max=3, p_max=12, model_mode=model_mode, save_dir=results_dir,
                        )
        poos_experiment(fl_master=fl_master, fl=fl, est_dates=est_dates, z_type=1, h=3, h_idx=1,
                        m_max=3, p_max=12, model_mode=model_mode, save_dir=results_dir,
                        )
        poos_experiment(fl_master=fl_master, fl=fl, est_dates=est_dates, z_type=1, h=6, h_idx=2,
                        m_max=3, p_max=12, model_mode=model_mode, save_dir=results_dir,
                        )
        poos_experiment(fl_master=fl_master, fl=fl, est_dates=est_dates, z_type=1, h=12, h_idx=3,
                        m_max=3, p_max=12, model_mode=model_mode, save_dir=results_dir,
                        )
        poos_experiment(fl_master=fl_master, fl=fl, est_dates=est_dates, z_type=1, h=24, h_idx=4,
                        m_max=3, p_max=12, model_mode=model_mode, save_dir=results_dir,
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
            results_dir=results_dir,
            model_mode=model_mode)

        model_mode = 'pca'
        results_dir = create_results_directory('./results/poos/{}'.format(var_name))
        fl = Fl_pca(val_split=None, x=None, yo=None, y=None,
                    time_stamp=None, time_idx=None,
                    features_names=fl_master.features_names, labels_names=fl_master.labels_names,
                    y_names=fl_master.y_names)
        poos_experiment(fl_master=fl_master, fl=fl, est_dates=est_dates, z_type=1, h=1, h_idx=0,
                        m_max=3, p_max=12, model_mode=model_mode, save_dir=results_dir,
                        )
        poos_experiment(fl_master=fl_master, fl=fl, est_dates=est_dates, z_type=1, h=3, h_idx=1,
                        m_max=3, p_max=12, model_mode=model_mode, save_dir=results_dir,
                        )
        poos_experiment(fl_master=fl_master, fl=fl, est_dates=est_dates, z_type=1, h=6, h_idx=2,
                        m_max=3, p_max=12, model_mode=model_mode, save_dir=results_dir,
                        )
        poos_experiment(fl_master=fl_master, fl=fl, est_dates=est_dates, z_type=1, h=12, h_idx=3,
                        m_max=3, p_max=12, model_mode=model_mode, save_dir=results_dir,
                        )
        poos_experiment(fl_master=fl_master, fl=fl, est_dates=est_dates, z_type=1, h=24, h_idx=4,
                        m_max=3, p_max=12, model_mode=model_mode, save_dir=results_dir,
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
            results_dir=results_dir,
            model_mode=model_mode)

    elif case == 5:
        # Forecast evaluation DM
        h_store = [1, 3, 6, 12, 24]
        var_name = 'PAY'
        results_dir = create_results_directory(f'./results/poos/model_eval_{var_name}')
        poos_model_evaluation(
            ar_store=[f'./results/poos/poos_{var_name}_ar/poos_ar_h{h}_analysis_results.pkl' for h in h_store],
            pca_store=[f'./results/poos/poos_{var_name}_pca/poos_pca_h{h}_analysis_results.pkl' for h in h_store],
            xgb_store=[f'./results/poos/poos_{var_name}_xgba/poos_xgb_h{h}_analysis_results.pkl' for h in h_store],
            results_dir=results_dir)
    elif case == 6:
        h_store = [1, 3, 6, 12, 24]
        var_name = kwargs['var_name']
        excel_dir = kwargs['excel_dir']
        results_dir = create_results_directory('./results/poos/shap_{}'.format(var_name))
        output = read_excel_dataloader(excel_dir=excel_dir)
        fl_master = Fl_master(x=output[0], features_names=output[1],
                              yo=output[2], labels_names=output[3],
                              y=output[4], y_names=output[5],
                              time_stamp=output[6])
        fl_xgb = Fl_xgb(val_split=None, x=None, yo=None, y=None,
                        time_stamp=None, time_idx=None,
                        features_names=fl_master.features_names, labels_names=fl_master.labels_names,
                        y_names=fl_master.y_names)
        poos_shap(fl_master=fl_master, fl=fl_xgb,
                  xgb_store=[f'./results/poos/{var_name}/poos_h{h}.pkl' for h in h_store],
                  results_dir=results_dir)


if __name__ == '__main__':
    selector(case=5, excel_dir='./excel/dataset2/INDPRO_data_loader.xlsx', var_name='poos_IND_ar')
