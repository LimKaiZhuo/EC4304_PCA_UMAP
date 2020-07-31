from own_package.boosting import run_testing
from own_package.poos import poos_analysis, poos_processed_data_analysis, poos_experiment
from own_package.features_labels import read_excel_dataloader, Fl_master, Fl_ar
from own_package.others import create_results_directory

class A(object):     # deriving from 'object' declares A as a 'new-style-class'
    def foo(self):
        print('foo')

class B(A):
    def doo(self):
        super().foo()   # calls 'A.foo()'
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
        h_store = [1]
        h_idx_store = [0]
        for h, h_idx in zip(h_store, h_idx_store):
            poos_analysis(fl_master=fl_master, h=h, h_idx=h_idx, model_mode='xgb',
                          results_dir='./results/poos/poos_CPIA_xgba',
                          save_dir=f'./results/poos/poos_CPIA_xgba/poos_h{h}.pkl')
    elif case == 3:
        poos_processed_data_analysis(save_dir_store=['./results/poos/poos_W875_xgba/poos_xgb_h1_analysis_results.pkl',
                                                     './results/poos/poos_W875_xgba/poos_xgb_h3_analysis_results.pkl',
                                                     './results/poos/poos_W875_xgba/poos_xgb_h6_analysis_results.pkl',
                                                     './results/poos/poos_W875_xgba/poos_xgb_h12_analysis_results.pkl',
                                                     './results/poos/poos_W875_xgba/poos_xgb_h24_analysis_results.pkl',
                                                     ],
                                     h_store=['1',
                                              '3',
                                              '6',
                                              '12',
                                              '24',
                                              ],
                                     results_dir='./results/poos/poos_W875_xgba')
    elif case == 4:
        # Run poos experiment
        var_name = kwargs['var_name']
        excel_dir = kwargs['excel_dir']
        results_dir = create_results_directory('./results/poos/{}'.format(var_name))
        output = read_excel_dataloader(excel_dir=excel_dir)
        fl_master = Fl_master(x=output[0], features_names=output[1],
                              yo=output[2], labels_names=output[3],
                              y=output[4], y_names=output[5],
                              time_stamp=output[6])
        fl_xgb = Fl_ar(val_split=None, x=None, yo=None, y=None,
                        time_stamp=None, time_idx=None,
                        features_names=fl_master.features_names, labels_names=fl_master.labels_names,
                        y_names=fl_master.y_names)

        est_dates = [f'{x}:12' for x in range(1969, 2020, 5)[:-1]]

        poos_experiment(fl_master=fl_master, fl=fl_xgb, est_dates=est_dates, z_type=1, h=1, h_idx=0,
                        m_max=12, p_max=24, model_mode='ar4', save_dir=results_dir,
                        )
        poos_experiment(fl_master=fl_master, fl=fl_xgb, est_dates=est_dates, z_type=1, h=3, h_idx=1,
                        m_max=12, p_max=24, model_mode='ar4', save_dir=results_dir,
                        )
        poos_experiment(fl_master=fl_master, fl=fl_xgb, est_dates=est_dates, z_type=1, h=6, h_idx=2,
                        m_max=12, p_max=24, model_mode='ar4', save_dir=results_dir,
                        )
        poos_experiment(fl_master=fl_master, fl=fl_xgb, est_dates=est_dates, z_type=1, h=12, h_idx=3,
                        m_max=12, p_max=24, model_mode='ar4', save_dir=results_dir,
                        )
        poos_experiment(fl_master=fl_master, fl=fl_xgb, est_dates=est_dates, z_type=1, h=24, h_idx=4,
                        m_max=12, p_max=24, model_mode='ar4', save_dir=results_dir,
                        )

if __name__ == '__main__':
    selector(case=2, excel_dir='./excel/dataset2/W875RX1_data_loader.xlsx', var_name='poos_AR4_W875')
