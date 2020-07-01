import pickle, os
import numpy as np
import pandas as pd
import openpyxl
from natsort import natsorted
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from own_package.features_labels import read_excel_data, read_excel_dataloader, Fl_master, Fl_cw_data_store
from own_package.postprocess import read_excel_to_df
from own_package.others import print_df_to_excel

def prepare_grand_data_store(dir_store, model):
    data_store = []
    if isinstance(dir_store, str):
        dir_store = [dir_store]

    if model == 'AR' or model == 'PLS':
        # Each pickle datastore is one complete OOS forecasting but with different lags (m or p).
        for dir_name in dir_store:
            filenames = natsorted(os.listdir(dir_name))
            for filename in filenames:
                if filename.endswith(".pkl"):
                    with open('{}/{}'.format(dir_name, filename), 'rb') as handle:
                        data_store.append(pickle.load(handle))
    elif model == 'CW' or model == 'CWd':
        # each pickle datastore is 5 OOS forecast steps. Extend them so that it is one list.
        for dir_name in dir_store:
            filenames = natsorted(os.listdir(dir_name))
            for filename in filenames:
                if filename.endswith(".pkl"):
                    with open('{}/{}'.format(dir_name, filename), 'rb') as handle:
                        data_store.extend(pickle.load(handle))
    return data_store


def cw_analysis(**kwargs):
    excel_dir = kwargs['excel_dir']
    results_dir = kwargs['results_dir']
    rolling = kwargs['rolling']
    h_idx = kwargs['h_idx']
    h = kwargs['h']
    p = kwargs['p']
    m = kwargs['m']
    z_type = kwargs['z_type']
    skip_first_val = kwargs['skip_first_val']
    output = read_excel_dataloader(excel_dir=excel_dir)
    data_store = prepare_grand_data_store([results_dir], model='CW')

    if skip_first_val:
        # old implementation of CW class did not save the first model instance nor the last 4 instances.
        data_store.insert(0, data_store[0])
        skip_idx = 1
    else:
        skip_idx = 0

    fl_master = Fl_master(x=output[0], features_names=output[1],
                          yo=output[2], labels_names=output[3],
                          y=output[4], y_names=output[5],
                          time_stamp=output[6])
    (f_tv, f_tt), (yo_tv, yo_t), (y_tv, y_tt), \
    (ts_tv, ts_tt), (tidx_tv, tidx_tt), (nobs_tv, nobs_tt) = fl_master.percentage_split(0)

    fl = Fl_cw_data_store(val_split=0.2, x=f_tv, yo=yo_tv, y=y_tv,
                          time_stamp=ts_tv, time_idx=tidx_tv,
                          features_names=fl_master.features_names, labels_names=fl_master.labels_names,
                          y_names=fl_master.y_names)

    y_hat, e_hat, _ = fl.pls_expanding_window(h=h, p=p, m=m,
                                              data_store=data_store,
                                              x_t=fl.x_t,
                                              x_v=fl.x_v,
                                              yo_t=fl.yo_t,
                                              y_t=fl.y_t[:, h_idx][..., None],
                                              yo_v=fl.yo_v,
                                              y_v=fl.y_v[:, h_idx][..., None],
                                              z_type=z_type,
                                              rolling=rolling,)

    m_star_store = [x['m_star'] for x in data_store]
    y_hat_star = [y[i_star-skip_idx] for y, i_star in zip(y_hat, m_star_store)]
    e_hat_star = [e[i_star - skip_idx] for e, i_star in zip(e_hat, m_star_store)]

    e_df = pd.DataFrame(e_hat).iloc[:,:]
    e_df['month'] = e_df.index

    x = e_df.index
    e_df = pd.melt(e_df, id_vars='month', var_name='m', value_name='e_hat')
    sns.lineplot(x='month', y='e_hat', hue='m', data=e_df)
    plt.plot(x, e_hat_star)
    plt.savefig('{}/e_hat.png'.format(results_dir))
    plt.close()
    sns.lineplot(x='month', y='e_hat', hue='m', data=e_df.loc[e_df['m']<=70])
    plt.plot(x, e_hat_star)
    plt.savefig('{}/e_hat_m70.png'.format(results_dir))
    plt.close()
    plt.plot(m_star_store)
    plt.savefig('{}/m_star_store.png'.format(results_dir))
    plt.close()


def combine_rmse_results(results_dir):
    df_store = []
    df_best_store = []
    for excel in os.listdir(results_dir):
        if 'testset_' in excel:
            print(f'Loading excel from {results_dir}/{excel}')
            df_store.append(read_excel_to_df(f'{results_dir}/{excel}'))
            df_best = []
            for df in df_store[-1]:
                df.insert(0,'model',excel.split('_')[2])
                if '_AR_' in excel:
                    df_best.append(df.iloc[[3,df['Val RMSE'].argmin()],:])
                elif '_PCA_' in excel:
                    df_best.append(df.iloc[[df['Val RMSE'].argmin()], :])
                else:
                    df_best.append(df)
            df_best_store.append(df_best)
    # transpose nested list
    df_store = list(map(list, zip(*df_store)))
    combined_df_store = []
    for df_h in df_store:
        combined_df_store.append(pd.concat(df_h).sort_values(by='Val RMSE'))

    wb = openpyxl.Workbook()
    for h,df in zip([1,3,6,12,24], combined_df_store):
        wb.create_sheet(f'h_{h}')
        ws = wb[f'h_{h}']
        print_df_to_excel(df=df, ws=ws)

    wb.save(f'{results_dir}/summary.xlsx')

    # transpose nested list
    df_store = list(map(list, zip(*df_best_store)))
    combined_df_store = []
    for df_h in df_store:
        combined_df_store.append(pd.concat(df_h).sort_values(by='Val RMSE'))

    wb = openpyxl.Workbook()
    for h,df in zip([1,3,6,12,24], combined_df_store):
        wb.create_sheet(f'h_{h}')
        ws = wb[f'h_{h}']
        print_df_to_excel(df=df, ws=ws)

    wb.save(f'{results_dir}/best summary.xlsx')



