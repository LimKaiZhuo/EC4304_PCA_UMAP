import pickle, os
import numpy as np
import pandas as pd
from natsort import natsorted

from own_package.features_labels import read_excel_data, read_excel_dataloader, Fl_master, Fl_cw_data_store

def prepare_grand_data_store(dir_store):
    data_store = []
    if isinstance(dir_store, str):
        dir_store = [dir_store]
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
    skip_first_val = kwargs['skip_first_val']
    output = read_excel_dataloader(excel_dir=excel_dir)
    data_store = prepare_grand_data_store([results_dir])

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
                                              rolling=rolling,)

    m_star_store = [x.m_star for x in data_store]
    y_hat_star = [y[i_star-skip_idx] for y, i_star in zip(y_hat, m_star_store)]
    e_hat_star = [np.abs(e[i_star - skip_idx]) for e, i_star in zip(e_hat, m_star_store)]



