from own_package.features_labels import read_excel_data, Fl_master, Fl_pca

def selector(case):
    if case == 1:
        # Testing filling out missing observation using iterated EM method
        excel_dir = './excel/dataset_1.xlsx'
        features, labels, time_stamp, features_names, labels_names = read_excel_data(excel_dir=excel_dir)
        fl_master = Fl_master(features, labels, time_stamp, features_names, labels_names)
        fl_master.iterated_em(features=features, labels=labels, pca_p=15, max_iter=10000, tol=0.1, excel_dir=excel_dir)
    elif case == 2:
        # Testing pca k selection using IC criteria
        excel_dir = './excel/dataset_2.xlsx'
        features, labels, time_stamp, features_names, labels_names = read_excel_data(excel_dir=excel_dir)
        fl_master = Fl_master(x=features, y=labels, time_stamp=time_stamp,
                              features_names=features_names, labels_names=labels_names)
        (f_tv, f_tt), (yo_tv, yo_t), (y_tv, y_tt),\
        (ts_tv, ts_tt), (tidx_tv, tidx_tt), (nobs_tv, nobs_tt) = fl_master.percentage_split(0.2)
        fl = Fl_pca(val_split=0.2, x=f_tv, yo=yo_tv, y=y_tv,
                    time_stamp=ts_tv, time_idx=tidx_tv, features_names=features_names, labels_names=labels_names)
        fl.pca_k_selection(lower_k=5, upper_k=fl.N-1)
    elif case == 3:
        excel_dir = './excel/dataset_2.xlsx'
        features, labels, time_stamp, features_names, labels_names = read_excel_data(excel_dir=excel_dir)
        fl_master = Fl_master(features, labels, time_stamp, features_names, labels_names)
        (f_tv, f_tt), (yo_tv, yo_t), (l_tv, l_tt),\
        (ts_tv, ts_tt), (tidx_tv, tidx_tt), (nobs_tv, nobs_tt) = fl_master.percentage_split(0.2)
        fl = Fl_pca(val_split=0.2, features=f_tv, yo=yo_tv, labels=l_tv,
                    time_stamp=ts_tv, time_idx=tidx_tv, features_names=features_names, labels_names=labels_names)
        factors, _ = fl.pca_factor_estimation(f_tv, 10)
        ff, fy, l = fl.pca_umap_prepare_data_matrix(factors, l_tv, l_tv, 10, 3, 2)
    pass

selector(3)