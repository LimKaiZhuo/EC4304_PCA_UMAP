from own_package.features_labels import read_excel_data, Fl_master

def selector(case):
    if case == 1:
        features, labels, time_stamp, features_names, labels_names = read_excel_data(excel_dir='./excel/dataset_1.xlsx')
        Fl_master(features, labels, time_stamp, features_names, labels_names)

selector(1)