from own_package.features_labels import read_excel_data, Fl_master, Fl_pca
from own_package.others import create_results_directory
import numpy as np
import pandas as pd
import openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows

def type_transformations(excel_dir, y_selection, h_steps):
    df = pd.read_excel(excel_dir, sheet_name='Master')
    names = df.columns.values.tolist()
    data = df.values
    data_type_store = np.copy(data[0,1:])
    time_stamps = np.copy(data[1:, 0])
    data = np.copy(data[1:,1:]).astype(np.float)

    x_store = []
    for _, (type, x) in enumerate(zip(data_type_store.tolist(), data.T.tolist())):
        if type == 1:
            x_store.append(x)
        elif type == 2:
            x_transformed = np.array(x)[1:] - np.array(x)[:-1]
            x_transformed = [np.nan] + x_transformed.tolist()
            x_store.append(x_transformed)
        elif type == 4:
            x_transformed = np.log(np.array(x)).tolist()
            x_store.append(x_transformed)
        elif type == 5:
            x_transformed = np.log(np.array(x)[1:]) - np.log(np.array(x)[:-1])
            x_transformed = [np.nan] + x_transformed.tolist()
            x_store.append(x_transformed)
        elif type == 6:
            x_transformed = np.log(np.array(x)[2:]) - 2 * np.log(np.array(x)[1:-1]) + np.log(np.array(x)[:-2])
            x_transformed = [np.nan, np.nan] + x_transformed.tolist()
            x_store.append(x_transformed)
        elif type == 7:
            x_transformed = np.array(x)[2:] / np.array(x)[1:-1] - np.array(x)[1:-1] / np.array(x)[:-2]
            x_transformed = [np.nan, np.nan] + x_transformed.tolist()
            x_store.append(x_transformed)
        else:
            pass

    x_store = np.array(x_store).T

    temp_names = names[1:]
    selection_idx = [i for i in range(len(temp_names)) if temp_names[i] in y_selection]

    y_transformed_names = []
    y_store = []
    for idx, selection in enumerate(selection_idx):
        yo = data[:,selection]
        type = data_type_store[selection]
        for h in h_steps:
            y_transformed_names.append('{}_h{}'.format(temp_names[selection], h))
            if type == 5:
                y_transformed = 1200 / h * np.log(yo[h:] / yo[:-h])
                y_transformed = [np.nan] * h + y_transformed.tolist()
                y_store.append(y_transformed)
            elif type == 6:
                y_transformed = 1200 / h * np.log(yo[h+1:] / yo[1:-h]) - 1200 * np.log(yo[1:-h]/yo[:-h-1])
                y_transformed = [np.nan] * (h+1) + y_transformed.tolist()
                y_store.append(y_transformed)
            else:
                raise KeyError('Label type is not 5 or 6')

    y_store = np.array(y_store).T
    x_store[:, selection_idx] = x_store[:, selection_idx] * 1200


    wb = openpyxl.load_workbook(excel_dir)
    wb.create_sheet('transformation')
    sheet_name = wb.sheetnames[-1]
    ws = wb[sheet_name]

    df = pd.DataFrame(data=np.concatenate((time_stamps[..., None], x_store), axis=1),
                      columns=names)

    for r in dataframe_to_rows(df, index=False, header=True):
        ws.append(r)

    wb.create_sheet('y transformed')
    sheet_name = wb.sheetnames[-1]
    ws = wb[sheet_name]
    df = pd.DataFrame(data=np.concatenate((time_stamps[..., None], y_store), axis=1),
                      columns=['Time Stamps'] + y_transformed_names)

    for r in dataframe_to_rows(df, index=False, header=True):
        ws.append(r)

    wb.save(excel_dir)
