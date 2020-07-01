from own_package.features_labels import read_excel_data, Fl_master, Fl_pca
from own_package.others import create_results_directory, print_df_to_excel, create_excel_file
import numpy as np
import pandas as pd
import openpyxl, math
from openpyxl.utils.dataframe import dataframe_to_rows
from sklearn.preprocessing import StandardScaler
from scipy.linalg import eigh

def type_transformations(excel_dir, results_dir, y_selection, h_steps):
    df = pd.read_excel(excel_dir, sheet_name='Master')
    names = df.columns.values.tolist()
    data = df.values
    data_type_store = np.copy(data[0,1:])
    time_stamps = np.copy(data[3:, 0])
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

    y_store = (np.array(y_store).T)[2:,:]
    x_store[:, selection_idx] = x_store[:, selection_idx] * 1200
    x_store = x_store[2:,:]

    x_store = iterated_em(all_x=x_store, pca_p=9, max_iter=1e4, tol=0.1)

    results_dir = create_results_directory(results_dir)
    wb = openpyxl.Workbook()
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

    wb.save('{}/transformed_data.xlsx'.format(results_dir))

    create_data_loader_excel(excel_dir='{}/transformed_data.xlsx'.format(results_dir), results_dir=results_dir)


def pca_factor_estimation(x, r, N, x_transformed_already=False):
    if not x_transformed_already:
        x_scaler = StandardScaler()
        x_scaler.fit(x)
        x = x_scaler.transform(x)

    w, v = eigh(x.T @ x)
    loadings = np.fliplr(v[:, -r:])
    loadings = loadings * math.sqrt(N)
    factors = x @ loadings / N
    loadings_T = loadings.T

    return factors, loadings_T

def iterated_em(all_x, pca_p, max_iter, tol):
        N = all_x.shape[1]
        nan_store = np.where(np.isnan(all_x))
        col_mean = np.nanmean(all_x, axis=0)
        all_x[nan_store] = np.take(col_mean, nan_store[1])

        diff = 1000
        iter = 0
        while iter < max_iter:
            if diff < tol:
                print('Tolerance met with maximum percentage error = {}%\n Number of iterations taken = {}'.format(
                    diff, iter))
                break
            all_x_scaler = StandardScaler()
            all_x_scaler.fit(all_x)
            all_x_norm = all_x_scaler.transform(all_x)

            factors, loadings_T = pca_factor_estimation(x=all_x_norm, r=pca_p, N=N, x_transformed_already=True)

            all_x_norm_1 = factors @ loadings_T
            all_x_1 = all_x_scaler.inverse_transform(all_x_norm_1)

            try:
                # diff = max((all_x_1[nan_store] - all_x[nan_store]) / all_x[nan_store] * 100)  # Tolerance is in percentage error
                diff = np.max((factors - factors_old) / factors_old * 100)
                factors_old = factors
            except:
                diff = 1000
                factors_old = factors
            all_x[nan_store] = all_x_1[nan_store]
            iter += 1
        else:
            raise ValueError(
                'During iterated EM method, maximum iteration of {} without meeting tolerance requirement.'
                ' Current maximum percentage error = {}%'.format(iter, diff))

        return all_x



def create_data_loader_excel(excel_dir, results_dir):
    ymain_df = pd.read_excel(excel_dir, sheet_name='y transformed', index_col=0)
    xmain_df = pd.read_excel(excel_dir, 'transformation', index_col=0)

    # Find unique var name for forecasting
    var_names = list(set([item.partition('_h')[0] for item in ymain_df.columns]))

    for var_name in var_names:
        excel_name = create_excel_file('{}/{}_data_loader.xlsx'.format(results_dir, var_name))
        wb = openpyxl.load_workbook(excel_name)
        wb.create_sheet('x')
        wb.create_sheet('yo')
        wb.create_sheet('y')
        print_df_to_excel(df=xmain_df.loc[:, xmain_df.columns != var_name], ws=wb['x'])
        print_df_to_excel(df=xmain_df.loc[:,[var_name]], ws=wb['yo'])
        mask = np.flatnonzero(np.core.defchararray.find(ymain_df.columns.values.astype(str),var_name)!=-1)
        print_df_to_excel(df=ymain_df.iloc[:,mask], ws=wb['y'])
        wb.save(excel_name)

    pass