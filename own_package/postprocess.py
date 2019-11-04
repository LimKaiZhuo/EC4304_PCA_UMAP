import numpy as np
import pandas as pd
import math
import cvxpy as cp
import openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows
from own_package.dm_test import dm_test



def read_excel_to_df(excel_dir):
    xls = pd.ExcelFile(excel_dir)
    sheet_names = xls.sheet_names
    df_store = []
    for sheet in sheet_names:
        if sheet == 'Sheet':
            pass
        else:
            df = pd.read_excel(excel_dir, sheet_name=sheet, skiprows=[0, 2],
                               index_col=0).sort_values(['m', 'p'])
            df = df.reset_index(drop=True)
            df_store.append(df)

    return df_store


def compile_pm_rm_excel(excel_dir_store):
    master_pm = [[] for x in range(5)]
    master_rm = [[] for x in range(5)]
    for excel_dir in excel_dir_store:
        xls = pd.ExcelFile(excel_dir)
        sheet_names = xls.sheet_names[1:]
        for sheet, pm_store, rm_store in zip(sheet_names, master_pm, master_rm):
            df = pd.read_excel(excel_dir, sheet_name=sheet).values
            pm_store.append(df[0:9,:])
            rm_store.append(df[10:,0][..., None])


    for idx, pm_h in enumerate(master_pm):
        pm = pm_h[0]
        for pm_hh in pm_h[1:]:
            pm = np.concatenate((pm, pm_hh), axis=1)
        master_pm[idx] = pm

    for idx, pm_h in enumerate(master_rm):
        rm = pm_h[0]
        for pm_hh in pm_h[1:]:
            rm = np.concatenate((rm, pm_hh), axis=1)
        master_rm[idx] = rm

    wb = openpyxl.Workbook()
    for idx, (pm, rm) in enumerate(zip(master_pm, master_rm)):

        pm_name = 'pm_h{}'.format([1,3,6,12,24][idx])
        rm_name = 'rm_h{}'.format([1, 3, 6, 12, 24][idx])
        wb.create_sheet(pm_name)
        wb.create_sheet(rm_name)

        ws = wb[pm_name]
        pm_df = pd.DataFrame(data=pm, columns=['m', 'p']*len(excel_dir_store))
        rows = dataframe_to_rows(pm_df, index=False)
        for r_idx, row in enumerate(rows, 1):
            for c_idx, value in enumerate(row, 1):
                ws.cell(row=r_idx + 1, column=c_idx, value=value)

        ws = wb[rm_name]
        rm_df = pd.DataFrame(data=rm, columns=['Relative RMSE']*len(excel_dir_store))
        rows = dataframe_to_rows(rm_df, index=False)
        for r_idx, row in enumerate(rows, 1):
            for c_idx, value in enumerate(row, 1):
                ws.cell(row=r_idx + 1 , column=c_idx, value=value)

    wb.save('./results/master_pm_rd.xlsx')

    pass

class Postdata:
    def __init__(self, results_dir, var_name):
        self.results_dir = results_dir
        # First 3 lines if a list of dataframes. Each df is one h step ahead, for h=1,3,6,12,24
        # 4th line is list of 1D ndarray for y values
        # 5th line is list of 2D ndarray for (models, y hat values)

        # self.testset_XXX_y is the list of  1d ndarray for the XXX model (e.g. PCA) actual y values
        # self.testset_XXX_y_hat is the list of  2d ndarray for the XXX model, size = (no. of models, no. of y values)
        # self.testset_XXX_AWA_y_hat is the list of 1d ndarray for the XXX model forecasted y values combined with AWA
        # self.testset_XXX_BWA_y_hat is the list of 1d ndarray for the XXX model forecasted y values combined with BWA
        self.AR_AIC_BIC = read_excel_to_df('{}/{}_AR_AIC_BIC.xlsx'.format(results_dir, var_name))
        self.AR_PLS = read_excel_to_df('{}/{}_AR_PLS.xlsx'.format(results_dir, var_name))
        self.testset_AR_PLS = read_excel_to_df('{}/testset_{}_AR_PLS.xlsx'.format(results_dir, var_name))
        self.testset_AR_y = [np.array(x.columns.tolist()[5:]) for x in self.testset_AR_PLS]
        self.testset_AR_y_hat = [x.iloc[:, 5:].values for x in self.testset_AR_PLS]

        self.PCA_AIC_BIC = read_excel_to_df('{}/{}_PCA_AIC_BIC.xlsx'.format(results_dir, var_name))
        self.PCA_PLS = read_excel_to_df('{}/{}_PCA_PLS.xlsx'.format(results_dir, var_name))
        self.testset_PCA_PLS = read_excel_to_df('{}/testset_{}_PCA_PLS.xlsx'.format(results_dir, var_name))
        self.testset_PCA_y = [np.array(x.columns.tolist()[5:]) for x in self.testset_PCA_PLS]
        self.testset_PCA_y_hat = [x.iloc[:, 5:].values for x in self.testset_PCA_PLS]

        self.UMAP_AIC_BIC = read_excel_to_df('{}/{}_UMAP_AIC_BIC.xlsx'.format(results_dir, var_name))
        self.UMAP_PLS = read_excel_to_df('{}/{}_UMAP_PLS.xlsx'.format(results_dir, var_name))
        self.testset_UMAP_PLS = read_excel_to_df('{}/testset_{}_UMAP_PLS.xlsx'.format(results_dir, var_name))
        self.testset_UMAP_y = [np.array(x.columns.tolist()[5:]) for x in self.testset_UMAP_PLS]
        self.testset_UMAP_y_hat = [x.iloc[:, 5:].values for x in self.testset_UMAP_PLS]

        self.num_h = len(self.AR_AIC_BIC)
        self.pm_store = [np.zeros((9, 2)) for x in range(self.num_h)]
        self.rm_store = [np.zeros((23)) for x in range(self.num_h)]
        self.benchmark_rmse = []
        self.benchmarky = []
        i = 0
        # Iterate through each h step ahead values for all AR. h = 1,3,6,12,24
        for idx, (aic, pls, test, pm, rm, yhat, y) in enumerate(
                zip(self.AR_AIC_BIC, self.AR_PLS, self.testset_AR_PLS, self.pm_store, self.rm_store, self.testset_AR_y_hat, self.testset_AR_y)):
            self.benchmark_forecasted_y_BIC = []
            min_BIC_idx = np.argmin(aic['BIC_t'])
            pm[1, 1] = aic['p'][min_BIC_idx]
            rm[1] = 1
            rmse_idx = test.index[test['p'] == pm[1, 1]].tolist()[0]
            self.benchmark_rmse.append(test['Val RMSE'][rmse_idx])
            self.benchmarky.append(yhat[rmse_idx])

            min_AIC_idx = np.argmin(aic['AIC_t'])
            pm[0, 1] = aic['p'][min_AIC_idx]
            rmse_idx2 = test.index[test['p'] == pm[0, 1]].tolist()[0]
            rmse = test['Val RMSE'][rmse_idx2]
            rm[0] = rmse / self.benchmark_rmse[-1]
            if rmse_idx != rmse_idx2:
                forecastedy = yhat[rmse_idx2]
                dm_r = dm_test(y, self.benchmarky[i], forecastedy, h=1, crit="MSE")
                pvalue = dm_r[1]
                if pvalue <= 0.05:
                    rm[0] = rm[0] + 500

            min_idx = np.argmin(pls['Val RMSE'])
            pm[2, 1] = pls['p'][min_idx]
            rmse_idx2 = test.index[test['p'] == pm[2, 1]].tolist()[0]
            rmse = test['Val RMSE'][rmse_idx2]
            rm[2] = rmse / self.benchmark_rmse[-1]
            if rmse_idx != rmse_idx2:
                forecastedy = yhat[rmse_idx2]
                dm_r = dm_test(y, self.benchmarky[i], forecastedy, h=1, crit="MSE")
                pvalue = dm_r[1]
                if pvalue <= 0.05:
                    rm[2] = rm[2] + 500

            i = i + 1

        i = 0
        # Iterate through each h step ahead values for all PCA. h = 1,3,6,12,24
        skip = 3
        skip2 = 7
        for idx, (aic, pls, test, pm, rm, yhat, y) in enumerate(
                zip(self.PCA_AIC_BIC, self.PCA_PLS, self.testset_PCA_PLS, self.pm_store, self.rm_store, self.testset_PCA_y_hat, self.testset_PCA_y)):
            min_BIC_idx = np.argmin(aic['BIC_t'])
            pm[1 + skip, 0] = aic['m'][min_BIC_idx]
            pm[1 + skip, 1] = aic['p'][min_BIC_idx]
            rmse_idx = test.index[(test['m'] == pm[1 + skip, 0]) & (test['p'] == pm[1 + skip, 1])].tolist()[0]
            rmse = test['Val RMSE'][rmse_idx]
            rm[1 + skip2] = rmse / self.benchmark_rmse[idx]
            forecastedy = yhat[rmse_idx]
            dm_r = dm_test(y, self.benchmarky[i], forecastedy, h=1, crit="MSE")
            pvalue = dm_r[1]
            if pvalue <= 0.05:
                rm[1 + skip2] = rm[1 + skip2] + 500



            min_AIC_idx = np.argmin(aic['AIC_t'])
            pm[0 + skip, 0] = aic['m'][min_AIC_idx]
            pm[0 + skip, 1] = aic['p'][min_AIC_idx]
            rmse_idx = test.index[(test['m'] == pm[0 + skip, 0]) & (test['p'] == pm[0 + skip, 1])].tolist()[0]
            rmse = test['Val RMSE'][rmse_idx]
            rm[0 + skip2] = rmse / self.benchmark_rmse[idx]
            forecastedy = yhat[rmse_idx]
            dm_r = dm_test(y, self.benchmarky[i], forecastedy, h=1, crit="MSE")
            pvalue = dm_r[1]
            if pvalue <= 0.05:
                rm[0 + skip2] = rm[0 + skip2] + 500

            min_idx = np.argmin(pls['Val RMSE'])
            pm[2 + skip, 0] = pls['m'][min_idx]
            pm[2 + skip, 1] = pls['p'][min_idx]
            rmse_idx = test.index[(test['m'] == pm[2 + skip, 0]) & (test['p'] == pm[2 + skip, 1])].tolist()[0]
            rmse = test['Val RMSE'][rmse_idx]
            rm[2 + skip2] = rmse / self.benchmark_rmse[idx]
            forecastedy = yhat[rmse_idx]
            dm_r = dm_test(y, self.benchmarky[i], forecastedy, h=1, crit="MSE")
            pvalue = dm_r[1]
            if pvalue <= 0.05:
                rm[2 + skip2] = rm[2 + skip2] + 500

            i = i + 1

        i = 0
        # Iterate through each h step ahead values for all UMAP. h = 1,3,6,12,24
        skip = 3 * 2
        skip2 = 7 * 2
        for idx, (aic, pls, test, pm, rm, yhat, y) in enumerate(
                zip(self.UMAP_AIC_BIC, self.UMAP_PLS, self.testset_UMAP_PLS, self.pm_store, self.rm_store, self.testset_UMAP_y_hat, self.testset_UMAP_y)):
            min_BIC_idx = np.argmin(aic['BIC_t'])
            pm[1 + skip, 0] = aic['m'][min_BIC_idx]
            pm[1 + skip, 1] = aic['p'][min_BIC_idx]
            rmse_idx = test.index[(test['m'] == pm[1 + skip, 0]) & (test['p'] == pm[1 + skip, 1])].tolist()[0]
            rmse = test['Val RMSE'][rmse_idx]
            rm[1 + skip2] = rmse / self.benchmark_rmse[idx]
            forecastedy = yhat[rmse_idx]
            dm_r = dm_test(y, self.benchmarky[i], forecastedy, h=1, crit="MSE")
            pvalue = dm_r[1]
            if pvalue <= 0.05:
                rm[1 + skip2] = rm[1 + skip2] + 500

            min_AIC_idx = np.argmin(aic['AIC_t'])
            pm[0 + skip, 0] = aic['m'][min_AIC_idx]
            pm[0 + skip, 1] = aic['p'][min_AIC_idx]
            rmse_idx = test.index[(test['m'] == pm[0 + skip, 0]) & (test['p'] == pm[0 + skip, 1])].tolist()[0]
            rmse = test['Val RMSE'][rmse_idx]
            rm[0 + skip2] = rmse / self.benchmark_rmse[idx]
            forecastedy = yhat[rmse_idx]
            dm_r = dm_test(y, self.benchmarky[i], forecastedy, h=1, crit="MSE")
            pvalue = dm_r[1]
            if pvalue <= 0.05:
                rm[0 + skip2] = rm[0 + skip2] + 500

            min_idx = np.argmin(pls['Val RMSE'])
            pm[2 + skip, 0] = pls['m'][min_idx]
            pm[2 + skip, 1] = pls['p'][min_idx]
            rmse_idx = test.index[(test['m'] == pm[2 + skip, 0]) & (test['p'] == pm[2 + skip, 1])].tolist()[0]
            rmse = test['Val RMSE'][rmse_idx]
            rm[2 + skip2] = rmse / self.benchmark_rmse[idx]
            forecastedy = yhat[rmse_idx]
            dm_r = dm_test(y, self.benchmarky[i], forecastedy, h=1, crit="MSE")
            pvalue = dm_r[1]
            if pvalue <= 0.05:
                rm[2 + skip2] = rm[2 + skip2] + 500

            i = i + 1

        pass

    def combination(self):
        """

        :param type: Either 'AIC_t' or 'BIC_t' for AWA and BWA respectively
        :return:
        """
        aic_bic_store = [self.AR_AIC_BIC, self.PCA_AIC_BIC, self.UMAP_AIC_BIC]
        pls_store = [self.AR_PLS, self.PCA_PLS, self.UMAP_PLS]
        testset_y_store = [self.testset_AR_y, self.testset_PCA_y, self.testset_UMAP_y]
        testset_y_hat_store = [self.testset_AR_y_hat, self.testset_PCA_y_hat, self.testset_UMAP_y_hat]
        self.testset_AR_AWA_y_hat = []
        self.testset_AR_BWA_y_hat = []
        self.testset_AR_AVG_y_hat = []
        self.testset_AR_GR_y_hat = []
        self.testset_PCA_AWA_y_hat = []
        self.testset_PCA_BWA_y_hat = []
        self.testset_PCA_AVG_y_hat = []
        self.testset_PCA_GR_y_hat = []
        self.testset_UMAP_AWA_y_hat = []
        self.testset_UMAP_BWA_y_hat = []
        self.testset_UMAP_AVG_y_hat = []
        self.testset_UMAP_GR_y_hat = []
        self.testset_PU_AVG_y_hat = []
        self.testset_PU_GR_y_hat = []

        for skip_idx, (aic_bic_all_h, pls_all_h, testset_y, testset_y_hat, awa_y_hat, bwa_y_hat, avg_y_hat, gr_y_hat) \
                in enumerate(zip(aic_bic_store, pls_store, testset_y_store, testset_y_hat_store,
                                 [self.testset_AR_AWA_y_hat, self.testset_PCA_AWA_y_hat, self.testset_UMAP_AWA_y_hat],
                                 [self.testset_AR_BWA_y_hat, self.testset_PCA_BWA_y_hat, self.testset_UMAP_BWA_y_hat],
                                 [self.testset_AR_AVG_y_hat, self.testset_PCA_AVG_y_hat, self.testset_UMAP_AVG_y_hat],
                                 [self.testset_AR_GR_y_hat, self.testset_PCA_GR_y_hat, self.testset_UMAP_GR_y_hat])):
            i = 0
            for idx, (ic, pls, y, y_hat, rm) in enumerate(
                    zip(aic_bic_all_h, pls_all_h, testset_y, testset_y_hat, self.rm_store)):
                # Simple average AVG
                t_idx = 3 + 7 * skip_idx
                y_combi_hat = np.mean(y_hat, axis=0)
                avg_y_hat.append(y_combi_hat)
                rmse_combi = math.sqrt(np.mean(np.array(y - y_combi_hat) ** 2))
                rm[t_idx] = rmse_combi / self.benchmark_rmse[idx]
                dm_r = dm_test(y, self.benchmarky[i], y_combi_hat, h=1, crit="MSE")
                pvalue = dm_r[1]
                if pvalue <= 0.05:
                    rm[t_idx] = rm[t_idx] + 500

                # AWA
                type = 'AIC_t'
                t_idx = 4 + 7 * skip_idx
                ic_values = ic[type].values
                min_ic = np.min(ic_values)
                ic_values += -min_ic
                weights = np.exp(-ic_values / 2)
                weights = weights / np.sum(weights)
                y_combi_hat = np.sum(y_hat * weights[:, None], axis=0)
                awa_y_hat.append(y_combi_hat)
                rmse_combi = math.sqrt(np.mean(np.array(y - y_combi_hat) ** 2))
                rm[t_idx] = rmse_combi / self.benchmark_rmse[idx]
                dm_r = dm_test(y, self.benchmarky[i], y_combi_hat, h=1, crit="MSE")
                pvalue = dm_r[1]
                if pvalue <= 0.05:
                    rm[t_idx] = rm[t_idx] + 500

                # BWA
                type = 'BIC_t'
                t_idx = 5 + 7 * skip_idx
                ic_values = ic[type].values
                min_ic = np.min(ic_values)
                ic_values += -min_ic
                weights = np.exp(-ic_values / 2)
                weights = weights / np.sum(weights)
                y_combi_hat = np.sum(y_hat * weights[:, None], axis=0)
                bwa_y_hat.append(y_combi_hat)
                rmse_combi = math.sqrt(np.mean(np.array(y - y_combi_hat) ** 2))
                rm[t_idx] = rmse_combi / self.benchmark_rmse[idx]
                dm_r = dm_test(y, self.benchmarky[i], y_combi_hat, h=1, crit="MSE")
                pvalue = dm_r[1]
                if pvalue <= 0.05:
                    rm[t_idx] = rm[t_idx] + 500

                # GR
                t_idx = 6 + 7 * skip_idx
                y_pls = np.array(pls.columns.tolist()[5:])
                y_hat_pls = pls.iloc[:, 5:].values
                m = np.shape(y_hat_pls)[0] + 1  # number of models + 1 constant term
                n = np.shape(y_hat_pls)[1]  # number of timesteps
                beta = cp.Variable(shape=(m, 1))

                pc_1 = np.ones((1, m - 1)) @ beta[1:, 0] == 1
                pc_2 = beta >= 0
                constraints = [pc_1, pc_2]

                X = np.concatenate((np.ones((n, 1)), y_hat_pls.T), axis=1)

                z = np.ones((1, n)) @ (y_pls[:, None] - X @ beta) ** 2
                objective = cp.Minimize(z)
                prob = cp.Problem(objective, constraints)

                prob.solve(solver='GUROBI')
                beta_hat = beta.value
                y_combi_hat = np.sum(y_hat * beta_hat[1:, 0][:, None] + beta_hat[0, 0], axis=0)
                gr_y_hat.append(y_combi_hat)
                rmse_combi = math.sqrt(np.mean(np.array(y - y_combi_hat) ** 2))
                rm[t_idx] = rmse_combi / self.benchmark_rmse[idx]
                dm_r = dm_test(y, self.benchmarky[i], y_combi_hat, h=1, crit="MSE")
                pvalue = dm_r[1]
                if pvalue <= 0.05:
                    rm[t_idx] = rm[t_idx] + 500

                i = i + 1
        i = 0
        # PCA+UMAP
        for idx, (pca_pls, umap_pls, y, pca_y_hat, umap_y_hat, rm) in enumerate(
                zip(self.PCA_PLS, self.UMAP_PLS, self.testset_PCA_y, self.testset_PCA_y_hat, self.testset_UMAP_y_hat,
                    self.rm_store)):
            # AVG
            y_pls = np.array(pca_pls.columns.tolist()[5:])
            pca_y_hat_pls = pca_pls.iloc[:, 5:].values
            umap_y_hat_pls = umap_pls.iloc[:, 5:].values

            y_combi_hat = np.mean(np.concatenate((pca_y_hat, umap_y_hat), axis=0), axis=0)
            self.testset_PU_AVG_y_hat.append(y_combi_hat)
            rmse_combi = math.sqrt(np.mean(np.array(y - y_combi_hat) ** 2))
            rm[21] = rmse_combi / self.benchmark_rmse[idx]
            dm_r = dm_test(y, self.benchmarky[i], y_combi_hat, h=1, crit="MSE")
            pvalue = dm_r[1]
            if pvalue <= 0.05:
                rm[21] = rm[21] + 500

            # GR
            y_hat_pls = np.concatenate((pca_y_hat_pls, umap_y_hat_pls), axis=0)
            m = np.shape(y_hat_pls)[0] + 1  # number of models + 1 constant term
            n = np.shape(y_hat_pls)[1]  # number of timesteps
            beta = cp.Variable(shape=(m, 1))

            pc_1 = np.ones((1, m - 1)) @ beta[1:, 0] == 1
            pc_2 = beta >= 0
            constraints = [pc_1, pc_2]

            X = np.concatenate((np.ones((n, 1)), y_hat_pls.T), axis=1)

            z = np.ones((1, n)) @ (y_pls[:, None] - X @ beta) ** 2
            objective = cp.Minimize(z)
            prob = cp.Problem(objective, constraints)

            prob.solve(solver='GUROBI')
            beta_hat = beta.value

            y_hat = np.concatenate((pca_y_hat, umap_y_hat), axis=0)
            y_combi_hat = np.sum(y_hat * beta_hat[1:, 0][:, None] + beta_hat[0, 0], axis=0)
            self.testset_PU_GR_y_hat.append(y_combi_hat)
            rmse_combi = math.sqrt(np.mean(np.array(y - y_combi_hat) ** 2))
            rm[22] = rmse_combi / self.benchmark_rmse[idx]
            dm_r = dm_test(y, self.benchmarky[i], y_combi_hat, h=1, crit="MSE")
            pvalue = dm_r[1]
            if pvalue <= 0.05:
                rm[22] = rm[22] + 500

            i = i + 1

        # Printing to excel
        wb = openpyxl.Workbook()
        for idx in range(len(self.pm_store)):
            wb.create_sheet('h = {}'.format([1,3,6,12,24][idx]))
        sheet_names = wb.sheetnames

        for sheet, pm, rm in zip(sheet_names[1:], self.pm_store, self.rm_store):
            ws = wb[sheet]

            pm_df = pd.DataFrame(data=pm, columns=['m', 'p'])
            rows = dataframe_to_rows(pm_df, index=False)
            for r_idx, row in enumerate(rows, 1):
                for c_idx, value in enumerate(row, 1):
                    ws.cell(row=r_idx + 1, column=c_idx, value=value)

            skip = len(pm_df.index) + 1
            rm_df = pd.DataFrame(data=rm, columns=['Relative RMSE'])
            rows = dataframe_to_rows(rm_df, index=False)
            for r_idx, row in enumerate(rows, 1):
                for c_idx, value in enumerate(row, 1):
                    ws.cell(row=r_idx + 1 + skip, column=c_idx, value=value)



        wb.save('{}/pm_rm_results.xlsx'.format(self.results_dir))


        pass

