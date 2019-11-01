import numpy as np
import pandas as pd
import math


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


class Postdata:
    def __init__(self, results_dir, var_name):
        # First 3 lines if a list of dataframes. Each df is one h step ahead, for h=1,3,6,12,24
        # 4th line is list of 1D ndarray for y values
        # 5th line is list of 2D ndarray for (models, y hat values)
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
        self.rm_store = [np.zeros((25)) for x in range(self.num_h)]
        self.benchmark_rmse = []

        # Iterate through each h step ahead values for all AR. h = 1,3,6,12,24
        for idx, (aic, pls, test, pm, rm) in enumerate(
                zip(self.AR_AIC_BIC, self.AR_PLS, self.testset_AR_PLS, self.pm_store, self.rm_store)):
            min_BIC_idx = np.argmin(aic['BIC_t'])
            pm[1, 1] = aic['p'][min_BIC_idx]
            rm[1] = 1
            rmse_idx = test.index[test['p'] == pm[1, 1]].tolist()[0]
            self.benchmark_rmse.append(test['Val RMSE'][rmse_idx])

            min_AIC_idx = np.argmin(aic['AIC_t'])
            pm[0, 1] = aic['p'][min_AIC_idx]
            rmse_idx = test.index[test['p'] == pm[0, 1]].tolist()[0]
            rmse = test['Val RMSE'][rmse_idx]
            rm[0] = rmse / self.benchmark_rmse[-1]

            min_idx = np.argmin(pls['Val RMSE'])
            pm[2, 1] = pls['p'][min_idx]
            rmse_idx = test.index[test['p'] == pm[2, 1]].tolist()[0]
            rmse = test['Val RMSE'][rmse_idx]
            rm[2] = rmse / self.benchmark_rmse[-1]

        # Iterate through each h step ahead values for all PCA. h = 1,3,6,12,24
        skip = 3
        skip2 = 7
        for idx, (aic, pls, test, pm, rm) in enumerate(
                zip(self.PCA_AIC_BIC, self.PCA_PLS, self.testset_PCA_PLS, self.pm_store, self.rm_store)):
            min_BIC_idx = np.argmin(aic['BIC_t'])
            pm[1 + skip, 0] = aic['m'][min_BIC_idx]
            pm[1 + skip, 1] = aic['p'][min_BIC_idx]
            rmse_idx = test.index[(test['m'] == pm[1 + skip, 0]) & (test['p'] == pm[1 + skip, 1])].tolist()[0]
            rmse = test['Val RMSE'][rmse_idx]
            rm[1 + skip2] = rmse / self.benchmark_rmse[idx]

            min_AIC_idx = np.argmin(aic['AIC_t'])
            pm[0 + skip, 0] = aic['m'][min_AIC_idx]
            pm[0 + skip, 1] = aic['p'][min_AIC_idx]
            rmse_idx = test.index[(test['m'] == pm[0 + skip, 0]) & (test['p'] == pm[0 + skip, 1])].tolist()[0]
            rmse = test['Val RMSE'][rmse_idx]
            rm[0 + skip2] = rmse / self.benchmark_rmse[idx]

            min_idx = np.argmin(pls['Val RMSE'])
            pm[2 + skip, 0] = pls['m'][min_idx]
            pm[2 + skip, 1] = pls['p'][min_idx]
            rmse_idx = test.index[(test['m'] == pm[2 + skip, 0]) & (test['p'] == pm[2 + skip, 1])].tolist()[0]
            rmse = test['Val RMSE'][rmse_idx]
            rm[2 + skip2] = rmse / self.benchmark_rmse[idx]

        # Iterate through each h step ahead values for all UMAP. h = 1,3,6,12,24
        skip = 3 * 2
        skip2 = 7 * 2
        for idx, (aic, pls, test, pm, rm) in enumerate(
                zip(self.UMAP_AIC_BIC, self.UMAP_PLS, self.testset_UMAP_PLS, self.pm_store, self.rm_store)):
            min_BIC_idx = np.argmin(aic['BIC_t'])
            pm[1 + skip, 0] = aic['m'][min_BIC_idx]
            pm[1 + skip, 1] = aic['p'][min_BIC_idx]
            rmse_idx = test.index[(test['m'] == pm[1 + skip, 0]) & (test['p'] == pm[1 + skip, 1])].tolist()[0]
            rmse = test['Val RMSE'][rmse_idx]
            rm[1 + skip2] = rmse / self.benchmark_rmse[idx]

            min_AIC_idx = np.argmin(aic['AIC_t'])
            pm[0 + skip, 0] = aic['m'][min_AIC_idx]
            pm[0 + skip, 1] = aic['p'][min_AIC_idx]
            rmse_idx = test.index[(test['m'] == pm[0 + skip, 0]) & (test['p'] == pm[0 + skip, 1])].tolist()[0]
            rmse = test['Val RMSE'][rmse_idx]
            rm[0 + skip2] = rmse / self.benchmark_rmse[idx]

            min_idx = np.argmin(pls['Val RMSE'])
            pm[2 + skip, 0] = pls['m'][min_idx]
            pm[2 + skip, 1] = pls['p'][min_idx]
            rmse_idx = test.index[(test['m'] == pm[2 + skip, 0]) & (test['p'] == pm[2 + skip, 1])].tolist()[0]
            rmse = test['Val RMSE'][rmse_idx]
            rm[2 + skip2] = rmse / self.benchmark_rmse[idx]

        pass

    def awa_bwa(self):
        """

        :param type: Either 'AIC_t' or 'BIC_t' for AWA and BWA respectively
        :return:
        """
        aic_bic_store = [self.AR_AIC_BIC, self.PCA_AIC_BIC, self.UMAP_AIC_BIC]
        testset_y_store = [self.testset_AR_y, self.testset_PCA_y, self.testset_UMAP_y]
        testset_y_hat_store = [self.testset_AR_y_hat, self.testset_PCA_y_hat, self.testset_UMAP_y_hat]
        self.testset_AR_AWA_y_hat = []
        self.testset_AR_BWA_y_hat = []
        self.testset_PCA_AWA_y_hat = []
        self.testset_PCA_BWA_y_hat = []
        self.testset_UMAP_AWA_y_hat = []
        self.testset_UMAP_BWA_y_hat = []

        for skip_idx ,(aic_bic, testset_y, testset_y_hat, awa_y_hat, bwa_y_hat)\
                in enumerate(zip(aic_bic_store, testset_y_store, testset_y_hat_store,
                                 [self.testset_AR_AWA_y_hat, self.testset_PCA_AWA_y_hat, self.testset_UMAP_AWA_y_hat],
                                 [self.testset_AR_BWA_y_hat, self.testset_PCA_BWA_y_hat, self.testset_UMAP_BWA_y_hat])):
            type = 'AIC_t'
            t_idx = 4 + 7 * skip_idx
            for idx, (ic, y, y_hat, rm) in enumerate(zip(aic_bic, testset_y, testset_y_hat, self.rm_store)):
                ic_values = ic[type].values
                min_ic = np.min(ic_values)
                ic_values += -min_ic
                weights = np.exp(-ic_values/2)
                weights = weights / np.sum(weights)
                y_combi_hat = np.sum(y_hat * weights[:, None], axis=0)
                awa_y_hat.append(y_combi_hat)
                rmse_combi = math.sqrt(np.mean(np.array(y-y_combi_hat) ** 2))
                rm[t_idx] = rmse_combi / self.benchmark_rmse[idx]

            type = 'BIC_t'
            t_idx = 5 + 7 * skip_idx
            for idx, (ic, y, y_hat, rm) in enumerate(zip(aic_bic, testset_y, testset_y_hat, self.rm_store)):
                ic_values = ic[type].values
                min_ic = np.min(ic_values)
                ic_values += -min_ic
                weights = np.exp(-ic_values/2)
                weights = weights / np.sum(weights)
                y_combi_hat = np.sum(y_hat * weights[:, None], axis=0)
                bwa_y_hat.append(y_combi_hat)
                rmse_combi = math.sqrt(np.mean(np.array(y-y_combi_hat) ** 2))
                rm[t_idx] = rmse_combi / self.benchmark_rmse[idx]


        pass





