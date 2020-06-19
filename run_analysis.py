from own_package.analysis import cw_analysis


def selector(case, **kwargs):
    if case == 1:
        cw_analysis(**kwargs)
        pass



selector(case=1, h_idx=0, h=1, m=3, p=12, rolling=True, results_dir='./results/testset_CMR',
         skip_first_val=True,
         excel_dir='./excel/CMR_data_loader.xlsx')
