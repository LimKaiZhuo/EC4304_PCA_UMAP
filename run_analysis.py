from own_package.analysis import cw_analysis, prepare_grand_data_store, combine_rmse_results
import itertools


def selector(case, **kwargs):
    if case == 0:
        data_store = prepare_grand_data_store('./results/testset_CMR11/AR', model='AR')
    elif case == 1:
        d = 1
        h = 6
        h_idx = 0
        cw_analysis(case=1, h_idx=h_idx, h=h, m=12, p=24, rolling=True,
                    results_dir=f'./results/testset_W8752/CWd1_h12',
                    z_type=d,
                    skip_first_val=False,
                    excel_dir='./excel/dataset2/W875RX1_data_loader.xlsx')
        pass
    elif case == 2:
        d = [6]
        h_store = zip([1, 3, 6, 12, 24], [0, 1, 2, 3, 4])
        params = itertools.product(d, h_store)

        for d, (h, h_idx) in params:
            cw_analysis(case=1, h_idx=h_idx, h=h, m=3, p=12, rolling=True,
                        results_dir=f'./results/testset_CPIA/CWd{d}_h{h}',
                        z_type=d,
                        skip_first_val=False,
                        excel_dir='./excel/dataset2/INDPRO_data_loader.xlsx')
    elif case == 3:
        combine_rmse_results('./results/testset_W875')


selector(1)
