from own_package.analysis import cw_analysis, xgb_analysis, prepare_grand_data_store, combine_rmse_results, \
    combine_best_summary_and_xgbs
import itertools


def selector(case, **kwargs):
    if case == 0:
        data_store = prepare_grand_data_store('./results/testset_CMR11/AR', model='AR')
    elif case == 1:
        d = 2
        h = 12
        h_idx = 3
        cw_analysis(case=1, h_idx=h_idx, h=h, m=3, p=12, rolling=True,
                    results_dir=f'./results/testset_W875test_dropout method 1 lr0.1/CWd2_h12',
                    z_type=d,
                    skip_first_val=False,
                    excel_dir='./excel/dataset2/W875RX1_data_loader.xlsx')
        pass
    elif case == 1.1:
        xgb_analysis('./results/testset_INDxgbd_rep_holdout_9')
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
    elif case == 4:
        best_summary_dir = './results/testset_CPIA3/best summary.xlsx'
        xgbs_dir = './results/testset_CPIAxgb/best_xgb.xlsx'
        results_dir = './results/testset_CPIA3'

        best_summary_dir = './results/testset_IND/best summary.xlsx'
        xgbs_dir = './results/testset_INDxgb/best_xgb.xlsx'
        results_dir = './results/testset_IND'

        #best_summary_dir = './results/testset_W875/best summary.xlsx'
        #xgbs_dir = './results/testset_W875xgb/best_xgb.xlsx'
        #results_dir = './results/testset_W875'

        combine_best_summary_and_xgbs(best_summary_dir=best_summary_dir,
                                      xgbs_dir=xgbs_dir,
                                      results_dir=results_dir)


selector(1.1)
