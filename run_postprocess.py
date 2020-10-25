from own_package.postprocess import Postdata, compile_pm_rm_excel, get_final_submission_excel
import pickle


def selector(case, var_name=None, **kwargs):
    if case == 1:
        results_dir = './results/{} Done'.format(var_name)
        print('Post data processing for {}'.format(var_name))
        post = Postdata(results_dir=results_dir, var_name=var_name, star=False)
        post.combination()
        # with open('{}/{}_data.pkl'.format(results_dir, var_name), 'wb') as output:
        #   pickle.dump(post, output, pickle.HIGHEST_PROTOCOL)
    elif case == 2:
        extension = kwargs['extensions']
        excel_dir_store = ['./results/W875RX1 Done/pm_rm_results.xlsx', './results/WPSFD49207 Done/pm_rm_results.xlsx',
                           './results/IND Done/pm_rm_results.xlsx', './results/PAY Done/pm_rm_results.xlsx',
                           './results/CMR Done/pm_rm_results.xlsx', './results/CPIAUCSL Done/pm_rm_results.xlsx',
                           './results/CPIULFSL Done/pm_rm_results.xlsx', './results/DPC Done/pm_rm_results.xlsx']

        excel_dir_store = ['./results/IND Done/pm_rm_results{}.xlsx'.format(extension),
                           './results/DPC Done/pm_rm_results{}.xlsx'.format(extension),
                           './results/CMR Done/pm_rm_results{}.xlsx'.format(extension),
                           './results/PAY Done/pm_rm_results{}.xlsx'.format(extension),
                           './results/CPIAUCSL Done/pm_rm_results{}.xlsx'.format(extension),
                           './results/W875RX1 Done/pm_rm_results{}.xlsx'.format(extension),
                           './results/CPIULFSL Done/pm_rm_results{}.xlsx'.format(extension),
                           './results/WPSFD49207 Done/pm_rm_results{}.xlsx'.format(extension)]
        compile_pm_rm_excel(excel_dir_store)
    elif case == 3:
        results_dir = './results/{} Done'.format(var_name)
        post = Postdata(results_dir=results_dir, var_name=var_name)
        post.combination()
        pm = post.pm_store
        for hstep in range(5):
            hsteppm = pm[hstep]
            for i in range(9):
                pmparams = hsteppm[i]
    elif case == 4:
        var_name = 'IND'
        get_final_submission_excel(excel_dir=f'./results/expt1/a_Final_submission_expt1/final_table_{var_name}.xlsx',
                                   read_excel_dir=f'./results/expt1/a_Final_submission_expt1/combined_poos_results_{var_name}.xlsx')


'''
selector(1, var_name='CPIAUCSL')
selector(1, var_name='CMR')
selector(1, var_name='CPIULFSL')
selector(1, var_name='DPC')
selector(1, var_name='IND')
selector(1, var_name='PAY')
selector(1, var_name='WPSFD49207')
selector(1, var_name='W875RX1')
selector(2, extensions=' - 4')
'''
selector(case=4)

