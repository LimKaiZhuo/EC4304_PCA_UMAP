from own_package.boosting import run_testing
from own_package.poos import poos_analysis, poos_processed_data_analysis
from own_package.features_labels import read_excel_dataloader, Fl_master

class A(object):     # deriving from 'object' declares A as a 'new-style-class'
    def foo(self):
        print('foo')

class B(A):
    def doo(self):
        super().foo()   # calls 'A.foo()'
        print('du')

def selector(case, **kwargs):
    if case == 1:
        run_testing()
    elif case == 2:
        excel_dir = kwargs['excel_dir']
        output = read_excel_dataloader(excel_dir=excel_dir)
        fl_master = Fl_master(x=output[0], features_names=output[1],
                              yo=output[2], labels_names=output[3],
                              y=output[4], y_names=output[5],
                              time_stamp=output[6])
        poos_analysis(fl_master=fl_master, h=1, h_idx=0, model_mode='xgb',
                      results_dir='./results/poos/poos_W875',
                      save_dir='./results/poos/poos_W875/poos_h1.pkl')
    elif case == 3:
        poos_processed_data_analysis(save_dir='./results/poos/poos_W875/poos_xgb_h1_analysis_results.pkl')

if __name__ == '__main__':
    selector(case=3, excel_dir='./excel/dataset2/W875RX1_data_loader.xlsx')
