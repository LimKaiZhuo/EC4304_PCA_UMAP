
from own_package.pre_processing import create_data_loader_excel, type_transformations


def selector(case, **kwargs):
    if case == 0:
        excel_dir = './excel/2020-07.xlsx'
        type_transformations(excel_dir=excel_dir, results_dir='./excel/dataset_0720',
                             y_selection=['W875RX1', 'DPCERA3M086SBEA', 'CMRMTSPLx', 'INDPRO',
                                          'PAYEMS', 'WPSFD49207', 'CPIAUCSL', 'CPIULFSL'],
                             h_steps=[1, 3, 6, 12, 24])
    elif case == 1:
        # Create excel loaders
        create_data_loader_excel(excel_dir='./excel/check/dataset_filled.xlsx', results_dir='./excel/check')


selector(0)

