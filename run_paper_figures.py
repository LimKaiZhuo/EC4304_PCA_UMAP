import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.tree import DecisionTreeRegressor, plot_tree
import xgboost as xgb
from own_package.others import create_results_directory


def selector(case):
    if case == 1:
        results_dir = create_results_directory('./results/paper/dtr_vs_xgb')
        x, y = load_boston(return_X_y=True)
        x = pd.DataFrame(x, columns=['crime', 'zn', 'indus', 'chas', 'nox', 'rm', 'age',
                                     'dis', 'rad', 'tax', 'ptratio', 'blacks', 'lstat'])
        dtr = DecisionTreeRegressor(max_depth=2)
        dtr.fit(x[['rm', 'lstat']], y)
        plt.close()
        plot_tree(dtr, impurity=False)
        plt.savefig(f'{results_dir}/dtr_visual.png')

        x_min = x.min(axis=0)
        x_max = x.max(axis=0)

        rm_linspace = np.linspace(x_min['rm'], x_max['rm'], 100)
        lstat_linspace = np.linspace(x_min['lstat'], x_max['lstat'], 100)

        rm, lstat = np.meshgrid(rm_linspace, lstat_linspace)
        points = np.stack(map(np.ravel, (rm, lstat)), axis=1)
        z = dtr.predict(points).reshape(rm.shape)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(rm, lstat, z, cmap=plt.cm.BuGn, linewidth=0.2, vmin=-50)
        ax.view_init(30, 135)
        plt.savefig(f'{results_dir}/dtr_prediction.png')

        hparams = {'seed': 42,
                   'booster': 'gbtree',
                   'learning_rate': 0.1,
                   'objective': 'reg:squarederror',
                   'verbosity': 0,
                   'subsample': 1,
                   'max_depth': 2,
                   'colsample_bytree': 0.5,
                   }
        dtrain = xgb.DMatrix(x[['rm', 'lstat']].values, label=y)
        model = xgb.train(hparams, dtrain=dtrain, num_boost_round=100, verbose_eval=False)
        z_xgb = model.predict(xgb.DMatrix(points)).reshape(rm.shape)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(rm, lstat, z_xgb, cmap=plt.cm.BuGn, linewidth=0.2, vmin=-50)
        ax.view_init(30, 135)
        plt.savefig(f'{results_dir}/xgb_prediction.png')

selector(case=1)
