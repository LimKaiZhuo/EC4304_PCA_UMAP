import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.linear_model import LinearRegression
from statsmodels.nonparametric.kernel_regression import KernelReg
import xgboost as xgb
from own_package.others import create_results_directory, set_matplotlib_style


def selector(case):
    if case == 1:
        results_dir = create_results_directory('./results/paper/dtr_vs_xgb')
        x, y = load_boston(return_X_y=True)
        x = pd.DataFrame(x, columns=['crime', 'zn', 'indus', 'chas', 'nox', 'rm', 'age',
                                     'dis', 'rad', 'tax', 'ptratio', 'blacks', 'lstat'])
        x = x[['rm', 'lstat']]
        df_all = x.copy()
        df_all['price'] = y

        # Plot 3D scatter
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(df_all['rm'], df_all['lstat'], df_all['price'])
        ax.view_init(30, 135)
        plt.savefig(f'{results_dir}/scatter.png')
        plt.close()

        dtr = DecisionTreeRegressor(max_depth=2)
        dtr.fit(x, y)
        plot_tree(dtr, impurity=False)
        plt.savefig(f'{results_dir}/dtr_visual.png')
        plt.close()

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
        plt.close()

        # Linear regression
        lr = LinearRegression().fit(x,y)
        z = lr.predict(points).reshape(rm.shape)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(rm, lstat, z, cmap=plt.cm.BuGn, linewidth=0.2, vmin=-50)
        ax.view_init(30, 135)
        plt.savefig(f'{results_dir}/lr_prediction.png')
        plt.close()

        # Linear regression
        kr = KernelReg(exog=x, endog=y, var_type='cc')
        z = kr.fit(points)[0].reshape(rm.shape)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(rm, lstat, z, cmap=plt.cm.BuGn, linewidth=0.2, vmin=-50)
        ax.view_init(30, 135)
        plt.savefig(f'{results_dir}/kr_prediction.png')
        plt.close()

        # XGB
        hparams = {'seed': 42,
                   'booster': 'gbtree',
                   'learning_rate': 0.1,
                   'objective': 'reg:squarederror',
                   'verbosity': 0,
                   'subsample': 1,
                   'max_depth': 2,
                   'colsample_bytree': 0.5,
                   }
        dtrain = xgb.DMatrix(x.values, label=y)
        model = xgb.train(hparams, dtrain=dtrain, num_boost_round=100, verbose_eval=False)
        z_xgb = model.predict(xgb.DMatrix(points)).reshape(rm.shape)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(rm, lstat, z_xgb, cmap=plt.cm.BuGn, linewidth=0.2, vmin=-50)
        ax.view_init(30, 135)
        plt.savefig(f'{results_dir}/xgb_prediction.png')

selector(case=1)
