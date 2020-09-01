import numpy as np
import statsmodels.api as sm
import pmdarima as pm
from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.tsa.api import SARIMAX
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted


class LocalLevel(sm.tsa.statespace.MLEModel):
    _start_params = [1., 1.]
    _param_names = ['var.level', 'var.irregular']

    def __init__(self, endog):
        super(LocalLevel, self).__init__(endog, k_states=1, initialization='diffuse')

        self['design', 0, 0] = 1
        self['transition', 0, 0] = 1
        self['selection', 0, 0] = 1

    def transform_params(self, unconstrained):
        return unconstrained ** 2

    def untransform_params(self, unconstrained):
        return unconstrained ** 0.5

    def update(self, params, **kwargs):
        params = super(LocalLevel, self).update(params, **kwargs)

        self['state_cov', 0, 0] = params[0]
        self['obs_cov', 0, 0] = params[1]


class SSMBase(BaseEstimator):

    def __init__(self, mode):
        self.mode = mode

    def fit(self, X, y=None):
        # Perform transformation if specified by *transformation
        if '*ln' in self.mode:
            self.X = np.log(np.array(X) + 1)
            mode = self.mode.partition('*ln')[0]
        elif '*bc' in self.mode:
            transformer = pm.preprocessing.BoxCoxEndogTransformer()
            self.X = transformer.fit_transform(y=X)
            self.transformer = transformer
            mode = self.mode.partition('*bc')[0]
        else:
            self.X = X
            mode = self.mode
        try:
            if mode == 'll':
                # Local Level
                model = LocalLevel(self.X)
                self.res_ = model.fit(disp=False)
                self.k_exog = None
            elif mode == 'lla':
                endog = X[2:]
                exog = np.column_stack((X[1:-1], X[:-2]))
                self.k_exog = exog.shape[1]
                model = UnobservedComponents(endog=endog, exog=exog, level='local level')
                self.res_ = model.fit(disp=False)
            elif mode == 'lls':
                self.k_exog = None
                model = SARIMAX(endog=self.X, order=(2, 0, 0), trend='c', measurement_error=True)
                self.res_ = model.fit(disp=False)
            elif mode == 'llt':
                # Local Linear Trend
                model = UnobservedComponents(endog=self.X, level='local linear trend')
                self.res_ = model.fit(disp=False)
            elif mode == 'llc':
                # Local Level Cycle
                model = UnobservedComponents(endog=self.X, level='local level', cycle=True, stochastic_cycle=True)
                self.res_ = model.fit(disp=False)
            elif mode == 'arima':
                self.res_ = pm.auto_arima(self.X, start_p=1, start_q=1, start_P=1, start_Q=1,
                                      max_p=5, max_q=5, max_P=5, max_Q=5, seasonal=True,
                                      stepwise=True, suppress_warnings=True, D=10, max_D=10,
                                      error_action='ignore')
        except np.linalg.LinAlgError:
            # Some kalman filter error ==> Use random walk
            print(f'Convergence failed for {mode}')
            self.converged = False
            return self
        try:
            self.converged = self.res_.mle_retvals['converged']
        except AttributeError:
            self.converged = True  # auto ARIMA from pmdarima should always converge
        return self

    def predict(self, X=None):
        # Get one step ahead prediction
        # Check is fit had been called
        #check_is_fitted(self)
        if not self.converged:
            # print('MLE convergence failed. Returning random walk forecast.')
            return self.X[-1]
        if 'lla' in self.mode:
            ret = self.res_.forecast(exog=self.X[-self.k_exog:])[0]
        elif 'arima' in self.mode:
            ret = self.res_.predict(n_periods=1, return_conf_int=False)
        else:
            ret = self.res_.forecast()[0]

        if '*ln' in self.mode:
            return np.exp(1) ** ret - 1
        elif '*bc' in self.mode:
            return self.transformer.inverse_transform(ret)
        else:
            return ret

    def score(self):
        pass
