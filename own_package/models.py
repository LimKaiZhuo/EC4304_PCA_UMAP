import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy, math
import time
import statsmodels.api as sm

from own_package.features_labels import Fl_master

class OLS:
    def __init__(self):
        self.model = None

    def train(self, x, y):
        self.model = sm.OLS(y, x)
        # What type of robust to use? There is HC0,1,2,3. Or should we use HAC?
        results = self.model.fit(cov_type='HC0')
        return results

    def eval(self, x):
        return self.model.eval(exog=x)




