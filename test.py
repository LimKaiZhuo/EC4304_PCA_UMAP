from own_package.boosting import run_testing
from own_package.poos import poos_analysis

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
        poos_analysis('./results/poos/poos_IND/poos_h1.pkl')
selector(case=2)
