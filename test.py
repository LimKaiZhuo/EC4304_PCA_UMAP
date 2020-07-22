from own_package.boosting import run_testing

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

selector(case=1)
