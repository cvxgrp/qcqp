from setuptools import setup

setup(
    name='qcqp',
    version='0.1',
    author='Jaehyun Park, Stephen Boyd',
     author_email='jpark@cs.stanford.edu, boyd@stanford.edu',
    packages=['qcqp'],
    license='GPLv3',
    zip_safe=False,
    install_requires=["cvxpy >= 0.3.5",
                      "munkres",
                      "CVXcanon >= 0.0.22"],
    use_2to3=True,
    url='http://github.com/cvxgrp/qcqp/',
    description='A CVXPY extension for nonconvex quadratically constrained quadratic programs.',
)
