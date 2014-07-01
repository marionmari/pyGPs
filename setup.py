
from setuptools import setup, find_packages

setup(
    name='pyGPs',
    version='1.1',
    author=['Marion Neumann',  'Shan Huang','Daniel Marthaler', 'Kristian Kersting'],
    author_email=['marion.neumann@uni-bonn.de.com', 'shan.huang@iais.fraunhofer.de', 'marthaler@ge.com', 'kristian.kersting@cs.tu-dortmund.de'],
    packages=find_packages(),
    url='https://github.com/marionmari/pyGPs',
    license='COPYRIGHT.txt',
    description='Gaussian Processes for Regression and Classification',
    long_description=open('README.md').read(),
    install_requires=[],
)
