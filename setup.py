from distutils.core import setup

setup(
    name='pyGP_OO',
    version='1.0.0',
    author=['Marion Neumann',  'Shan Huang','Daniel Marthaler', 'Kristian Kersting'],
    author_email=['marion.neumann@uni-bonn.de.com', 'shan.huang@iais.fraunhofer.de', 'marthaler@ge.com', 'kristian.kersting@cs.tu-dortmund.de'],
    packages=['pyGP_OO','pyGP_OO.Optimization','pyGP_OO.Valid','pyGP_OO.Core'],
    url='https://github.com/marionmari/pyGP_OO',
    license='COPYRIGHT.txt',
    description='Functional Gaussian Processes',
    long_description=open('README.md').read(),
    install_requires=[
        "Python >= 2.6",
        "Numpy >= 1.7.1",
        "Scipy >= 0.12.0",
        "matplotlib >= 1.2.1",
    ],
)
