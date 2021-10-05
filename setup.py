from setuptools import setup
import os

with open("README.md", "r") as fh:
    long_description = fh.read()


license="GPL"
packages_dir = 'pyMBIR'
packages = [packages_dir]
MAKE_PATH = os.path.join('.',packages_dir,'prior_model')
os.system('make -C '+ MAKE_PATH)
exec_file1 = 'prior_model/mrf3d_grad_linear.so'
exec_file2 = 'prior_model/mrf_prior.py'

setup(
    name='pyMBIR',
    version='0.1',
    description='Python code for model-based tomographic reconstruction for different source types and scan geometries',
    license=license,
    author='S.V. Venkatakrishnan',
    author_email='venkatakrisv@ornl.gov',
    url='https://github.com/svvenkatakrishnan/pyMBIR',
    packages=packages,
    package_data={'pyMBIR': [exec_file1,exec_file2]},
    install_requires=['astra-toolbox','numpy', 'scipy', 'matplotlib', 'futures', 'psutil']
)
