from distutils.core import setup

setup(
    name='Diagnoses_Analysis',
    version='0.1.0',
    description='Data analysis',
    author='Lokesh Dondapati',
    author_email='ldondapa@mail.yu.edu',
    license='MIT',
    url='https://github.com/LokeshDondapati/HIV-AIDS-Daignoses-Analysis',
    packages=['diagnoses_analysis'],
    install_requires=[
        'matplotlib>=3.0.2',
        'numpy>=1.15.2',
        'pandas>=0.23.4',
        'seaborn>=0.11.0',
        'scikit-learn>=1.3.2'
    ],
)
