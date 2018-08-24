from setuptools import setup, find_packages

setup(
    name = 'simple-classifier',
    version = '0.0.1',
    url = 'https://github.com/pillyshi/simple-classifier',
    author = 'pillyshi',
    author_email = 'pillyshi21@gmail.com',
    description = 'An implementation of Simple Classifier',
    packages = find_packages(),
    install_requires = ['numpy', 'scikit-learn', 'scipy'],
)
