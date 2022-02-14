from setuptools import setup

setup(
    name='supplement_package',
    version='0.0.45',
    packages=['supplement_package'],
    install_requires=[
        'pandas',
        'numpy',
        'gurobipy'
    ],
)
