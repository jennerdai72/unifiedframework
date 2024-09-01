# setup.py
from setuptools import setup, find_packages

setup(
    name='distmetrics',
    version='0.1',
    packages=find_packages(),
    description='A eXtensible Algebraic Framework from an Unified 6-Set Discrete Probability Algebra For Unified Algebraic Measures: Euclidean Distance, Entropy, Divergence, Wasserstein, MMD, Unified Composite Entropy, Unified Composite Measure, and other measurements for neural network outputs',
    long_description=open('README.md').read(),
    author='Li Dai',
    author_email='jennerdai@gmail.com',
    install_requires=[
        'numpy',             # Required for numerical operations
        'torch',             # PyTorch for neural network functionality
        'matplotlib'         # For plotting and visualizations
        'scipy'              # For wasserstein distance optimization
    ],
)
