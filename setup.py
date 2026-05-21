from setuptools import find_packages, setup

setup(
    name='lstnn',
    version='1.0.0',
    description='Code for "Aligning Transformer Circuit Mechanisms to Neural Representations in Relational Reasoning"',
    author='Luke Hartleyp-Speirs et al.',
    url='https://github.com/anomalyco/LSTNN_public',
    license='MIT',
    packages=find_packages(include=['lstnn']),
    python_requires='>=3.10',
    install_requires=[
        'torch>=1.10.0',
        'numpy',
        'pandas',
        'scipy',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'tqdm',
        'joblib',
        'rsatoolbox',
        'pingouin',
        'statsmodels',
        'torchlens>=1.0,<2.0',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='transformer, fMRI, representational similarity analysis, reasoning',
)
