from setuptools import setup, find_packages

setup(
    name='demba',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'opencv-python',
        'torch',
        'scikit-learn',
        'matplotlib',
        'tqdm',
    ],
    python_requires='>=3.8',
)