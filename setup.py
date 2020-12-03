from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='spektral',
    version='1.0.2',
    packages=find_packages(),
    install_requires=['joblib',
                      'lxml',
                      'networkx',
                      'numpy',
                      'pandas',
                      'requests',
                      'scikit-learn',
                      'scipy',
                      'tensorflow>=2.1.0',
                      'tqdm'],
    url='https://github.com/danielegrattarola/spektral',
    license='MIT',
    author='Daniele Grattarola',
    author_email='daniele.grattarola@gmail.com',
    description='Graph Neural Networks with Keras and Tensorflow 2.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3.5"
    ],
)
