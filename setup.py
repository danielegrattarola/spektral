from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='spektral',
    version='0.6.1',
    packages=find_packages(),
    install_requires=['tensorflow>=2.1.0',
                      'networkx',
                      'pandas',
                      'lxml',
                      'joblib',
                      'numpy',
                      'scipy',
                      'requests',
                      'scikit-learn'],
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
