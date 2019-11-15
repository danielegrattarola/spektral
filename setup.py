from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='spektral',
    version='0.1.1',
    packages=find_packages(),
    install_requires=['keras<2.3',
                      'tensorflow<2.0.0',
                      'networkx',
                      'pandas',
                      'joblib',
                      'pygraphviz',
                      'numpy',
                      'scipy',
                      'requests',
                      'scikit-learn'],
    url='https://github.com/danielegrattarola/spektral',
    license='MIT',
    author='Daniele Grattarola',
    author_email='daniele.grattarola@gmail.com',
    description='Graph Neural Networks with Keras and Tensorflow.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
)
