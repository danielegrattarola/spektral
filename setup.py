from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='spektral',
    version='0.0.11',
    packages=find_packages(),
    install_requires=['keras', 'networkx', 'pandas', 'joblib', 'matplotlib',
                      'tqdm', 'pygraphviz', 'numpy', 'scipy', 'requests'],
    url='https://github.com/danielegrattarola/spektral',
    license='MIT',
    author='Daniele Grattarola',
    author_email='daniele.grattarola@gmail.com',
    description='A Python framework for relational representation learning',
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3.5",
    ],
)
