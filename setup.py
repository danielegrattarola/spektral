from setuptools import find_packages, setup
import platform

requirements = [
    "joblib",
    "lxml",
    "networkx",
    "numpy",
    "pandas",
    "requests",
    "scikit-learn",
    "scipy",
    "tqdm",
]

if platform.processor() != "arm":
    requirements.append("tensorflow>=2.2.0")
else:
    requirements.append("tensorflow-macos>=2.5.0")

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="spektral",
    version="1.2.0",
    packages=find_packages(),
    install_requires=requirements,
    url="https://github.com/danielegrattarola/spektral",
    license="MIT",
    author="Daniele Grattarola",
    author_email="daniele.grattarola@gmail.com",
    description="Graph Neural Networks with Keras and Tensorflow 2.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
