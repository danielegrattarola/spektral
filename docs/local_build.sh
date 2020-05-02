#!/bin/bash

# install current branch
cd ../
# python setup.py install
cd docs/

# delete old docs
rm -r sources/

# generate new docs
python autogen.py

# serve new docs
python -m mkdocs serve
