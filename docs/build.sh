#!/bin/bash

rm -r sources/
python autogen.py
mkdocs build
cp CNAME sources/
mkdocs gh-deploy
