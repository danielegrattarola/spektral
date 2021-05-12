#!/bin/bash

rm -r sources/
python autogen.py
mkdocs build
cp CNAME sources/
cp -r sources/.well-known/ site/
mkdocs gh-deploy
