#!/bin/bash

rm -r sources/
python autogen.py
python -m mkdocs serve
