#!/usr/bin/env bash

# python3 -m unittest -v tests/test_s2_tiles_supres.py
python3 -m pytest --pylint --pylint-rcfile=../../pylintrc --mypy --mypy-ignore-missing-imports --cov=src/