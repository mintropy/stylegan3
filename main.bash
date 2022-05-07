#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh

conda activate stylegan3-fastapi
python setup.py
uvicorn main:app --host 0.0.0.0 --port 8010