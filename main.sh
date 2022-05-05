#!/bin/sh
conda activate
conda activate stylegan3-fastapi
uvicorn main:app --host 0.0.0.0 --port 8010