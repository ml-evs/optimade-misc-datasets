#!/bin/bash
set -e
cd camd_tri_dataset && python utils.py && gunicorn -w 1 -k uvicorn.workers.UvicornWorker --timeout 600 "app:app"
