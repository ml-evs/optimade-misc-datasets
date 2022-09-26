#!/bin/bash
set -e
cd optimade_misc_datasets && gunicorn -w 1 -k uvicorn.workers.UvicornWorker --timeout 600 "application:app"
