#!/bin/bash
mkdir -p logs
mkdir -p /tmp

touch logs/odbx_misc_access.log
touch logs/odbx_misc_error.log

tail -f -n 20 logs/odbx_misc_access.log logs/odbx_misc_error.log &

gunicorn \
    -w 1 \
    -k uvicorn.workers.UvicornWorker \
    --error-logfile logs/odbx_misc_error.log \
    --access-logfile logs/odbx_misc_access.log \
    --capture-output \
    --access-logformat "%(t)s: %(h)s %(l)s %(u)s %(r)s %(s)s %(b)s %(f)s %(a)s" \
    -b unix:/tmp/gunicorn.sock optimade_misc_datasets.application:app
