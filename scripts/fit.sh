#!/usr/bin/env bash
# POST /fit

curl -X POST \
  -H 'Content-Type: application/json' \
  "${URL:-http://localhost:8000/fit}"
echo