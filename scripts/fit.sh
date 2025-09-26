#!/usr/bin/env bash

curl -X POST \
  -H 'Content-Type: application/json' \
  "${URL:-http://localhost:8000/fit}"
echo