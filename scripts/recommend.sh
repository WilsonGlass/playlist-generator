#!/usr/bin/env bash

if [ -z "$1" ]; then
  echo "usage: ./recommend.sh \"your playlist prompt\"" >&2
  exit 1
fi

PROMPT="$1"
curl -X POST \
  -H 'Content-Type: application/json' \
  -d "{\"prompt\":\"$PROMPT\"}" \
  "${URL:-http://localhost:8000/recommend}"
echo
