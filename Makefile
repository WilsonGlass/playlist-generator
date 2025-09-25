# Makefile at playlist-generator/

PYTHON = python
BACKEND_DIR = backend
APP = src.api.api

.PHONY: run

# Default target: run the API with backend as PYTHONPATH
run:
	PYTHONPATH=$(BACKEND_DIR) $(PYTHON) -m $(APP)

.DEFAULT_GOAL := run
