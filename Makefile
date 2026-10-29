# Works with GNU make and mingw32-make (Windows). Recipes avoid shell-specific syntax.
# Keep targets in sync with CLAUDE.md "Commands".

ifeq ($(OS),Windows_NT)
PY := $(CURDIR)/.venv/Scripts/python.exe
else
PY := $(CURDIR)/.venv/bin/python
endif

.PHONY: help setup data train evaluate test lint api app mlflow docker-up

help:
	@echo Targets: setup data train evaluate test lint api app mlflow docker-up

setup:
	uv venv --python 3.11 --allow-existing .venv || python -m venv .venv
	uv pip install --python "$(PY)" -e ".[dev]" || "$(PY)" -m pip install -e ".[dev]"

data:
	"$(PY)" scripts/download_data.py
	"$(PY)" -m churnguard.data.prepare

# Stubs below fail loudly until their block lands (see BUILD_PROMPTS.md).
train:
	"$(PY)" scripts/not_implemented.py train "Blocks 3-4"

evaluate:
	"$(PY)" scripts/not_implemented.py evaluate "Block 4"

test:
	"$(PY)" -m pytest

lint:
	"$(PY)" -m ruff check . && "$(PY)" -m black --check .

api:
	"$(PY)" scripts/not_implemented.py api "Block 6"

app:
	"$(PY)" scripts/not_implemented.py app "Block 7"

mlflow:
	"$(PY)" -m mlflow ui --backend-store-uri ./mlruns

docker-up:
	"$(PY)" scripts/not_implemented.py docker-up "Block 8"
