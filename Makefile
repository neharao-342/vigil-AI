.RECIPEPREFIX = >
.PHONY: help install dev up down validate test lint fmt

help:
>@echo "install  - install python deps (editable + dev)"
>@echo "up       - docker compose up -d"
>@echo "down     - docker compose down"
>@echo "validate - run repo structure + DAG validator"
>@echo "test     - run pytest"
>@echo "lint     - ruff check"
>@echo "fmt      - ruff format"

install:
>pip install -e ".[dev]"

dev:
>uvicorn server.main:app --reload --host 0.0.0.0 --port 8000

up:
>docker compose up -d

down:
>docker compose down

validate:
>python tools/validate.py

test:
>pytest -q

lint:
>ruff check .

fmt:
>ruff format .
