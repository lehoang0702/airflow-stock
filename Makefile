.PHONY: up down build test lint logs clean

up:
	docker compose up -d

down:
	docker compose down

build:
	docker compose build

test:
	pytest tests/ -v

lint:
	python -m py_compile dags/*.py dashboard/app.py tests/*.py

logs:
	docker compose logs -f --tail=100

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name '*.pyc' -delete 2>/dev/null || true
