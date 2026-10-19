.PHONY: up down build test test-docker lint logs clean

up:
	docker compose up -d

down:
	docker compose down

build:
	docker compose build

test:
	pytest tests/ -v

test-docker:
	docker run --rm --network none --entrypoint python \
		-e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/opt/airflow:/opt/airflow/dags \
		-e AIRFLOW_HOME=/tmp/airflow-tests \
		-e AIRFLOW__LOGGING__BASE_LOG_FOLDER=/tmp/airflow-tests/logs \
		-e AIRFLOW__LOGGING__DAG_PROCESSOR_MANAGER_LOG_LOCATION=/tmp/airflow-tests/logs/manager.log \
		-v "$(CURDIR):/opt/airflow:ro" custom-airflow-xgboost:latest \
		-m pytest /opt/airflow/tests -v -p no:cacheprovider

lint:
	python -m py_compile dags/*.py dashboard/*.py tests/*.py

logs:
	docker compose logs -f --tail=100

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name '*.pyc' -delete 2>/dev/null || true
