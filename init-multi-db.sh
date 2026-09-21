#!/bin/bash
# Tạo thêm database MLflow khi PostgreSQL khởi tạo lần đầu.
# File được mount vào /docker-entrypoint-initdb.d/ của container postgres.

set -e

if [ -n "$POSTGRES_MULTIPLE_DATABASES" ]; then
    for db in $(echo "$POSTGRES_MULTIPLE_DATABASES" | tr ',' ' '); do
        echo "  Creating database '$db'..."
        psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
            SELECT 'CREATE DATABASE $db'
            WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '$db')\gexec
EOSQL
    done
fi
