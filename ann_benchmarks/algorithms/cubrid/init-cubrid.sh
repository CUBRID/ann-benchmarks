#!/bin/bash
# First-run initialization for CUBRID in ann-benchmarks.
#
# The DB is created at container start (not at docker build) so the preallocated
# data/log volume files live in the overlayfs upper layer. This avoids the
# synchronous copy-up triggered by open(O_RDWR) that otherwise dominates server
# boot time (~125 MB/s copy of a 3 GB volume = ~25 s).
#
# Semantic parity with pgvector:
#   pgvector  -> initdb + CREATE USER + CREATE EXTENSION at docker build time
#   cubrid    -> createdb + CREATE USER at container first-start (equivalent
#                "one-shot setup" cost that is not included in fit()).
# At runtime, fit() only starts the server/broker -- same as pgvector's
# `service postgresql start`.

set -e

INITDB="${ANN_BENCHMARKS_CUB_DB_PATH}/initdb"
DBNAME="${ANN_BENCHMARKS_CUB_DBNAME}"
DBUSER="${ANN_BENCHMARKS_CUB_USER}"
DBPASS="${ANN_BENCHMARKS_CUB_PASSWORD}"

if [ ! -f "${INITDB}/${DBNAME}" ]; then
    echo "[init-cubrid] Creating database '${DBNAME}' under ${INITDB} ..."
    mkdir -p "${INITDB}"
    cd "${INITDB}"
    cubrid createdb --db-volume-size=3G --log-volume-size=1G "${DBNAME}" en_US.utf8

    cubrid server start "${DBNAME}"
    until cubrid server status | grep -q "Server ${DBNAME}"; do
        echo "[init-cubrid] Waiting for server '${DBNAME}' to start ..."
        sleep 1
    done

    csql -u dba "${DBNAME}" -c "CREATE USER ${DBUSER};"
    csql -u dba "${DBNAME}" -c "ALTER USER ${DBUSER} PASSWORD '${DBPASS}';"

    cubrid server stop "${DBNAME}"
    echo "[init-cubrid] Database '${DBNAME}' initialized."
else
    echo "[init-cubrid] Database '${DBNAME}' already exists, skipping init."
fi

cd /home/app
exec python -u run_algorithm.py "$@"
