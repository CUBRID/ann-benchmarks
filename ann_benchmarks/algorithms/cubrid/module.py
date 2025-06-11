"""
This module supports connecting to a CUBRID instance and performing vector
indexing and search using the pgvector extension. The default behavior uses
the "ann" value of CUBRID user name, password, and database name, as well
as the default host and port values of the psycopg driver.

If CUBRID is managed externally, e.g. in a cloud DBaaS environment, the
environment variable overrides listed below are available for setting CUBRID
connection parameters:

ANN_BENCHMARKS_CUB_USER
ANN_BENCHMARKS_CUB_PASSWORD
ANN_BENCHMARKS_CUB_DBNAME
ANN_BENCHMARKS_CUB_HOST
ANN_BENCHMARKS_CUB_PORT

This module starts the CUBRID service automatically using the "service"
command. The environment variable ANN_BENCHMARKS_CUB_START_SERVICE could be set
to "false" (or e.g. "0" or "no") in order to disable this behavior.

This module will also attempt to create the pgvector extension inside the
target database, if it has not been already created.
"""


import os
import subprocess
import sys
import time
import contextlib
import CUBRIDdb

from typing import Dict, Any, Optional

from ..base.module import BaseANN

METRIC_PROPERTIES = {
    "angular": {
        "distance_operator": "<c>",
        "ops_type": "COSINE",
    },
    "euclidean": {
        "distance_operator": "<->",
        "ops_type": "EUCLIDEAN",
    }
}


def get_cub_param_env_var_name(pg_param_name: str) -> str:
    return f'ANN_BENCHMARKS_CUB_{pg_param_name.upper()}'


def get_cub_conn_param(
        cub_param_name: str,
        default_value: Optional[str] = None) -> Optional[str]:
    env_var_name = get_cub_param_env_var_name(cub_param_name)
    env_var_value = os.getenv(env_var_name, default_value)
    if env_var_value is None or len(env_var_value.strip()) == 0:
        return default_value
    return env_var_value

@contextlib.contextmanager
def open_connection(host, port, database, user, password):
    url = f"CUBRID:{host}:{port}:{database}:::"
    conn = CUBRIDdb.connect(url, user, password or '')
    try:
        yield conn
    finally:
        conn.close()

@contextlib.contextmanager
def open_cursor(conn):
    cur = conn.cursor()
    try:
        yield cur
    finally:
        cur.close()

class CUBVEC(BaseANN):
    def __init__(self, metric, method_param):
        self._metric = metric
        self._m = method_param['M']
        self._ef_construction = method_param['efConstruction']
        self._cur = None

        if metric == "angular":
            self._query = "SELECT id FROM items ORDER BY embedding <c> %s LIMIT %s"
        elif metric == "euclidean":
            self._query = "SELECT id FROM items ORDER BY embedding <-> %s LIMIT %s"
        else:
            raise RuntimeError(f"unknown metric {metric}")

    def get_metric_properties(self) -> Dict[str, str]:
        """
        Get properties of the metric type associated with this index.

        Returns:
            A dictionary with keys distance_operator and ops_type.
        """
        if self._metric not in METRIC_PROPERTIES:
            raise ValueError(
                "Unknown metric: {}. Valid metrics: {}".format(
                    self._metric,
                    ', '.join(sorted(METRIC_PROPERTIES.keys()))
                ))
        return METRIC_PROPERTIES[self._metric]

    def fit(self, X):
        cubrid_connect_kwargs: Dict[str, Any] = dict(
            autocommit=True,
        )
        for arg_name in ['user', 'password', 'dbname']:
            # The default value is "ann" for all of these parameters.
            cubrid_connect_kwargs[arg_name] = get_cub_conn_param(
                arg_name, 'ann')

        # If host/port are not specified, leave the default choice to the
        # psycopg driver.
        cub_host: Optional[str] = get_cub_conn_param('host')
        if cub_host is not None:
            cubrid_connect_kwargs['host'] = cub_host

        cub_port_str: Optional[str] = get_cub_conn_param('port')
        if cub_port_str is not None:
            cubrid_connect_kwargs['port'] = int(cub_port_str)

        should_start_service = True
        if should_start_service:
            subprocess.run(
                "cubrid service start",
                shell=True,
                check=True,
                stdout=sys.stdout,
                stderr=sys.stderr)
        else:
            print(
                "Assuming that CUBRID service is managed externally. "
                "Not attempting to start the service.")
            
        conn = open_connection(cubrid_connect_kwargs['host'], cubrid_connect_kwargs['port'], cubrid_connect_kwargs['dbname'], cubrid_connect_kwargs['user'], cubrid_connect_kwargs['password'])
        cur = open_cursor(conn)
        cur.execute("DROP TABLE IF EXISTS items;")
        cur.execute("CREATE TABLE items (id int, embedding vector(%d));" % X.shape[1])
        print("creating index...")
        sys.stdout.flush()
        create_index_str = \
            "CREATE INDEX ON items(embedding %s) " \
            "WITH (m = %d, ef_construction = %d);" % (
                self.get_metric_properties()["ops_type"],
                self._m,
                self._ef_construction
            )
        cur.execute(create_index_str)
        print("copying data...")
        sys.stdout.flush()
        num_rows = 0
        insert_start_time_sec = time.time()
        for i, vec in enumerate(X):
            cur.execute("INSERT INTO items (id, embedding) VALUES (%d, %s);" % (i, vec))
            num_rows += 1
        insert_elapsed_time_sec = time.time() - insert_start_time_sec

        print("inserted {} rows into table in {:.3f} seconds".format(
            num_rows, insert_elapsed_time_sec))

        self._cur = cur

    def set_query_arguments(self, ef_search):
        self._ef_search = ef_search
        self._cur.execute("SET SYSTEM PARAMETERS 'hnsw_ef_search=%d'" % ef_search)

    def query(self, v, n):
        self._cur.execute(self._query, (v, n), binary=True, prepare=True)
        return [id for id, in self._cur.fetchall()]

    def get_memory_usage(self):
        # TODO: Implement this
        return 0

    def __str__(self):
        return f"CUBVEC(m={self._m}, ef_construction={self._ef_construction}, ef_search={self._ef_search})"
