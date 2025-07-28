"""
This module supports connecting to a CUBRID instance and performing vector
indexing and search. The default behavior uses the "ann" value of CUBRID user name, password, and database name.
and the default host and port values are localhost and 33000.

If CUBRID is managed externally, e.g. in a cloud DBaaS environment, the
environment variable overrides listed below are available for setting CUBRID
connection parameters:

ANN_BENCHMARKS_CUB_USER
ANN_BENCHMARKS_CUB_PASSWORD
ANN_BENCHMARKS_CUB_DBNAME
ANN_BENCHMARKS_CUB_HOST
ANN_BENCHMARKS_CUB_PORT
ANN_BENCHMARKS_CUB_SERVER_PORT
ANN_BENCHMARKS_CUB_NUM_CAS

This module starts the CUBRID server and broker automatically using the "cubrid"
command.
"""

import os
import subprocess
import sys
import time
import contextlib
import io
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

def get_cub_conn_param(cub_param_name: str, default_value: Optional[str] = None) -> Optional[str]:
    env_var_name = get_cub_param_env_var_name(cub_param_name)
    env_var_value = os.getenv(env_var_name, default_value)
    if env_var_value is None or len(env_var_value.strip()) == 0:
        return default_value
    return env_var_value
class CUBVEC(BaseANN):

    def done(self) -> None:
        print("### done")

    def __init__(self, metric, method_param):
        self._metric = metric
        self._m = method_param['M']
        self._ef_construction = method_param['efConstruction']
        self._cur = None
        self.is_prepared = False

        self._signature_base = metric + "_" + str(self._m) + "_" + str(self._ef_construction)
        self._signature = self._signature_base

        if metric == "angular":
            self._query = "SELECT /*+ no_parallel_heap_scan */ id FROM {} ORDER BY embedding <c> ? LIMIT 10"
        elif metric == "euclidean":
            self._query = "SELECT /*+ no_parallel_heap_scan */ id FROM {} ORDER BY embedding <-> ? LIMIT 10"
        else:
            raise RuntimeError(f"unknown metric {metric}")

    def get_metric_properties(self) -> Dict[str, str]:
        if self._metric not in METRIC_PROPERTIES:
            raise ValueError("Unknown metric: {}. Valid metrics: {}".format(
                self._metric, ', '.join(sorted(METRIC_PROPERTIES.keys()))))
        return METRIC_PROPERTIES[self._metric]

    def fit(self, X):
        self._prepare_signature(X)
        self._start_cubrid_services()
        conn = self._connect_to_db()
        cur = self._open_cursor_primitive(conn)

        try:
            self._prepare_object_files(X)
            if self._table_exists(cur, self._signature, X.shape[0]):
                print("Table already exists with correct row count. Skipping.")
                return

            self._create_table_and_index(cur, X.shape[1])
            self._insert_data(X)
        finally:
            cur.close()

        self._cur = self._open_cursor_primitive(conn)

    def set_query_arguments(self, ef_search):
        self._ef_search = ef_search
        self._cur.execute("SET SYSTEM PARAMETERS 'hnsw_ef_search=%d'" % ef_search)

        print("### preparing query...")
        query_str = self._query.format(self._signature)
        self._cur._cs.prepare(query_str)

    def query(self, v, n):
        vector_str = "[" + ",".join(map(str, v)) + "]"
        cur = self._cur

        # args = [vector_str, n] # this reduces QPS from 3500 to 600
        args = [vector_str]
        set_type = None
        if args is not None:
            cur._bind_params(args, set_type)
        r = cur._cs.execute()
        cur.rowcount = cur._cs.rowcount
        cur.description = cur._cs.description
        # return r

        res = [id for id, in cur.fetchall()]
        return res

    def _open_connection_primitive(self, host, port, database, user, password):
        url = f"CUBRID:{host}:{port}:{database}:::"
        return CUBRIDdb.connect(url, user, password or '')

    def _open_cursor_primitive(self, conn):
        return conn.cursor()

    def _connect_to_db(self):
        kwargs = { 'autocommit': True }
        for arg in ['user', 'password', 'dbname']:
                kwargs[arg] = get_cub_conn_param(arg, 'ann')

        host = get_cub_conn_param('host')
        if host: kwargs['host'] = host

        port = get_cub_conn_param('port')
        if port: kwargs['port'] = int(port)

        print(kwargs)
        return self._open_connection_primitive(kwargs.get('host', 'localhost'),
                                        kwargs.get('port', 33000),
                                        kwargs['dbname'],
                                        kwargs['user'],
                                        kwargs['password'])

    def _start_cubrid_services(self):
        try:
                subprocess.run(["cubrid", "server", "start", "ann"], check=True)
                print("CUBRID server 'ann' started.")
        except subprocess.CalledProcessError as e:
                print("Failed to start CUBRID server:", e)

        try:
                subprocess.run(["cubrid", "broker", "start"], check=True)
                print("CUBRID broker started.")
        except subprocess.CalledProcessError as e:
                print("Failed to start CUBRID broker:", e)

    def _prepare_signature(self, X):
        total_rows, dim = X.shape
        if self._signature == self._signature_base:
                self._signature += f"_{total_rows}_{dim}"

    def _prepare_object_files(self, X):
        total_rows, dim = X.shape
        batch_size = 50000
        header = f"%id {self._signature} 0\n%class {self._signature} ([id] [embedding])\n"

        for start in range(0, total_rows, batch_size):
            end = min(start + batch_size, total_rows)
            object_file_path = f"/tmp/{self._signature}_object_{start}_{end}"
            if os.path.exists(object_file_path + ".flag"):
                print(f"Skipping batch {start}-{end}")
                continue

            buffer = io.StringIO()
            lines = [
                f"{i} '[{','.join(map(str, vec))}]'\n"
                for i, vec in enumerate(X[start:end], start=start)
            ]
            buffer.write(header)
            buffer.writelines(lines)

            with open(object_file_path, "w") as f:
                f.write(buffer.getvalue())
            with open(object_file_path + ".flag", "w") as f:
                f.write("success")

            print(f"Prepared object file for batch {start}-{end}")

    def _table_exists(self, cur, table_name, expected_count) -> bool:
        try:
                cur.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = cur.fetchone()[0]
                return count == expected_count
        except Exception:
                return False

    def _create_table_and_index(self, cur, dim):
        print(f"Creating table and index: {self._signature}")
        cur.execute(f"DROP TABLE IF EXISTS {self._signature};")
        cur.execute(f"CREATE TABLE {self._signature} (id int, embedding vector({dim}));")

        idx_stmt = (
                "CREATE VECTOR INDEX idx_v ON %s(embedding %s) "
                "WITH (m = %d, ef_construction = %d);" % (
                self._signature,
                self.get_metric_properties()["ops_type"],
                self._m,
                self._ef_construction
                )
        )
        cur.execute(idx_stmt)

    def _insert_data(self, X):
        total_rows = X.shape[0]
        batch_size = 50000
        start_time = time.time()

        for start in range(0, total_rows, batch_size):
            end = min(start + batch_size, total_rows)
            object_file_path = f"/tmp/{self._signature}_object_{start}_{end}"
            try:
                subprocess.run([
                    "cubrid", "loaddb",
                    "-C", get_cub_conn_param('dbname', 'ann'),
                    "-u", "ann",
                    "-p", "ann",
                    "-d", object_file_path,
                    "-c", str(batch_size),
                    "--estimated-size", str(batch_size),
                    "--no-statistics",
                    "--no-user-specified-name"
                ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                print(f"Inserted rows {start}-{end}")
            except subprocess.CalledProcessError as e:
                print("loaddb failed with error:\n", e.stderr)
                raise

        print("Total insert time: {:.3f} sec".format(time.time() - start_time))

    def __str__(self):
        return f"CUBVEC(m={self._m}, ef_construction={self._ef_construction}, ef_search={self._ef_search})"
