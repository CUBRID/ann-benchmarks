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
ANN_BENCHMARKS_CUB_DB_PATH

This module starts the CUBRID server and broker automatically using the "cubrid"
command.
"""

import os
import subprocess
import sys
import time
import io
import CUBRIDdb
import shutil
import signal

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
    def __init__(self, metric, method_param):
        self._metric = metric
        self._m = method_param['M']
        self._ef_construction = method_param['efConstruction']
        self._cur = None

        # for statdump
        self._statdump_mode = True
        self._statdump_proc = None

        # for perf
        self._perf_pid = None

        if metric == "angular":
            self._query = "SELECT /*+ no_parallel_heap_scan */ id FROM items ORDER BY embedding <c> ? LIMIT 10"
        elif metric == "euclidean":
            self._query = "SELECT /*+ no_parallel_heap_scan */ id FROM items ORDER BY embedding <-> ? LIMIT 10"
        else:
            raise RuntimeError(f"unknown metric {metric}")

    def done(self) -> None:
        if self._perf_pid:
          self._stop_perf_marker("query")

        if self._statdump_mode:
          statdump = self._stop_and_collect_statdump("query")
          print(statdump)

        self._perf_pid = None

    def get_metric_properties(self) -> Dict[str, str]:
        if self._metric not in METRIC_PROPERTIES:
            raise ValueError("Unknown metric: {}. Valid metrics: {}".format(
                self._metric, ', '.join(sorted(METRIC_PROPERTIES.keys()))))
        return METRIC_PROPERTIES[self._metric]

    def fit(self, X):
        if self._perf_pid != None:
          self._stop_perf_marker("query")

        success = self._create_db(X)
        if success:
          print("Database created successfully")
        else:
          print("Failed to create database")

        # restart db to save vector index
        # self._start_cubrid_services("start")
        conn = self._connect_to_db()
        self._cur = self._open_cursor_primitive(conn)

        if self._statdump_mode:
          self._statdump_proc = self._run_statdump("query")
          if self._statdump_proc is None:
            print("Failed to run statdump for query")

        self._perf_pid = self.get_cubrid_server_pid("ann")
        self._start_perf_marker("query")

    def set_query_arguments(self, ef_search):
        self._ef_search = ef_search
        self._cur.execute("SET SYSTEM PARAMETERS 'hnsw_ef_search=%d'" % ef_search)

        self._cur._cs.prepare(self._query)

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

        rows = cur.fetchall()

        #for i, (id,) in enumerate(rows):
        #    if id is None:
        #        print(f"[DEBUG] NULL id at row {i}")

        return [id for (id,) in rows if id is not None]

    def _create_db(self, X):
        success = False
        try:
            print("Database does not exist. Creating new database...")

            self._start_cubrid_services("start")

            if self._statdump_mode:
              self._statdump_proc = self._run_statdump("build")
              if self._statdump_proc is None:
                print("Failed to run statdump for build")

            conn = self._connect_to_db()
            cur = self._open_cursor_primitive(conn)

            self._prepare_object_files(X)
            self._create_table_and_index(cur, X.shape[1])

            start_time = time.time()
            self._insert_data(X)

            print("Total inserting data time: {:.3f} sec".format(time.time() - start_time))

            self._perf_pid = self.get_cubrid_server_pid("ann")
            self._start_perf_marker("build")
            idx_stmt = (
                "CREATE VECTOR INDEX vidx_v ON items(embedding %s) "
                "WITH (m = %d, ef_construction = %d);" % (
                self.get_metric_properties()["ops_type"],
                self._m,
                self._ef_construction
                )
            )
            cur.execute(idx_stmt)

            print("Total building index time: {:.3f} sec".format(time.time() - start_time))

            if self._statdump_mode:
              statdump = self._stop_and_collect_statdump("build")
              print(statdump)

            self._stop_perf_marker("build")

            success = True
        finally:
            pass

        return success

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

    def _start_cubrid_services(self, command):
        try:
                subprocess.run(["cubrid", "server", command, "ann"], check=True)
                print("CUBRID server 'ann' started.")
        except subprocess.CalledProcessError as e:
                print("Failed to start CUBRID server:", e)

        try:
                subprocess.run(["cubrid", "broker", command], check=True)
                print("CUBRID broker started.")
        except subprocess.CalledProcessError as e:
                print("Failed to start CUBRID broker:", e)

    def _prepare_object_files(self, X):
        total_rows, dim = X.shape
        batch_size = 50000
        header = f"%id items 0\n%class items ([id] [embedding])\n"

        for start in range(0, total_rows, batch_size):
            end = min(start + batch_size, total_rows)
            object_file_path = f"/tmp/items_object_{start}_{end}"
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

    # for debugging
    def _table_exists_and_has_correct_count(self, cur, table_name, expected_count) -> bool:
        try:
                cur.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = cur.fetchone()[0]
                if int(count) == int(expected_count):
                    print(f"[REUSABLE] Table {table_name} exists with {count} rows")
                    return True
                else:
                    print(f"[NON-REUSABLE] Table {table_name} exists with {count} rows, expected {expected_count}")
                    return False
        except Exception: 
                print(f"[NON-REUSABLE] Table {table_name} does not exist")
                return False

    def _create_table_and_index(self, cur, dim):
        print(f"Creating table and index: items")
        cur.execute(f"DROP TABLE IF EXISTS items;")
        cur.execute(f"CREATE TABLE items (id int, embedding vector({dim}) );")
    
    # for debugging
    def _insert_data_sql(self, cur, X):
        total_rows = X.shape[0]

        for i in range(total_rows):
                vec = X[i]
                vector_str = "'[" + ",".join(map(str, vec)) + "]'"

                sql = (
                    f"INSERT INTO items "
                    f"VALUES ({i}, {vector_str})"
                )

                try:
                    cur.execute(sql)

                    if  i % 1000 == 0:
                        print(f"INSERT success at row {i}")

                except Exception as e:
                    print(f"INSERT failed at row {i}: {e}")
                    raise

    def _insert_data(self, X):
        total_rows = X.shape[0]
        batch_size = 50000

        print(f"test insert")

        for start in range(0, total_rows, batch_size):
            end = min(start + batch_size, total_rows)
            object_file_path = f"/tmp/items_object_{start}_{end}"
            try:
                subprocess.run([
                    "cubrid", "loaddb",
                    "-C", get_cub_conn_param('dbname', 'ann'),
                    "-u", "ann",
                    "-p", "ann",
                    "-d", object_file_path,
                    "-c", str(int(batch_size / 5)),
                    "--estimated-size", str(batch_size),
                    "--no-statistics",
                    "--no-user-specified-name"
                ], check=True)
                print(f"Inserted rows {start}-{end}")
            except subprocess.CalledProcessError as e:
                print("loaddb failed with error:\n", e.stderr)
                raise

    def _print_statdump(self, q_str):
        try:
            subprocess.run([
                "cubrid", "statdump",
                "-s", q_str,
                "-c",
                get_cub_conn_param('dbname', 'ann')
            ], check=True)
        except subprocess.CalledProcessError as e:
            print("statdump failed with error:\n", e.stderr)
            raise

    def _run_statdump(self, q_str):
        proc = subprocess.Popen(
            [
                "cubrid", "statdump",
                "-i", "10",
                "-c", 
                "-o", f"/tmp/statdump_{q_str}.txt",
                get_cub_conn_param('dbname', 'ann')
            ],
            text=True
        )

        return proc

    def _stop_and_collect_statdump(self, q_str):
        if self._statdump_proc is None:
            return "No statdump process"

        if self._statdump_proc.poll() is None:
            self._statdump_proc.send_signal(signal.SIGINT)
            self._statdump_proc.wait()

        with open(f"/tmp/statdump_{q_str}.txt", "r") as f:
            text = f.read()

        last = self._extract_last_block(text)
        if last is None:
            return "No text"
        filtered = self._filter_hnsw_page(last)

        # delete the file
        os.remove(f"/tmp/statdump_{q_str}.txt")

        self._statdump_proc = None
        return filtered

    def _extract_last_block(self, text):
        blocks = text.split("*** SERVER EXECUTION STATISTICS ***")

        if len(blocks) < 2:
            return None

        last = blocks[-1]
        return "*** SERVER EXECUTION STATISTICS ***" + last

    def _filter_hnsw_page(self, block: str) -> str:
        if not block:
            return ""

        lines = block.splitlines()
        filtered = [
            line for line in lines
            if ("hnsw" in line or "page" in line)
        ]
        return "\n".join(filtered)

    # for perf
    def _start_perf_marker(self, phase: str):
        pid = self._perf_pid
        print(f"[PERF_HINT] START phase={phase} pid={pid}", flush=True)

    # for perf
    def _stop_perf_marker(self, phase: str):
        pid = self._perf_pid
        print(f"[PERF_HINT] STOP phase={phase} pid={pid}", flush=True)

    # for perf
    def get_cubrid_server_pid(self, dbname):
        out = subprocess.check_output(
            ["cubrid", "server", "status"],
            text=True
        )
        for line in out.splitlines():
            if f"Server {dbname}" in line:
                # 예: Server ann (pid 12345)
                return int(line.split("pid")[1].strip(" )"))
        raise RuntimeError("CUBRID server PID not found")

    def __str__(self):
        return f"CUBVEC(m={self._m}, ef_construction={self._ef_construction}, ef_search={self._ef_search})"
