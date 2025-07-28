#!/bin/bash

# For local testing setup
# This script is used to setup the CUBRID server and broker for local testing.
#   python run.py --algorithm cubrid --local

export ANN_BENCHMARKS_CUB_USER=ann
export ANN_BENCHMARKS_CUB_PASSWORD=ann
export ANN_BENCHMARKS_CUB_DBNAME=ann
export ANN_BENCHMARKS_CUB_HOST=localhost
export ANN_BENCHMARKS_CUB_SERVER_PORT=1523
export ANN_BENCHMARKS_CUB_PORT=33000
export ANN_BENCHMARKS_CUB_NUM_CAS=50
export ANN_BENCHMARKS_CUB_DB_PATH=/tmp/ann

# to create a bare database, run the following command:
# mkdir -p $ANN_BENCHMARKS_CUB_DB_PATH/initdb && cd $ANN_BENCHMARKS_CUB_DB_PATH/initdb && \
# cubrid createdb --db-volume-size=100M --log-volume-size=100M ${ANN_BENCHMARKS_CUB_DBNAME} en_US.utf8

echo "[@ann]" >> $CUBRID/conf/cubrid.conf
echo "data_buffer_size=16G" >> $CUBRID/conf/cubrid.conf
sed -i "s/^cubrid_port_id *= *.*/cubrid_port_id = ${ANN_BENCHMARKS_CUB_SERVER_PORT}/" $CUBRID/conf/cubrid.conf

awk ' \
BEGIN { skip=0 } \
/^\[%query_editor\]/ { skip=1; next } \
/^\[/ && skip { skip=0 } \
!skip \
' $CUBRID/conf/cubrid_broker.conf > /tmp/cubrid_broker.conf
mv /tmp/cubrid_broker.conf $CUBRID/conf/cubrid_broker.conf

sed -i "s/^MIN_NUM_APPL_SERVER[ \t]*=.*/MIN_NUM_APPL_SERVER = ${ANN_BENCHMARKS_CUB_NUM_CAS}/" $CUBRID/conf/cubrid_broker.conf
sed -i "s/^MAX_NUM_APPL_SERVER[ \t]*=.*/MAX_NUM_APPL_SERVER = ${ANN_BENCHMARKS_CUB_NUM_CAS}/" $CUBRID/conf/cubrid_broker.conf
sed -i "s/^BROKER_PORT[ \t]*=.*/BROKER_PORT = ${ANN_BENCHMARKS_CUB_PORT}/" $CUBRID/conf/cubrid_broker.conf

cubrid server restart ${ANN_BENCHMARKS_CUB_DBNAME} && cubrid broker restart
