#!/bin/bash

# For local testing

export ANN_BENCHMARKS_CUB_USER=ann
export ANN_BENCHMARKS_CUB_PASSWORD=ann
export ANN_BENCHMARKS_CUB_DBNAME=ann
export ANN_BENCHMARKS_CUB_HOST=localhost
export ANN_BENCHMARKS_CUB_SERVER_PORT=1523
export ANN_BENCHMARKS_CUB_PORT=33000
export ANN_BENCHMARKS_CUB_NUM_CAS=50

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

cubrid server restart ann && cubrid broker restart