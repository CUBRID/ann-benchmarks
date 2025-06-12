FROM ann-benchmarks

# https://github.com/pgvector/pgvector/blob/master/Dockerfile

RUN git clone https://github.com/cubrid/cubrid /tmp/cubrid
RUN git clone https://github.com/cubrid/cubrid-python /tmp/cubrid-python

RUN DEBIAN_FRONTEND=noninteractive apt-get -y install tzdata
RUN apt-get update && apt-get install -y software-properties-common && add-apt-repository ppa:ubuntu-toolchain-r/test && apt-get update
RUN apt-get install -y --no-install-recommends build-essential
RUN apt-get install -y --no-install-recommends wget git cmake systemtap systemtap-sdt-dev bison flex
RUN apt-get install -y --no-install-recommends gcc-9 g++-9 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 90 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 90
RUN apt-get install -y --no-install-recommends libopenblas-dev liblapack-dev

RUN PRESET=release_avx2 && \
    BUILD_DIR="build_preset_${PRESET}" && \
    INSTALL_DIR="/tmp/cubrid_install" && \
    cd /tmp/cubrid && \
    git checkout cubvec/cubvec && \
    cmake --preset "$PRESET" && \
    cmake --build --preset "$PRESET" -j$(nproc) && \
    cmake --install "$BUILD_DIR" --prefix "$INSTALL_DIR"

# do we need to run as cubrid?
# USER cubrid
USER root
RUN cd /tmp/cubrid-python && \
    # hacky way to fix submodule
    git submodule set-url cci-src https://github.com/CUBRID/cubrid-cci.git && \
    git submodule update --init --recursive && \
    python setup_3.py build && \
    python setup_3.py install

RUN cd /tmp && \
    cubrid createdb --db-volume-size=100M --log-volume-size=100M ann en_US.utf8 && \
    echo "[ann]" >> $CUBRID/conf/cubrid.conf && \
    echo "data_buffer_size=16G" >> $CUBRID/conf/cubrid.conf

RUN cubrid server start ann && cubrid broker start && \
    csql -u dba ann -c "CREATE USER ann;" && \
    csql -u dba ann -c "ALTER USER ann PASSWORD 'ann';"

ENV CUBRID /tmp/cubrid_install
ENV CUBRID_DATABASES $CUBRID/databases
ENV LD_LIBRARY_PATH $CUBRID/lib:$CUBRID/cci/lib:$LD_LIBRARY_PATH
ENV SHLIB_PATH $LD_LIBRARY_PATH
ENV LIBPATH $LD_LIBRARY_PATH
ENV PATH $CUBRID/bin:$PATH

ENV ANN_BENCHMARKS_CUB_USER ann
ENV ANN_BENCHMARKS_CUB_PASSWORD ann
ENV ANN_BENCHMARKS_CUB_DBNAME ann
ENV ANN_BENCHMARKS_CUB_HOST localhost
ENV ANN_BENCHMARKS_CUB_PORT 33000
