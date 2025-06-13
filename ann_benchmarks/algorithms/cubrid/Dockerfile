FROM ann-benchmarks

# https://github.com/pgvector/pgvector/blob/master/Dockerfile

RUN git clone https://github.com/cubrid/cubrid /tmp/cubrid
RUN git clone https://github.com/cubrid/cubrid-python /tmp/cubrid-python

RUN DEBIAN_FRONTEND=noninteractive apt-get -y install tzdata
RUN apt-get update && apt-get install -y software-properties-common && add-apt-repository ppa:ubuntu-toolchain-r/test && apt-get update
RUN apt-get install -y --no-install-recommends build-essential
RUN apt-get install -y --no-install-recommends wget git cmake systemtap systemtap-sdt-dev flex ninja-build
RUN apt-get install -y --no-install-recommends gcc-9 g++-9 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 90 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 90
RUN apt-get install -y --no-install-recommends libopenblas-dev liblapack-dev
RUN apt-get install -y --no-install-recommends openjdk-8-jdk
RUN apt-get install -y --no-install-recommends ncurses-dev
RUN apt-get install -y --no-install-recommends systemtap-sdt-dev
RUN apt-get install -y --no-install-recommends libelf-dev
RUN apt-get install -y --no-install-recommends libtool m4 curl
RUN apt-get install -y --no-install-recommends libfl-dev # flexlexer.h

ENV MAKEFLAGS -j80

# Install Bison 3.0.5 manually
ENV BISON_VERSION=3.0.5
RUN curl -L https://ftp.gnu.org/gnu/bison/bison-${BISON_VERSION}.tar.gz | tar xzvf - && \
    cd bison-${BISON_VERSION} && \
    ./configure --prefix=/usr && \
    make -j$(nproc) && \
    make install && \
    cd .. && \
    rm -rf bison-${BISON_VERSION}


ENV DEBIAN_FRONTEND=noninteractive
ENV CMAKE_VERSION=3.26.3

# Download and install CMake
RUN curl -L https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz | \
    tar --strip-components=1 -xz -C /usr --wildcards 'cmake-*/*'

RUN cd /tmp/cubrid && git fetch --all --prune

USER root
RUN PRESET=release_avx2 && \
    BUILD_DIR="build_preset_${PRESET}" && \
    INSTALL_DIR="/tmp/cubrid_install" && \
    cd /tmp/cubrid && \
    git checkout cubvec/experiment && \
    git submodule update --init cubrid-cci && \
    cmake --preset "$PRESET" && \
    cmake --build --preset "$PRESET" -j$(nproc) && \
    cmake --install "$BUILD_DIR" --prefix "$INSTALL_DIR"

# do we need to run as cubrid?
# USER cubrid
RUN cd /tmp/cubrid-python && \
    git submodule set-url cci-src https://github.com/CUBRID/cubrid-cci.git && \
    git submodule update --init --recursive && \
    cd ./cci-src && ./build.sh && cd .. && \
    python setup.py build && \
    python setup.py install

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
ENV ANN_BENCHMARKS_CUB_SERVER_PORT 5558

# broker port
ENV ANN_BENCHMARKS_CUB_PORT 33003
# number of cas
ENV ANN_BENCHMARKS_CUB_NUM_CAS 50

RUN cd /tmp && \
    cubrid createdb --db-volume-size=100M --log-volume-size=100M ann en_US.utf8 && \
    echo "[ann]" >> $CUBRID/conf/cubrid.conf && \
    echo "data_buffer_size=16G" >> $CUBRID/conf/cubrid.conf

RUN sed -i "s/^cubrid_port_id *= *.*/cubrid_port_id = ${ANN_BENCHMARKS_CUB_SERVER_PORT}/" $CUBRID/conf/cubrid.conf && \
    cubrid server start ann && \
    until cubrid server status | grep -q "Server ann"; do \
        echo "Waiting for 'ann' DB to start..."; \
        sleep 1; \
    done && \
    awk ' \
      BEGIN { skip=0 } \
      /^\[%query_editor\]/ { skip=1; next } \
      /^\[/ && skip { skip=0 } \
      !skip \
    ' $CUBRID/conf/cubrid_broker.conf > /tmp/cubrid_broker.conf && \
    mv /tmp/cubrid_broker.conf $CUBRID/conf/cubrid_broker.conf && \
    sed -i "s/^MIN_NUM_APPL_SERVER[ \t]*=.*/MIN_NUM_APPL_SERVER = ${ANN_BENCHMARKS_CUB_NUM_CAS}/" $CUBRID/conf/cubrid_broker.conf && \
    sed -i "s/^MAX_NUM_APPL_SERVER[ \t]*=.*/MAX_NUM_APPL_SERVER = ${ANN_BENCHMARKS_CUB_NUM_CAS}/" $CUBRID/conf/cubrid_broker.conf && \
    sed -i "s/^BROKER_PORT[ \t]*=.*/BROKER_PORT = ${ANN_BENCHMARKS_CUB_PORT}/" $CUBRID/conf/cubrid_broker.conf && \
    cubrid broker start && \
    until cubrid broker status | grep -q "% broker1"; do \
        echo "Waiting for broker1 to start..."; \
        sleep 1; \
    done && \
    csql -u dba ann -c "CREATE USER ann;" && \
    csql -u dba ann -c "ALTER USER ann PASSWORD 'ann';"
