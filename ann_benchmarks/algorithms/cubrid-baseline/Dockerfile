FROM ann-benchmarks

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        tzdata \
        software-properties-common \
        wget \
        git \
        curl \
        build-essential \
        cmake \
        ninja-build \
        flex \
        libfl-dev \
        libtool \
        m4 \
        ncurses-dev \
        libelf-dev \
        libopenblas-dev \
        liblapack-dev \
        openjdk-8-jdk \
    && rm -rf /var/lib/apt/lists/*

RUN add-apt-repository ppa:ubuntu-toolchain-r/test && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc-9 \
        g++-9 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 90 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 90 \
    && rm -rf /var/lib/apt/lists/*

ENV MAKEFLAGS=-j80

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

ARG GIT_REMOTE_URL=https://github.com/CUBRID/cubrid.git
ARG GIT_BRANCH=cubvec/cubvec
ARG GIT_COMMIT=
ARG PRESET=release_avx2

RUN git clone ${GIT_REMOTE_URL} /tmp/cubrid
RUN git clone https://github.com/cubrid/cubrid-python /tmp/cubrid-python

RUN cd /tmp/cubrid

USER root

RUN BUILD_DIR="build_preset_${PRESET}" && \
    INSTALL_DIR="/tmp/cubrid_install" && \
    cd /tmp/cubrid && \
    git fetch origin && \
    if [ -n "$GIT_COMMIT" ]; then \
        echo "Checkout commit: $GIT_COMMIT" && \
        git checkout "$GIT_COMMIT"; \
    else \
        echo "Checkout branch: origin/$GIT_BRANCH" && \
        git checkout --detach "origin/$GIT_BRANCH"; \
    fi && \
    git submodule update --init cubrid-cci && \
    cmake --preset "$PRESET" -DENABLE_SYSTEMTAP=OFF && \
    cmake --build --preset "$PRESET" -j4 && \
    cmake --install "$BUILD_DIR" --prefix "$INSTALL_DIR"

# do we need to run as cubrid?
# USER cubrid
RUN cd /tmp/cubrid-python && \
    git checkout develop && \
    git submodule set-url cci-src https://github.com/CUBRID/cubrid-cci.git && \
    git submodule update --init --recursive && \
    ln -sf /bin/bash /bin/sh && \
    python setup.py build && \
    python setup.py install

ENV CUBRID=/tmp/cubrid_install
ENV CUBRID_DATABASES=${CUBRID}/databases
ENV LD_LIBRARY_PATH=${CUBRID}/lib:${CUBRID}/cci/lib:${LD_LIBRARY_PATH}
ENV SHLIB_PATH=${LD_LIBRARY_PATH}
ENV LIBPATH=${LD_LIBRARY_PATH}
ENV PATH=${CUBRID}/bin:$PATH

ENV ANN_BENCHMARKS_CUB_USER=ann
ENV ANN_BENCHMARKS_CUB_PASSWORD=ann
ENV ANN_BENCHMARKS_CUB_DBNAME=ann
ENV ANN_BENCHMARKS_CUB_HOST=localhost
ENV ANN_BENCHMARKS_CUB_SERVER_PORT=1523
ENV ANN_BENCHMARKS_CUB_DB_PATH=/tmp/ann

# broker port
ENV ANN_BENCHMARKS_CUB_PORT=33000
# number of cas
ENV ANN_BENCHMARKS_CUB_NUM_CAS=3
# required to prevent CAS memory hard limit, due to faiss 3rdparty dependency
ENV OPENBLAS_NUM_THREADS=1

RUN mkdir -p ${ANN_BENCHMARKS_CUB_DB_PATH}/initdb && cd ${ANN_BENCHMARKS_CUB_DB_PATH}/initdb && \
    cubrid createdb --db-volume-size=3G --log-volume-size=1G ${ANN_BENCHMARKS_CUB_DBNAME} en_US.utf8 && \
    echo "[@${ANN_BENCHMARKS_CUB_DBNAME}]" >> $CUBRID/conf/cubrid.conf && \
    echo "data_buffer_size=16G" >> $CUBRID/conf/cubrid.conf && \
    echo "log_buffer_size=256M" >> $CUBRID/conf/cubrid.conf && \
    echo "sort_buffer_size=64M" >> $CUBRID/conf/cubrid.conf && \
    echo "double_write_buffer_size=0" >> $CUBRID/conf/cubrid.conf && \
    echo "stored_procedure=no" >> $CUBRID/conf/cubrid.conf && \
    echo "max_clients=10" >> $CUBRID/conf/cubrid.conf

RUN sed -i "s/^cubrid_port_id *= *.*/cubrid_port_id = ${ANN_BENCHMARKS_CUB_SERVER_PORT}/" $CUBRID/conf/cubrid.conf && \
    cubrid server start ${ANN_BENCHMARKS_CUB_DBNAME} && \
    until cubrid server status | grep -q "Server ${ANN_BENCHMARKS_CUB_DBNAME}"; do \
        echo "Waiting for '${ANN_BENCHMARKS_CUB_DBNAME}' DB to start..."; \
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
    csql -u dba ${ANN_BENCHMARKS_CUB_DBNAME} -c "CREATE USER ${ANN_BENCHMARKS_CUB_USER};" && \
    csql -u dba ${ANN_BENCHMARKS_CUB_DBNAME} -c "ALTER USER ${ANN_BENCHMARKS_CUB_USER} PASSWORD '${ANN_BENCHMARKS_CUB_PASSWORD}';"
