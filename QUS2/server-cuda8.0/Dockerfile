# Install Bazel, https://bazel.build/versions/master/docs/install.html#ubuntu
FROM nvidia/cuda:8.0-cudnn5-devel

### Install Prerequesites ###
RUN apt-get update \
  && apt-get install -y \  
  build-essential \
  openjdk-8-jdk \
  git \
  vim \
  tar \
  wget \
  python-numpy \
  python-dev \
  python-pip \
  python-wheel \
  pkg-config \
  zip \
  unzip \
  g++ \
  zlib1g-dev \
  autotools-dev \
  libicu-dev \
  libbz2-dev \
  libboost-all-dev \
  && rm -rf /var/lib/apt/lists/*

RUN pip install \
  six \
  numpy \
  wheel \
  mock
  
  
### Install Bazel ###
WORKDIR /

RUN wget -O bazel-0.4.5-installer-linux-x86_64.sh https://github.com/bazelbuild/bazel/releases/download/0.4.5/bazel-0.4.5-installer-linux-x86_64.sh

RUN chmod +x bazel-0.4.5-installer-linux-x86_64.sh

RUN ./bazel-0.4.5-installer-linux-x86_64.sh \
  && rm bazel-0.4.5-installer-linux-x86_64.sh
  
RUN echo "startup --max_idle_secs=100000000" >> ~/.bazelrc


### Install Boost ###
WORKDIR /

RUN apt-get update \
  && apt-get install -y
  
RUN wget -O boost_1_58_0.tar.gz https://sourceforge.net/projects/boost/files/boost/1.58.0/boost_1_58_0.tar.gz/download \
  && tar xzvf boost_1_58_0.tar.gz

WORKDIR boost_1_58_0

RUN ./bootstrap.sh --prefix=/boost/ \
  && ./b2 install \
  && rm /boost_1_58_0.tar.gz


### Build QUS + Tensorflow from sources ###
ENV PYTHON_BIN_PATH=/usr/bin/python \
  PYTHON_LIB_PATH=/usr/local/lib/python2.7/dist-packages \
  TF_NEED_MKL=0 \
  TF_NEED_JEMALLOC=0 \
  TF_NEED_GCP=0 \
  TF_NEED_HDFS=0 \
  TF_NEED_S3=0 \
  TF_NEED_KAFKA=0 \
  TF_NEED_GDR=0 \
  TF_NEED_VERBS=0 \
  TF_ENABLE_XLA=0 \
  TF_NEED_OPENCL=0 \
  TF_NEED_OPENCL_SYCL=0 \
  TF_NEED_CUDA=1 \
  TF_CUDA_VERSION='8.0' \
  CUDA_TOOLKIT_PATH=/usr/local/cuda \
  TF_CUDNN_VERSION='5' \
  CUDNN_INSTALL_PATH=/usr/lib/x86_64-linux-gnu \
  TF_CUDA_CLANG=0 \
  GCC_HOST_COMPILER_PATH='/usr/bin/gcc' \ 
  TF_NEED_TENSORRT=0 \
  TF_NCCL_VERSION=1 \
  TF_CUDA_COMPUTE_CAPABILITIES='6.1' \
  LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64 \
  TF_NEED_MPI=0 \
  TF_SET_ANDROID_WORKSPACE=0 \ 
  CC_OPT_FLAGS=" "
  

RUN mkdir -p /root/.ssh/

ADD id_rsa /root/.ssh/id_rsa

RUN ssh-keyscan git.assembla.com >> /root/.ssh/known_hosts

WORKDIR /

RUN git clone git@git.assembla.com:ece-rcl/quantitative-ultrasound-server.git -b prostate-purang29

WORKDIR quantitative-ultrasound-server

RUN ./configure

RUN bazel build --config=cuda QUSWrapper/main:qus_main

