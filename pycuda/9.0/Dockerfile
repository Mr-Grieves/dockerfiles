# Start with NVidia CUDA base image
FROM nvidia/cuda:9.0-devel-ubuntu16.04

RUN apt-get update

RUN apt-get install -y --no-install-recommends \
	build-essential\
	git\
	vim\
	tar\
	wget\
	unzip\
	python3-dev\
	python3-pip\
	python3-numpy\
	python3-tk\
	python3-setuptools

RUN pip3 --no-cache-dir install \
	numpy\
	scipy\
	matplotlib\
	tifffile\
	tqdm

WORKDIR /

RUN wget https://files.pythonhosted.org/packages/b3/30/9e1c0a4c10e90b4c59ca7aa3c518e96f37aabcac73ffe6b5d9658f6ef843/pycuda-2017.1.1.tar.gz \
	&& tar xfz pycuda-2017.1.1.tar.gz

WORKDIR /pycuda-2017.1.1	

ENV CUDA_INC_DIR=i/usr/local/cuda-9.0/targets/x86_64-linux/include

RUN python3 configure.py --cuda-root=/usr/local/cuda-9.0/targets/x86_64-linux \
	&& make install


