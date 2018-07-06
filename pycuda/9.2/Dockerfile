# Start with NVidia CUDA base image
FROM nvidia/cuda:9.2-base-ubuntu16.04

RUN apt-get update

RUN apt-get install -y --no-install-recommends \
	build-essential\
	git\
	vim\
	wget\
	unzip\
	python3-dev\
	python3-numpy\
	python3-pip\
	python3-tk\
	python3-setuptools

RUN echo "keyboard-configuration  keyboard-configuration/layout   select  English (US)" | debconf-set-selections &&\
	DEBIAN_FRONTEND=noninteractive \
	apt-get install -y --no-install-recommends \
	cuda\
	python3-pycuda

RUN pip3 --no-cache-dir install \
	numpy\
	scipy\
	matplotlib\
	tifffile\
	tqdm

RUN echo "Hello world!"
