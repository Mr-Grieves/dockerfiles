FROM tensorflow/tensorflow:1.9.0-rc1-gpu-py3

RUN apt-get update && \
	apt-get install -y \
	libsm6 \
	libxext6 \
	python3-tk

RUN pip3 install \
	tensorlayer \
	scikit-image \
	SimpleITK \
	opencv-python
	
