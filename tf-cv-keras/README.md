lasagne-cpu
============

Ubuntu Core 16.04 + CUDA 9.0 + Python 3.5 + Tensorflow 1.8.0 + OpenCV 3.4.1 + Keras 

Usage
-----
Use nvidia-docker: ``nvidia-docker run -it ubcrcl/tf-cv-keras``

To access port 50051, add option: ``-p 50051:50051``

To mount your home directory inside the container, add option: ``--volume=$HOME:/workspace``

