tf-cv-itk
============

Ubuntu Core 16.04 + CUDA 9.0 + Python 3.5 + Tensorflow 1.9.0 + OpenCV (pip) + SimpleITK + tensorlayer 

Usage
-----
Use nvidia-docker: ``nvidia-docker run -it ubcrcl/tf-cv-itk bash``

To access port 50051, add option: ``-p 50051:50051``

To mount your home directory inside the container, add option: ``--volume=$HOME:/workspace``

To specify which GPU to isolate, add prefix: ``NV_GPU=<X>`` e.g. ``NV_GPU=1 nvidia-docker run ...``
