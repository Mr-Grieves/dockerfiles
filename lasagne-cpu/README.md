lasagne-cpu
============

Ubuntu Core 14.04 + [Theano](http://www.deeplearning.net/software/theano/) + [Lasagne](http://lasagne.readthedocs.org/) + [GRPC](https://grpc.io/docs/tutorials/basic/python.html)


Usage
-----
Use regular Docker: ``docker run -it ubcrcl/lasagne-cpu``

To access port 50051, add option: ``-p 50051:50051``

To mount pwd inside the container, add option: ``--volume=$HOME:/workspace

