FROM ubuntu:12.04

RUN apt-get update
RUN apt-get install -y git
RUN mkdir -p /root/.ssh/
ADD ./id_rsa /root/.ssh/id_rsa
RUN ssh-keyscan git.assembla.com >> /root/.ssh/known_hosts
#RUN ssh -v git@git.assembla.com
RUN git clone git@git.assembla.com:ece-rcl/quantitative-ultrasound-server.git 
