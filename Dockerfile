FROM ubuntu:bionic
ENV TZ=Asia/Shanghai
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y apt-transport-https ca-certificates gpg-agent curl software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa \
    && curl https://mirrors.aliyun.com/docker-ce/linux/ubuntu/gpg -o /tmp/gpg.key && apt-key add /tmp/gpg.key \
    && add-apt-repository "deb [arch=amd64] https://mirrors.aliyun.com/docker-ce/linux/ubuntu $(lsb_release -cs) stable" \
    && apt-get update && apt-get install -y cmake python3.8 && ln -s /usr/bin/python3.8 /usr/bin/python \
    && apt-get install -y python3-pip python3.8-dev \
    && python -m pip install -U pip \
    && rm -f /usr/bin/pip && ln -s /usr/bin/pip3 /usr/bin/pip
RUN apt-get update && apt-get install -y python3-opencv libglib2.0-0 libsm6 libxext6 libxrender-dev


RUN pip install -U pip && pip install torch==1.11.0 cnocr==2.2
