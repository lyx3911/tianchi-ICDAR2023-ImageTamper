FROM ubuntu:20.04
MAINTAINER liyingxuan 913797866@qq.com

RUN  apt-get update
RUN  apt-get upgrade -y

# Install python3
RUN apt-get install -y python3.7
RUN apt-get install -y libgl1-mesa-dev
RUN DEBIAN_FRONTEND="noninteractive" apt-get install -y libglib2.0-dev

ADD . /
WORKDIR /

# Install pip
RUN apt-get install -y python3-pip 
RUN pip install --upgrade pip
RUN chmod +x requirements.txt
RUN pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
