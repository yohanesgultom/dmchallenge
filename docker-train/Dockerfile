# Docker image for training

FROM nvidia/cuda:7.5-cudnn5-runtime

# configure cuda
RUN echo("configure cuda path..")
COPY .bashrc /root
RUN source /root/.bashrc

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

RUN curl -qsSLkO \
    https://repo.continuum.io/miniconda/Miniconda-latest-Linux-`uname -p`.sh \
  && bash Miniconda-latest-Linux-`uname -p`.sh -b \
  && rm Miniconda-latest-Linux-`uname -p`.sh

ENV PATH=/root/miniconda2/bin:$PATH

RUN conda install -y \
    h5py \
    pandas \
    scikit-learn \
    theano \
  && conda clean --yes --tarballs --packages --source-cache \
  && pip install --upgrade pip \
  && pip install --upgrade -I setuptools \
  && pip install --upgrade keras

# configure keras for Theano
RUN echo("configure Keras with Theano..")
RUN mkdir /root/.keras
COPY keras.json /root/.keras
COPY .theanorc /root

# copy training and test files
RUN echo("copy training & test files..")
COPY train.py /train.py
COPY train.sh /train.sh
COPY test.py /test.py
COPY test.sh /test.sh
COPY vgg16_weights.h5 /vgg16_weights.h5

RUN echo("chmod training & test files..")
RUN chmod 777 /train.py
RUN chmod 777 /train.sh
RUN chmod 777 /test.py
RUN chmod 777 /test.sh
RUN chmod 777 /vgg16_weights.h5

RUN echo("complete.")
