# Docker image for training

FROM nvidia/cuda:7.5-cudnn4-devel

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    ca-certificates \
    time \
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
    pytables \
  && conda clean --yes --tarballs --packages --source-cache \
  && pip install --upgrade pip \
  && pip install --upgrade -I setuptools \
  && pip install git+git://github.com/fchollet/keras.git

# configure keras for Theano
RUN echo "configure Keras with Theano.."
RUN mkdir /root/.keras
RUN mkdir /root/.keras/models
COPY keras.json /root/.keras
COPY vgg16_weights_th_dim_ordering_th_kernels_notop.h5 /root/.keras/models
COPY .theanorc /root

# copy training and test files
RUN echo "copy training & test files.."
COPY train.py /train.py
COPY train.sh /train.sh
COPY test.py /test.py
COPY test.sh /test.sh
COPY config /config

RUN echo "chmod training & test files.."
RUN chmod 777 /train.py
RUN chmod 777 /train.sh
RUN chmod 777 /test.py
RUN chmod 777 /test.sh
RUN chmod 777 /config

RUN echo "complete."
