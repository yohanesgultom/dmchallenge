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
    pytables \
  && conda clean --yes --tarballs --packages --source-cache \
  && pip install --upgrade pip \
  && pip install --upgrade -I setuptools \
  && pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git \
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
COPY *.py /
COPY *.sh /

RUN echo "chmod training & test files.."
RUN chmod 777 /*.py
RUN chmod 777 /*.sh

RUN echo "complete."
