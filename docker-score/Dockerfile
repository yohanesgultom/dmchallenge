# Docker image for scoring

FROM nvidia/cuda:7.5-cudnn4-devel

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    ca-certificates \
    time \
    libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

RUN curl -qsSLkO \
    https://repo.continuum.io/miniconda/Miniconda-latest-Linux-`uname -p`.sh \
  && bash Miniconda-latest-Linux-`uname -p`.sh -b \
  && rm Miniconda-latest-Linux-`uname -p`.sh

ENV PATH=/root/miniconda2/bin:$PATH

RUN conda install -y \
    h5py \
    scipy \
    theano \
    opencv \
    pytables \
  && conda clean --yes --tarballs --packages --source-cache \
  && pip install --upgrade pip \
  && pip install --upgrade -I setuptools \
  && pip install --upgrade pydicom
  && pip install git+git://github.com/fchollet/keras.git

# configure keras for Theano
RUN echo "configure Keras with Theano.."
RUN mkdir /root/.keras
RUN mkdir /root/.keras/models
COPY keras.json /root/.keras
COPY .theanorc /root

# copy scoring files
RUN echo "copy training & test files.."
COPY sc1_infer.py /sc1_infer.py
COPY sc1_infer.sh /sc1_infer.sh
COPY sc2_infer.py /sc2_infer.py
COPY sc2_infer.sh /sc2_infer.sh

RUN echo "chmod training & test files.."
RUN chmod 777 /sc1_infer.py
RUN chmod 777 /sc1_infer.sh
RUN chmod 777 /sc2_infer.py
RUN chmod 777 /sc2_infer.sh

RUN echo "complete."
