FROM  ubuntu:18.04

ENV LANG C.UTF-8

RUN apt-get update &&\
    apt-get -y install build-essential yasm nasm cmake unzip git wget \
    sysstat libtcmalloc-minimal4 pkgconf autoconf libtool flex bison \
    python3 python3-pip python3-dev python3-setuptools \
    libsm6 libxext6 libxrender1 libssl-dev libx264-dev &&\
    ln -s /usr/bin/python3 /usr/bin/python &&\
    ln -s /usr/bin/pip3 /usr/bin/pip &&\
    apt-get clean &&\
    apt-get autoremove &&\
    rm -rf /var/lib/apt/lists/* &&\
    rm -rf /var/cache/apt/archives/*

# if you work with GPU use nvidia headers

#RUN git clone -b sdk/8.2 --single-branch https://git.videolan.org/git/ffmpeg/nv-codec-headers.git &&\
#    cd nv-codec-headers && make install &&\
#    cd .. && rm -rf nv-codec-headers

#Use this installation of ffmpeg to get encoders and decoders for video
RUN apt-get update
#https://linuxconfig.org/install-ffmpeg-on-ubuntu-18-04-bionic-beaver-linux
RUN git clone --depth 1 -b release/4.2 --single-branch https://github.com/FFmpeg/FFmpeg.git &&\
    cd FFmpeg &&\
    mkdir ffmpeg_build && cd ffmpeg_build &&\
    ../configure \
#    --enable-cuda \ #for nvidia
#    --enable-cuvid \
    --enable-shared \
    --disable-static \
    --disable-doc \
    --extra-cflags=-I/usr/local/cuda/include \
    --extra-ldflags=-L/usr/local/cuda/lib64 \
    --enable-nonfree \
    --enable-gpl \
    --enable-openssl \
    --enable-libx264 \
    --extra-libs=-lpthread \
    --nvccflags="-gencode arch=compute_75,code=sm_75" &&\
    make -j$(nproc) && make install && ldconfig &&\
    cd ../.. && rm -rf FFmpeg


ENV PYTHONPATH $PYTHONPATH:/workdir/src
ENV TORCH_HOME=/workdir/data/.torch


COPY . /workdir

WORKDIR /workdir