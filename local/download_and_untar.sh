#!/usr/bin/env bash

# This is my own work (fs2776).
# It downloads and untar the data file.

data=$1
version=$2

mkdir -p $data
pushd $data

if [ ! -f speech_commands_v0.0${version}.tar.gz ]; then
    wget http://download.tensorflow.org/data/speech_commands_v0.0${version}.tar.gz || exit 1
    tar xf speech_commands_v0.0${version}.tar.gz
  fi
