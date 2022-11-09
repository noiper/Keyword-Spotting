#!/usr/bin/env bash

data=$1

mkdir -p $data
pushd $data

if [ ! -f speech_commands_v0.01.tar.gz ]; then
    wget http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz || exit 1
    tar xf speech_commands_v0.01.tar.gz
  fi
