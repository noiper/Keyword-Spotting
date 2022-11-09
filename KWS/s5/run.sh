#!/usr/bin/env bash

# Location for the data
data=./db/

. ./cmd.sh
. ./path.sh

stage=0
nj=40

. utils/parse_options.sh

set -euo pipefail

if [ $stage -eq 0 ]; then
  echo a
  local/download_and_untar.sh $data
fi

if [ $stage -eq 1 ]; then
  local/prepare_data.sh $data ./data
fi

if [ $stage -eq 2 ]; then
  for x in train validation test; do 
    steps/make_mfcc.sh --nj $nj data/$x exp/make_mfcc/$x mfcc
    steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x mfcc
    utils/fix_data_dir.sh data/$x
  done
fi
